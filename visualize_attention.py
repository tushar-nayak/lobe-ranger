import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F

from dataset import LungHistPairedDataset, get_transforms
from model import MOPFN

def denormalize_image(tensor):
    """Converts a normalized PyTorch image tensor back to a displayable NumPy array"""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = std * img + mean
    img = np.clip(img, 0, 1)
    return img

def main():
    # 1. Device Selection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    model_path = "best_mopfn.pth"
    csv_path = "data/data/data.csv"
    data_dir = "data/data"
    
    # Auto-resolve metadata paths if nested differently
    if not os.path.exists(csv_path):
        found_csv = False
        for root, dirs, files in os.walk("data"):
            for file in files:
                if file.endswith(".csv"):
                    csv_path = os.path.join(root, file)
                    data_dir = root
                    found_csv = True
                    print(f"Auto-resolved metadata path to: {csv_path}")
                    break
            if found_csv:
                break
        if not found_csv:
            print("Error: Could not find CSV metadata.")
            return
            
    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}. Please train first.")
        return
        
    # Load dataset
    dataset = LungHistPairedDataset(
        csv_path=csv_path,
        data_dir=data_dir,
        transform=get_transforms(224, 'val'),
        split='all'
    )
    
    # Initialize model
    model = MOPFN(pretrained=False, fusion_mode='mopfn').to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Find a malignant sample (preferably Adenocarcinoma or SCC) for clear visualization
    sample = None
    for i in range(len(dataset)):
        s = dataset[i]
        if s['is_malignant'].item() == 1:
            sample = s
            break
            
    if sample is None:
        sample = dataset[0]
        
    # Prepare inputs
    img_20_tensor = sample['image_20x'].unsqueeze(0).to(device)
    img_40_tensor = sample['image_40x'].unsqueeze(0).to(device)
    
    # Forward pass to extract attention weights
    with torch.no_grad():
        outputs = model(img_20_tensor, img_40_tensor, return_attention=True)
        
    # Extract attention weights
    # Shape: [batch_size, num_heads, 197, 197]
    attn_20_40 = outputs['attn_20_40']
    attn_40_20 = outputs['attn_40_20']
    
    # Average across heads (dim 1)
    attn_20_40_avg = attn_20_40.mean(dim=1)[0] # [197, 197]
    attn_40_20_avg = attn_40_20.mean(dim=1)[0] # [197, 197]
    
    # Isolate spatial patch-to-patch attention (exclude CLS token at index 0)
    spatial_20_40 = attn_20_40_avg[1:, 1:] # [196, 196]
    spatial_40_20 = attn_40_20_avg[1:, 1:] # [196, 196]
    
    # 20x <- 40x cross attention map (20x queries 40x details)
    # Average attention each 20x patch puts on all 40x patches
    map_20x = spatial_20_40.mean(dim=-1).cpu().numpy() # [196]
    # Average attention all 20x patches put on each 40x patch
    map_40x_focus = spatial_20_40.mean(dim=-2).cpu().numpy() # [196]
    
    # Reshape to [14, 14]
    map_20x = map_20x.reshape(14, 14)
    map_40x_focus = map_40x_focus.reshape(14, 14)
    
    # Normalize
    map_20x = (map_20x - map_20x.min()) / (map_20x.max() - map_20x.min() + 1e-8)
    map_40x_focus = (map_40x_focus - map_40x_focus.min()) / (map_40x_focus.max() - map_40x_focus.min() + 1e-8)
    
    # Denormalize original images for plotting
    img_20_np = denormalize_image(sample['image_20x'])
    img_40_np = denormalize_image(sample['image_40x'])
    
    # Plot premium comparative attention dashboard
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 20x Original
    axes[0, 0].imshow(img_20_np)
    axes[0, 0].set_title("20x Original Image", fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # 20x Attention Heatmap (how much 20x queries 40x)
    im1 = axes[0, 1].imshow(map_20x, cmap='jet')
    axes[0, 1].set_title("20x Spatial Query Focus Map", fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # 20x Overlay
    axes[0, 2].imshow(img_20_np)
    # Upscale map to 224x224
    map_20x_resized = np.array(Image.fromarray((map_20x * 255).astype(np.uint8)).resize((224, 224), Image.Resampling.BILINEAR)) / 255.0
    axes[0, 2].imshow(map_20x_resized, cmap='jet', alpha=0.45)
    axes[0, 2].set_title("20x Cross-Scale Focus Overlay", fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    
    # 40x Original
    axes[1, 0].imshow(img_40_np)
    axes[1, 0].set_title("40x Original Image", fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    # 40x Attention Heatmap (where 20x queries attend in 40x)
    im2 = axes[1, 1].imshow(map_40x_focus, cmap='jet')
    axes[1, 1].set_title("40x Key Attention Map (Attended Areas)", fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    fig.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    # 40x Overlay
    axes[1, 2].imshow(img_40_np)
    map_40x_resized = np.array(Image.fromarray((map_40x_focus * 255).astype(np.uint8)).resize((224, 224), Image.Resampling.BILINEAR)) / 255.0
    axes[1, 2].imshow(map_40x_resized, cmap='jet', alpha=0.45)
    axes[1, 2].set_title("40x Cross-Scale Focus Overlay", fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')
    
    plt.suptitle("Lobe Ranger: Bidirectional Spatial Patch-Token Cross-Scale Attention Maps", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save visualizations
    os.makedirs("docs", exist_ok=True)
    plt.savefig("docs/patch_token_attention.png", dpi=300, bbox_inches='tight')
    plt.savefig("xai_patch_attention.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Successfully generated and saved spatial patch-token attention visualizations!")
    
if __name__ == "__main__":
    main()
