import os
import random
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

from dataset import LungHistPairedDataset, get_transforms
from model import MOPFN

def generate_saliency_map(model, img_20x, img_40x, device):
    """
    Computes input-gradient saliency maps for both 20x and 40x paired images
    with respect to the Malignancy detection score.
    """
    model.eval()
    
    # Enable gradients on the input tensors
    img_20x.requires_grad_()
    img_40x.requires_grad_()
    
    # Forward pass
    outputs = model(img_20x, img_40x)
    logits_malignancy = outputs['malignancy']
    
    # Grab the logit corresponding to 'Malignant' (class 1)
    target_score = logits_malignancy[0, 1]
    
    # Zero all gradients
    model.zero_grad()
    
    # Backward pass to compute gradients with respect to inputs
    target_score.backward()
    
    # Saliency is the absolute value of the input gradients
    saliency_20x = img_20x.grad.data.abs().cpu().numpy()[0]
    saliency_40x = img_40x.grad.data.abs().cpu().numpy()[0]
    
    # Take the maximum across color channels
    saliency_20x = np.max(saliency_20x, axis=0)
    saliency_40x = np.max(saliency_40x, axis=0)
    
    # Normalize to [0, 1]
    saliency_20x = (saliency_20x - saliency_20x.min()) / (saliency_20x.max() - saliency_20x.min() + 1e-8)
    saliency_40x = (saliency_40x - saliency_40x.min()) / (saliency_40x.max() - saliency_40x.min() + 1e-8)
    
    return saliency_20x, saliency_40x

def denormalize_image(tensor):
    """Converts a normalized PyTorch image tensor back to a displayable NumPy array"""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = std * img + mean
    img = np.clip(img, 0, 1)
    return img

def save_xai_visualization(img_20x_tensor, img_40x_tensor, sal_20x, sal_40x, save_path, sample_title=""):
    """Generates and saves a premium 3-column comparative XAI layout"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    img_20x_np = denormalize_image(img_20x_tensor[0])
    img_40x_np = denormalize_image(img_40x_tensor[0])
    
    # Row 0: 20x Magnification (Macro-structure)
    axes[0, 0].imshow(img_20x_np)
    axes[0, 0].set_title("20x Original Image", fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    im_sal_20 = axes[0, 1].imshow(sal_20x, cmap='jet')
    axes[0, 1].set_title("20x Saliency Heatmap", fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    fig.colorbar(im_sal_20, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # Overlay saliency map on top of the original image
    axes[0, 2].imshow(img_20x_np)
    axes[0, 2].imshow(sal_20x, cmap='jet', alpha=0.5)
    axes[0, 2].set_title("20x Diagnostic Focus Overlay", fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Row 1: 40x Magnification (Micro-cytology)
    axes[1, 0].imshow(img_40x_np)
    axes[1, 0].set_title("40x Original Image", fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    im_sal_40 = axes[1, 1].imshow(sal_40x, cmap='jet')
    axes[1, 1].set_title("40x Saliency Heatmap", fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    fig.colorbar(im_sal_40, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    axes[1, 2].imshow(img_40x_np)
    axes[1, 2].imshow(sal_40x, cmap='jet', alpha=0.5)
    axes[1, 2].set_title("40x Diagnostic Focus Overlay", fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')
    
    plt.suptitle(f"Lobe Ranger XAI Interpretability: {sample_title}", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved XAI Interpretability visualization to: {save_path}")

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
    
    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}. Please train first.")
        return
        
    if not os.path.exists("split_info.pth"):
        print("Error: split_info.pth not found. Please run train.py first.")
        return
        
    # Load splits
    split_info = torch.load("split_info.pth")
    test_patients = split_info['test_patients']
    print(f"Running XAI interpretation on test split patients: {test_patients}")
    
    # Load test dataset
    test_dataset = LungHistPairedDataset(
        csv_path=csv_path,
        data_dir=data_dir,
        patient_ids=test_patients,
        transform=get_transforms(224, 'test'),
        split='test'
    )
    
    # Initialize model
    model = MOPFN(pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Grab a malignant and a normal sample from the test set for comparison
    malignant_sample = None
    normal_sample = None
    
    for i in range(len(test_dataset)):
        sample = test_dataset[i]
        is_mal = sample['is_malignant'].item()
        
        if is_mal == 1 and malignant_sample is None:
            malignant_sample = sample
        elif is_mal == 0 and normal_sample is None:
            normal_sample = sample
            
        if malignant_sample is not None and normal_sample is not None:
            break
            
    # Compute and save saliency maps
    if malignant_sample is not None:
        # Prepare inputs
        img_20 = malignant_sample['image_20x'].unsqueeze(0).to(device)
        img_40 = malignant_sample['image_40x'].unsqueeze(0).to(device)
        
        sal_20, sal_40 = generate_saliency_map(model, img_20, img_40, device)
        
        # Resolve subtype name
        sub_map = {0: "Adenocarcinoma", 1: "Squamous Cell Carcinoma"}
        subtype_label = sub_map.get(malignant_sample['subtype'].item(), "Carcinoma")
        
        save_path = "xai_malignant_sample.png"
        save_xai_visualization(
            img_20.detach(), img_40.detach(), sal_20, sal_40, 
            save_path, sample_title=f"Malignant Case ({subtype_label})"
        )
        
    if normal_sample is not None:
        # Reset gradients
        img_20 = normal_sample['image_20x'].unsqueeze(0).to(device)
        img_40 = normal_sample['image_40x'].unsqueeze(0).to(device)
        
        sal_20, sal_40 = generate_saliency_map(model, img_20, img_40, device)
        
        save_path = "xai_normal_sample.png"
        save_xai_visualization(
            img_20.detach(), img_40.detach(), sal_20, sal_40, 
            save_path, sample_title="Normal Pulmonary Structure"
        )
        
if __name__ == "__main__":
    main()
