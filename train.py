import os
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Let's import our classes
from dataset import LungHistPairedDataset, get_transforms
from model import MOPFN, CoralOrdinalLoss, predict_ordinal_class

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser(description="Train MOPFN on LungHist700")
    parser.add_argument("--csv_path", type=str, default="data/LungHist700/LungHist700.csv", help="Path to metadata CSV")
    parser.add_argument("--data_dir", type=str, default="data/LungHist700", help="Directory where images are extracted")
    parser.add_argument("--epochs", type=type(1), default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=type(1), default=4, help="Batch size for training")
    parser.add_argument("--lr", type=type(1e-5), default=2e-5, help="Learning rate")
    parser.add_argument("--img_size", type=type(224), default=224, help="Input image size")
    parser.add_argument("--seed", type=type(42), default=42, help="Random seed")
    parser.add_argument("--save_path", type=str, default="best_mopfn.pth", help="Path to save best model weights")
    args = parser.parse_args()

    set_seed(args.seed)
    
    # 1. Device Selection (MPS on Apple Silicon, CUDA, or CPU)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU Acceleration (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using NVIDIA GPU Acceleration (CUDA)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Resolve CSV and data dir paths if nested
    csv_path = args.csv_path
    data_dir = args.data_dir
    
    # Let's check where the CSV is
    if not os.path.exists(csv_path):
        # Walk and find it
        found_csv = False
        for root, dirs, files in os.walk("data"):
            for file in files:
                if file.endswith(".csv") and "lunghist" in file.lower():
                    csv_path = os.path.join(root, file)
                    data_dir = root
                    found_csv = True
                    print(f"Auto-resolved metadata path to: {csv_path}")
                    break
            if found_csv:
                break
        if not found_csv:
            print(f"Error: Could not find CSV at {args.csv_path}. Please run download_data.py first.")
            return

    # 2. Stratified Patient-Wise Splitting (to prevent data leakage and balance clinical classes)
    df_meta = pd.read_csv(csv_path)
    df_meta.columns = [c.strip().lower() for c in df_meta.columns]
    df_meta['subclass'] = df_meta['subclass'].fillna('').astype(str).str.lower().str.strip()
    df_meta['superclass'] = df_meta['superclass'].astype(str).str.lower().str.strip()
    
    # Classify each patient into a clean stratum
    patient_strata = {}
    patient_col = 'patient_id' if 'patient_id' in df_meta.columns else 'patient'
    for patient, group in df_meta.groupby(patient_col):
        sups = group['superclass'].unique()
        if 'aca' in sups and 'scc' in sups:
            patient_strata[patient] = 'mixed'
        elif 'aca' in sups:
            patient_strata[patient] = 'aca'
        elif 'scc' in sups:
            patient_strata[patient] = 'scc'
        else:
            patient_strata[patient] = 'nor'
            
    # Group patient IDs by strata
    strata_groups = {'nor': [], 'aca': [], 'scc': [], 'mixed': []}
    for p, s in patient_strata.items():
        strata_groups[s].append(p)
        
    # Shuffle each stratum with the seed for reproducibility
    for s in strata_groups:
        random.shuffle(strata_groups[s])
        
    # Perform stratified split (approx 80/10/10)
    train_patients, val_patients, test_patients = [], [], []
    
    # Stratified distribution:
    # nor: 5 patients -> 3 train, 1 val, 1 test
    # aca: 19 patients -> 15 train, 2 val, 2 test
    # scc: 19 patients -> 15 train, 2 val, 2 test
    # mixed: 2 patients -> 1 train, 1 val (or test)
    for s, p_list in strata_groups.items():
        n = len(p_list)
        if n >= 3:
            n_val = max(1, int(0.1 * n))
            n_test = max(1, int(0.1 * n))
            n_train = n - n_val - n_test
            
            train_patients.extend(p_list[:n_train])
            val_patients.extend(p_list[n_train:n_train+n_val])
            test_patients.extend(p_list[n_train+n_val:])
        else:
            # Handle small strata like 'mixed'
            train_patients.extend(p_list[:1])
            if n > 1:
                val_patients.extend(p_list[1:])
                
    print(f"Total patients in dataset: {len(patient_strata)}")
    print(f"Train patients: {len(train_patients)} | Val patients: {len(val_patients)} | Test patients: {len(test_patients)}")
    print(f"Test split patients: {test_patients}")

    # 3. Create Datasets & Dataloaders
    train_dataset = LungHistPairedDataset(
        csv_path=csv_path,
        data_dir=data_dir,
        patient_ids=train_patients,
        transform=get_transforms(args.img_size, 'train'),
        split='train'
    )
    
    val_dataset = LungHistPairedDataset(
        csv_path=csv_path,
        data_dir=data_dir,
        patient_ids=val_patients,
        transform=get_transforms(args.img_size, 'val'),
        split='val'
    )
    
    # We set pin_memory=True if running on GPU for faster transfers
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 4. Initialize MOPFN Model & Multi-Task Losses
    model = MOPFN(pretrained=True).to(device)
    
    criterion_malignancy = nn.CrossEntropyLoss()
    criterion_subtype = nn.CrossEntropyLoss(reduction='none')
    
    # Custom Coral Ordinal Loss
    criterion_ordinal = CoralOrdinalLoss()
    # Let's create an element-wise version of Coral loss for masking
    bce_elementwise = nn.BCEWithLogitsLoss(reduction='none')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float('inf')
    
    print("\n--- Starting MOPFN Training Loop ---")
    for epoch in range(1, args.epochs + 1):
        # 5. Training Phase
        model.train()
        train_loss = 0.0
        train_mal_correct = 0
        train_mal_total = 0
        
        for batch_idx, batch in enumerate(train_loader):
            img_20 = batch['image_20x'].to(device)
            img_40 = batch['image_40x'].to(device)
            is_mal = batch['is_malignant'].to(device)
            subtype = batch['subtype'].to(device)
            diff = batch['differentiation'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(img_20, img_40)
            
            # Loss A: Malignancy
            loss_mal = criterion_malignancy(outputs['malignancy'], is_mal)
            
            # Loss B: Subtype (Masked: only computed for malignant samples)
            loss_sub_raw = criterion_subtype(outputs['subtype'], subtype)
            loss_sub_masked = (loss_sub_raw * is_mal.float()).sum() / (is_mal.float().sum() + 1e-8)
            
            # Loss C: Ordinal Differentiation (Masked: only computed for malignant samples)
            # Expand diff target for CORAL
            binary_diff = torch.zeros_like(outputs['ordinal'])
            for i in range(binary_diff.size(1)):
                binary_diff[:, i] = (diff > i).float()
            
            loss_ord_raw = bce_elementwise(outputs['ordinal'], binary_diff).mean(dim=-1)
            loss_ord_masked = (loss_ord_raw * is_mal.float()).sum() / (is_mal.float().sum() + 1e-8)
            
            # Joint Multi-Task Loss
            loss = loss_mal + loss_sub_masked + loss_ord_masked
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Accuracies
            pred_mal = outputs['malignancy'].argmax(dim=-1)
            train_mal_correct += (pred_mal == is_mal).sum().item()
            train_mal_total += is_mal.size(0)
            
            if (batch_idx + 1) % max(1, len(train_loader) // 3) == 0:
                print(f"Epoch {epoch}/{args.epochs} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f} (Mal: {loss_mal.item():.3f}, Sub: {loss_sub_masked.item():.3f}, Ord: {loss_ord_masked.item():.3f})")
                
        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)
        train_mal_acc = train_mal_correct / train_mal_total
        
        # 6. Validation Phase
        model.eval()
        val_loss = 0.0
        val_mal_correct = 0
        val_mal_total = 0
        val_sub_correct = 0
        val_sub_total = 0
        val_ord_correct = 0
        val_ord_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                img_20 = batch['image_20x'].to(device)
                img_40 = batch['image_40x'].to(device)
                is_mal = batch['is_malignant'].to(device)
                subtype = batch['subtype'].to(device)
                diff = batch['differentiation'].to(device)
                
                outputs = model(img_20, img_40)
                
                loss_mal = criterion_malignancy(outputs['malignancy'], is_mal)
                
                loss_sub_raw = criterion_subtype(outputs['subtype'], subtype)
                loss_sub_masked = (loss_sub_raw * is_mal.float()).sum() / (is_mal.float().sum() + 1e-8)
                
                binary_diff = torch.zeros_like(outputs['ordinal'])
                for i in range(binary_diff.size(1)):
                    binary_diff[:, i] = (diff > i).float()
                
                loss_ord_raw = bce_elementwise(outputs['ordinal'], binary_diff).mean(dim=-1)
                loss_ord_masked = (loss_ord_raw * is_mal.float()).sum() / (is_mal.float().sum() + 1e-8)
                
                loss = loss_mal + loss_sub_masked + loss_ord_masked
                val_loss += loss.item()
                
                # Malignancy Acc
                pred_mal = outputs['malignancy'].argmax(dim=-1)
                val_mal_correct += (pred_mal == is_mal).sum().item()
                val_mal_total += is_mal.size(0)
                
                # Subtype Acc (Only count malignant)
                pred_sub = outputs['subtype'].argmax(dim=-1)
                mal_mask = (is_mal == 1)
                if mal_mask.sum() > 0:
                    val_sub_correct += (pred_sub[mal_mask] == subtype[mal_mask]).sum().item()
                    val_sub_total += mal_mask.sum().item()
                    
                # Ordinal Differentiation Acc (Only count malignant)
                pred_ord = predict_ordinal_class(outputs['ordinal'])
                if mal_mask.sum() > 0:
                    val_ord_correct += (pred_ord[mal_mask] == diff[mal_mask]).sum().item()
                    val_ord_total += mal_mask.sum().item()
                    
        avg_val_loss = val_loss / len(val_loader)
        val_mal_acc = val_mal_correct / val_mal_total
        val_sub_acc = val_sub_correct / max(1, val_sub_total)
        val_ord_acc = val_ord_correct / max(1, val_ord_total)
        
        print(f"--> Epoch {epoch} Results:")
        print(f"    [TRAIN] Loss: {avg_train_loss:.4f} | Malignancy Acc: {train_mal_acc*100:.1f}%")
        print(f"    [VAL]   Loss: {avg_val_loss:.4f} | Malignancy Acc: {val_mal_acc*100:.1f}% | Subtype Acc: {val_sub_acc*100:.1f}% | Ordinal Acc: {val_ord_acc*100:.1f}%")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), args.save_path)
            print(f"    *** New Best Model Saved to {args.save_path} (Val Loss: {best_val_loss:.4f}) ***")
            
    print("\n--- Training Finished! ---")
    print(f"Best validation loss achieved: {best_val_loss:.4f}")
    
    # Save the split lists for evaluate.py to use the same split
    split_info = {
        'train_patients': train_patients,
        'val_patients': val_patients,
        'test_patients': test_patients
    }
    torch.save(split_info, "split_info.pth")
    print("Saved patient splits to split_info.pth")

if __name__ == "__main__":
    main()
