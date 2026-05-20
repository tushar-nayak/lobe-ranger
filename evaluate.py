import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from dataset import LungHistPairedDataset, get_transforms
from model import MOPFN, predict_ordinal_class

def main():
    parser = argparse.ArgumentParser(description="Evaluate MOPFN on LungHist700 Test Split")
    parser.add_argument("--csv_path", type=str, default="data/LungHist700/LungHist700.csv", help="Path to metadata CSV")
    parser.add_argument("--data_dir", type=str, default="data/LungHist700", help="Directory where images are extracted")
    parser.add_argument("--model_path", type=str, default="best_mopfn.pth", help="Path to trained model weights")
    parser.add_argument("--img_size", type=type(224), default=224, help="Input image size")
    parser.add_argument("--batch_size", type=type(1), default=1, help="Batch size (1 is ideal for sample-wise attention analysis)")
    args = parser.parse_args()

    # 1. Device Selection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU Acceleration (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using NVIDIA GPU Acceleration (CUDA)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # 2. Check model weights exist
    if not os.path.exists(args.model_path):
        print(f"Error: Model weights not found at {args.model_path}. Please train the model first.")
        return

    # Check split info exists
    if not os.path.exists("split_info.pth"):
        print("Error: split_info.pth not found. Please run train.py to establish splits.")
        return

    # Resolve CSV and data dir paths if nested
    csv_path = args.csv_path
    data_dir = args.data_dir
    if not os.path.exists(csv_path):
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
            print("Error: Could not find CSV metadata.")
            return

    # 3. Load Splits & Test Dataset
    split_info = torch.load("split_info.pth")
    test_patients = split_info['test_patients']
    print(f"Loaded test split with {len(test_patients)} patients: {test_patients}")

    test_dataset = LungHistPairedDataset(
        csv_path=csv_path,
        data_dir=data_dir,
        patient_ids=test_patients,
        transform=get_transforms(args.img_size, 'test'),
        split='test'
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 4. Load Model
    model = MOPFN(pretrained=False).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print(f"Loaded model weights from: {args.model_path}")

    # 5. Inference Loop
    y_mal_true, y_mal_pred, y_mal_probs = [], [], []
    y_sub_true, y_sub_pred, y_sub_probs = [], [], []
    y_ord_true, y_ord_pred = [], []
    
    # Track scale attention coefficients
    # Format: {class_label: [list of attention values]}
    attn_stats = {
        'normal': {'20_to_40': [], '40_to_20': []},
        'aca': {'20_to_40': [], '40_to_20': []},
        'scc': {'20_to_40': [], '40_to_20': []}
    }

    print("\n--- Running Inference on Test Split ---")
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            img_20 = batch['image_20x'].to(device)
            img_40 = batch['image_40x'].to(device)
            is_mal = batch['is_malignant'].to(device)
            subtype = batch['subtype'].to(device)
            diff = batch['differentiation'].to(device)
            
            # Forward pass with attention extraction
            outputs = model(img_20, img_40, return_attention=True)
            
            # Extract Predictions
            pred_mal = outputs['malignancy'].argmax(dim=-1).cpu().numpy()
            prob_mal = torch.softmax(outputs['malignancy'], dim=-1)[:, 1].cpu().numpy()
            
            pred_sub = outputs['subtype'].argmax(dim=-1).cpu().numpy()
            prob_sub = torch.softmax(outputs['subtype'], dim=-1)[:, 1].cpu().numpy()
            
            pred_ord = predict_ordinal_class(outputs['ordinal']).cpu().numpy()
            
            # Accumulate
            y_mal_true.extend(is_mal.cpu().numpy())
            y_mal_pred.extend(pred_mal)
            y_mal_probs.extend(prob_mal)
            
            # Masked metrics: accumulate only for true malignant cases
            for i in range(is_mal.size(0)):
                if is_mal[i] == 1:
                    y_sub_true.append(subtype[i].cpu().item())
                    y_sub_pred.append(pred_sub[i])
                    y_sub_probs.append(prob_sub[i])
                    
                    y_ord_true.append(diff[i].cpu().item())
                    y_ord_pred.append(pred_ord[i])

            # Extract cross scale attention weights
            # Shape of attn_20_40: [batch_size, num_heads, seq_len_q, seq_len_kv] -> [1, 4, 1, 1]
            attn_20_40 = outputs['attn_20_40'].mean().cpu().item() # average over heads and batch
            attn_40_20 = outputs['attn_40_20'].mean().cpu().item()
            
            # Group by class for explainability audit
            for i in range(is_mal.size(0)):
                cls_str = 'normal'
                if is_mal[i] == 1:
                    cls_str = 'aca' if subtype[i] == 0 else 'scc'
                attn_stats[cls_str]['20_to_40'].append(attn_20_40)
                attn_stats[cls_str]['40_to_20'].append(attn_40_20)

    # 6. Compute & Print Metrics
    print("\n==============================================")
    print("             MOPFN EVALUATION RESULTS          ")
    print("==============================================")
    
    # Task A: Malignancy
    acc_mal = accuracy_score(y_mal_true, y_mal_pred)
    prec_mal = precision_score(y_mal_true, y_mal_pred, zero_division=0)
    rec_mal = recall_score(y_mal_true, y_mal_pred, zero_division=0)
    f1_mal = f1_score(y_mal_true, y_mal_pred, zero_division=0)
    auc_mal = roc_auc_score(y_mal_true, y_mal_probs) if len(np.unique(y_mal_true)) > 1 else 1.0
    cm_mal = confusion_matrix(y_mal_true, y_mal_pred)
    
    print("\n--- TASK A: Malignancy Detection (Normal vs. Carcinoma) ---")
    print(f"Accuracy:    {acc_mal*100:.2f}%")
    print(f"Precision:   {prec_mal*100:.2f}%")
    print(f"Recall:      {rec_mal*100:.2f}%")
    print(f"F1-Score:    {f1_mal*100:.2f}%")
    print(f"ROC-AUC:     {auc_mal:.4f}")
    print("Confusion Matrix:")
    print(cm_mal)
    
    # Task B: Carcinoma Subtype (Adenocarcinoma vs. Squamous Cell)
    if y_sub_true:
        acc_sub = accuracy_score(y_sub_true, y_sub_pred)
        f1_sub = f1_score(y_sub_true, y_sub_pred, average='macro', zero_division=0)
        cm_sub = confusion_matrix(y_sub_true, y_sub_pred)
        print("\n--- TASK B: Carcinoma Subtype (Adenocarcinoma vs. Squamous Cell) ---")
        print(f"Accuracy:    {acc_sub*100:.2f}%")
        print(f"Macro F1:    {f1_sub*100:.2f}%")
        print("Confusion Matrix (0: ACA, 1: SCC):")
        print(cm_sub)
    else:
        print("\n--- TASK B: Carcinoma Subtype ---")
        print("No malignant samples present in the test set to evaluate.")

    # Task C: Ordinal Tumor Differentiation Grade (Well, Moderate, Poor)
    if y_ord_true:
        acc_ord = accuracy_score(y_ord_true, y_ord_pred)
        mae_ord = np.mean(np.abs(np.array(y_ord_true) - np.array(y_ord_pred)))
        cm_ord = confusion_matrix(y_ord_true, y_ord_pred)
        
        print("\n--- TASK C: Ordinal Differentiation Level (Well -> Mod -> Poor) ---")
        print(f"Accuracy:    {acc_ord*100:.2f}%")
        print(f"Mean Absolute Error (MAE): {mae_ord:.3f} grades")
        print("Confusion Matrix (0: Well, 1: Moderate, 2: Poor):")
        print(cm_ord)
    else:
        print("\n--- TASK C: Ordinal Differentiation ---")
        print("No malignant samples present in the test set to evaluate.")

    # Explainability Audit: Scale-interaction Analysis
    print("\n==============================================")
    print("      CROSS-SCALE ATTENTION EXPLAINABILITY     ")
    print("==============================================")
    print("This audits how information flows between scales during prediction.")
    print(" - 20x queries 40x (20x <- 40x): How much the 40x cytologic detail guides the 20x architecture.")
    print(" - 40x queries 20x (40x <- 20x): How much the 20x structural context guides the 40x details.")
    print("A higher score denotes stronger interaction / reliance on that cross-scale information flow.\n")
    
    for cls_name in ['normal', 'aca', 'scc']:
        stats = attn_stats[cls_name]
        if stats['20_to_40']:
            mean_20_40 = np.mean(stats['20_to_40'])
            mean_40_20 = np.mean(stats['40_to_20'])
            print(f"Class [{cls_name.upper()}]:")
            print(f"  - 40x micro details guiding 20x macro view (20x <- 40x): {mean_20_40:.4f}")
            print(f"  - 20x macro context guiding 40x micro view (40x <- 20x): {mean_40_20:.4f}")
        else:
            print(f"Class [{cls_name.upper()}]: No test samples evaluated.")
            
    print("\n==============================================")
    print("Evaluation completed successfully!")

if __name__ == "__main__":
    main()
