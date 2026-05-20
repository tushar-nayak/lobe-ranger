import os
import random
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from dataset import LungHistPairedDataset, get_transforms
from model import MOPFN, CoralOrdinalLoss, predict_ordinal_class

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_stratified_folds(csv_path, num_folds=5, seed=42):
    """
    Groups patients by strata and divides them deterministically into 5 folds.
    Strata definition:
    - Mixed: Patients who have both aca and scc images.
    - ACA: Patients who have adenocarcinoma images only.
    - SCC: Patients who have squamous cell carcinoma images only.
    - Normal: Patients who have normal tissue images only.
    """
    random.seed(seed)
    df_meta = pd.read_csv(csv_path)
    df_meta.columns = [c.strip().lower() for c in df_meta.columns]
    df_meta['subclass'] = df_meta['subclass'].fillna('').astype(str).str.lower().str.strip()
    df_meta['superclass'] = df_meta['superclass'].astype(str).str.lower().str.strip()
    
    patient_col = 'patient_id' if 'patient_id' in df_meta.columns else 'patient'
    patient_strata = {}
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
            
    strata_groups = {'nor': [], 'aca': [], 'scc': [], 'mixed': []}
    for p, s in patient_strata.items():
        strata_groups[s].append(p)
        
    for s in strata_groups:
        strata_groups[s].sort()  # Ensure lexicographical determinism before shuffle
        random.shuffle(strata_groups[s])
        
    folds = [[] for _ in range(num_folds)]
    for s, p_list in strata_groups.items():
        for idx, p in enumerate(p_list):
            folds[idx % num_folds].append(p)
            
    print("\n--- Stratified Patient-Level Fold Distribution ---")
    for f_idx, fold in enumerate(folds):
        print(f"Fold {f_idx + 1}: {len(fold)} patients | IDs: {fold}")
    return folds

def compute_bootstrap_ci(y_true, y_pred, metric_fn, num_bootstraps=1000, seed=42, is_prob=False):
    """
    Computes 95% Confidence Intervals using bootstrapping.
    """
    np.random.seed(seed)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    bootstrapped_scores = []
    
    n_samples = len(y_true)
    if n_samples == 0:
        return 0.0, 0.0, 0.0
        
    for _ in range(num_bootstraps):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        # Handle ROC-AUC edge cases
        if is_prob:
            if len(np.unique(y_true[indices])) > 1:
                score = metric_fn(y_true[indices], y_pred[indices])
                bootstrapped_scores.append(score)
        else:
            score = metric_fn(y_true[indices], y_pred[indices])
            bootstrapped_scores.append(score)
            
    if not bootstrapped_scores:
        val_score = metric_fn(y_true, y_pred) if not is_prob or len(np.unique(y_true)) > 1 else 1.0
        return val_score, val_score, val_score
        
    bootstrapped_scores = np.sort(bootstrapped_scores)
    percentile_2_5 = np.percentile(bootstrapped_scores, 2.5)
    percentile_97_5 = np.percentile(bootstrapped_scores, 97.5)
    mean_score = np.mean(bootstrapped_scores)
    
    return mean_score, percentile_2_5, percentile_97_5

def plot_and_save_confusion_matrix(y_true, y_pred, classes, title, save_path):
    """
    Plots a highly polished, premium confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    # Check if empty
    if cm.size == 0 or len(y_true) == 0:
        return
    plt.figure(figsize=(6, 5))
    sns.set_theme(style='white')
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes,
                cbar=False, annot_kws={"size": 14, "weight": "bold"})
    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Predicted Label', fontsize=12, labelpad=10)
    plt.ylabel('True Label', fontsize=12, labelpad=10)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11, rotation=0)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def main():
    seed = 42
    set_seed(seed)
    
    # 1. Device Selection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS) for training.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using NVIDIA GPU (CUDA) for training.")
    else:
        device = torch.device("cpu")
        print("Using CPU for training.")
        
    # Paths setup
    csv_path = "data/data/data.csv"
    data_dir = "data/data"
    
    # Auto-resolve CSV path
    if not os.path.exists(csv_path):
        found_csv = False
        for root, dirs, files in os.walk("data"):
            for file in files:
                if file.endswith(".csv"):
                    csv_path = os.path.join(root, file)
                    data_dir = root
                    found_csv = True
                    print(f"Auto-resolved data CSV to: {csv_path}")
                    break
            if found_csv:
                break
        if not found_csv:
            raise FileNotFoundError("Could not locate LungHist700 metadata CSV.")

    # 2. Get Stratified Patient-Wise Folds
    folds = get_stratified_folds(csv_path, num_folds=5, seed=seed)
    
    # 3. Models to evaluate
    fusion_modes = ['20x_only', '40x_only', 'late_fusion', 'concat_fusion', 'mopfn']
    
    # Hyperparameters
    epochs = 3
    batch_size = 8
    lr = 1e-4
    img_size = 224
    
    # Results dictionary
    all_results = {}
    
    for mode in fusion_modes:
        print(f"\n=======================================================")
        print(f" TRAINING BASELINE ARCHITECTURE: {mode.upper()}")
        print(f"=======================================================")
        
        # Accumulate out-of-fold predictions
        oof_mal_true, oof_mal_pred, oof_mal_probs = [], [], []
        oof_sub_true, oof_sub_pred, oof_sub_probs = [], [], []
        oof_ord_true, oof_ord_pred = [], []
        
        for fold_idx in range(5):
            print(f"\n--- Model: {mode} | Fold {fold_idx + 1}/5 ---")
            
            # Setup split patient lists
            val_patients = folds[fold_idx]
            train_patients = [p for i, f in enumerate(folds) if i != fold_idx for p in f]
            
            # Datasets
            train_dataset = LungHistPairedDataset(
                csv_path=csv_path,
                data_dir=data_dir,
                patient_ids=train_patients,
                transform=get_transforms(img_size, 'train'),
                split=f"train_f{fold_idx+1}"
            )
            val_dataset = LungHistPairedDataset(
                csv_path=csv_path,
                data_dir=data_dir,
                patient_ids=val_patients,
                transform=get_transforms(img_size, 'val'),
                split=f"val_f{fold_idx+1}"
            )
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            # Instantiate Model (ImageNet ViT Backbone is frozen internally in model.py)
            model = MOPFN(pretrained=True, fusion_mode=mode).to(device)
            
            criterion_malignancy = nn.CrossEntropyLoss()
            criterion_subtype = nn.CrossEntropyLoss(reduction='none')
            criterion_ordinal = CoralOrdinalLoss()
            bce_elementwise = nn.BCEWithLogitsLoss(reduction='none')
            
            # Only optimize parameters with requires_grad=True (fusion & heads)
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
            
            # Training Loop
            for epoch in range(1, epochs + 1):
                model.train()
                train_loss = 0.0
                for batch in train_loader:
                    img_20 = batch['image_20x'].to(device)
                    img_40 = batch['image_40x'].to(device)
                    is_mal = batch['is_malignant'].to(device)
                    subtype = batch['subtype'].to(device)
                    diff = batch['differentiation'].to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(img_20, img_40)
                    
                    # Multi-task Loss A: Malignancy
                    loss_mal = criterion_malignancy(outputs['malignancy'], is_mal)
                    
                    # Multi-task Loss B: Subtype (Masked: malignant only)
                    loss_sub_raw = criterion_subtype(outputs['subtype'], subtype)
                    loss_sub_masked = (loss_sub_raw * is_mal.float()).sum() / (is_mal.float().sum() + 1e-8)
                    
                    # Multi-task Loss C: Ordinal Grade (Masked: malignant only)
                    binary_diff = torch.zeros_like(outputs['ordinal'])
                    for i in range(binary_diff.size(1)):
                        binary_diff[:, i] = (diff > i).float()
                    loss_ord_raw = bce_elementwise(outputs['ordinal'], binary_diff).mean(dim=-1)
                    loss_ord_masked = (loss_ord_raw * is_mal.float()).sum() / (is_mal.float().sum() + 1e-8)
                    
                    loss = loss_mal + loss_sub_masked + loss_ord_masked
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                scheduler.step()
                
            # Validation (aggregating out-of-fold predictions)
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    img_20 = batch['image_20x'].to(device)
                    img_40 = batch['image_40x'].to(device)
                    is_mal = batch['is_malignant'].to(device)
                    subtype = batch['subtype'].to(device)
                    diff = batch['differentiation'].to(device)
                    
                    outputs = model(img_20, img_40)
                    
                    # Extract outputs
                    prob_mal = torch.softmax(outputs['malignancy'], dim=-1)[:, 1].cpu().numpy()
                    pred_mal = outputs['malignancy'].argmax(dim=-1).cpu().numpy()
                    
                    prob_sub = torch.softmax(outputs['subtype'], dim=-1)[:, 1].cpu().numpy()
                    pred_sub = outputs['subtype'].argmax(dim=-1).cpu().numpy()
                    
                    pred_ord = predict_ordinal_class(outputs['ordinal']).cpu().numpy()
                    
                    # Accumulate task outputs
                    oof_mal_true.extend(is_mal.cpu().numpy())
                    oof_mal_pred.extend(pred_mal)
                    oof_mal_probs.extend(prob_mal)
                    
                    for idx_sample in range(is_mal.size(0)):
                        if is_mal[idx_sample] == 1:
                            oof_sub_true.append(subtype[idx_sample].cpu().item())
                            oof_sub_pred.append(pred_sub[idx_sample])
                            oof_sub_probs.append(prob_sub[idx_sample])
                            
                            oof_ord_true.append(diff[idx_sample].cpu().item())
                            oof_ord_pred.append(pred_ord[idx_sample])
                            
        # 4. Out-of-fold Evaluation & Bootstrapping for current architecture
        print(f"\nComputing aggregate out-of-fold metrics and bootstrap CIs for: {mode}")
        
        # Helper metrics functions
        fn_acc = lambda t, p: accuracy_score(t, p)
        fn_f1 = lambda t, p: f1_score(t, p, zero_division=0)
        fn_f1_macro = lambda t, p: f1_score(t, p, average='macro', zero_division=0)
        fn_auc = lambda t, p: roc_auc_score(t, p)
        fn_mae = lambda t, p: np.mean(np.abs(np.array(t) - np.array(p)))
        
        # Calculate scores and 95% Confidence Intervals
        mal_acc_m, mal_acc_l, mal_acc_h = compute_bootstrap_ci(oof_mal_true, oof_mal_pred, fn_acc)
        mal_f1_m, mal_f1_l, mal_f1_h = compute_bootstrap_ci(oof_mal_true, oof_mal_pred, fn_f1)
        mal_auc_m, mal_auc_l, mal_auc_h = compute_bootstrap_ci(oof_mal_true, oof_mal_probs, fn_auc, is_prob=True)
        
        sub_acc_m, sub_acc_l, sub_acc_h = compute_bootstrap_ci(oof_sub_true, oof_sub_pred, fn_acc)
        sub_f1_m, sub_f1_l, sub_f1_h = compute_bootstrap_ci(oof_sub_true, oof_sub_pred, fn_f1_macro)
        
        ord_acc_m, ord_acc_l, ord_acc_h = compute_bootstrap_ci(oof_ord_true, oof_ord_pred, fn_acc)
        ord_mae_m, ord_mae_l, ord_mae_h = compute_bootstrap_ci(oof_ord_true, oof_ord_pred, fn_mae)
        
        all_results[mode] = {
            'malignancy': {
                'accuracy': {'mean': mal_acc_m, 'ci_lower': mal_acc_l, 'ci_upper': mal_acc_h},
                'f1': {'mean': mal_f1_m, 'ci_lower': mal_f1_l, 'ci_upper': mal_f1_h},
                'auc': {'mean': mal_auc_m, 'ci_lower': mal_auc_l, 'ci_upper': mal_auc_h}
            },
            'subtype': {
                'accuracy': {'mean': sub_acc_m, 'ci_lower': sub_acc_l, 'ci_upper': sub_acc_h},
                'f1_macro': {'mean': sub_f1_m, 'ci_lower': sub_f1_l, 'ci_upper': sub_f1_h}
            },
            'differentiation': {
                'accuracy': {'mean': ord_acc_m, 'ci_lower': ord_acc_l, 'ci_upper': ord_acc_h},
                'mae': {'mean': ord_mae_m, 'ci_lower': ord_mae_l, 'ci_upper': ord_mae_h}
            }
        }
        
        # Plot and save confusion matrices
        plot_and_save_confusion_matrix(
            oof_mal_true, oof_mal_pred, ['Normal', 'Malignant'],
            f'{mode.replace("_", " ").title()} - Task A: Malignancy',
            f'docs/cm_{mode}_malignancy.png'
        )
        plot_and_save_confusion_matrix(
            oof_sub_true, oof_sub_pred, ['ACA', 'SCC'],
            f'{mode.replace("_", " ").title()} - Task B: Carcinoma Subtype',
            f'docs/cm_{mode}_subtype.png'
        )
        plot_and_save_confusion_matrix(
            oof_ord_true, oof_ord_pred, ['Well', 'Moderate', 'Poor'],
            f'{mode.replace("_", " ").title()} - Task C: Differentiation Grade',
            f'docs/cm_{mode}_differentiation.png'
        )
        
        print(f"Task A (Malignancy) -> Acc: {mal_acc_m*100:.2f}% (95% CI: [{mal_acc_l*100:.1f}%, {mal_acc_h*100:.1f}%]) | ROC-AUC: {mal_auc_m:.4f} (95% CI: [{mal_auc_l:.3f}, {mal_auc_h:.3f}])")
        print(f"Task B (Subtype)    -> Acc: {sub_acc_m*100:.2f}% (95% CI: [{sub_acc_l*100:.1f}%, {sub_acc_h*100:.1f}%]) | F1 Macro: {sub_f1_m*100:.2f}%")
        print(f"Task C (Ordinal)    -> Acc: {ord_acc_m*100:.2f}% (95% CI: [{ord_acc_l*100:.1f}%, {ord_acc_h*100:.1f}%]) | MAE: {ord_mae_m:.3f} (95% CI: [{ord_mae_l:.3f}, {ord_mae_h:.3f}])")
        
    # 5. Write final results to JSON file
    with open("result.json", "w") as f:
        json.dump(all_results, f, indent=4)
    print("\nSaved comprehensive cross-validation results table to result.json.")
    
    # 6. Generate Premium Locked Results Markdown Comparison Table
    print("\n=======================================================")
    print(" LOCKED RESULTS COMPARISON TABLE (OOF CROSS-VALIDATION) ")
    print("=======================================================")
    headers = ["Architecture", "Task A Acc (95% CI)", "Task A ROC-AUC (95% CI)", "Task B Acc (95% CI)", "Task C Acc (95% CI)", "Task C MAE (95% CI)"]
    print(f"| {' | '.join(headers)} |")
    print("|" + "---|"*len(headers))
    for mode in fusion_modes:
        res = all_results[mode]
        mal_acc_str = f"{res['malignancy']['accuracy']['mean']*100:.2f}% ({res['malignancy']['accuracy']['ci_lower']*100:.1f}-{res['malignancy']['accuracy']['ci_upper']*100:.1f})"
        mal_auc_str = f"{res['malignancy']['auc']['mean']:.3f} ({res['malignancy']['auc']['ci_lower']:.3f}-{res['malignancy']['auc']['ci_upper']:.3f})"
        sub_acc_str = f"{res['subtype']['accuracy']['mean']*100:.2f}% ({res['subtype']['accuracy']['ci_lower']*100:.1f}-{res['subtype']['accuracy']['ci_upper']*100:.1f})"
        ord_acc_str = f"{res['differentiation']['accuracy']['mean']*100:.2f}% ({res['differentiation']['accuracy']['ci_lower']*100:.1f}-{res['differentiation']['accuracy']['ci_upper']*100:.1f})"
        ord_mae_str = f"{res['differentiation']['mae']['mean']:.3f} ({res['differentiation']['mae']['ci_lower']:.3f}-{res['differentiation']['mae']['ci_upper']:.3f})"
        
        mode_label = mode.replace('_', ' ').upper()
        print(f"| {mode_label} | {mal_acc_str} | {mal_auc_str} | {sub_acc_str} | {ord_acc_str} | {ord_mae_str} |")
        
if __name__ == "__main__":
    main()
