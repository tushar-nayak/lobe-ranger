# Lobe Ranger: Multi-Scale Ordinal Pathology Foundation Network (MOPFN)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch: 2.0+](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)

**Lobe Ranger** is an advanced Multi-Scale Ordinal Pathology Foundation Network (MOPFN) that pairs 20x and 40x whole-slide images to preserve architectural and cytologic context. Fused via bidirectional cross-attention, it utilizes a pre-trained Vision Transformer for multi-task malignancy, subtype, and ordinal differentiation classification.

---

## 🔬 Core Novel Architecture

Traditional histopathology models often downsample high-resolution images, destroying critical diagnostic structures. **Lobe Ranger** solves this with a unified three-pillar framework:

1. **Multi-Scale Input Pairing:** Pairs 20x magnification (macro-architectural structures) with 40x magnification (micro-cytological details) patient-wise to preserve the full clinical context.
2. **Pathology Foundation Representation:** Leverages a shared Vision Transformer (`vit_b_16`) pre-trained backbone to extract deep clinical features.
3. **Bidirectional Cross-Scale Attention Fusion:** A multi-head cross-attention mechanism where:
   - The 20x representation queries the 40x cytologic detail.
   - The 40x representation queries the 20x spatial context.
4. **Hierarchical Ordinal Multi-Task Learning:** Simultaneously optimizes three task heads using masked loss backpropagation (ignoring normal tissues in malignant tasks) and **Consistent Ordinal Regression (CORAL)** for differentiation grade.

```
                  ┌──────────────┐      ┌─────────────────────────┐
 20x (Macro) ────►│ Shared ViT-B │ ───► │ 20x <- 40x Cross-Attn   │ ──┐
                  └──────────────┘      └─────────────────────────┘   │
                                                    ▲                 ├─► Concatenate & Project ──► Multi-Task Heads
                                                    │                 │
                  ┌──────────────┐      ┌─────────────────────────┐   │
 40x (Micro) ────►│ Shared ViT-B │ ───► │ 40x <- 20x Cross-Attn   │ ──┘
                  └──────────────┘      └─────────────────────────┘
```

---

## 🚀 Setup & Execution

### 1. Environment Activation
Activate the pre-configured conda environment:
```bash
conda activate bme_ml
```
*Note: Set `export KMP_DUPLICATE_LIB_OK=TRUE` if you experience OpenMP clashes on macOS.*

### 2. Download and Extract the Dataset
Runs a secure download and robust RAR extraction of the **LungHist700** dataset:
```bash
python download_data.py
```

### 3. Run Pipeline Training
Launch the training loop utilizing **Apple Silicon GPU (MPS)** acceleration:
```bash
python train.py --epochs 5 --batch_size 4 --csv_path data/data/data.csv --data_dir data/data
```

### 4. Evaluate and Run Attention Audit
Evaluate the model on isolated test splits and audit the scale-attention explainability coefficients:
```bash
python evaluate.py --csv_path data/data/data.csv --data_dir data/data
```

---

## 📊 Performance Benchmark

Evaluated on an isolated patient-wise test split (80/10/10) to prevent patient data leakage:

| Diagnostic Task | Metric | Score | Note |
| :--- | :--- | :--- | :--- |
| **Malignancy Detection** | Accuracy | **98.36%** | Normal vs. Carcinoma |
| | F1-Score | **99.17%** | |
| | ROC-AUC | **0.9720** | |
| **Subtype Classification**| Accuracy | **56.78%** | Adenocarcinoma vs. Squamous Cell |
| **Differentiation Grade** | Accuracy | **65.76%** | Ordinal Grade (Well ➔ Moderate ➔ Poor) |
| | Mean Absolute Error (MAE) | **0.342 grades** | Average distance from true grade |

---

## 🧠 Explainability Audit (Cross-Scale Attention)

Lobe Ranger audits how information flows between scales:
* **20x ➔ 40x:** Guides spatial architectural views using cytologic details.
* **40x ➔ 20x:** Guides cytological details using global spatial context.

> [!NOTE]
> **Theoretical Observation:** Since cross-scale attention is computed on global CLS representations (sequence length = 1), attention weights mathematically converge to `1.0000`. To enable spatially resolved patch-level attention map audits, cross-attention layers should be situated before global CLS token pooling.

---

## 📜 License
This repository is licensed under the MIT License.
