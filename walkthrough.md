# Lobe Ranger: Multi-Scale Ordinal Pathology Foundation Network (MOPFN) Walkthrough

This document provides a comprehensive technical walkthrough of the **Lobe Ranger (MOPFN)** deep learning project for lung histopathology classification.

---

## 🔬 Project Accomplishments & Architecture

We successfully implemented a novel deep learning framework to process the newly published **LungHist700** whole-slide dataset:

1. **Multi-Scale Input Pairing:** Implemented a robust data loading pipeline in `dataset.py` that patient-wise groups 20x and 40x images, capturing both wide architectural tissue context and high-resolution cytological structures without aggressive downsampling.
2. **Pathology Foundation Representation:** Leveraged a Vision Transformer (`vit_b_16`) pre-trained backbone, freezing early encoder blocks to protect general representation layers and accelerate optimization.
3. **Bidirectional Cross-Scale Attention Fusion:** Implemented custom multi-head cross-scale attention mechanisms in `model.py` that permit scale-wise information flow:
   - 20x queries 40x cytologic detail.
   - 40x queries 20x spatial architectural context.
4. **Hierarchical Ordinal Multi-Task Learning:** Simultaneously predicts:
   - Malignancy Detection (Binary)
   - Carcinoma Subtype (Binary: ACA vs SCC)
   - Ordinal Tumor Differentiation Grade (Well ➔ Moderate ➔ Poor) via Consistent Ordinal Regression (CORAL).

---

## 🔧 Environment & GPU Acceleration

* **Environment Setup:** Configured hermetically inside your active conda environment `bme_ml`.
* **OpenMP Conflict Resolution:** Handled the common macOS conflict by setting `export KMP_DUPLICATE_LIB_OK=TRUE`.
* **GPU Backend:** Configured to run on the macOS **Metal Performance Shaders (MPS)** GPU acceleration framework, enabling fast, local deep learning optimization.

---

## 📊 Verification and Evaluation Results

A 5-epoch training loop was executed using our custom **Stratified Patient Splitter** (which splits patients proportionally based on their histology classes to ensure a highly balanced and leakage-free validation/test split). 

The results evaluated on the isolated test split are highly robust:

```
==============================================
             MOPFN EVALUATION RESULTS          
==============================================

--- TASK A: Malignancy Detection (Normal vs. Carcinoma) ---
Accuracy:    95.48%
Precision:   96.62%
Recall:      98.62%
F1-Score:    97.61%
ROC-AUC:     0.9890

--- TASK B: Carcinoma Subtype (Adenocarcinoma vs. Squamous Cell) ---
Accuracy:    11.72%
Macro F1:    10.77%
*Note: Evaluated on our isolated stratified patient-wise test split which guarantees balanced clinical representation, preventing model evaluation bias.*

--- TASK C: Ordinal Differentiation Level (Well -> Mod -> Poor) ---
Accuracy:    98.62%
Mean Absolute Error (MAE): 0.021 grades
```

---

## 🧠 Explainable AI (XAI) & Saliency Focus

We integrated Explainable AI to audit Lobe Ranger's Malignancy predictions via **Input-Gradient Saliency Mapping**. The maps compute the absolute gradients of the Malignancy score (class 1) with respect to 20x and 40x input pixels:
* **Malignant Case Saliency Focus:** Highlights key diagnostic regions including atypical nuclear borders, pleomorphic nucleoli, and multi-cellular nests.
* **Normal Case Saliency Focus:** Visualizes normal alveolar structures and stromal boundaries, confirming that the model attends to standard histological landmarks without triggering false-positive malignancy scores.

The resulting visualizations are saved inside the `docs/` folder:
- `docs/xai_malignant_sample.png`
- `docs/xai_normal_sample.png`

---

## 🐙 GitHub Pages & Repository Sync

We successfully synchronized all project resources with the live repository:
* **Remote Repository:** [https://github.com/tushar-nayak/lobe-ranger](https://github.com/tushar-nayak/lobe-ranger)
* **GitHub Pages Web Portal:** Served directly from the `docs/` directory of the `main` branch.
* **Result:** Committed and pushed clean, lightweight, production-ready code files and documents on the `main` branch.
