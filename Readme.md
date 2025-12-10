# Ensemble Deep‐Learning Networks for Automated Knee Osteoarthritis Grading

[![Paper](https://img.shields.io/badge/Paper-Nature%20Scientific%20Reports-green)](https://www.nature.com/articles/s41598-023-50210-4)
[![Dataset](https://img.shields.io/badge/Dataset-Mendeley%20Data-blue)](https://data.mendeley.com/datasets/56rmx5bjcr/1)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-orange)](https://pytorch.org/)

## 📝 Overview
Osteoarthritis (OA) is a degenerative joint disease affecting millions worldwide. The **Kellgren-Lawrence (KL) grading system** is the gold standard for diagnosing severity, yet manual grading suffers from high inter-observer variability. This project introduces a robust **Ensemble Deep Learning Framework** capable of automating KL grading with high accuracy and consistency, bridging the gap between "black box" AI and clinical interpretability.

## 🌟 Key Features
* **Heterogeneous Ensemble:** Combines 8 diverse CNN architectures (EfficientNet, DenseNet, ResNet variants).
* **Optimized Resolution:** Each model is trained on input resolutions specifically optimized for its architecture.
* **Mix Voting Strategy:** A novel aggregation method combining hard voting (majority rule) and soft voting (probability averaging) for robust predictions.
* **Explainable AI (XAI):** Integrated Grad-CAM and LIME visualizations to validate clinical relevance (e.g., joint space narrowing detection).

## 📂 Dataset
We utilized the **Osteoarthritis Initiative (OAI)** dataset, comprising **8,260 knee X-ray images** labeled with KL grades 0–4.

* **Source:** [Mendeley Data](https://data.mendeley.com/datasets/56rmx5bjcr/1)
* **Preprocessing:** Images were preprocessed with histogram equalization and augmented using rotation, flipping, and noise injection to handle class imbalance.

## 🏗️ Model Architecture
Our ensemble aggregates predictions from the following `torchvision` backbones:

| Model | Variant |
| :--- | :--- |
| **EfficientNet** | B5, V2-Small |
| **ResNet** | ResNet-101, Wide-ResNet-50-2 |
| **DenseNet** | DenseNet-161 |
| **RegNet** | Y-8GF |
| **ResNext** | 50-32x4d |
| **ShuffleNet** | V2-x2-0 |

### Training Strategy
1.  **Resolution Optimization:** Systematic ablation study to find the best input size for each architecture.
2.  **Two-Stage Training:**
    * *Stage 1:* Frozen backbone, trained FC layers (LR = 0.01).
    * *Stage 2:* Unfrozen full fine-tuning with progressive learning rate decay.
3.  **Cross-Validation:** Stratified 5-fold CV to ensure class balance across folds.

## 📊 Results
The ensemble framework outperforms individual models and previous state-of-the-art methods.

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **76.93%** |
| **F1-Score** | **0.7665** |

## 🧠 Visualization & Explainability
To ensure clinical trust, we employ **Grad-CAM** and **LIME** to generate visual explanations for model decisions. These heatmaps confirm the model focuses on relevant biomarkers such as:
* Osteophyte formation
* Joint space narrowing (JSN)
* Sclerosis

## 🚀 Getting Started

### Prerequisites
* Python 3.8+
* PyTorch
* Torchvision
* Scikit-learn
* Lime

### Installation
```bash
git clone [https://github.com/your-username/your-repo.git](https://github.com/your-username/your-repo.git)
cd your-repo
pip install -r requirements.txt
