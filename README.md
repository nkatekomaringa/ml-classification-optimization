# Comparative Machine Learning: Optimization, Support Vectors, and Deep Architectures

An empirical investigation into fundamental machine learning tradeoffs, comparing traditional kernel methods (Support Vector Machines), unsupervised clustering (k-Means), spatial deep architectures (Convolutional Neural Networks), and gradient dynamics across deep feedforward networks.

---

## 📌 Project Overview

This repository provides a modular, production-ready codebase analyzing critical machine learning concepts across five core domains:
1. **SVM Regularization & Margins:** Evaluating soft vs. hard margins ($C=0.001$ vs $C=10$) on non-linearly separable data.
2. **Hyperparameter Tuning & Kernels:** Comparing Linear vs. RBF kernels tuned via 5-Fold Grid Search Cross-Validation.
3. **Distance Metrics & Scaling:** Investigating the impact of feature standardization on $k$-Means clustering over pixel features.
4. **Spatial Feature Representation:** Comparing traditional SVMs against PyTorch Convolutional Neural Networks (`DigitCNN`) on image classification tasks.
5. **Gradient Flow Dynamics:** Quantifying vanishing gradients across 10-layer deep networks comparing Sigmoid and ReLU activations.

---

## 📊 Key Findings & Results Summary

| Analysis | Experiment / Setting | Metric | Result | Insights |
|---|---|---|---|---|
| **SVM Margin** | Soft Margin ($C=0.001$) | Support Vectors / Test Acc | 100 SVs / 90.0% | Wider margin, lower variance, less sensitive to individual outliers. |
| **SVM Margin** | Hard Margin ($C=10.0$) | Support Vectors / Test Acc | 60 SVs / 86.7% | Narrower margin, higher variance, prone to overfitting noisy points. |
| **Kernel Tuning** | RBF Kernel ($C=1, \gamma=\text{'scale'}$) | 5-Fold CV Test Accuracy | **90.5%** | Captures non-linear decision boundaries effectively. |
| **Kernel Tuning** | Linear Kernel | Test Accuracy | 85.7% | Limited by linear hyperplanes on complex datasets (e.g., Moons). |
| **Clustering** | $k$-Means (Unscaled Digits) | Majority Voting Accuracy | **79.4%** | Features retain natural intensity scale variations. |
| **Clustering** | $k$-Means (Standardized Digits) | Majority Voting Accuracy | 68.2% | Zero-variance background pixels get inflated, distorting Euclidean distance. |
| **Image Classif.** | Support Vector Classifier (RBF) | Test Accuracy | **98.3%** | Highly effective on small $8\times 8$ benchmark digit datasets. |
| **Image Classif.** | PyTorch `DigitCNN` (15 Epochs) | Test Accuracy | 94.3% | Spatial convolutions capture local structure effectively without manual feature scaling. |
| **Gradient Flow** | 10-Layer Net (Sigmoid) | 1st Layer Grad Norm | $8.311 \times 10^{-10}$ | Severe vanishing gradient due to derivative saturation ($\le 0.25$). |
| **Gradient Flow** | 10-Layer Net (ReLU) | 1st Layer Grad Norm | $6.752 \times 10^{-5}$ | Non-saturating gradient preserves derivative propagation backwards. |

---

## 📁 Repository Structure

```text
ml-classification-optimization/
├── README.md                  # Project overview and key empirical results
├── requirements.txt           # Python dependencies
├── src/
│   ├── svm_analysis.py        # Margin width, soft vs. hard C, and RBF kernel grid search
│   ├── kmeans_clustering.py   # k-Means clustering on raw vs. standardized digit features
│   ├── cnn_vs_svm.py          # PyTorch DigitCNN vs SVC benchmark on 8x8 digits
│   └── vanishing_gradients.py # Gradient flow quantification across 10-layer MLPs
```

---

## 🛠️ Installation & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/ml-classification-optimization.git
cd ml-classification-optimization
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Experiments
```bash
python src/svm_analysis.py
python src/kmeans_clustering.py
python src/cnn_vs_svm.py
python src/vanishing_gradients.py
```

---

## 📜 License
This project is released under the [MIT License](LICENSE).
