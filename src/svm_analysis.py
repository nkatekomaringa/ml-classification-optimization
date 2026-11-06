"""
SVM Regularization and Kernel Tuning Analysis.

Examines soft-margin vs hard-margin trade-offs and hyperparameter search
for RBF vs Linear kernels using scikit-learn.
"""

import numpy as np
from sklearn.datasets import make_blobs, make_moons
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler


def analyze_svm_margins():
    """Evaluates Support Vector Machine behavior under varying C parameters."""
    X, y = make_blobs(n_samples=200, centers=2, cluster_std=4.0, random_state=7)
    y = np.where(y == 0, -1, 1)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Soft Margin (C = 0.001)
    svm_soft = SVC(kernel="linear", C=0.001).fit(Xtr, ytr)
    n_sv_soft = len(svm_soft.support_vectors_)
    acc_soft = svm_soft.score(Xte, yte)

    # Hard Margin (C = 10.0)
    svm_hard = SVC(kernel="linear", C=10.0).fit(Xtr, ytr)
    n_sv_hard = len(svm_hard.support_vectors_)
    acc_hard = svm_hard.score(Xte, yte)

    print("--- SVM Margin Analysis ---")
    print(f"C=0.001 (Soft): Support Vectors = {n_sv_soft}, Test Accuracy = {acc_soft:.4f}")
    print(f"C=10.0  (Hard): Support Vectors = {n_sv_hard}, Test Accuracy = {acc_hard:.4f}\n")


def tune_rbf_kernel():
    """Performs 5-Fold Grid Search Cross-Validation for RBF Kernel SVM."""
    Xm, ym = make_moons(n_samples=140, noise=0.3, random_state=6)
    Xtr_m, Xte_m, ytr_m, yte_m = train_test_split(Xm, ym, test_size=0.3, random_state=42, stratify=ym)
    scaler = StandardScaler().fit(Xtr_m)
    Xtr_s, Xte_s = scaler.transform(Xtr_m), scaler.transform(Xte_m)

    param_grid = {
        "C": [0.1, 1, 10],
        "gamma": ["scale", 0.1, 1, 10]
    }
    grid = GridSearchCV(SVC(kernel="rbf"), param_grid=param_grid, cv=5).fit(Xtr_s, ytr_m)
    
    best_c = grid.best_params_["C"]
    rbf_acc = grid.score(Xte_s, yte_m)

    # Compare with Linear Kernel
    svm_lin = SVC(kernel="linear").fit(Xtr_s, ytr_m)
    lin_acc = svm_lin.score(Xte_s, yte_m)

    print("--- RBF vs Linear Kernel Evaluation ---")
    print(f"Best Hyperparameters : {grid.best_params_}")
    print(f"Tuned RBF Test Acc   : {rbf_acc:.4f}")
    print(f"Linear Kernel Test Acc: {lin_acc:.4f}\n")


if __name__ == "__main__":
    analyze_svm_margins()
    tune_rbf_kernel()
