"""
Unsupervised Digit Clustering Analysis.

Evaluates k-Means performance on raw vs. standardized feature spaces
using majority-vote label mapping.
"""

import numpy as np
from scipy.stats import mode
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


def map_to_labels(labels: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Maps cluster IDs to their majority target class."""
    mapped = np.zeros_like(labels)
    for k in range(10):
        mask = (labels == k)
        if np.any(mask):
            mapped[mask] = mode(y_true[mask], keepdims=True).mode[0]
    return mapped


def run_kmeans_experiments():
    """Compares unscaled vs. scaled k-Means performance on 8x8 handwritten digits."""
    digits = load_digits()
    X, y = digits.data, digits.target

    # Unscaled Clustering
    kmeans_unscaled = KMeans(n_clusters=10, n_init=10, random_state=42).fit(X)
    pred_unscaled = map_to_labels(kmeans_unscaled.labels_, y)
    acc_unscaled = accuracy_score(y, pred_unscaled)

    # Scaled Clustering
    X_scaled = StandardScaler().fit_transform(X)
    kmeans_scaled = KMeans(n_clusters=10, n_init=10, random_state=42).fit(X_scaled)
    pred_scaled = map_to_labels(kmeans_scaled.labels_, y)
    acc_scaled = accuracy_score(y, pred_scaled)

    print("--- k-Means Clustering Analysis ---")
    print(f"Unscaled Label Accuracy : {acc_unscaled:.4f}")
    print(f"Scaled Label Accuracy   : {acc_scaled:.4f}\n")


if __name__ == "__main__":
    run_kmeans_experiments()
