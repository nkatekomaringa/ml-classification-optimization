"""
Convolutional Neural Network vs Support Vector Machine Benchmark.

Compares PyTorch 2D Convolution architecture with Standardized RBF SVM
on 8x8 pixel digit image classification.
"""

import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score


def as_images(flat_pixels: np.ndarray) -> torch.Tensor:
    """Converts 64 flat pixel vectors to (N, 1, 8, 8) normalized tensors."""
    return torch.tensor(flat_pixels, dtype=torch.float32).div(16).view(-1, 1, 8, 8)


class DigitCNN(nn.Module):
    """Lightweight 2D Convolutional Neural Network for 8x8 digits."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 12, kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(12, 10)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_and_eval_cnn(Xtr, Xte, ytr, yte, epochs: int = 15) -> float:
    """Trains PyTorch DigitCNN and returns test accuracy."""
    torch.manual_seed(0)
    device = "cpu"
    model = DigitCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    train_ds = torch.utils.data.TensorDataset(as_images(Xtr), torch.tensor(ytr, dtype=torch.long))
    loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=False)

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb.to(device)), yb.to(device))
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(as_images(Xte)).argmax(dim=1)
        correct = (preds == torch.tensor(yte)).sum().item()

    acc = correct / len(yte)
    return acc


def eval_svm(Xtr, Xte, ytr, yte):
    """Evaluates RBF SVC performance on standardized features."""
    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    Xte_s = scaler.transform(Xte)

    clf = SVC().fit(Xtr_s, ytr)
    preds = clf.predict(Xte_s)
    acc = accuracy_score(yte, preds)

    print("--- SVM Classification Report ---")
    print(classification_report(yte, preds, digits=4))
    return acc


if __name__ == "__main__":
    import numpy as np
    digits = load_digits()
    Xtr, Xte, ytr, yte = train_test_split(
        digits.data, digits.target, test_size=0.3, random_state=42, stratify=digits.target
    )

    cnn_acc = train_and_eval_cnn(Xtr, Xte, ytr, yte)
    svm_acc = eval_svm(Xtr, Xte, ytr, yte)

    print(f"CNN Test Accuracy : {cnn_acc:.4f}")
    print(f"SVM Test Accuracy : {svm_acc:.4f}\n")
