"""
Gradient Flow Analysis in Deep Neural Networks.

Quantifies first-layer weight gradient decay in deep architectures
under Sigmoid vs. ReLU activation functions.
"""

import torch
import torch.nn as nn


def build_and_compute_gradient(activation_cls: nn.Module) -> float:
    """Builds a 10-layer network, performs a pass, and returns 1st-layer gradient norm."""
    torch.manual_seed(0)
    layers = []
    for _ in range(10):
        layers.extend([nn.Linear(20, 20), activation_cls()])
    layers.append(nn.Linear(20, 1))

    net = nn.Sequential(*layers)
    x = torch.randn(64, 20)
    target = torch.randn(64, 1)

    loss = ((net(x) - target) ** 2).mean()
    loss.backward()

    grad_norm = float(net[0].weight.grad.norm())
    return grad_norm


if __name__ == "__main__":
    sigmoid_norm = build_and_compute_gradient(nn.Sigmoid)
    relu_norm = build_and_compute_gradient(nn.ReLU)

    print("--- Vanishing Gradient Quantification ---")
    print(f"Sigmoid First-Layer Gradient Norm : {sigmoid_norm:.3e}")
    print(f"ReLU First-Layer Gradient Norm    : {relu_norm:.3e}\n")
