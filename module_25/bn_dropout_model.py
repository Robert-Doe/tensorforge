"""module_25/bn_dropout_model.py — CNN with Batch Normalisation and Dropout.

Adds BatchNorm2d after each conv layer and BatchNorm1d after fc1.
BatchNorm normalises each mini-batch's activations to zero-mean, unit-variance,
then applies learnable scale (gamma) and shift (beta).

Why this order: Conv → BN → ReLU (not Conv → ReLU → BN).
BN before ReLU means we normalise raw activations — ReLU then clips only the
negative half of a well-centred distribution, which is the intended behaviour.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DigitCNNwithBN(nn.Module):
    """DigitCNN (Module 24) extended with BatchNorm2d and configurable Dropout.

    Attributes:
        conv1, bn1:    first conv block + batch norm
        conv2, bn2:    second conv block + batch norm
        pool:          shared MaxPool (no learnable params)
        fc1, bn_fc:    first dense layer + batch norm
        dropout:       inverted dropout before output layer
        fc2:           output layer
    """

    FEATURE_H = 7
    FEATURE_W = 7
    CONV2_CHANNELS = 32

    def __init__(self, dropout_p: float = 0.4):
        """Build the model.

        Args:
            dropout_p: dropout probability (0 = no dropout, 0.5 = heavy regularisation)
        """
        super().__init__()

        # ── Conv block 1 ───────────────────────────────────────────────────────
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(16)    # 16 = number of feature maps from conv1

        # ── Conv block 2 ───────────────────────────────────────────────────────
        self.conv2 = nn.Conv2d(16, self.CONV2_CHANNELS, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(self.CONV2_CHANNELS)

        self.pool = nn.MaxPool2d(2, 2)

        # ── Dense classifier ───────────────────────────────────────────────────
        flat_size    = self.CONV2_CHANNELS * self.FEATURE_H * self.FEATURE_W   # 1568
        self.fc1     = nn.Linear(flat_size, 128)
        self.bn_fc   = nn.BatchNorm1d(128)   # 1d because input is (batch, features)
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc2     = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with BN before ReLU in each block.

        Args:
            x: (batch, 1, 28, 28) normalised pixel tensor

        Returns:
            (batch, 10) log-probabilities
        """
        # Conv → BN → ReLU → Pool
        x = self.pool(F.relu(self.bn1(self.conv1(x))))   # (B,16,14,14)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))   # (B,32,7,7)

        x = x.view(x.size(0), -1)                        # (B,1568)

        # Dense → BN → ReLU → Dropout
        x = F.relu(self.bn_fc(self.fc1(x)))              # (B,128)
        x = self.dropout(x)
        x = self.fc2(x)                                  # (B,10)

        return F.log_softmax(x, dim=1)


def count_parameters(model: nn.Module) -> int:
    """Print trainable parameter count per layer and return total.

    Args:
        model: any nn.Module

    Returns:
        total number of trainable parameters
    """
    total = 0
    print("\nParameter count:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            n = param.numel()
            total += n
            print(f"  {name:<30} {str(list(param.shape)):<20} → {n:,}")
    print(f"  {'TOTAL':<30}                      → {total:,}")
    return total
