"""module_24/cnn_model.py — Full CNN in PyTorch for MNIST digit classification.

Architecture:
  Conv(1→16, 3×3) → ReLU → MaxPool(2×2)   [28×28 → 14×14, 16 channels]
  Conv(16→32, 3×3) → ReLU → MaxPool(2×2)  [14×14 → 7×7,  32 channels]
  Flatten → Linear(32×7×7→128) → ReLU → Dropout(0.3) → Linear(128→10) → LogSoftmax

Ten output classes (digits 0-9). Loss: NLLLoss (pairs with LogSoftmax).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DigitCNN(nn.Module):
    """Convolutional neural network for MNIST digit classification.

    Attributes:
        conv1:    first convolutional layer (1→16 feature maps)
        conv2:    second convolutional layer (16→32 feature maps)
        pool:     2×2 max-pooling layer (shared between both conv layers)
        fc1:      first fully-connected layer (flatten → 128 hidden units)
        dropout:  dropout regularisation before the output layer
        fc2:      output layer (128 → 10 class scores)
    """

    # Spatial dimensions after two MaxPool(2×2): 28 → 14 → 7
    FEATURE_H = 7
    FEATURE_W = 7
    CONV2_CHANNELS = 32

    def __init__(self, dropout_p=0.3):
        """Build the CNN.

        Args:
            dropout_p: dropout probability before the output layer
        """
        super().__init__()

        # ── Convolutional feature extractor ──────────────────────────────────
        # padding=1 keeps spatial size the same after each conv (same-padding)
        self.conv1 = nn.Conv2d(
            in_channels=1,   # MNIST is grayscale — 1 input channel
            out_channels=16, # learn 16 different 3×3 filters
            kernel_size=3,
            padding=1,       # output stays 28×28 after conv
        )
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=self.CONV2_CHANNELS,
            kernel_size=3,
            padding=1,       # output stays 14×14 after conv
        )
        # MaxPool halves spatial dimensions: (H,W) → (H/2, W/2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # ── Fully-connected classifier ────────────────────────────────────────
        flat_size = self.CONV2_CHANNELS * self.FEATURE_H * self.FEATURE_W  # 32×7×7 = 1568
        self.fc1     = nn.Linear(flat_size, 128)
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc2     = nn.Linear(128, 10)          # 10 digit classes

    def forward(self, x):
        """Forward pass: conv → pool × 2 → flatten → dense → output.

        Args:
            x: torch.Tensor of shape (batch, 1, 28, 28) — normalised pixel values

        Returns:
            log_probs: (batch, 10) — log-probabilities for each digit class
        """
        # Block 1: conv → relu → pool
        x = self.pool(F.relu(self.conv1(x)))   # (B,1,28,28) → (B,16,14,14)

        # Block 2: conv → relu → pool
        x = self.pool(F.relu(self.conv2(x)))   # (B,16,14,14) → (B,32,7,7)

        # Flatten spatial dimensions into one vector per image
        x = x.view(x.size(0), -1)             # (B,32,7,7) → (B,1568)

        # Fully-connected classifier
        x = F.relu(self.fc1(x))               # (B,1568) → (B,128)
        x = self.dropout(x)
        x = self.fc2(x)                        # (B,128) → (B,10) raw logits

        # Log-softmax for numerical stability with NLLLoss
        return F.log_softmax(x, dim=1)         # (B,10) log-probabilities


def count_parameters(model):
    """Count and print trainable parameters per layer.

    Args:
        model: nn.Module

    Returns:
        total: int — total number of trainable parameters
    """
    total = 0
    print("\nParameter count by layer:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            n = param.numel()
            total += n
            print(f"  {name:<25} {list(param.shape)}  → {n:,} params")
    print(f"  {'TOTAL':<25}              → {total:,} params")
    return total
