"""module_26/transfer_model.py — Transfer learning with pretrained ResNet-18.

Strategy:
  1. Load ResNet-18 pretrained on ImageNet (1.28M images, 1000 classes).
  2. Freeze the entire convolutional backbone — those weights are not updated.
  3. Replace the final fully-connected layer with a new head sized for our task.
  4. Train only the new head for a few epochs.

Why ResNet-18?
  Smallest standard ResNet — downloads fast (~45 MB), runs on CPU in < 2 min/epoch.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import List


def build_feature_extractor(freeze: bool = True) -> nn.Module:
    """Load pretrained ResNet-18 and optionally freeze its backbone.

    Args:
        freeze: if True, set requires_grad=False on all backbone params

    Returns:
        model with original fc layer still intact (caller replaces it)
    """
    # weights=DEFAULT fetches the latest recommended pretrained weights
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    if freeze:
        for param in model.parameters():
            param.requires_grad = False   # backbone weights won't move during fine-tuning

    return model


def build_transfer_model(num_classes: int, hidden_size: int = 256,
                         dropout_p: float = 0.4, freeze_backbone: bool = True) -> nn.Module:
    """Build a ResNet-18 with a custom classification head.

    Replaces ResNet's final Linear(512, 1000) with:
      Linear(512, hidden_size) → BN → ReLU → Dropout → Linear(hidden_size, num_classes)

    Args:
        num_classes:     number of output classes for our dataset
        hidden_size:     hidden units in the new head
        dropout_p:       dropout probability
        freeze_backbone: if True, only the new head is trained

    Returns:
        nn.Module ready for training
    """
    model = build_feature_extractor(freeze=freeze_backbone)

    # ResNet's penultimate layer outputs 512 features (for ResNet-18/34)
    in_features = model.fc.in_features   # 512

    # Build a deeper head than the default single linear layer
    model.fc = nn.Sequential(
        nn.Linear(in_features, hidden_size),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout_p),
        nn.Linear(hidden_size, num_classes),
    )

    return model


def count_trainable(model: nn.Module) -> None:
    """Print total vs trainable parameter counts.

    Shows the contrast between the frozen backbone and the small trainable head.

    Args:
        model: nn.Module
    """
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = total - trainable

    print(f"\nParameter summary:")
    print(f"  Total      : {total:>10,}")
    print(f"  Frozen     : {frozen:>10,}  (backbone — not updated)")
    print(f"  Trainable  : {trainable:>10,}  (new head — updated each step)")
    print(f"  Ratio      : {trainable/total*100:.1f}% of parameters are trained")
