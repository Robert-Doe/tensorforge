"""module_22/model.py — M18 MLP rewritten in PyTorch.

Every line maps directly to a scratch implementation:
  nn.Linear      ↔  DenseLayer (M18)
  nn.ReLU        ↔  ReLULayer  (M18)
  nn.Dropout     ↔  DropoutLayer (M21)
  nn.Sigmoid     ↔  SigmoidLayer (M18)
  loss.backward()↔  MLP.backward() + chain rule (M17)
  optimizer.step()↔ MLP._update_weights() (M18)
"""

import torch
import torch.nn as nn


class DetectiveMLP(nn.Module):
    """Binary classifier built with PyTorch nn.Module.

    Mirrors the RegularisedMLP from M21 architecture:
        Linear(6→16) → ReLU → Dropout → Linear(16→8) → ReLU → Dropout → Linear(8→1) → Sigmoid

    Attributes:
        net: nn.Sequential container of all layers
    """

    def __init__(self, n_features=6, hidden_sizes=None, dropout_p=0.2):
        """Build the layer stack.

        Args:
            n_features:   number of input features
            hidden_sizes: list of hidden widths (default [16, 8])
            dropout_p:    dropout probability for hidden layers
        """
        super().__init__()                  # mandatory for nn.Module subclasses
        if hidden_sizes is None:
            hidden_sizes = [16, 8]

        layers = []
        sizes  = [n_features] + hidden_sizes
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))   # ↔ DenseLayer
            layers.append(nn.ReLU())                            # ↔ ReLULayer
            if dropout_p > 0.0:
                layers.append(nn.Dropout(p=dropout_p))         # ↔ DropoutLayer

        layers.append(nn.Linear(sizes[-1], 1))                 # output neuron
        layers.append(nn.Sigmoid())                             # ↔ SigmoidLayer

        self.net = nn.Sequential(*layers)   # chains all layers in order

    def forward(self, x):
        """Run input through every layer.

        PyTorch calls this automatically — never call forward() directly;
        call the model instance: output = model(x).

        Args:
            x: torch.Tensor of shape (batch_size, n_features)

        Returns:
            out: torch.Tensor of shape (batch_size, 1) — probabilities
        """
        return self.net(x)
