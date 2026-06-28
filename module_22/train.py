"""module_22/train.py — PyTorch training loop with autograd.

Demonstrates:
  - Converting NumPy arrays to torch.Tensor
  - model.train() / model.eval() toggle
  - loss.backward() for automatic gradient computation
  - optimizer.step() + optimizer.zero_grad()
  - Recording train and val loss per epoch
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# ── Constants ────────────────────────────────────────────────────────────────
RANDOM_SEED = 42
BATCH_SIZE  = 32
EPOCHS      = 150
LEARNING_RATE = 0.05


def numpy_to_tensor(X, y):
    """Convert NumPy arrays to float32 torch Tensors.

    PyTorch default dtype is float32. NumPy default is float64.
    Mismatched dtypes cause runtime errors — always convert explicitly.

    Args:
        X: (n_samples, n_features) NumPy array
        y: (n_samples,)            NumPy array

    Returns:
        X_t: FloatTensor (n_samples, n_features)
        y_t: FloatTensor (n_samples, 1)  — column vector for BCELoss
    """
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)   # (n,) → (n,1)
    return X_t, y_t


def make_dataloader(X_t, y_t, batch_size, shuffle=True):
    """Wrap tensors in a DataLoader for mini-batch iteration.

    DataLoader handles shuffling, batching, and dropping the last incomplete
    batch if drop_last=True (we leave it True to keep batch sizes consistent).

    Args:
        X_t:        FloatTensor
        y_t:        FloatTensor
        batch_size: int
        shuffle:    bool — True for training, False for validation

    Returns:
        torch.utils.data.DataLoader
    """
    dataset = torch.utils.data.TensorDataset(X_t, y_t)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,       # keep the last (smaller) batch
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )


def train_epoch(model, loader, criterion, optimizer):
    """Run one epoch of mini-batch training.

    Args:
        model:     nn.Module (DetectiveMLP)
        loader:    DataLoader over training data
        criterion: loss function (nn.BCELoss)
        optimizer: optim.SGD or optim.Adam

    Returns:
        mean_loss: float — average batch loss for this epoch
    """
    model.train()                          # activates Dropout, BatchNorm (train mode)
    batch_losses = []

    for X_batch, y_batch in loader:
        optimizer.zero_grad()              # clear gradients from previous step
        # ↑ forgetting this causes gradient accumulation — a common bug

        preds = model(X_batch)             # forward pass (calls model.forward)
        loss  = criterion(preds, y_batch)  # compute scalar loss

        loss.backward()                    # autograd: compute all gradients
        # ↑ this replaces our entire backward() implementation from M17-M21

        optimizer.step()                   # apply gradients to weights
        # ↑ this replaces _update_weights() from M18

        batch_losses.append(loss.item())   # .item() extracts Python float from tensor

    return float(np.mean(batch_losses))


def eval_epoch(model, loader, criterion):
    """Evaluate loss on a dataset without computing gradients.

    Args:
        model:     nn.Module
        loader:    DataLoader
        criterion: loss function

    Returns:
        mean_loss: float
    """
    model.eval()                           # deactivates Dropout (eval mode)
    losses = []
    with torch.no_grad():                  # disables gradient tracking (saves memory)
        for X_batch, y_batch in loader:
            preds = model(X_batch)
            loss  = criterion(preds, y_batch)
            losses.append(loss.item())
    return float(np.mean(losses))


def train(model, X_train, y_train, X_val, y_val,
          epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE,
          weight_decay=0.0):
    """Full training loop with train/val loss tracking.

    Args:
        model:        DetectiveMLP instance
        X_train, y_train: training NumPy arrays
        X_val,   y_val:   validation NumPy arrays
        epochs:       number of epochs
        batch_size:   mini-batch size
        lr:           learning rate
        weight_decay: L2 regularisation (passed to optimizer as weight_decay)

    Returns:
        history: dict with 'train' and 'val' loss lists
    """
    X_tr_t, y_tr_t = numpy_to_tensor(X_train, y_train)
    X_va_t, y_va_t = numpy_to_tensor(X_val,   y_val)

    train_loader = make_dataloader(X_tr_t, y_tr_t, batch_size, shuffle=True)
    val_loader   = make_dataloader(X_va_t, y_va_t, batch_size, shuffle=False)

    criterion = nn.BCELoss()               # binary cross-entropy (same as M16-M21)
    optimizer = optim.SGD(                 # stochastic gradient descent
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,         # L2 penalty — same as l2_lambda in M21
        momentum=0.9,                      # accumulate velocity for faster convergence
    )

    history = {"train": [], "val": []}

    for epoch in range(epochs):
        tr_loss = train_epoch(model, train_loader, criterion, optimizer)
        va_loss = eval_epoch(model,  val_loader,   criterion)
        history["train"].append(tr_loss)
        history["val"].append(va_loss)

        if (epoch + 1) % 30 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}  "
                  f"train={tr_loss:.4f}  val={va_loss:.4f}")

    return history


def accuracy(model, X, y):
    """Compute accuracy of model on NumPy arrays.

    Args:
        model: trained DetectiveMLP
        X:     (n_samples, n_features) NumPy array
        y:     (n_samples,)            NumPy array

    Returns:
        float: fraction of correct predictions
    """
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        probs = model(X_t).squeeze().numpy()    # (n,1) → (n,) NumPy
    preds = (probs >= 0.5).astype(int)
    return float(np.mean(preds == y))
