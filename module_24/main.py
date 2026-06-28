"""module_24/main.py — Full CNN trained on MNIST.

Downloads MNIST via torchvision (first run only, ~12 MB).
Trains for 5 epochs — expect ~98% test accuracy.
Run: python main.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cnn_model import DigitCNN, count_parameters

# ── Constants ────────────────────────────────────────────────────────────────
RANDOM_SEED  = 42
BATCH_SIZE   = 64
EPOCHS       = 5
LEARNING_RATE = 1e-3      # Adam default — much lower than our M22 SGD rate
DATA_DIR     = "data"     # MNIST downloads here
PLOTS_DIR    = "plots"
FIGURE_DPI   = 120


def get_dataloaders():
    """Download MNIST and return train/test DataLoaders.

    Normalisation: mean=0.1307, std=0.3081 are the MNIST channel statistics.
    Computed once over the full training set and hardcoded here.

    Returns:
        train_loader, test_loader: DataLoader objects
    """
    transform = transforms.Compose([
        transforms.ToTensor(),                           # PIL image → (1,28,28) tensor in [0,1]
        transforms.Normalize((0.1307,), (0.3081,)),     # zero-mean, unit-std per channel
    ])

    train_set = datasets.MNIST(DATA_DIR, train=True,  download=True, transform=transform)
    test_set  = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              generator=torch.Generator().manual_seed(RANDOM_SEED))
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader


def train_epoch(model, loader, criterion, optimizer, device):
    """One epoch of training.

    Args:
        model:     DigitCNN
        loader:    training DataLoader
        criterion: nn.NLLLoss
        optimizer: optim.Adam
        device:    'cpu' or 'cuda'

    Returns:
        (mean_loss, accuracy): floats
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        log_probs = model(X_batch)                      # (B,10) log-probabilities
        loss      = criterion(log_probs, y_batch)       # NLLLoss expects log-probs
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y_batch)
        preds       = log_probs.argmax(dim=1)           # class with highest log-prob
        correct    += (preds == y_batch).sum().item()
        total      += len(y_batch)

    return total_loss / total, correct / total


def eval_epoch(model, loader, criterion, device):
    """Evaluate on a DataLoader without updating weights."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            log_probs  = model(X_batch)
            loss       = criterion(log_probs, y_batch)
            total_loss += loss.item() * len(y_batch)
            preds       = log_probs.argmax(dim=1)
            correct    += (preds == y_batch).sum().item()
            total      += len(y_batch)

    return total_loss / total, correct / total


def plot_training_curves(history):
    """Save loss and accuracy curves."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor="#0d1117")
    for ax, (tr_key, va_key, ylabel) in zip(axes, [
        ("train_loss", "val_loss", "Loss"),
        ("train_acc",  "val_acc",  "Accuracy"),
    ]):
        ax.set_facecolor("#0d1117")
        ax.plot(epochs, history[tr_key], "o-", color="#58a6ff", label="Train")
        ax.plot(epochs, history[va_key], "s-", color="#f85149", label="Val")
        ax.set_xlabel("Epoch",  color="#e6edf3")
        ax.set_ylabel(ylabel,   color="#e6edf3")
        ax.set_title(f"CNN {ylabel}", color="#e6edf3")
        ax.legend(facecolor="#161b22", labelcolor="#e6edf3")
        ax.tick_params(colors="#e6edf3")
        ax.spines[:].set_color("#30363d")

    plt.tight_layout()
    fname = f"{PLOTS_DIR}/cnn_training.png"
    plt.savefig(fname, dpi=FIGURE_DPI, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved → {fname}")


def plot_predictions(model, test_loader, device):
    """Save a grid of 16 test images with predicted and true labels."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    model.eval()
    X_batch, y_batch = next(iter(test_loader))
    X_batch = X_batch[:16]
    y_batch = y_batch[:16]

    with torch.no_grad():
        log_probs = model(X_batch.to(device))
        preds     = log_probs.argmax(dim=1).cpu()

    fig, axes = plt.subplots(4, 4, figsize=(8, 8), facecolor="#0d1117")
    for ax, img, pred, true in zip(axes.ravel(), X_batch, preds, y_batch):
        ax.set_facecolor("#0d1117")
        ax.imshow(img.squeeze(), cmap="gray")
        colour = "#3fb950" if pred == true else "#f85149"
        ax.set_title(f"pred={pred.item()} true={true.item()}",
                     color=colour, fontsize=9)
        ax.axis("off")

    plt.suptitle("CNN Predictions (green=correct, red=wrong)", color="#e6edf3")
    plt.tight_layout()
    fname = f"{PLOTS_DIR}/cnn_predictions.png"
    plt.savefig(fname, dpi=FIGURE_DPI, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved → {fname}")


def main():
    torch.manual_seed(RANDOM_SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 52)
    print("MODULE 24 — Full CNN in PyTorch (MNIST)")
    print("=" * 52)
    print(f"Device: {device}")

    # ── Data ─────────────────────────────────────────────────────────────────
    print("\nLoading MNIST (downloads ~12 MB on first run)...")
    train_loader, test_loader = get_dataloaders()
    print(f"Train: {len(train_loader.dataset):,}  Test: {len(test_loader.dataset):,}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = DigitCNN(dropout_p=0.3).to(device)
    count_parameters(model)

    criterion = nn.NLLLoss()                          # expects log-probabilities
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ── Training ──────────────────────────────────────────────────────────────
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    print(f"\nTraining for {EPOCHS} epochs (Adam lr={LEARNING_RATE}):")
    print(f"{'Epoch':>6} {'Tr Loss':>9} {'Tr Acc':>8} {'Va Loss':>9} {'Va Acc':>8}")
    print("-" * 46)

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = eval_epoch(model,  test_loader,  criterion,            device)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        print(f"{epoch:>6} {tr_loss:>9.4f} {tr_acc:>8.4f} {va_loss:>9.4f} {va_acc:>8.4f}")

    print(f"\nFinal test accuracy: {history['val_acc'][-1]*100:.2f}%")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_training_curves(history)
    plot_predictions(model, test_loader, device)
    print("\nDone.")


if __name__ == "__main__":
    main()
