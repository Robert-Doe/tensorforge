"""module_26/main.py — Transfer learning demo on CIFAR-10.

Uses CIFAR-10 (60,000 32×32 colour images, 10 classes) as a proxy for a
"small custom dataset." Pretrained ResNet-18 is fine-tuned for 5 epochs.

Three experiments:
  1. Frozen backbone — only new head trained (fast, ~95% acc)
  2. Full fine-tuning — all weights updated with a low LR (slower, often +1%)
  3. From scratch — ResNet-18 with random init (shows how much pretraining helps)

Run: python main.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from transfer_model import build_transfer_model, count_trainable

# ── Constants ────────────────────────────────────────────────────────────────
RANDOM_SEED    = 42
BATCH_SIZE     = 64
EPOCHS         = 5
LR_HEAD        = 1e-3     # learning rate when only head is trained
LR_FINETUNE    = 1e-4     # low LR for full fine-tuning (backbone already near optimal)
NUM_CLASSES    = 10       # CIFAR-10 classes
DATA_DIR       = "data"
PLOTS_DIR      = "plots"
FIGURE_DPI     = 120

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# ResNet was pretrained on ImageNet; use ImageNet mean/std for normalisation
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def get_dataloaders() -> tuple:
    """Download CIFAR-10 and return train/test DataLoaders.

    Resizes 32×32 images to 224×224 because ResNet-18 expects at least 224 input.
    On CPU this makes the run slower but is required for correct feature maps.

    Returns:
        (train_loader, test_loader)
    """
    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),      # cheap augmentation
        transforms.RandomCrop(224, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_set = datasets.CIFAR10(DATA_DIR, train=True,  download=True, transform=train_transform)
    test_set  = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=val_transform)

    g = torch.Generator().manual_seed(RANDOM_SEED)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  generator=g, num_workers=0)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return train_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, device) -> tuple:
    """One epoch of training.

    Returns:
        (mean_loss, accuracy)
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out  = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct    += (out.argmax(1) == y).sum().item()
        total      += len(y)
    return total_loss / total, correct / total


def eval_one_epoch(model, loader, criterion, device) -> tuple:
    """Evaluate without updating weights.

    Returns:
        (mean_loss, accuracy)
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out  = model(X)
            total_loss += criterion(out, y).item() * len(y)
            correct    += (out.argmax(1) == y).sum().item()
            total      += len(y)
    return total_loss / total, correct / total


def run_experiment(name, model, train_loader, test_loader, device, lr) -> dict:
    """Train model and return history.

    Args:
        name:  label for printing
        model: nn.Module
        lr:    learning rate for Adam
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()   # raw logits — ResNet doesn't apply softmax
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),  # only trainable params
        lr=lr,
    )

    history = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}
    print(f"\n{'─'*52}")
    print(f"Experiment: {name}")
    count_trainable(model)
    print(f"{'─'*52}")
    print(f"{'Epoch':>6} {'Tr Loss':>9} {'Tr Acc':>8} {'Va Loss':>9} {'Va Acc':>8}")
    print("-" * 48)

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = eval_one_epoch (model, test_loader,  criterion,            device)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)
        print(f"{epoch:>6} {tr_loss:>9.4f} {tr_acc:>8.4f} {va_loss:>9.4f} {va_acc:>8.4f}")

    return history


def plot_comparison(results: dict):
    """Save bar chart + learning curves."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    names   = list(results.keys())
    accs    = [results[n]["val_acc"][-1] * 100 for n in names]
    colours = ["#58a6ff", "#3fb950", "#f85149"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor="#0d1117")

    ax = axes[0]
    ax.set_facecolor("#0d1117")
    bars = ax.bar(names, accs, color=colours[:len(names)])
    ax.set_ylabel("Val Accuracy (%)", color="#e6edf3")
    ax.set_title("Transfer Learning Comparison", color="#e6edf3")
    ax.tick_params(colors="#e6edf3")
    ax.spines[:].set_color("#30363d")
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f"{acc:.1f}%", ha="center", color="#e6edf3", fontsize=9)

    ax = axes[1]
    ax.set_facecolor("#0d1117")
    epochs = range(1, EPOCHS + 1)
    for (name, hist), colour in zip(results.items(), colours):
        ax.plot(epochs, [a * 100 for a in hist["val_acc"]], "o-", color=colour, label=name)
    ax.set_xlabel("Epoch", color="#e6edf3")
    ax.set_ylabel("Val Accuracy (%)", color="#e6edf3")
    ax.set_title("Learning Curves", color="#e6edf3")
    ax.legend(facecolor="#161b22", labelcolor="#e6edf3", fontsize=8)
    ax.tick_params(colors="#e6edf3")
    ax.spines[:].set_color("#30363d")

    plt.tight_layout()
    fname = f"{PLOTS_DIR}/transfer_comparison.png"
    plt.savefig(fname, dpi=FIGURE_DPI, facecolor=fig.get_facecolor())
    plt.close()
    print(f"\nSaved → {fname}")


def main():
    torch.manual_seed(RANDOM_SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 54)
    print("MODULE 26 — Transfer Learning (ResNet-18 on CIFAR-10)")
    print("=" * 54)
    print(f"Device: {device}")
    print("Note: CIFAR-10 images are resized 32→224 for ResNet input.")
    print("      Downloading ResNet weights on first run (~45 MB).")

    train_loader, test_loader = get_dataloaders()
    print(f"\nTrain: {len(train_loader.dataset):,}  Test: {len(test_loader.dataset):,}")

    # ── Three experiments ─────────────────────────────────────────────────────
    results = {}

    # 1. Frozen backbone — fastest
    torch.manual_seed(RANDOM_SEED)
    frozen_model = build_transfer_model(NUM_CLASSES, freeze_backbone=True)
    results["Frozen backbone"] = run_experiment(
        "Frozen backbone", frozen_model, train_loader, test_loader, device, LR_HEAD
    )

    # 2. Full fine-tuning — unfreeze everything, very low LR
    torch.manual_seed(RANDOM_SEED)
    finetune_model = build_transfer_model(NUM_CLASSES, freeze_backbone=False)
    results["Full fine-tune"] = run_experiment(
        "Full fine-tune", finetune_model, train_loader, test_loader, device, LR_FINETUNE
    )

    # 3. From scratch — no pretrained weights
    torch.manual_seed(RANDOM_SEED)
    from torchvision import models
    scratch_model = models.resnet18(weights=None)
    scratch_model.fc = nn.Linear(scratch_model.fc.in_features, NUM_CLASSES)
    results["From scratch"] = run_experiment(
        "From scratch", scratch_model, train_loader, test_loader, device, LR_HEAD
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 54)
    print("Final test accuracy summary:")
    for name, hist in results.items():
        print(f"  {name:<22} {hist['val_acc'][-1]*100:.2f}%")

    plot_comparison(results)
    print("Done.")


if __name__ == "__main__":
    main()
