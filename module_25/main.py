"""module_25/main.py — BatchNorm + Dropout ablation study on MNIST.

Trains three model variants back-to-back and compares their validation accuracy:
  1. Baseline CNN (Module 24 architecture — no BN, dropout=0.3)
  2. BN only (no dropout)
  3. BN + Dropout=0.4

Saves a bar-chart comparison and training curves.
Run: python main.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
# also need Module 24's cnn_model for the baseline
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "module_24"))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cnn_model import DigitCNN                  # Module 24 baseline
from bn_dropout_model import DigitCNNwithBN, count_parameters

# ── Constants ────────────────────────────────────────────────────────────────
RANDOM_SEED   = 42
BATCH_SIZE    = 64
EPOCHS        = 5
LR            = 1e-3
DATA_DIR      = os.path.join(os.path.dirname(__file__), "..", "module_24", "data")
PLOTS_DIR     = "plots"
FIGURE_DPI    = 120


def get_dataloaders() -> tuple:
    """Return MNIST train/test DataLoaders.

    Reuses the same DATA_DIR as Module 24 to avoid re-downloading.

    Returns:
        (train_loader, test_loader)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_set = datasets.MNIST(DATA_DIR, train=True,  download=True, transform=transform)
    test_set  = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)

    g = torch.Generator().manual_seed(RANDOM_SEED)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  generator=g)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, device) -> tuple:
    """Train for one epoch.

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
    """Evaluate without gradient computation.

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


def run_experiment(name, model, train_loader, test_loader, device) -> dict:
    """Train a model for EPOCHS and return history dict.

    Args:
        name:         label for printing
        model:        nn.Module
        train_loader: DataLoader
        test_loader:  DataLoader
        device:       'cpu' or 'cuda'

    Returns:
        dict with keys train_acc, val_acc, train_loss, val_loss (lists of length EPOCHS)
    """
    model = model.to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    history = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}
    print(f"\n{'─'*50}")
    print(f"Experiment: {name}")
    print(f"{'─'*50}")
    print(f"{'Epoch':>6} {'Tr Loss':>9} {'Tr Acc':>8} {'Va Loss':>9} {'Va Acc':>8}")
    print("-" * 46)

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
    """Save bar chart of final validation accuracy per variant.

    Args:
        results: {name: history_dict}
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)
    names  = list(results.keys())
    accs   = [results[n]["val_acc"][-1] * 100 for n in names]
    colours = ["#58a6ff", "#3fb950", "#bc8cff"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor="#0d1117")

    # Bar chart — final accuracy
    ax = axes[0]
    ax.set_facecolor("#0d1117")
    bars = ax.bar(names, accs, color=colours[:len(names)])
    ax.set_ylim(97, 100)
    ax.set_ylabel("Val Accuracy (%)", color="#e6edf3")
    ax.set_title("Final Validation Accuracy", color="#e6edf3")
    ax.tick_params(colors="#e6edf3")
    ax.spines[:].set_color("#30363d")
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{acc:.2f}%", ha="center", va="bottom", color="#e6edf3", fontsize=9)

    # Learning curves — val accuracy
    ax = axes[1]
    ax.set_facecolor("#0d1117")
    epochs = range(1, EPOCHS + 1)
    for (name, hist), colour in zip(results.items(), colours):
        ax.plot(epochs, [a * 100 for a in hist["val_acc"]],
                "o-", color=colour, label=name)
    ax.set_xlabel("Epoch", color="#e6edf3")
    ax.set_ylabel("Val Accuracy (%)", color="#e6edf3")
    ax.set_title("Learning Curves", color="#e6edf3")
    ax.legend(facecolor="#161b22", labelcolor="#e6edf3")
    ax.tick_params(colors="#e6edf3")
    ax.spines[:].set_color("#30363d")

    plt.tight_layout()
    fname = f"{PLOTS_DIR}/bn_dropout_comparison.png"
    plt.savefig(fname, dpi=FIGURE_DPI, facecolor=fig.get_facecolor())
    plt.close()
    print(f"\nSaved → {fname}")


def demo_bn_statistics(model, loader, device):
    """Show that BN normalises activations to ~N(0,1) before ReLU.

    Runs one batch forward, hooks conv1 output before and after BN,
    and prints mean/std of both.
    """
    model.eval()
    pre_bn_stats  = {}
    post_bn_stats = {}

    def make_hook(store, key):
        def hook(module, inp, out):
            store[key] = (out.mean().item(), out.std().item())
        return hook

    h1 = model.conv1.register_forward_hook(make_hook(pre_bn_stats,  "conv1"))
    h2 = model.bn1.register_forward_hook(  make_hook(post_bn_stats, "bn1"))

    X, _ = next(iter(loader))
    with torch.no_grad():
        model(X.to(device))

    h1.remove()
    h2.remove()

    print("\nBatch Norm effect on conv1 activations:")
    print(f"  Before BN: mean={pre_bn_stats['conv1'][0]:+.4f}  std={pre_bn_stats['conv1'][1]:.4f}")
    print(f"  After  BN: mean={post_bn_stats['bn1'][0]:+.4f}  std={post_bn_stats['bn1'][1]:.4f}")
    print("  → BN re-centres activations near 0 and rescales to unit variance.")


def main():
    torch.manual_seed(RANDOM_SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 52)
    print("MODULE 25 — BatchNorm + Dropout (MNIST ablation)")
    print("=" * 52)
    print(f"Device: {device}")

    train_loader, test_loader = get_dataloaders()
    print(f"Train: {len(train_loader.dataset):,}  Test: {len(test_loader.dataset):,}")

    # Show parameter structure of the BN model
    count_parameters(DigitCNNwithBN())

    # ── Three experiments ─────────────────────────────────────────────────────
    experiments = {
        "Baseline (no BN)":   DigitCNN(dropout_p=0.3),
        "BN only":            DigitCNNwithBN(dropout_p=0.0),
        "BN + Dropout=0.4":   DigitCNNwithBN(dropout_p=0.4),
    }

    results = {}
    for name, model in experiments.items():
        torch.manual_seed(RANDOM_SEED)   # same init for fair comparison
        results[name] = run_experiment(name, model, train_loader, test_loader, device)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 52)
    print("Final test accuracy summary:")
    for name, hist in results.items():
        print(f"  {name:<25} {hist['val_acc'][-1]*100:.2f}%")

    # ── BN statistics demo ────────────────────────────────────────────────────
    bn_model = DigitCNNwithBN(dropout_p=0.4).to(device)
    # quick 1-epoch train so BN running stats are initialised
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(bn_model.parameters(), lr=LR)
    train_one_epoch(bn_model, train_loader, criterion, optimizer, device)
    demo_bn_statistics(bn_model, test_loader, device)

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_comparison(results)
    print("Done.")


if __name__ == "__main__":
    main()
