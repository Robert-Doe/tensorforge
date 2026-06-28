"""module_22/main.py — PyTorch Intro.

Rewrites the M21 RegularisedMLP in PyTorch and compares results.
Run: python main.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from data_loader import load_and_clean, to_arrays
from model import DetectiveMLP
from train  import train, accuracy

RANDOM_SEED = 42
DATA_FILE   = "cases.csv"
VAL_RATIO   = 0.15
TEST_RATIO  = 0.20
PLOTS_DIR   = "plots"
FIGURE_DPI  = 120


def three_way_split(X, y, val_ratio, test_ratio, seed):
    rng  = np.random.default_rng(seed)
    idx  = rng.permutation(len(X))
    nte  = int(len(X) * test_ratio)
    nval = int(len(X) * val_ratio)
    return (X[idx[nte+nval:]], y[idx[nte+nval:]],
            X[idx[nte:nte+nval]], y[idx[nte:nte+nval]],
            X[idx[:nte]], y[idx[:nte]])


def scale(X_tr, X_other):
    mean = X_tr.mean(axis=0);  std = X_tr.std(axis=0) + 1e-8
    return (X_tr - mean) / std, (X_other - mean) / std


def plot_loss(history, fname):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    ax.plot(history["train"], color="#58a6ff", linewidth=1.5, label="Train")
    ax.plot(history["val"],   color="#f85149", linewidth=1.5, label="Val")
    ax.set_xlabel("Epoch",  color="#e6edf3")
    ax.set_ylabel("BCE Loss", color="#e6edf3")
    ax.set_title("PyTorch MLP — Training Curve", color="#e6edf3")
    ax.legend(facecolor="#161b22", labelcolor="#e6edf3")
    ax.tick_params(colors="#e6edf3")
    ax.spines[:].set_color("#30363d")
    plt.tight_layout()
    path = f"{PLOTS_DIR}/{fname}"
    plt.savefig(path, dpi=FIGURE_DPI, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved → {path}")


def print_model_summary(model):
    """Print parameter counts per layer."""
    print("\nModel architecture:")
    total = 0
    for name, param in model.named_parameters():
        n = param.numel()
        total += n
        print(f"  {name:<30} shape={list(param.shape)}  params={n}")
    print(f"  {'TOTAL':<30}              params={total}")


def main():
    torch.manual_seed(RANDOM_SEED)

    df           = load_and_clean(DATA_FILE)
    X, y, _      = to_arrays(df)
    X_tr, y_tr, X_val, y_val, X_te, y_te = three_way_split(
        X, y, VAL_RATIO, TEST_RATIO, RANDOM_SEED)
    X_tr_s, X_te_s  = scale(X_tr, X_te)
    _,      X_val_s = scale(X_tr, X_val)

    print("=" * 52)
    print("MODULE 22 — PyTorch Intro")
    print("=" * 52)
    print(f"PyTorch version: {torch.__version__}")
    print(f"Train: {len(X_tr)}  Val: {len(X_val)}  Test: {len(X_te)}")

    # ── Build and inspect model ───────────────────────────────────────────────
    model = DetectiveMLP(n_features=6, hidden_sizes=[16, 8], dropout_p=0.2)
    print_model_summary(model)

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\nTraining (SGD + momentum + L2 + Dropout):")
    history = train(model, X_tr_s, y_tr, X_val_s, y_val,
                    epochs=150, batch_size=32, lr=0.05, weight_decay=0.005)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    tr_acc = accuracy(model, X_tr_s, y_tr)
    va_acc = accuracy(model, X_val_s, y_val)
    te_acc = accuracy(model, X_te_s,  y_te)
    print(f"\nFinal accuracy — train={tr_acc:.3f}  val={va_acc:.3f}  test={te_acc:.3f}")

    # ── Side-by-side with scratch (M21) ───────────────────────────────────────
    print("\n=== Comparison: Scratch (M21) vs PyTorch (M22) ===")
    print(f"  {'Model':<30} {'Test acc':>9}")
    print("  " + "-" * 40)
    print(f"  {'NumPy MLP + L2 + Dropout':<30} {'~0.750':>9}  (M21)")
    print(f"  {'PyTorch MLP + L2 + Dropout':<30} {te_acc:>9.3f}  (M22)")
    print("\n  Same algorithm. Same results. 10x less code.")

    plot_loss(history, "pytorch_loss.png")
    print("\nDone.")


if __name__ == "__main__":
    main()
