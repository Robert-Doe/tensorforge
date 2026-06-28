"""module_20/main.py — Training Loops & Mini-Batches.

Compares full-batch GD vs mini-batch SGD and shows how
validation loss reveals overfitting.
Run: python main.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_loader import load_and_clean, to_arrays
from training_loop import MiniBatchTrainer

# ── Constants ────────────────────────────────────────────────────────────────
DATA_FILE  = "cases.csv"
RANDOM_SEED = 42
VAL_RATIO   = 0.15
TEST_RATIO  = 0.20
PLOTS_DIR   = "plots"
FIGURE_DPI  = 120


def scale(X_tr, X_other):
    """Standard-scale using X_tr statistics only."""
    mean = X_tr.mean(axis=0)
    std  = X_tr.std(axis=0) + 1e-8
    return (X_tr - mean) / std, (X_other - mean) / std


def three_way_split(X, y, val_ratio, test_ratio, seed):
    """Split into train / val / test sets."""
    rng      = np.random.default_rng(seed)
    idx      = rng.permutation(len(X))
    n_test   = int(len(X) * test_ratio)
    n_val    = int(len(X) * val_ratio)
    test_idx = idx[:n_test]
    val_idx  = idx[n_test:n_test + n_val]
    tr_idx   = idx[n_test + n_val:]
    return (X[tr_idx], y[tr_idx],
            X[val_idx], y[val_idx],
            X[test_idx], y[test_idx])


def plot_loss_curves(trainer_full, trainer_mini):
    """Save training loss comparison: full-batch vs mini-batch."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor="#0d1117")

    for ax, trainer, title in zip(axes,
        [trainer_full, trainer_mini],
        ["Full-batch GD (all samples per update)",
         "Mini-batch SGD (batch_size=32)"]):
        ax.set_facecolor("#0d1117")
        ax.plot(trainer.train_loss_per_epoch, color="#58a6ff",
                linewidth=1.5, label="Train loss")
        if trainer.val_loss_per_epoch:
            ax.plot(trainer.val_loss_per_epoch, color="#f85149",
                    linewidth=1.5, label="Val loss")
        ax.set_title(title, color="#e6edf3")
        ax.set_xlabel("Epoch",  color="#e6edf3")
        ax.set_ylabel("Loss",   color="#e6edf3")
        ax.legend(facecolor="#161b22", labelcolor="#e6edf3")
        ax.tick_params(colors="#e6edf3")
        ax.spines[:].set_color("#30363d")

    plt.tight_layout()
    fname = f"{PLOTS_DIR}/minibatch_loss.png"
    plt.savefig(fname, dpi=FIGURE_DPI, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved → {fname}")


def plot_step_loss(step_losses):
    """Save the noisy per-step loss curve — shows SGD noise visually."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 3), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    ax.plot(step_losses, color="#d29922", linewidth=0.8, alpha=0.7)
    ax.set_xlabel("Step (mini-batch update)", color="#e6edf3")
    ax.set_ylabel("Batch loss",              color="#e6edf3")
    ax.set_title("SGD Step-Level Loss (noisy but fast)", color="#e6edf3")
    ax.tick_params(colors="#e6edf3")
    ax.spines[:].set_color("#30363d")
    plt.tight_layout()
    fname = f"{PLOTS_DIR}/step_loss.png"
    plt.savefig(fname, dpi=FIGURE_DPI, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved → {fname}")


def main():
    """Compare full-batch vs mini-batch training."""
    df              = load_and_clean(DATA_FILE)
    X, y, _         = to_arrays(df)

    X_tr, y_tr, X_val, y_val, X_te, y_te = three_way_split(
        X, y, VAL_RATIO, TEST_RATIO, RANDOM_SEED
    )
    X_tr_s,  X_te_s  = scale(X_tr, X_te)
    _,       X_val_s = scale(X_tr, X_val)

    print("=" * 52)
    print("MODULE 20 — Training Loops & Mini-Batches")
    print("=" * 52)
    print(f"Train: {len(X_tr)}  Val: {len(X_val)}  Test: {len(X_te)}")

    # ── Full-batch baseline ───────────────────────────────────────────────────
    print("\n--- Full-batch GD (all samples at once) ---")
    full = MiniBatchTrainer(n_features=6, hidden_sizes=[16, 8],
                            lr=0.05, epochs=100,
                            batch_size=len(X_tr))   # batch = entire dataset
    full.fit(X_tr_s, y_tr, X_val_s, y_val)
    full_acc = np.mean(full.predict(X_te_s) == y_te)
    print(f"Test accuracy (full-batch) : {full_acc:.3f}")

    # ── Mini-batch SGD ────────────────────────────────────────────────────────
    print("\n--- Mini-batch SGD (batch_size=32) ---")
    mini = MiniBatchTrainer(n_features=6, hidden_sizes=[16, 8],
                            lr=0.05, epochs=100, batch_size=32)
    mini.fit(X_tr_s, y_tr, X_val_s, y_val)
    mini_acc = np.mean(mini.predict(X_te_s) == y_te)
    print(f"Test accuracy (mini-batch) : {mini_acc:.3f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    n_steps_full = 100 * 1      # 1 update per epoch
    n_steps_mini = len(mini.step_losses)
    print(f"\nSteps taken:")
    print(f"  Full-batch: {n_steps_full} updates (1 per epoch)")
    print(f"  Mini-batch: {n_steps_mini} updates ({n_steps_mini//100} per epoch)")
    print(f"  Mini-batch took {n_steps_mini//n_steps_full}x more steps "
          f"but each step is {len(X_tr)//32}x faster.")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_loss_curves(full, mini)
    plot_step_loss(mini.step_losses)
    print("\nDone.")


if __name__ == "__main__":
    main()
