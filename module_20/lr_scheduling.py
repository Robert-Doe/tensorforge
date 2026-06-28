"""module_20/lr_scheduling.py — Learning Rate Scheduling & Early Stopping.

Covers:
  - Step decay, cosine annealing, warmup + cosine (modern default)
  - Cyclic learning rates (CLR)
  - Early stopping with patience
  - Gradient norm monitoring

All schedulers are implemented from scratch AND shown with PyTorch equivalents.

Run standalone: python lr_scheduling.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "module_18"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import (
    StepLR, CosineAnnealingLR, OneCycleLR, CyclicLR
)

RANDOM_SEED = 42
PLOTS_DIR   = "plots"
FIGURE_DPI  = 120
torch.manual_seed(RANDOM_SEED)
os.makedirs(PLOTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. LEARNING RATE SCHEDULE FUNCTIONS (from scratch)
# ─────────────────────────────────────────────────────────────────────────────

def step_decay(initial_lr: float, epoch: int,
               drop: float = 0.5, epochs_drop: int = 10) -> float:
    """Reduce LR by 'drop' every 'epochs_drop' epochs.

    Args:
        initial_lr:   starting learning rate
        epoch:        current epoch (0-indexed)
        drop:         multiplicative factor (0.5 = halve)
        epochs_drop:  how often to drop

    Returns:
        current learning rate
    """
    return initial_lr * (drop ** (epoch // epochs_drop))


def cosine_annealing(initial_lr: float, epoch: int, T_max: int) -> float:
    """Cosine annealing: smoothly decays LR from initial to near-zero.

    lr(t) = lr_min + 0.5*(lr_max - lr_min)*(1 + cos(π*t/T_max))

    Args:
        initial_lr: maximum learning rate
        epoch:      current epoch
        T_max:      number of epochs for one cosine period

    Returns:
        current learning rate
    """
    return initial_lr * 0.5 * (1 + np.cos(np.pi * epoch / T_max))


def warmup_cosine(epoch: int, warmup_epochs: int,
                  total_epochs: int, peak_lr: float,
                  min_lr: float = 1e-6) -> float:
    """Linear warmup followed by cosine decay (transformer training default).

    Warmup: LR increases linearly from 0 to peak_lr over warmup_epochs.
    Cosine:  LR decays from peak_lr to min_lr over remaining epochs.

    This is the schedule used for BERT, GPT, ViT, and most modern models.

    Args:
        epoch:         current epoch
        warmup_epochs: number of warmup epochs
        total_epochs:  total training epochs
        peak_lr:       maximum learning rate
        min_lr:        minimum learning rate

    Returns:
        current learning rate
    """
    if epoch < warmup_epochs:
        return peak_lr * (epoch / warmup_epochs)
    progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    return min_lr + 0.5 * (peak_lr - min_lr) * (1 + np.cos(np.pi * progress))


def cyclic_lr(epoch: int, step_size: int,
              base_lr: float, max_lr: float) -> float:
    """Triangular cyclic learning rate (CLR).

    LR oscillates between base_lr and max_lr in triangular waves.
    Periodic LR spikes help escape sharp local minima.

    Args:
        epoch:     current step/epoch
        step_size: half-cycle length
        base_lr:   minimum LR
        max_lr:    maximum LR

    Returns:
        current learning rate
    """
    cycle  = np.floor(1 + epoch / (2 * step_size))
    x      = abs(epoch / step_size - 2 * cycle + 1)
    return base_lr + (max_lr - base_lr) * max(0, 1 - x)


def plot_schedules():
    """Plot all four schedules for 100 epochs."""
    total  = 100
    epochs = np.arange(total)

    schedules = {
        "Step Decay (drop=0.5/10 epochs)":  [step_decay(0.1, e) for e in epochs],
        "Cosine Annealing":                  [cosine_annealing(0.1, e, total) for e in epochs],
        "Warmup + Cosine (warm=10)":         [warmup_cosine(e, 10, total, 0.1) for e in epochs],
        "Cyclic LR (step=10)":               [cyclic_lr(e, 10, 0.001, 0.1) for e in epochs],
    }

    colours = ["#58a6ff", "#3fb950", "#f85149", "#bc8cff"]
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    for (name, lrs), colour in zip(schedules.items(), colours):
        ax.plot(epochs, lrs, color=colour, linewidth=2, label=name)
    ax.set_xlabel("Epoch", color="#e6edf3")
    ax.set_ylabel("Learning Rate", color="#e6edf3")
    ax.set_title("Learning Rate Schedules", color="#e6edf3")
    ax.legend(facecolor="#161b22", labelcolor="#e6edf3", fontsize=9)
    ax.tick_params(colors="#e6edf3"); ax.spines[:].set_color("#30363d")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/lr_schedules.png", dpi=FIGURE_DPI,
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved → {PLOTS_DIR}/lr_schedules.png")


# ─────────────────────────────────────────────────────────────────────────────
# 2. EARLY STOPPING
# ─────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    """Stop training when validation loss stops improving.

    Attributes:
        patience:      epochs to wait after last improvement
        min_delta:     minimum improvement to count as improvement
        best_loss:     best validation loss seen so far
        counter:       epochs since last improvement
        should_stop:   flag set to True when training should halt
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = float("inf")
        self.counter    = 0
        self.should_stop = False
        self.best_epoch  = 0

    def step(self, val_loss: float, epoch: int) -> bool:
        """Check if training should stop.

        Args:
            val_loss: current validation loss
            epoch:    current epoch number

        Returns:
            True if training should stop
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.counter    = 0
            self.best_epoch = epoch
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


def demo_early_stopping():
    """Simulate training with early stopping on synthetic loss curves."""
    print("\n── 2. Early Stopping Demo ──────────────────────────")
    rng_es = np.random.default_rng(RANDOM_SEED)

    # Simulate: val loss decreases until epoch 25, then starts rising (overfit)
    true_best = 25
    epochs    = 100
    val_losses = [1.0 / (1 + np.exp(-0.3*(e-true_best))) + 0.1 +
                  rng_es.normal(0, 0.02)
                  for e in range(epochs)]
    # Make it U-shaped: decreasing then increasing
    val_losses = [2 - 0.05*e + 0.001*e**2 + rng_es.normal(0, 0.03)
                  for e in range(epochs)]

    es = EarlyStopping(patience=10, min_delta=1e-3)
    stopped_at = None
    for epoch, vl in enumerate(val_losses):
        if es.step(vl, epoch):
            stopped_at = epoch
            break

    print(f"  True minimum at epoch: {np.argmin(val_losses)}")
    print(f"  Early stopping triggered at: {stopped_at}")
    print(f"  Best epoch detected: {es.best_epoch}")
    print(f"  Saved {epochs - stopped_at - 1} unnecessary epochs.")


# ─────────────────────────────────────────────────────────────────────────────
# 3. PYTORCH SCHEDULERS — practical usage
# ─────────────────────────────────────────────────────────────────────────────

def demo_pytorch_schedulers():
    """Show PyTorch LR scheduler API."""
    print("\n── 3. PyTorch Scheduler API ────────────────────────")

    model     = nn.Linear(10, 1)
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    examples = [
        ("StepLR(step=10, gamma=0.5)",
         StepLR(optimizer, step_size=10, gamma=0.5)),
        ("CosineAnnealingLR(T_max=50)",
         CosineAnnealingLR(optimizer, T_max=50)),
    ]

    for name, scheduler in examples:
        optimizer.param_groups[0]["lr"] = 0.1   # reset
        lrs = []
        for _ in range(50):
            lrs.append(optimizer.param_groups[0]["lr"])
            scheduler.step()
        print(f"  {name}: start={lrs[0]:.4f}  mid={lrs[25]:.4f}  end={lrs[-1]:.4f}")

    # OneCycleLR — the recommended schedule for training from scratch
    optimizer2   = optim.SGD(model.parameters(), lr=0.01)
    one_cycle    = OneCycleLR(optimizer2, max_lr=0.1, total_steps=100)
    lrs_one = []
    for _ in range(100):
        lrs_one.append(optimizer2.param_groups[0]["lr"])
        one_cycle.step()
    print(f"  OneCycleLR: start={lrs_one[0]:.4f}  "
          f"peak={max(lrs_one):.4f}  end={lrs_one[-1]:.6f}")
    print("  OneCycleLR: warmup to max_lr, then cosine decay to near-zero.")
    print("  Developed by fastai; achieves state-of-art in 1/3 the epochs.")


# ─────────────────────────────────────────────────────────────────────────────
# 4. GRADIENT NORM MONITORING
# ─────────────────────────────────────────────────────────────────────────────

def demo_gradient_norm():
    """Show how to track gradient norms to detect vanishing/exploding gradients."""
    print("\n── 4. Gradient Norm Monitoring ─────────────────────")
    model     = nn.Sequential(nn.Linear(10, 32), nn.ReLU(),
                               nn.Linear(32, 16), nn.ReLU(),
                               nn.Linear(16, 1))
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    X = torch.randn(64, 10)
    y = torch.randn(64, 1)

    print(f"  {'Step':>5}  {'Loss':>8}  {'Grad Norm':>12}")
    for step in range(1, 6):
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()

        # Total gradient norm across all parameters
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5

        optimizer.step()
        print(f"  {step:>5}  {loss.item():>8.4f}  {total_norm:>12.4f}")

    print("\n  Exploding gradients: norm > 100 → use gradient clipping.")
    print("  Vanishing gradients: norm < 1e-4 → use ReLU, BatchNorm, residual connections.")


def main():
    print("=" * 54)
    print("MODULE 20 — Learning Rate Scheduling & Early Stopping")
    print("=" * 54)
    print("\n── 1. LR Schedule Plots ────────────────────────────")
    plot_schedules()
    demo_early_stopping()
    demo_pytorch_schedulers()
    demo_gradient_norm()
    print("\nDone.")


if __name__ == "__main__":
    main()
