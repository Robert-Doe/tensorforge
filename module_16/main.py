"""module_16/main.py — One Neuron from Scratch.

Shows that a single sigmoid neuron = Logistic Regression,
demonstrates all three activation functions, plots loss curve.
Run: python main.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_loader import load_and_clean, to_arrays
from neuron import Neuron, sigmoid, relu, sigmoid_derivative

# ── Constants ────────────────────────────────────────────────────────────────
DATA_FILE  = "cases.csv"
RANDOM_SEED = 42
TEST_RATIO  = 0.2
PLOTS_DIR   = "plots"
FIGURE_DPI  = 120


def plot_activation_functions():
    """Save a figure showing sigmoid and ReLU side by side."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    z = np.linspace(-6, 6, 300)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor="#0d1117")
    for ax in axes:
        ax.set_facecolor("#0d1117")
        ax.tick_params(colors="#e6edf3")
        ax.spines[:].set_color("#30363d")
        ax.axhline(0, color="#30363d", linewidth=0.8)
        ax.axvline(0, color="#30363d", linewidth=0.8)

    axes[0].plot(z, sigmoid(z), color="#58a6ff", linewidth=2)
    axes[0].set_title("Sigmoid σ(z) = 1/(1+e⁻ᶻ)", color="#e6edf3")
    axes[0].set_xlabel("z", color="#e6edf3")
    axes[0].set_ylabel("σ(z)", color="#e6edf3")

    axes[1].plot(z, relu(z), color="#3fb950", linewidth=2)
    axes[1].set_title("ReLU max(0, z)", color="#e6edf3")
    axes[1].set_xlabel("z", color="#e6edf3")
    axes[1].set_ylabel("ReLU(z)", color="#e6edf3")

    plt.tight_layout()
    fname = f"{PLOTS_DIR}/activation_functions.png"
    plt.savefig(fname, dpi=FIGURE_DPI, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved → {fname}")


def plot_loss_curve(history):
    """Save training loss curve."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    ax.plot(history, color="#58a6ff", linewidth=1.5)
    ax.set_xlabel("Epoch",              color="#e6edf3")
    ax.set_ylabel("Binary Cross-Entropy", color="#e6edf3")
    ax.set_title("Single Neuron — Training Loss", color="#e6edf3")
    ax.tick_params(colors="#e6edf3")
    ax.spines[:].set_color("#30363d")
    plt.tight_layout()
    fname = f"{PLOTS_DIR}/neuron_loss_curve.png"
    plt.savefig(fname, dpi=FIGURE_DPI, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved → {fname}")


def main():
    """Demonstrate a single neuron on detective data."""
    df           = load_and_clean(DATA_FILE)
    X, y, feats  = to_arrays(df)

    # manual split (no sklearn — we're building from scratch here)
    rng      = np.random.default_rng(RANDOM_SEED)
    idx      = rng.permutation(len(X))
    split    = int(len(X) * (1 - TEST_RATIO))
    X_train, X_test = X[idx[:split]], X[idx[split:]]
    y_train, y_test = y[idx[:split]], y[idx[split:]]

    # scale manually so the neuron trains well
    mean   = X_train.mean(axis=0)
    std    = X_train.std(axis=0) + 1e-8
    X_train_s = (X_train - mean) / std
    X_test_s  = (X_test  - mean) / std

    print("=" * 52)
    print("MODULE 16 — One Neuron from Scratch")
    print("=" * 52)

    # ── Train neuron ─────────────────────────────────────────────────────────
    neuron = Neuron(lr=0.1, epochs=500)
    neuron.fit(X_train_s, y_train)

    train_acc = np.mean(neuron.predict(X_train_s) == y_train)
    test_acc  = np.mean(neuron.predict(X_test_s)  == y_test)

    print(f"\nSingle Neuron Results:")
    print(f"  Train accuracy : {train_acc:.3f}")
    print(f"  Test  accuracy : {test_acc:.3f}")
    print(f"  Final loss     : {neuron.history_[-1]:.4f}")

    print(f"\nLearnt weights (one per feature):")
    for fname, w in zip(feats, neuron.w):
        bar = "█" * int(abs(w) * 10)
        sign = "+" if w >= 0 else "-"
        print(f"  {fname:<22} {sign}{abs(w):.3f}  {bar}")
    print(f"  {'bias':<22} {neuron.b:+.3f}")

    # ── Activation function demo ─────────────────────────────────────────────
    print("\nActivation function demo (z=-3, 0, +3):")
    for z in [-3.0, 0.0, 3.0]:
        print(f"  z={z:+.0f}  sigmoid={sigmoid(z):.3f}  relu={relu(z):.3f}")

    print("\nThis neuron IS Logistic Regression — same math, different framing.")
    print("The difference starts in Module 18 when we stack neurons in layers.")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_activation_functions()
    plot_loss_curve(neuron.history_)

    print("\nDone.")


if __name__ == "__main__":
    main()
