"""module_19/main.py — Activation Functions deep dive.

Run: python main.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from activations import (
    sigmoid, sigmoid_grad, tanh, tanh_grad,
    relu, relu_grad, leaky_relu, leaky_relu_grad, elu,
    vanishing_gradient_demo, relu_gradient_demo,
)

# ── Constants ────────────────────────────────────────────────────────────────
PLOTS_DIR  = "plots"
FIGURE_DPI = 120
Z_RANGE    = np.linspace(-4, 4, 400)
N_LAYERS   = 12   # depth for vanishing gradient demo


def plot_activations():
    """Save a 2×3 grid showing each activation function and its gradient."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

    fns = [
        ("Sigmoid",     sigmoid,     sigmoid_grad,     "#58a6ff"),
        ("Tanh",        tanh,        tanh_grad,        "#3fb950"),
        ("ReLU",        relu,        relu_grad,        "#d29922"),
        ("Leaky ReLU",  leaky_relu,  leaky_relu_grad,  "#bc8cff"),
        ("ELU",         elu,         elu_grad,         "#f85149"),
    ]

    fig, axes = plt.subplots(2, 5, figsize=(18, 6), facecolor="#0d1117")
    fig.suptitle("Activation Functions (top) and their Gradients (bottom)",
                 color="#e6edf3", fontsize=13)

    for col, (name, fn, gfn, colour) in enumerate(fns):
        for row, (func, ylabel) in enumerate([(fn, "f(z)"), (gfn, "f′(z)")]):
            ax = axes[row][col]
            ax.set_facecolor("#0d1117")
            ax.plot(Z_RANGE, func(Z_RANGE), color=colour, linewidth=2)
            ax.axhline(0, color="#30363d", linewidth=0.8)
            ax.axvline(0, color="#30363d", linewidth=0.8)
            ax.set_title(name if row == 0 else "", color="#e6edf3", fontsize=11)
            ax.set_xlabel("z",      color="#e6edf3", fontsize=9)
            ax.set_ylabel(ylabel,   color="#e6edf3", fontsize=9)
            ax.tick_params(colors="#e6edf3", labelsize=8)
            ax.spines[:].set_color("#30363d")

    plt.tight_layout()
    fname = f"{PLOTS_DIR}/activation_comparison.png"
    plt.savefig(fname, dpi=FIGURE_DPI, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved → {fname}")


def plot_vanishing_gradient():
    """Save gradient magnitude vs. layer depth for sigmoid vs ReLU."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

    sig_grads  = vanishing_gradient_demo(N_LAYERS)
    relu_grads = relu_gradient_demo(N_LAYERS)
    layers     = list(range(N_LAYERS + 1))

    fig, ax = plt.subplots(figsize=(8, 4), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    ax.semilogy(layers, sig_grads,  "o-", color="#f85149", label="Sigmoid (vanishes)")
    ax.semilogy(layers, relu_grads, "s-", color="#3fb950", label="ReLU (stable)")
    ax.set_xlabel("Layer depth (from output)", color="#e6edf3")
    ax.set_ylabel("Gradient magnitude (log scale)", color="#e6edf3")
    ax.set_title("Vanishing Gradient: Sigmoid vs. ReLU", color="#e6edf3")
    ax.legend(facecolor="#161b22", labelcolor="#e6edf3")
    ax.tick_params(colors="#e6edf3")
    ax.spines[:].set_color("#30363d")
    plt.tight_layout()
    fname = f"{PLOTS_DIR}/vanishing_gradient.png"
    plt.savefig(fname, dpi=FIGURE_DPI, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved → {fname}")


def print_gradient_table():
    """Print max gradient for each activation at key z values."""
    print("\n=== Maximum Gradient Values ===")
    print(f"  {'Activation':<14} {'Max gradient':>14} {'At z':>8} {'Problem'}")
    print("  " + "-" * 60)
    rows = [
        ("Sigmoid",    0.25,  "0",    "Vanishes in deep nets (×0.25 per layer)"),
        ("Tanh",       1.0,   "0",    "Vanishes too, but less severe"),
        ("ReLU",       1.0,   "z>0",  "Dead neurons (z<0 → gradient=0)"),
        ("Leaky ReLU", 1.0,   "z>0",  "Tiny leak (0.01) for z<0 — no dead neurons"),
        ("ELU",        1.0,   "z>0",  "Smooth negative side, near-zero mean"),
    ]
    for name, maxg, atz, problem in rows:
        print(f"  {name:<14} {maxg:>14.2f} {atz:>8}  {problem}")


def main():
    """Print gradient table, show vanishing gradient, save plots."""
    print("=" * 52)
    print("MODULE 19 — Activation Functions")
    print("=" * 52)

    print_gradient_table()

    # ── Vanishing gradient numbers ────────────────────────────────────────────
    sig_grads = vanishing_gradient_demo(N_LAYERS)
    print(f"\n=== Vanishing Gradient through {N_LAYERS} Sigmoid Layers ===")
    print(f"  Layer 0 (output): gradient = {sig_grads[0]:.6f}")
    for i in [1, 3, 6, 9, 12]:
        if i < len(sig_grads):
            print(f"  Layer {i:2d}        : gradient = {sig_grads[i]:.8f}")
    print(f"\n  After {N_LAYERS} sigmoid layers, gradient ≈ 0.25^{N_LAYERS} = {0.25**N_LAYERS:.2e}")
    print("  Weights in early layers barely move — network cannot learn deep representations.")

    # ── Activation value demo ─────────────────────────────────────────────────
    print("\n=== Activation values at z = -3, 0, +3 ===")
    print(f"  {'z':>4}  {'sigmoid':>9}  {'tanh':>8}  {'relu':>8}  {'leaky':>8}  {'elu':>8}")
    print("  " + "-" * 54)
    for z in [-3.0, 0.0, 3.0]:
        print(f"  {z:>4.0f}  {sigmoid(z):>9.4f}  {tanh(z):>8.4f}  "
              f"{relu(z):>8.4f}  {leaky_relu(z):>8.4f}  {elu(z):>8.4f}")

    # ── Plots ──────────────────────────────────────────────────────────────────
    plot_activations()
    plot_vanishing_gradient()
    print("\nDone.")


if __name__ == "__main__":
    main()
