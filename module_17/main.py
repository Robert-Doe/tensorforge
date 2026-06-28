"""module_17/main.py — Backpropagation through a 2-layer network.

Run: python main.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_loader import load_and_clean, to_arrays
from backprop import TwoLayerNet

# ── Constants ────────────────────────────────────────────────────────────────
DATA_FILE   = "cases.csv"
RANDOM_SEED = 42
TEST_RATIO  = 0.2
PLOTS_DIR   = "plots"
FIGURE_DPI  = 120


def scale(X_train, X_test):
    """Standard-scale using training statistics only."""
    mean = X_train.mean(axis=0)
    std  = X_train.std(axis=0) + 1e-8
    return (X_train - mean) / std, (X_test - mean) / std


def plot_loss_comparison(history_1layer, history_2layer):
    """Save side-by-side loss curves for 1-layer and 2-layer networks."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    ax.plot(history_1layer, color="#58a6ff", linewidth=1.5, label="1-layer (M16 neuron)")
    ax.plot(history_2layer, color="#3fb950", linewidth=1.5, label="2-layer (backprop)")
    ax.set_xlabel("Epoch",              color="#e6edf3")
    ax.set_ylabel("Binary Cross-Entropy", color="#e6edf3")
    ax.set_title("1-layer vs 2-layer Loss", color="#e6edf3")
    ax.legend(facecolor="#161b22", labelcolor="#e6edf3")
    ax.tick_params(colors="#e6edf3")
    ax.spines[:].set_color("#30363d")
    plt.tight_layout()
    fname = f"{PLOTS_DIR}/backprop_loss.png"
    plt.savefig(fname, dpi=FIGURE_DPI, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved → {fname}")


def print_gradient_trace(net, X_sample, y_sample):
    """Show one backward pass with actual gradient values, step by step."""
    print("\n=== Gradient trace (1 sample, 1 pass) ===")
    X1 = X_sample[0:1]   # single sample
    y1 = y_sample[0:1]

    Z1, A1, Z2, A2 = net.forward(X1)
    grads = net.backward(X1, y1, Z1, A1, A2)

    print(f"  Input X         : {X1.ravel()[:3]}...")
    print(f"  Hidden Z1 (first 3): {Z1.ravel()[:3]}")
    print(f"  Hidden A1 (first 3): {A1.ravel()[:3]}")
    print(f"  Output Z2       : {Z2.ravel()[0]:.4f}")
    print(f"  Output A2 (prob): {A2.ravel()[0]:.4f}  (true label={y1[0]})")
    print(f"  dZ2 (error)     : {(A2-y1.reshape(-1,1)).ravel()[0]:.4f}")
    print(f"  ||dW2||         : {np.linalg.norm(grads['dW2']):.4f}")
    print(f"  ||dW1||         : {np.linalg.norm(grads['dW1']):.4f}")
    print("  Gradient flows back: output → hidden → input")


def main():
    """Train 2-layer net, compare to 1-layer, show gradient trace."""
    df           = load_and_clean(DATA_FILE)
    X, y, feats  = to_arrays(df)

    rng    = np.random.default_rng(RANDOM_SEED)
    idx    = rng.permutation(len(X))
    split  = int(len(X) * (1 - TEST_RATIO))
    X_tr, X_te = X[idx[:split]], X[idx[split:]]
    y_tr, y_te = y[idx[:split]], y[idx[split:]]
    X_tr_s, X_te_s = scale(X_tr, X_te)

    print("=" * 52)
    print("MODULE 17 — Backpropagation")
    print("=" * 52)

    # ── 1-layer baseline (M16 Neuron) ─────────────────────────────────────
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "module_16"))
    from neuron import Neuron
    n1 = Neuron(lr=0.1, epochs=1000)
    n1.fit(X_tr_s, y_tr)
    acc_1 = np.mean(n1.predict(X_te_s) == y_te)
    print(f"\n1-layer Neuron (M16)   Test acc: {acc_1:.3f}")

    # ── 2-layer network with backprop ─────────────────────────────────────
    net = TwoLayerNet(n_features=X.shape[1], hidden_size=8, lr=0.05, epochs=1000)
    net.fit(X_tr_s, y_tr)
    acc_2 = np.mean(net.predict(X_te_s) == y_te)
    print(f"2-layer Net (backprop) Test acc: {acc_2:.3f}")
    print(f"\nParameter count:")
    print(f"  1-layer: {X.shape[1]+1} params  (w + b)")
    print(f"  2-layer: {X.shape[1]*8 + 8 + 8 + 1} params  (W1,b1,W2,b2)")

    # ── Gradient trace ────────────────────────────────────────────────────
    print_gradient_trace(net, X_tr_s, y_tr)

    # ── Loss plot ─────────────────────────────────────────────────────────
    plot_loss_comparison(n1.history_, net.history_)
    print("\nDone.")


if __name__ == "__main__":
    main()
