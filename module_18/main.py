"""module_18/main.py — Multi-Layer Network from Scratch.

Demonstrates:
  1. XOR problem — single neuron CANNOT solve it; 2-layer MLP can
  2. Detective case classification with MLP vs. single neuron
Run: python main.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_loader import load_and_clean, to_arrays
from layers import MLP

# ── Constants ────────────────────────────────────────────────────────────────
DATA_FILE   = "cases.csv"
RANDOM_SEED = 42
TEST_RATIO  = 0.2
PLOTS_DIR   = "plots"
FIGURE_DPI  = 120


def xor_demo():
    """Train single neuron vs MLP on XOR — show that depth matters."""
    # XOR truth table: output is 1 when inputs differ
    X_xor = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
    y_xor = np.array([0, 1, 1, 0])

    # Single neuron (from M16)
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "module_16"))
    from neuron import Neuron
    n1 = Neuron(lr=0.5, epochs=2000)
    n1.fit(X_xor, y_xor)
    preds_1 = n1.predict(X_xor)
    acc_1   = np.mean(preds_1 == y_xor)

    # 2-layer MLP [2 → 4 → 1]
    mlp = MLP(n_in=2, hidden_sizes=[4], lr=0.5, epochs=2000)
    mlp.fit(X_xor, y_xor)
    preds_m = mlp.predict(X_xor)
    acc_m   = np.mean(preds_m == y_xor)

    print("=== XOR Problem ===")
    print(f"  Single Neuron accuracy : {acc_1:.2f}  (max possible: 0.75 — cannot solve XOR)")
    print(f"  2-layer MLP accuracy   : {acc_m:.2f}  (should reach 1.00)")
    print()
    print(f"  {'Input':>10} {'True':>6} {'1-layer':>8} {'MLP':>6}")
    print("  " + "-" * 34)
    for x, yt, p1, pm in zip(X_xor, y_xor, preds_1, preds_m):
        print(f"  {str(x.astype(int)):>10} {int(yt):>6} {int(p1):>8} {int(pm):>6}")
    return mlp.history_


def scale(X_tr, X_te):
    """Standard-scale using training statistics only."""
    mean = X_tr.mean(axis=0)
    std  = X_tr.std(axis=0) + 1e-8
    return (X_tr - mean) / std, (X_te - mean) / std


def plot_loss_curves(xor_history, det_history):
    """Save loss curves for XOR and detective tasks."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor="#0d1117")
    for ax, hist, title in zip(axes,
                                [xor_history, det_history],
                                ["MLP Loss — XOR", "MLP Loss — Detective Cases"]):
        ax.set_facecolor("#0d1117")
        ax.plot(hist, color="#3fb950", linewidth=1.5)
        ax.set_title(title, color="#e6edf3")
        ax.set_xlabel("Epoch", color="#e6edf3")
        ax.set_ylabel("Loss",  color="#e6edf3")
        ax.tick_params(colors="#e6edf3")
        ax.spines[:].set_color("#30363d")
    plt.tight_layout()
    fname = f"{PLOTS_DIR}/mlp_loss_curves.png"
    plt.savefig(fname, dpi=FIGURE_DPI, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved → {fname}")


def main():
    """Run XOR demo then detective classification."""
    print("=" * 52)
    print("MODULE 18 — Multi-Layer Network from Scratch")
    print("=" * 52)

    # ── XOR demo ─────────────────────────────────────────────────────────────
    xor_history = xor_demo()

    # ── Detective classification ─────────────────────────────────────────────
    df           = load_and_clean(DATA_FILE)
    X, y, feats  = to_arrays(df)

    rng   = np.random.default_rng(RANDOM_SEED)
    idx   = rng.permutation(len(X))
    split = int(len(X) * (1 - TEST_RATIO))
    X_tr, X_te = X[idx[:split]], X[idx[split:]]
    y_tr, y_te = y[idx[:split]], y[idx[split:]]
    X_tr_s, X_te_s = scale(X_tr, X_te)

    print("\n=== Detective Case Classification ===")
    results = []
    for arch in [[8], [16, 8], [32, 16, 8]]:
        mlp     = MLP(n_in=6, hidden_sizes=arch, lr=0.05, epochs=1500)
        mlp.fit(X_tr_s, y_tr)
        tr_acc  = np.mean(mlp.predict(X_tr_s) == y_tr)
        te_acc  = np.mean(mlp.predict(X_te_s) == y_te)
        n_param = sum(l.W.size + l.b.size for l in mlp.layers
                      if hasattr(l, "W"))
        results.append((arch, tr_acc, te_acc, n_param, mlp.history_))
        print(f"  arch={arch}  params={n_param:4d}  train={tr_acc:.3f}  test={te_acc:.3f}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    det_history = results[1][4]   # [16,8] arch
    plot_loss_curves(xor_history, det_history)
    print("\nDone.")


if __name__ == "__main__":
    main()
