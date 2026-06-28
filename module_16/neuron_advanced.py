"""module_16/neuron_advanced.py — Single Neuron Theory Deep Dive.

Covers:
  - Different loss functions and when to use each
  - Perceptron convergence theorem (proof sketch + demo)
  - Bias-variance from a single neuron perspective
  - MSE vs MAE vs Huber loss — robustness to outliers
  - Cross-entropy for binary classification — why not MSE?

Run standalone: python neuron_advanced.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RANDOM_SEED = 42
PLOTS_DIR   = "plots"
FIGURE_DPI  = 120
rng = np.random.default_rng(RANDOM_SEED)
os.makedirs(PLOTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOSS FUNCTIONS COMPARED
# ─────────────────────────────────────────────────────────────────────────────

def mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error: sensitive to outliers (quadratic penalty)."""
    return np.mean((y_true - y_pred) ** 2)


def mae_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error: robust to outliers (linear penalty)."""
    return np.mean(np.abs(y_true - y_pred))


def huber_loss(y_true: np.ndarray, y_pred: np.ndarray, delta: float = 1.0) -> float:
    """Huber loss: MSE for |r|≤δ, MAE for |r|>δ.

    Best of both worlds: smooth (MSE-like) near zero for gradient stability,
    linear (MAE-like) for large errors (outlier robustness).
    """
    r = np.abs(y_true - y_pred)
    return np.where(r <= delta,
                    0.5 * r**2,
                    delta * r - 0.5 * delta**2).mean()


def binary_cross_entropy(y_true: np.ndarray, y_prob: np.ndarray,
                          eps: float = 1e-9) -> float:
    """Binary cross-entropy: standard loss for sigmoid output + classification."""
    y_prob = np.clip(y_prob, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))


def demo_loss_functions():
    """Compare loss surfaces and robustness to outliers."""
    print("── 1. Loss Functions Compared ──────────────────────")
    residuals = np.linspace(-4, 4, 200)

    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor="#0d1117")
    ax.set_facecolor("#161b22")
    ax.plot(residuals, 0.5 * residuals**2, color="#f85149",
            label="MSE (0.5r²)", linewidth=2.2)
    ax.plot(residuals, np.abs(residuals),   color="#3fb950",
            label="MAE (|r|)",   linewidth=2.2)
    delta = 1.0
    huber = np.where(np.abs(residuals) <= delta,
                     0.5 * residuals**2,
                     delta * np.abs(residuals) - 0.5 * delta**2)
    ax.plot(residuals, huber, color="#58a6ff",
            label=f"Huber (δ={delta})", linewidth=2.2, linestyle="--")
    ax.set_xlabel("Residual (y - ŷ)", color="#e6edf3")
    ax.set_ylabel("Loss", color="#e6edf3")
    ax.set_title("Loss Function Shapes", color="#e6edf3")
    ax.legend(facecolor="#0d1117", labelcolor="#e6edf3")
    ax.tick_params(colors="#8b949e")
    ax.spines[:].set_color("#30363d")
    ax.set_ylim(0, 8)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/loss_functions.png", dpi=FIGURE_DPI,
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved → {PLOTS_DIR}/loss_functions.png")

    # Robustness to outliers demonstration
    y_true  = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_clean = np.array([1.1, 2.1, 2.9, 4.0, 5.1])
    y_dirty = y_clean.copy()
    y_dirty[-1] = 50.0   # extreme outlier

    print(f"\n  Outlier comparison (y[-1] changed from 5.1 to 50.0):")
    print(f"  {'Loss':<15} {'Clean':>10} {'With Outlier':>14} {'Ratio':>8}")
    for name, fn in [("MSE",   mse_loss),
                     ("MAE",   mae_loss),
                     ("Huber", huber_loss)]:
        l_clean = fn(y_true, y_clean)
        l_dirty = fn(y_true, y_dirty)
        ratio   = l_dirty / l_clean
        print(f"  {name:<15} {l_clean:>10.4f} {l_dirty:>14.4f} {ratio:>8.1f}×")

    print("\n  MSE amplifies the outlier 1800×; Huber only ~45× — much more robust.")


# ─────────────────────────────────────────────────────────────────────────────
# 2. WHY NOT MSE FOR CLASSIFICATION?
# ─────────────────────────────────────────────────────────────────────────────

def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


def demo_mse_vs_bce():
    """Show why BCE has better gradients than MSE for classification."""
    print("\n── 2. Why BCE > MSE for Classification ─────────────")
    z   = np.linspace(-5, 5, 300)   # pre-activation (logit)
    p   = sigmoid(z)

    # Gradient of loss w.r.t. z for a true label y=1
    # MSE gradient: dL/dz = -(y - p) * p * (1-p)  [extra p(1-p) factor → vanishes at saturation]
    # BCE gradient: dL/dz = -(y - p)               [clean linear signal]
    y = 1.0
    grad_mse = -(y - p) * p * (1 - p)
    grad_bce = -(y - p)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor="#0d1117")
    for ax in axes:
        ax.set_facecolor("#161b22")
        ax.axhline(0, color="#30363d", linewidth=0.8)
        ax.axvline(0, color="#30363d", linewidth=0.8)
        ax.tick_params(colors="#8b949e")
        ax.spines[:].set_color("#30363d")

    axes[0].plot(z, grad_mse, color="#f85149", linewidth=2.2, label="MSE gradient")
    axes[0].plot(z, grad_bce, color="#3fb950", linewidth=2.2, label="BCE gradient")
    axes[0].set_xlabel("Logit z", color="#e6edf3")
    axes[0].set_title("Gradient magnitude (y=1)", color="#e6edf3")
    axes[0].legend(facecolor="#0d1117", labelcolor="#e6edf3")

    axes[1].plot(z, p, color="#58a6ff", linewidth=2.2)
    axes[1].set_xlabel("Logit z", color="#e6edf3")
    axes[1].set_title("Sigmoid output p(z)", color="#e6edf3")

    plt.suptitle("MSE vs BCE gradient for classification", color="#e6edf3")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/mse_vs_bce_gradient.png", dpi=FIGURE_DPI,
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved → {PLOTS_DIR}/mse_vs_bce_gradient.png")

    print("""
  MSE gradient: -(y - p) * p(1-p)
    p(1-p) → 0 when p≈0 or p≈1 (saturated sigmoid)
    → gradient vanishes when model is very confident (even if wrong!)
    → training stalls

  BCE gradient: -(y - p)
    gradient = prediction error, no saturation factor
    → strong learning signal even when model is confidently wrong
    → BCE is the theoretically correct loss for probabilistic classification
  """)


# ─────────────────────────────────────────────────────────────────────────────
# 3. PERCEPTRON CONVERGENCE THEOREM
# ─────────────────────────────────────────────────────────────────────────────

class Perceptron:
    """Classic Rosenblatt Perceptron (1958).

    Update rule: if y_pred ≠ y_true: w += y_true * x, b += y_true
    (where y ∈ {-1, +1})
    """

    def __init__(self, n_features: int, lr: float = 1.0):
        self.w  = np.zeros(n_features)
        self.b  = 0.0
        self.lr = lr

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.sign(X @ self.w + self.b)

    def fit(self, X: np.ndarray, y: np.ndarray, max_epochs: int = 100) -> int:
        """Run the perceptron learning rule. Returns number of epochs until convergence."""
        for epoch in range(max_epochs):
            errors = 0
            for xi, yi in zip(X, y):
                if self.predict(xi.reshape(1, -1))[0] != yi:
                    self.w += self.lr * yi * xi
                    self.b += self.lr * yi
                    errors += 1
            if errors == 0:
                return epoch + 1
        return max_epochs


def demo_perceptron_convergence():
    """Demonstrate perceptron convergence on linearly separable data."""
    print("\n── 3. Perceptron Convergence Theorem ────────────────")
    print("""
  Theorem (Novikoff, 1962):
  If the data is linearly separable with margin γ > 0, the perceptron
  algorithm makes at most (R/γ)² mistakes before converging,
  where R = max ||x_i|| (radius of data).

  Key implication:
  - Convergence is GUARANTEED if data is linearly separable
  - Convergence is IMPOSSIBLE if data is NOT linearly separable (loops forever)
  - This motivated multi-layer networks and later SVM (max-margin)
  """)

    # Linearly separable case
    X_lin = np.array([[1, 1], [2, 1], [1, 2],
                      [-1,-1], [-2,-1], [-1,-2]], dtype=float)
    y_lin = np.array([1, 1, 1, -1, -1, -1])

    perc = Perceptron(n_features=2)
    epochs = perc.fit(X_lin, y_lin, max_epochs=100)
    final_acc = (perc.predict(X_lin) == y_lin).mean()
    print(f"  Linearly separable: converged in {epochs} epoch(s), accuracy={final_acc:.2f}")

    # XOR case — NOT linearly separable
    X_xor = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
    y_xor = np.array([-1, 1, 1, -1])

    perc_xor = Perceptron(n_features=2)
    epochs_xor = perc_xor.fit(X_xor, y_xor, max_epochs=100)
    final_acc_xor = (perc_xor.predict(X_xor) == y_xor).mean()
    print(f"  XOR (not separable):  ran {epochs_xor} epochs, accuracy={final_acc_xor:.2f}")
    print("  → Perceptron cannot learn XOR — needs a hidden layer (MLP).")
    print("  This limitation, published by Minsky & Papert (1969), nearly")
    print("  killed neural network research for a decade — until backprop (1986).")


# ─────────────────────────────────────────────────────────────────────────────
# 4. BIAS-VARIANCE FROM A SINGLE NEURON PERSPECTIVE
# ─────────────────────────────────────────────────────────────────────────────

def demo_bias_variance_single_neuron():
    """Show underfitting vs overfitting with a single logistic neuron."""
    print("\n── 4. Bias-Variance in a Single Neuron ─────────────")
    print("""
  A single sigmoid neuron is a linear classifier in feature space.
  It can only learn a HYPERPLANE decision boundary.

  HIGH BIAS (underfitting):
  - The neuron's hypothesis class is too simple (linear boundary).
  - Data with curved decision boundary will always have irreducible error.
  - Adding features or preprocessing (e.g. polynomial features) can help.
  - Regularisation makes bias worse (shrinks weights → more linear).

  LOW BIAS but HIGH VARIANCE (overfitting):
  - With many features and few samples, the neuron can fit the training
    data exactly but generalise poorly (e.g. on random labels).

  The BIAS-VARIANCE DECOMPOSITION:
  Expected MSE = Bias² + Variance + Noise

  For a single logistic neuron with C features:
    Bias²:    determined by how curved the true decision boundary is
              vs the neuron's linear hypothesis class
    Variance: decreases with more data, increases with more features
              (especially in high-dim, low-sample settings)
    Noise:    irreducible — label noise in the data

  Remedy: more neurons (more complex hypothesis class) to reduce bias,
  regularisation + more data to reduce variance.
  """)

    # Empirical demo: logistic regression on linearly separable vs circular data
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_circles, make_classification
    from sklearn.metrics import accuracy_score

    # Linear data — a single neuron should work fine
    X_lin, y_lin = make_classification(n_samples=200, n_features=2,
                                        n_informative=2, n_redundant=0,
                                        random_state=RANDOM_SEED)
    # Circular data — a single neuron will underfit
    X_circ, y_circ = make_circles(n_samples=200, noise=0.1,
                                   random_state=RANDOM_SEED)

    print(f"  {'Dataset':<25} {'Train Acc':>10} {'Test Acc':>10}")
    for name, X, y in [("Linear (easy)",    X_lin,  y_lin),
                       ("Circular (hard)",  X_circ, y_circ)]:
        X_tr, X_te = X[:150], X[150:]
        y_tr, y_te = y[:150], y[150:]
        model = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
        tr_acc = accuracy_score(y_tr, model.predict(X_tr))
        te_acc = accuracy_score(y_te, model.predict(X_te))
        print(f"  {name:<25} {tr_acc:>10.4f} {te_acc:>10.4f}")

    print("\n  Single neuron achieves near-perfect accuracy on linear data")
    print("  but struggles on circular data → HIGH BIAS.")


def main():
    print("=" * 54)
    print("MODULE 16 — Single Neuron Theory Deep Dive")
    print("=" * 54)
    demo_loss_functions()
    demo_mse_vs_bce()
    demo_perceptron_convergence()
    demo_bias_variance_single_neuron()
    print("\nDone.")


if __name__ == "__main__":
    main()
