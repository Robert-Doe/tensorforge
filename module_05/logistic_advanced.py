"""module_05/logistic_advanced.py — Advanced Logistic Regression.

Covers:
  - Multiclass classification via Softmax (one-vs-rest extension)
  - Regularisation paths (how C controls decision boundary)
  - Class imbalance handling (class_weight, threshold tuning)
  - Calibration: are predicted probabilities trustworthy?

Run standalone: python logistic_advanced.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, roc_auc_score

RANDOM_SEED = 42
PLOTS_DIR   = "plots"
FIGURE_DPI  = 120
rng = np.random.default_rng(RANDOM_SEED)
os.makedirs(PLOTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. SOFTMAX REGRESSION (multiclass)
# ─────────────────────────────────────────────────────────────────────────────

def softmax(Z: np.ndarray) -> np.ndarray:
    """Numerically stable softmax.

    Args:
        Z: (n_samples, n_classes) raw logits

    Returns:
        (n_samples, n_classes) probabilities summing to 1 per row
    """
    Z_shifted = Z - Z.max(axis=1, keepdims=True)   # subtract max for stability
    exp_Z = np.exp(Z_shifted)
    return exp_Z / exp_Z.sum(axis=1, keepdims=True)


class SoftmaxClassifier:
    """Multinomial logistic regression trained with gradient descent.

    Weight matrix W has shape (n_features, n_classes).
    For each sample: logits = X @ W + b  → softmax → probabilities.
    Loss: cross-entropy  = -mean(sum_k y_k * log(p_k))
    Gradient: dL/dW = X.T @ (P - Y) / n  (same elegant form as binary case)
    """

    def __init__(self, n_classes: int, lr: float = 0.1, epochs: int = 500):
        self.n_classes = n_classes
        self.lr        = lr
        self.epochs    = epochs

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train with gradient descent.

        Args:
            X: (n, d) feature matrix
            y: (n,) integer class labels 0..n_classes-1
        """
        n, d = X.shape
        self.W = np.zeros((d, self.n_classes))
        self.b = np.zeros(self.n_classes)

        # One-hot encode y
        Y = np.zeros((n, self.n_classes))
        Y[np.arange(n), y] = 1

        self.losses = []
        for _ in range(self.epochs):
            logits = X @ self.W + self.b          # (n, K)
            P      = softmax(logits)              # (n, K)

            # Cross-entropy loss
            loss = -np.mean(np.sum(Y * np.log(P + 1e-9), axis=1))
            self.losses.append(loss)

            # Gradient
            dlogits = (P - Y) / n                 # (n, K)
            self.W -= self.lr * (X.T @ dlogits)
            self.b -= self.lr * dlogits.sum(0)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return softmax(X @ self.W + self.b)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)


def demo_softmax():
    """Train SoftmaxClassifier on 4-class synthetic data."""
    print("── 1. Softmax Regression (4 classes) ──────────────")
    X, y = make_blobs(n_samples=400, centers=4, cluster_std=1.5,
                      random_state=RANDOM_SEED)
    X = (X - X.mean(0)) / X.std(0)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                               random_state=RANDOM_SEED)
    clf = SoftmaxClassifier(n_classes=4, lr=0.2, epochs=300)
    clf.fit(X_tr, y_tr)

    acc = (clf.predict(X_te) == y_te).mean()
    print(f"Test accuracy: {acc:.4f}")
    print(f"Final loss: {clf.losses[-1]:.4f}")
    print(f"Weight matrix W shape: {clf.W.shape}  (features × classes)")


# ─────────────────────────────────────────────────────────────────────────────
# 2. REGULARISATION PATH
# ─────────────────────────────────────────────────────────────────────────────

def demo_regularisation_path():
    """Show how the inverse regularisation parameter C affects weights."""
    print("\n── 2. Regularisation Path ──────────────────────────")
    X, y = make_classification(n_samples=300, n_features=10, n_informative=5,
                               random_state=RANDOM_SEED)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                               random_state=RANDOM_SEED)

    Cs = np.logspace(-3, 3, 20)
    train_accs, test_accs, weight_norms = [], [], []

    for C in Cs:
        model = LogisticRegression(C=C, max_iter=1000, random_state=RANDOM_SEED)
        model.fit(X_tr, y_tr)
        train_accs.append(model.score(X_tr, y_tr))
        test_accs.append(model.score(X_te, y_te))
        weight_norms.append(np.linalg.norm(model.coef_))

    best_C = Cs[np.argmax(test_accs)]
    print(f"Best C by test accuracy: {best_C:.4f}")
    print(f"  Small C → heavy regularisation → small weights → underfitting")
    print(f"  Large C → weak regularisation  → large weights → overfitting")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor="#0d1117")
    for ax in axes: ax.set_facecolor("#0d1117")

    axes[0].semilogx(Cs, train_accs, "o-", color="#58a6ff", label="Train")
    axes[0].semilogx(Cs, test_accs,  "s-", color="#3fb950", label="Test")
    axes[0].axvline(best_C, color="#f85149", ls="--", label=f"Best C={best_C:.3f}")
    axes[0].set_xlabel("C (inverse regularisation)", color="#e6edf3")
    axes[0].set_ylabel("Accuracy", color="#e6edf3")
    axes[0].set_title("Regularisation Path", color="#e6edf3")
    axes[0].legend(facecolor="#161b22", labelcolor="#e6edf3")
    axes[0].tick_params(colors="#e6edf3"); axes[0].spines[:].set_color("#30363d")

    axes[1].semilogx(Cs, weight_norms, "o-", color="#bc8cff")
    axes[1].set_xlabel("C", color="#e6edf3")
    axes[1].set_ylabel("||w||₂", color="#e6edf3")
    axes[1].set_title("Weight Norm vs C", color="#e6edf3")
    axes[1].tick_params(colors="#e6edf3"); axes[1].spines[:].set_color("#30363d")

    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/logistic_reg_path.png", dpi=FIGURE_DPI,
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved → {PLOTS_DIR}/logistic_reg_path.png")


# ─────────────────────────────────────────────────────────────────────────────
# 3. CLASS IMBALANCE — class_weight and threshold tuning
# ─────────────────────────────────────────────────────────────────────────────

def demo_class_imbalance():
    """Show that default threshold (0.5) hurts on imbalanced data."""
    print("\n── 3. Class Imbalance & Threshold Tuning ───────────")
    X, y = make_classification(n_samples=1000, n_features=10,
                               weights=[0.9, 0.1],   # 90% class 0, 10% class 1
                               random_state=RANDOM_SEED)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                               stratify=y, random_state=RANDOM_SEED)

    def evaluate(model, X_te, y_te, threshold=0.5):
        proba  = model.predict_proba(X_te)[:, 1]
        preds  = (proba >= threshold).astype(int)
        tp = ((preds == 1) & (y_te == 1)).sum()
        fp = ((preds == 1) & (y_te == 0)).sum()
        fn = ((preds == 0) & (y_te == 1)).sum()
        precision = tp / (tp + fp + 1e-9)
        recall    = tp / (tp + fn + 1e-9)
        f1        = 2 * precision * recall / (precision + recall + 1e-9)
        return precision, recall, f1

    # Default (no class weight)
    m1 = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    m1.fit(X_tr, y_tr)

    # Balanced class weight — sklearn computes n_samples / (n_classes * count_k)
    m2 = LogisticRegression(class_weight="balanced", max_iter=1000,
                            random_state=RANDOM_SEED)
    m2.fit(X_tr, y_tr)

    for name, model in [("Default (threshold=0.5)", m1),
                        ("Balanced weight  (0.5)", m2),
                        ("Default (threshold=0.2)", m1)]:
        threshold = 0.2 if "0.2" in name else 0.5
        p, r, f1 = evaluate(model, X_te, y_te, threshold)
        print(f"  {name:<32}  P={p:.3f}  R={r:.3f}  F1={f1:.3f}")

    print("\n  On imbalanced data:")
    print("  - Default threshold → high accuracy but misses minority class")
    print("  - class_weight='balanced' → penalises minority misclassification more")
    print("  - Lower threshold → higher recall at cost of precision")


# ─────────────────────────────────────────────────────────────────────────────
# 4. CALIBRATION — are probabilities trustworthy?
# ─────────────────────────────────────────────────────────────────────────────

def demo_calibration():
    """Plot reliability diagram to check if predicted probabilities are calibrated."""
    print("\n── 4. Probability Calibration ──────────────────────")
    X, y = make_classification(n_samples=2000, n_features=15,
                               random_state=RANDOM_SEED)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                               random_state=RANDOM_SEED)

    model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]

    # calibration_curve bins predictions and computes fraction of true positives
    frac_pos, mean_pred = calibration_curve(y_te, proba, n_bins=10)

    # A well-calibrated model has points on the diagonal: predicted_prob ≈ actual_prob
    fig, ax = plt.subplots(figsize=(6, 5), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    ax.plot([0, 1], [0, 1], "k--", color="#8b949e", label="Perfect calibration")
    ax.plot(mean_pred, frac_pos, "o-", color="#58a6ff", label="Logistic Regression")
    ax.set_xlabel("Mean predicted probability", color="#e6edf3")
    ax.set_ylabel("Fraction of positives",      color="#e6edf3")
    ax.set_title("Calibration (Reliability) Diagram", color="#e6edf3")
    ax.legend(facecolor="#161b22", labelcolor="#e6edf3")
    ax.tick_params(colors="#e6edf3"); ax.spines[:].set_color("#30363d")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/calibration.png", dpi=FIGURE_DPI,
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved → {PLOTS_DIR}/calibration.png")
    print("  Logistic regression is naturally well-calibrated (points near diagonal).")
    print("  Random Forests and SVMs are often miscalibrated — use CalibratedClassifierCV.")


def main():
    print("=" * 54)
    print("MODULE 05 — Advanced Logistic Regression")
    print("=" * 54)
    demo_softmax()
    demo_regularisation_path()
    demo_class_imbalance()
    demo_calibration()
    print("\nDone.")


if __name__ == "__main__":
    main()
