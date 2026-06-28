"""module_11/svm_advanced.py — Advanced SVM: Kernels, SVR, Multi-Class.

Covers:
  - Kernel trick derivation (why we can replace dot products with k(xi, xj))
  - RBF, polynomial, and sigmoid kernels
  - Support Vector Regression (SVR) — epsilon-insensitive tube
  - Multi-class SVM strategies (OvO, OvR)
  - Kernel alignment score — measuring how well a kernel fits the data

Run standalone: python svm_advanced.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.svm import SVC, SVR
from sklearn.datasets import make_circles, make_blobs, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.pipeline import Pipeline

RANDOM_SEED = 42
PLOTS_DIR   = "plots"
FIGURE_DPI  = 120
os.makedirs(PLOTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. KERNEL TRICK — why kernels work
# ─────────────────────────────────────────────────────────────────────────────

def demo_kernel_trick():
    """Show that kernels implicitly lift data to a higher-dimensional space."""
    print("── 1. Kernel Trick Intuition ───────────────────────")
    print("""
  THE KERNEL TRICK:
  SVM with linear kernel: f(x) = Σ α_i y_i <x_i, x> + b
  We only ever need the DOT PRODUCT between examples, never the features themselves.

  KEY INSIGHT: Replace <x_i, x_j> with k(x_i, x_j) = <φ(x_i), φ(x_j)>
  where φ maps to a (possibly infinite-dimensional) feature space.

  We never compute φ explicitly — just the kernel function k(xi, xj).
  This is the "kernel trick": work in high-dim space at low-dim cost.

  Common kernels:
  ────────────────────────────────────────────────────────────────
  Linear:       k(xi, xj) = xi·xj
  Polynomial:   k(xi, xj) = (γ·xi·xj + r)^d     (adds all d-th degree features)
  RBF/Gaussian: k(xi, xj) = exp(-γ||xi - xj||²)  (infinite-dimensional φ!)
  Sigmoid:      k(xi, xj) = tanh(γ·xi·xj + r)    (like a neural network layer)

  Mercer's theorem: k is a valid kernel iff its Gram matrix is PSD for all data.
  This is what guarantees the kernel corresponds to a real dot product in some φ.
  """)

    # Demonstrate RBF lifting XOR (not linearly separable in 2D) to separable
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([-1, 1, 1, -1])

    linear_svm = SVC(kernel="linear").fit(X, y)
    rbf_svm    = SVC(kernel="rbf", gamma=1.0).fit(X, y)

    print(f"  XOR problem (NOT linearly separable):")
    print(f"  Linear SVM accuracy: {accuracy_score(y, linear_svm.predict(X)):.2f}")
    print(f"  RBF SVM accuracy:    {accuracy_score(y, rbf_svm.predict(X)):.2f}")
    print("  RBF maps XOR to a space where it IS linearly separable.")


# ─────────────────────────────────────────────────────────────────────────────
# 2. KERNEL COMPARISON ON CIRCLES DATASET
# ─────────────────────────────────────────────────────────────────────────────

def plot_svm_boundaries():
    """Visualise decision boundaries for different kernels."""
    X, y = make_circles(n_samples=200, noise=0.1, factor=0.3, random_state=RANDOM_SEED)
    scaler = StandardScaler().fit(X)
    X_s    = scaler.transform(X)

    kernels = ["linear", "poly", "rbf", "sigmoid"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), facecolor="#0d1117")

    x_min, x_max = X_s[:,0].min()-0.5, X_s[:,0].max()+0.5
    y_min, y_max = X_s[:,1].min()-0.5, X_s[:,1].max()+0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 120),
                         np.linspace(y_min, y_max, 120))

    for ax, kernel in zip(axes, kernels):
        model = SVC(kernel=kernel, gamma="auto", C=1.0)
        model.fit(X_s, y)
        acc   = accuracy_score(y, model.predict(X_s))

        Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        ax.set_facecolor("#161b22")
        ax.contourf(xx, yy, Z, alpha=0.3, cmap="RdBu")
        ax.scatter(X_s[y==0,0], X_s[y==0,1], color="#f85149", s=18, edgecolor="none")
        ax.scatter(X_s[y==1,0], X_s[y==1,1], color="#58a6ff", s=18, edgecolor="none")
        # Mark support vectors
        sv = model.support_vectors_
        ax.scatter(sv[:,0], sv[:,1], s=60, facecolors="none", edgecolors="#f0883e", linewidths=1.5)
        ax.set_title(f"{kernel.upper()} kernel\nAcc={acc:.2f}", color="#e6edf3", fontsize=9)
        ax.tick_params(colors="#8b949e")
        ax.spines[:].set_color("#30363d")

    plt.suptitle("SVM Kernels on Circles — circles = support vectors",
                 color="#e6edf3", fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/svm_kernels.png", dpi=FIGURE_DPI,
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n── 2. Kernel Comparison ────────────────────────────")
    print(f"  Saved → {PLOTS_DIR}/svm_kernels.png")


# ─────────────────────────────────────────────────────────────────────────────
# 3. SVR — Support Vector Regression
# ─────────────────────────────────────────────────────────────────────────────

def demo_svr():
    """Fit SVR with epsilon-insensitive tube and compare to linear regression."""
    print("\n── 3. Support Vector Regression (SVR) ──────────────")
    print("""
  SVR CONCEPT:
  Instead of minimising prediction error directly, SVR finds a function
  f(x) that lies within ε of ALL training points (the ε-tube).
  Points inside the tube contribute ZERO loss.
  Points outside the tube get a linear penalty (like a 1D margin).

  The ε-tube gives SVR natural robustness to noise near the boundary,
  and only the points on/outside the tube become support vectors.

  Hyperparameters:
    C:   regularisation (same role as in SVC — trades off margin vs violations)
    ε:   tube width (larger ε → fewer support vectors, smoother fit)
    kernel: same choices as SVC
  """)

    X, y = make_regression(n_samples=100, n_features=1, noise=15.0,
                           random_state=RANDOM_SEED)
    # Add some outliers
    X_noisy = np.vstack([X, [[2.5]], [[3.0]]])
    y_noisy  = np.append(y, [300.0, -300.0])   # extreme outliers

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_noisy, y_noisy, test_size=0.3, random_state=RANDOM_SEED)

    from sklearn.linear_model import LinearRegression
    models = {
        "Linear Regression": LinearRegression(),
        "SVR (ε=10, C=10)":  SVR(kernel="rbf", epsilon=10, C=10),
        "SVR (ε=50, C=100)": SVR(kernel="rbf", epsilon=50, C=100),
    }
    print(f"  {'Model':<22} {'Train MSE':>12} {'Test MSE':>12} {'SVs':>8}")
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        tr_mse = mean_squared_error(y_tr, model.predict(X_tr))
        te_mse = mean_squared_error(y_te, model.predict(X_te))
        n_sv   = getattr(model, "n_support_vectors_", None)
        if n_sv is None:
            n_sv_str = "N/A"
        else:
            n_sv_str = str(n_sv)
        svs = getattr(model, "support_", None)
        n_sv_str = str(len(svs)) if svs is not None else "N/A"
        print(f"  {name:<22} {tr_mse:>12.1f} {te_mse:>12.1f} {n_sv_str:>8}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. MULTI-CLASS SVM STRATEGIES
# ─────────────────────────────────────────────────────────────────────────────

def demo_multiclass():
    """Compare OvO vs OvR multi-class SVM strategies."""
    print("\n── 4. Multi-Class SVM (OvO vs OvR) ─────────────────")
    print("""
  SVM is inherently BINARY — it finds a hyperplane between 2 classes.
  For K classes, two strategies:

  One-vs-One (OvO):
    Train C(K,2) = K(K-1)/2 binary classifiers (one per pair of classes).
    Predict: each binary classifier votes; take the class with most votes.
    Pros: each sub-problem has half the data, often faster per classifier.
    Cons: K(K-1)/2 classifiers to train and store. For K=100: 4950 models.

  One-vs-Rest (OvR / OvA):
    Train K binary classifiers (each: class k vs all others).
    Predict: take argmax of K decision function values.
    Pros: K classifiers only.
    Cons: class imbalance (1 vs K-1 in each sub-problem).

  sklearn's SVC uses OvO by default (historically better calibrated).
  LinearSVC uses OvR by default (scales better with K).
  """)

    X, y = make_blobs(n_samples=300, centers=5, cluster_std=0.8,
                       random_state=RANDOM_SEED)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_SEED)
    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_te_s = scaler.transform(X_te)

    K = len(np.unique(y))
    print(f"  K={K} classes, OvO uses {K*(K-1)//2} classifiers, OvR uses {K}")

    for name, decision in [("OvO (sklearn SVC default)", "ovo"),
                            ("OvR (one-vs-rest)",         "ovr")]:
        model = SVC(kernel="rbf", decision_function_shape=decision, C=1.0, gamma="auto")
        model.fit(X_tr_s, y_tr)
        acc   = accuracy_score(y_te, model.predict(X_te_s))
        print(f"  {name}: accuracy={acc:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. KERNEL ALIGNMENT SCORE
# ─────────────────────────────────────────────────────────────────────────────

def kernel_alignment(K_mat: np.ndarray, y: np.ndarray) -> float:
    """Compute kernel alignment score between a kernel matrix and the ideal kernel.

    Alignment(K, y) = <K, yy^T>_F / (||K||_F * ||yy^T||_F)

    Higher alignment means the kernel's geometry better matches the label structure.
    Used to select kernels without cross-validation (Cristianini et al. 2002).

    Args:
        K_mat: (n, n) kernel Gram matrix
        y:     (n,) label vector in {-1, +1}

    Returns:
        Alignment score in [-1, 1]
    """
    y = y.astype(float)
    K_ideal = np.outer(y, y)
    numerator   = np.trace(K_mat @ K_ideal)
    denominator = np.linalg.norm(K_mat, "fro") * np.linalg.norm(K_ideal, "fro")
    return numerator / (denominator + 1e-12)


def demo_kernel_alignment():
    """Score kernels on circles vs blobs to find the best match."""
    print("\n── 5. Kernel Alignment Score ────────────────────────")
    X_circles, y_circles = make_circles(n_samples=100, noise=0.1,
                                         factor=0.3, random_state=RANDOM_SEED)
    y_circles = 2 * y_circles - 1   # {0,1} → {-1,+1}

    scaler = StandardScaler().fit(X_circles)
    X_s    = scaler.transform(X_circles)

    def rbf_kernel(X, gamma):
        sq_dists = np.sum((X[:, None] - X[None, :]) ** 2, axis=2)
        return np.exp(-gamma * sq_dists)

    def linear_kernel(X):
        return X @ X.T

    print(f"  {'Kernel':<25} {'Alignment':>10}  (circles dataset — curved boundary)")
    print(f"  {'-'*37}")
    print(f"  {'Linear':<25} {kernel_alignment(linear_kernel(X_s), y_circles):>10.4f}")
    for gamma in [0.1, 0.5, 1.0, 2.0, 5.0]:
        K   = rbf_kernel(X_s, gamma)
        aln = kernel_alignment(K, y_circles)
        print(f"  {'RBF (γ='+str(gamma)+')':<25} {aln:>10.4f}")

    print("\n  Higher alignment → kernel geometry better fits the class structure.")
    print("  Use alignment to narrow down hyperparameter search before cross-validation.")


def main():
    print("=" * 54)
    print("MODULE 11 — Advanced SVM: Kernels, SVR, Multi-Class")
    print("=" * 54)
    demo_kernel_trick()
    plot_svm_boundaries()
    demo_svr()
    demo_multiclass()
    demo_kernel_alignment()
    print("\nDone.")


if __name__ == "__main__":
    main()
