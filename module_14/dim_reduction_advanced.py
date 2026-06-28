"""module_14/dim_reduction_advanced.py — Advanced Dimensionality Reduction.

Covers:
  - t-SNE (t-distributed Stochastic Neighbour Embedding) for visualisation
  - Kernel PCA (non-linear PCA via kernel trick)
  - Whitening (ZCA/PCA whitening — decorrelate and normalise variance)
  - Truncated SVD (for sparse data, no centering required)
  - Explained variance: when is PCA enough?

Run standalone: python dim_reduction_advanced.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll, load_digits
from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42
PLOTS_DIR   = "plots"
FIGURE_DPI  = 120
rng = np.random.default_rng(RANDOM_SEED)
os.makedirs(PLOTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. PCA vs KERNEL PCA — linear vs non-linear
# ─────────────────────────────────────────────────────────────────────────────

def demo_kernel_pca():
    """Show that linear PCA can't unroll a Swiss Roll; Kernel PCA can."""
    print("── 1. Kernel PCA vs Linear PCA ─────────────────────")
    X, color = make_swiss_roll(n_samples=800, noise=0.1, random_state=RANDOM_SEED)

    # Linear PCA projects to top 2 eigenvectors
    pca   = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # RBF kernel PCA maps to high-dim feature space, then does PCA there
    kpca      = KernelPCA(n_components=2, kernel="rbf", gamma=0.04,
                          random_state=RANDOM_SEED)
    X_kpca    = kpca.fit_transform(X)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor="#0d1117")
    for ax in axes: ax.set_facecolor("#0d1117")

    for ax, X_2d, title in zip(axes,
                                 [X_pca, X_kpca],
                                 ["Linear PCA (manifold not unrolled)",
                                  "Kernel PCA RBF (manifold unrolled)"]):
        sc = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=color, cmap="viridis", s=8)
        ax.set_title(title, color="#e6edf3", fontsize=10)
        ax.tick_params(colors="#e6edf3"); ax.spines[:].set_color("#30363d")

    plt.colorbar(sc, ax=axes[-1])
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/kernel_pca.png", dpi=FIGURE_DPI,
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved → {PLOTS_DIR}/kernel_pca.png")
    print("  Kernel PCA uses the kernel trick: never computes the high-dim coordinates")
    print("  explicitly — only pairwise kernel values k(x_i, x_j) = exp(-γ||x_i-x_j||²).")


# ─────────────────────────────────────────────────────────────────────────────
# 2. t-SNE — visualise high-dimensional data in 2D
# ─────────────────────────────────────────────────────────────────────────────

def demo_tsne():
    """Apply t-SNE to MNIST digits and compare with PCA 2D projection."""
    print("\n── 2. t-SNE on MNIST Digits ────────────────────────")
    digits = load_digits()
    X, y   = digits.data, digits.target

    # PCA to 2D first (optional preprocessing that speeds t-SNE)
    X_pca50 = PCA(n_components=50, random_state=RANDOM_SEED).fit_transform(X)
    pca2    = PCA(n_components=2,  random_state=RANDOM_SEED).fit_transform(X)

    # t-SNE: preserve local neighbourhood structure in 2D
    # perplexity ≈ effective number of neighbours (5–50 typical)
    tsne   = TSNE(n_components=2, perplexity=30, random_state=RANDOM_SEED,
                  n_iter=500)
    X_tsne = tsne.fit_transform(X_pca50)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor="#0d1117")
    for ax in axes: ax.set_facecolor("#0d1117")
    cmap = plt.cm.tab10

    for ax, X_2d, title in zip(axes, [pca2, X_tsne],
                                 ["PCA (linear projection)", "t-SNE (perplexity=30)"]):
        sc = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap=cmap,
                        s=8, alpha=0.7)
        ax.set_title(title, color="#e6edf3")
        ax.tick_params(colors="#e6edf3"); ax.spines[:].set_color("#30363d")

    plt.colorbar(sc, ax=axes[-1], ticks=range(10))
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/tsne_digits.png", dpi=FIGURE_DPI,
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved → {PLOTS_DIR}/tsne_digits.png")
    print("  t-SNE clusters similar digits together much better than linear PCA.")
    print("  CAUTION: t-SNE distances between clusters are NOT meaningful —")
    print("  only local structure is preserved. Do not use for ML features (use PCA).")


# ─────────────────────────────────────────────────────────────────────────────
# 3. WHITENING — decorrelate and normalise variance
# ─────────────────────────────────────────────────────────────────────────────

def pca_whitening(X: np.ndarray) -> np.ndarray:
    """PCA whitening: rotate to eigenbasis, then scale each dimension to unit variance.

    After whitening:
      - Features are decorrelated (covariance matrix = identity)
      - All features have unit variance
      - Very useful before ICA and some neural network layers

    Args:
        X: (n_samples, n_features) centred input

    Returns:
        X_white: whitened matrix
    """
    cov = np.cov(X, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Sort descending (eigh returns ascending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues  = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Scale eigenvectors by 1/sqrt(eigenvalue) → unit variance in each PC direction
    W = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues + 1e-8))
    return X @ W


def demo_whitening():
    """Show that whitening decorrelates features."""
    print("\n── 3. PCA Whitening ────────────────────────────────")
    # Correlated 2D data
    mean  = [0, 0]
    cov   = [[3, 2], [2, 2]]
    X     = rng.multivariate_normal(mean, cov, size=500)
    X    -= X.mean(0)

    X_w = pca_whitening(X)

    print(f"  Original covariance:\n{np.cov(X, rowvar=False).round(2)}")
    print(f"\n  After whitening:\n{np.cov(X_w, rowvar=False).round(2)}")
    print("  Covariance ≈ identity: features decorrelated, unit variance.")


# ─────────────────────────────────────────────────────────────────────────────
# 4. TRUNCATED SVD — for sparse data (text, recommender systems)
# ─────────────────────────────────────────────────────────────────────────────

def demo_truncated_svd():
    """Truncated SVD does NOT centre the data — safe for sparse matrices."""
    print("\n── 4. Truncated SVD (for sparse data) ──────────────")
    # Simulate a sparse document-term matrix (mostly zeros)
    X_sparse = rng.exponential(0.3, size=(200, 500))
    X_sparse[X_sparse < 0.5] = 0    # ~60% sparsity

    svd = TruncatedSVD(n_components=50, random_state=RANDOM_SEED)
    X_reduced = svd.fit_transform(X_sparse)

    ev_ratio = svd.explained_variance_ratio_
    print(f"  Input shape:   {X_sparse.shape}")
    print(f"  Reduced shape: {X_reduced.shape}")
    print(f"  Variance explained by first 50 components: {ev_ratio.sum()*100:.1f}%")
    print(f"  Cumulative variance (first 5): {ev_ratio[:5].cumsum().round(3)}")
    print("\n  Unlike PCA, TruncatedSVD skips mean-centering — centering a sparse")
    print("  matrix makes it dense, destroying memory efficiency.")


# ─────────────────────────────────────────────────────────────────────────────
# 5. HOW MANY COMPONENTS? — explained variance heuristics
# ─────────────────────────────────────────────────────────────────────────────

def demo_explained_variance():
    """Show three common rules for choosing the number of PCA components."""
    print("\n── 5. Choosing Number of PCA Components ────────────")
    digits = load_digits()
    X      = digits.data   # (1797, 64)
    X_scaled = StandardScaler().fit_transform(X)

    pca  = PCA(random_state=RANDOM_SEED)
    pca.fit(X_scaled)
    ev   = pca.explained_variance_ratio_
    cev  = np.cumsum(ev)

    # Rule 1: 95% explained variance threshold
    n_95 = np.argmax(cev >= 0.95) + 1

    # Rule 2: Kaiser criterion — keep eigenvalues > 1 (before standardising, > mean)
    n_kaiser = (pca.explained_variance_ > 1).sum()

    # Rule 3: Elbow in scree plot
    diffs     = np.diff(ev)
    n_elbow   = np.argmin(diffs) + 1   # first big drop-off

    print(f"  95% variance threshold: {n_95} components (from 64)")
    print(f"  Kaiser criterion:       {n_kaiser} components")
    print(f"  Scree plot elbow:       {n_elbow} components")

    fig, ax = plt.subplots(figsize=(8, 4), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    ax.bar(range(1, 31), ev[:30], color="#58a6ff", alpha=0.8, label="Per-component")
    ax.plot(range(1, 31), cev[:30], "o-", color="#3fb950", label="Cumulative")
    ax.axhline(0.95, color="#f85149", ls="--", label="95% threshold")
    ax.set_xlabel("Component", color="#e6edf3")
    ax.set_ylabel("Explained variance ratio", color="#e6edf3")
    ax.set_title("PCA Scree Plot — Digits Dataset", color="#e6edf3")
    ax.legend(facecolor="#161b22", labelcolor="#e6edf3")
    ax.tick_params(colors="#e6edf3"); ax.spines[:].set_color("#30363d")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/pca_scree_advanced.png", dpi=FIGURE_DPI,
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved → {PLOTS_DIR}/pca_scree_advanced.png")


def main():
    print("=" * 54)
    print("MODULE 14 — Advanced Dimensionality Reduction")
    print("=" * 54)
    demo_kernel_pca()
    demo_tsne()
    demo_whitening()
    demo_truncated_svd()
    demo_explained_variance()
    print("\nDone.")


if __name__ == "__main__":
    main()
