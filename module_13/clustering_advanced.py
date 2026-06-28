"""module_13/clustering_advanced.py — Advanced Clustering.

Covers:
  - k-Means++ initialisation (smarter centroid seeding)
  - DBSCAN (density-based, handles arbitrary shapes and noise)
  - Agglomerative Hierarchical Clustering + Dendrogram
  - Silhouette score (cluster quality without ground truth)
  - Gaussian Mixture Models (soft cluster assignments)

Run standalone: python clustering_advanced.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_blobs
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

RANDOM_SEED = 42
PLOTS_DIR   = "plots"
FIGURE_DPI  = 120
rng = np.random.default_rng(RANDOM_SEED)
os.makedirs(PLOTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. K-MEANS++ INITIALISATION
# ─────────────────────────────────────────────────────────────────────────────

def kmeans_plus_plus_init(X: np.ndarray, k: int) -> np.ndarray:
    """k-Means++ centroid initialisation.

    Selects centroids probabilistically: each new centroid is chosen with
    probability proportional to its distance² from the nearest existing centroid.
    This spreads centroids out, avoiding the pathological convergence of random init.

    Args:
        X: (n_samples, n_features)
        k: number of clusters

    Returns:
        centroids: (k, n_features)
    """
    n = len(X)
    # First centroid: uniform random
    first = rng.integers(0, n)
    centroids = [X[first]]

    for _ in range(1, k):
        # Distance² from each point to its nearest centroid
        dists = np.array([
            min(np.sum((x - c)**2) for c in centroids)
            for x in X
        ])
        # Sample proportional to distance²
        probs = dists / dists.sum()
        idx   = rng.choice(n, p=probs)
        centroids.append(X[idx])

    return np.array(centroids)


def demo_kmeans_plusplus():
    """Compare random init vs k-Means++ on convergence speed."""
    print("── 1. k-Means++ Initialisation ─────────────────────")
    X, _ = make_blobs(n_samples=500, centers=5, cluster_std=1.2,
                      random_state=RANDOM_SEED)

    # sklearn KMeans with both init strategies
    km_rand = KMeans(n_clusters=5, init="random", n_init=1, random_state=RANDOM_SEED)
    km_pp   = KMeans(n_clusters=5, init="k-means++", n_init=1, random_state=RANDOM_SEED)

    km_rand.fit(X); km_pp.fit(X)

    print(f"  Random init  — inertia: {km_rand.inertia_:.1f}  "
          f"n_iter: {km_rand.n_iter_}")
    print(f"  k-Means++ init — inertia: {km_pp.inertia_:.1f}  "
          f"n_iter: {km_pp.n_iter_}")
    print("  k-Means++ converges in fewer iterations and to a better optimum.")


# ─────────────────────────────────────────────────────────────────────────────
# 2. DBSCAN — density-based clustering
# ─────────────────────────────────────────────────────────────────────────────

def demo_dbscan():
    """Show DBSCAN handling non-convex shapes and noise that k-Means fails on."""
    print("\n── 2. DBSCAN ────────────────────────────────────────")
    X, y_true = make_moons(n_samples=300, noise=0.1, random_state=RANDOM_SEED)

    # k-Means fails on moons — it assumes convex, spherical clusters
    km  = KMeans(n_clusters=2, n_init=10, random_state=RANDOM_SEED)
    km.fit(X)

    # DBSCAN: core points have >= min_samples neighbours within eps
    # Labels: -1 = noise, 0/1/... = cluster ids
    db = DBSCAN(eps=0.2, min_samples=5)
    db_labels = db.fit_predict(X)

    n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    n_noise    = (db_labels == -1).sum()
    print(f"  DBSCAN found {n_clusters} clusters, {n_noise} noise points")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor="#0d1117")
    for ax in axes: ax.set_facecolor("#0d1117")

    colours = ["#58a6ff", "#3fb950", "#f85149", "#bc8cff"]
    for ax, labels, title in zip(axes, [km.labels_, db_labels],
                                  ["k-Means (fails)", "DBSCAN (correct)"]):
        for label in set(labels):
            mask  = labels == label
            col   = "#8b949e" if label == -1 else colours[label % len(colours)]
            name  = "Noise" if label == -1 else f"Cluster {label}"
            ax.scatter(X[mask, 0], X[mask, 1], c=col, s=15, label=name, alpha=0.8)
        ax.set_title(title, color="#e6edf3")
        ax.legend(facecolor="#161b22", labelcolor="#e6edf3", fontsize=8)
        ax.tick_params(colors="#e6edf3"); ax.spines[:].set_color("#30363d")

    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/dbscan_vs_kmeans.png", dpi=FIGURE_DPI,
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved → {PLOTS_DIR}/dbscan_vs_kmeans.png")
    print("  eps controls the neighbourhood radius; min_samples controls density.")
    print("  eps too large → everything is one cluster. Too small → everything is noise.")


# ─────────────────────────────────────────────────────────────────────────────
# 3. HIERARCHICAL CLUSTERING + DENDROGRAM
# ─────────────────────────────────────────────────────────────────────────────

def demo_hierarchical():
    """Build a dendrogram to visualise the merging hierarchy."""
    print("\n── 3. Hierarchical Clustering ──────────────────────")
    X, _ = make_blobs(n_samples=40, centers=4, cluster_std=0.8,
                      random_state=RANDOM_SEED)

    # scipy linkage computes the full merge tree
    Z = linkage(X, method="ward")   # Ward: minimise within-cluster variance

    fig, ax = plt.subplots(figsize=(10, 4), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    dendrogram(Z, ax=ax, color_threshold=5, above_threshold_color="#8b949e")
    ax.set_title("Hierarchical Clustering Dendrogram (Ward linkage)",
                 color="#e6edf3")
    ax.set_xlabel("Sample index", color="#e6edf3")
    ax.set_ylabel("Distance", color="#e6edf3")
    ax.tick_params(colors="#e6edf3"); ax.spines[:].set_color("#30363d")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/dendrogram.png", dpi=FIGURE_DPI,
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved → {PLOTS_DIR}/dendrogram.png")
    print("  Cut the dendrogram at a height to choose number of clusters.")
    print("  Large vertical gaps before the cut → natural cluster structure.")

    # sklearn implementation for labels
    hc = AgglomerativeClustering(n_clusters=4, linkage="ward")
    labels = hc.fit_predict(X)
    print(f"  AgglomerativeClustering(k=4) assigned: {np.bincount(labels)}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. SILHOUETTE SCORE — cluster quality metric
# ─────────────────────────────────────────────────────────────────────────────

def demo_silhouette():
    """Use silhouette score to select k without ground truth labels."""
    print("\n── 4. Silhouette Score (choose k) ──────────────────")
    X, _ = make_blobs(n_samples=300, centers=4, cluster_std=1.0,
                      random_state=RANDOM_SEED)

    # silhouette(x_i) = (b_i - a_i) / max(a_i, b_i)
    # a_i = mean intra-cluster distance, b_i = mean distance to nearest other cluster
    # Range [-1, 1]: 1 = well-separated, 0 = on boundary, -1 = wrong cluster
    ks = range(2, 9)
    scores = []
    for k in ks:
        km     = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_SEED)
        labels = km.fit_predict(X)
        scores.append(silhouette_score(X, labels))
        print(f"  k={k}: silhouette={scores[-1]:.4f}")

    best_k = ks[np.argmax(scores)]
    print(f"\n  Best k by silhouette: {best_k}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. GAUSSIAN MIXTURE MODELS — soft assignments
# ─────────────────────────────────────────────────────────────────────────────

def demo_gmm():
    """GMM assigns soft membership probabilities; k-Means gives hard assignments."""
    print("\n── 5. Gaussian Mixture Models ──────────────────────")
    X, _ = make_blobs(n_samples=300, centers=3, cluster_std=1.5,
                      random_state=RANDOM_SEED)

    gmm = GaussianMixture(n_components=3, covariance_type="full",
                          random_state=RANDOM_SEED)
    gmm.fit(X)
    proba  = gmm.predict_proba(X)   # (n, k) soft memberships
    labels = gmm.predict(X)          # (n,) hard assignments

    print(f"  Component means:\n{gmm.means_.round(2)}")
    print(f"\n  Sample soft memberships (first 5 points):")
    for i in range(5):
        print(f"    x={X[i].round(2)}  →  {proba[i].round(3)}")

    # BIC to choose number of components
    print(f"\n  BIC (Bayesian Information Criterion) — lower is better:")
    for k in [2, 3, 4, 5]:
        bic = GaussianMixture(n_components=k, random_state=RANDOM_SEED).fit(X).bic(X)
        print(f"    k={k}: BIC={bic:.1f}")
    print("  Minimum BIC selects the model that best balances fit and complexity.")


def main():
    print("=" * 54)
    print("MODULE 13 — Advanced Clustering")
    print("=" * 54)
    demo_kmeans_plusplus()
    demo_dbscan()
    demo_hierarchical()
    demo_silhouette()
    demo_gmm()
    print("\nDone.")


if __name__ == "__main__":
    main()
