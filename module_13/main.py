"""module_13/main.py — k-Means Clustering from scratch.

Clusters detective cases without using the 'solved' label,
then checks how well the clusters align with the true labels.
Run: python main.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_loader import load_and_clean, to_arrays
from kmeans import KMeans, elbow_sweep

# ── Constants ────────────────────────────────────────────────────────────────
DATA_FILE  = "cases.csv"
K_RANGE    = range(2, 11)    # sweep k from 2 to 10
K_CHOSEN   = 3               # the k we'll use for detailed analysis
PLOTS_DIR  = "plots"
FIGURE_DPI = 120


def plot_elbow(k_range, inertias):
    """Save elbow curve: inertia vs k."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    ks = list(k_range)
    fig, ax = plt.subplots(figsize=(7, 4), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    ax.plot(ks, inertias, "o-", color="#58a6ff")
    ax.set_xlabel("k (number of clusters)", color="#e6edf3")
    ax.set_ylabel("Inertia",                color="#e6edf3")
    ax.set_title("Elbow Curve",             color="#e6edf3")
    ax.tick_params(colors="#e6edf3")
    ax.spines[:].set_color("#30363d")
    plt.tight_layout()
    fname = f"{PLOTS_DIR}/kmeans_elbow.png"
    plt.savefig(fname, dpi=FIGURE_DPI, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved → {fname}")


def plot_clusters_2d(X, labels, centroids, feat_names):
    """Project to first two features and colour by cluster."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    colours = ["#58a6ff", "#3fb950", "#d29922", "#f85149", "#bc8cff"]
    k = centroids.shape[0]

    fig, ax = plt.subplots(figsize=(7, 5), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    for c in range(k):
        mask = labels == c
        ax.scatter(X[mask, 0], X[mask, 1],
                   color=colours[c % len(colours)], alpha=0.7,
                   label=f"Cluster {c}", s=40)
        ax.scatter(centroids[c, 0], centroids[c, 1],
                   color=colours[c % len(colours)], marker="X", s=200,
                   edgecolors="#e6edf3", linewidths=1.5)
    ax.set_xlabel(feat_names[0], color="#e6edf3")
    ax.set_ylabel(feat_names[1], color="#e6edf3")
    ax.set_title(f"k-Means k={k} (first 2 features)", color="#e6edf3")
    ax.tick_params(colors="#e6edf3")
    ax.legend(facecolor="#161b22", labelcolor="#e6edf3")
    ax.spines[:].set_color("#30363d")
    plt.tight_layout()
    fname = f"{PLOTS_DIR}/kmeans_clusters.png"
    plt.savefig(fname, dpi=FIGURE_DPI, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved → {fname}")


def cluster_purity(labels_pred, labels_true, k):
    """Measure how much each cluster overlaps with a single true class.

    Pure cluster = all members share the same true label.
    Purity = (sum of majority class per cluster) / total samples.
    """
    total_correct = 0
    for c in range(k):
        members_true = labels_true[labels_pred == c]
        if len(members_true) == 0:
            continue
        majority_count = np.bincount(members_true).max()
        total_correct += majority_count
    return total_correct / len(labels_true)


def main():
    """Load data, run elbow sweep, fit k=3, evaluate purity."""
    df              = load_and_clean(DATA_FILE)
    X, y, feat_cols = to_arrays(df)

    print("=" * 52)
    print("MODULE 13 — k-Means Clustering from Scratch")
    print("=" * 52)
    print(f"Data: {X.shape[0]} cases, {X.shape[1]} features (no labels used)")

    # ── Elbow sweep ───────────────────────────────────────────────────────────
    print(f"\n=== Elbow Sweep k=2..10 ===")
    inertias = elbow_sweep(X, K_RANGE)
    plot_elbow(K_RANGE, inertias)

    # ── Fit chosen k ─────────────────────────────────────────────────────────
    print(f"\n=== k={K_CHOSEN} Detailed Analysis ===")
    km = KMeans(k=K_CHOSEN)
    km.fit(X)
    print(f"Converged in {km.n_iter_} iterations")
    print(f"Final inertia: {km.inertia_:.2f}")

    # ── Cluster composition (compare to true labels) ──────────────────────────
    print(f"\nCluster composition (true label breakdown):")
    for c in range(K_CHOSEN):
        members_y = y[km.labels_ == c]
        solved   = members_y.sum()
        unsolved = len(members_y) - solved
        print(f"  Cluster {c}: {len(members_y):3d} cases — "
              f"{solved} solved, {unsolved} unsolved")

    purity = cluster_purity(km.labels_, y, K_CHOSEN)
    print(f"\nCluster purity (vs. true labels): {purity:.3f}")
    print("  (1.0 = perfect separation, 0.5 = random)")

    # ── Centroid positions ────────────────────────────────────────────────────
    print(f"\nCentroid positions (feature means per cluster):")
    print(f"  {'Feature':<22} " + "  ".join(f"C{c}" for c in range(K_CHOSEN)))
    print("  " + "-" * 42)
    for fi, fname in enumerate(feat_cols):
        vals = "  ".join(f"{km.centroids_[c, fi]:6.2f}" for c in range(K_CHOSEN))
        print(f"  {fname:<22} {vals}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_clusters_2d(X, km.labels_, km.centroids_, feat_cols)
    print("\nDone.")


if __name__ == "__main__":
    main()
