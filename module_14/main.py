"""module_14/main.py — PCA from Scratch.

Reduces 6-feature detective data to 2D, visualises it coloured
by true labels and by k-Means clusters.
Run: python main.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_loader import load_and_clean, to_arrays
from pca import PCA

# ── Constants ────────────────────────────────────────────────────────────────
DATA_FILE    = "cases.csv"
N_COMPONENTS = 2
PLOTS_DIR    = "plots"
FIGURE_DPI   = 120


def plot_pca_2d(X_2d, colour_by, title, fname, cmap_labels=None):
    """Save a 2D scatter plot of PCA-projected data.

    Args:
        X_2d:        (n_samples, 2) projected data
        colour_by:   (n_samples,) int array — used to colour points
        title:       plot title string
        fname:       output file path
        cmap_labels: optional dict {int: label_string} for legend
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)
    colours = ["#58a6ff", "#3fb950", "#d29922", "#f85149"]

    fig, ax = plt.subplots(figsize=(7, 5), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")

    for cls in np.unique(colour_by):
        mask  = colour_by == cls
        label = cmap_labels[cls] if cmap_labels else str(cls)
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   color=colours[int(cls) % len(colours)],
                   alpha=0.75, s=40, label=label)

    ax.set_xlabel("PC1", color="#e6edf3")
    ax.set_ylabel("PC2", color="#e6edf3")
    ax.set_title(title,  color="#e6edf3")
    ax.tick_params(colors="#e6edf3")
    ax.legend(facecolor="#161b22", labelcolor="#e6edf3")
    ax.spines[:].set_color("#30363d")

    plt.tight_layout()
    plt.savefig(fname, dpi=FIGURE_DPI, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved → {fname}")


def plot_scree(pca_full):
    """Save a scree plot showing explained variance for all components.

    Args:
        pca_full: fitted PCA with n_components = n_features
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)
    ratios = pca_full.explained_variance_ratio_
    cumul  = np.cumsum(ratios)
    xs     = range(1, len(ratios) + 1)

    fig, ax = plt.subplots(figsize=(7, 4), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    ax.bar(xs, ratios * 100, color="#58a6ff", alpha=0.8, label="Individual")
    ax.plot(xs, cumul * 100, "o-", color="#3fb950", label="Cumulative")
    ax.axhline(90, color="#d29922", linestyle="--", linewidth=1, label="90% threshold")
    ax.set_xlabel("Principal Component", color="#e6edf3")
    ax.set_ylabel("Variance Explained (%)", color="#e6edf3")
    ax.set_title("Scree Plot", color="#e6edf3")
    ax.tick_params(colors="#e6edf3")
    ax.legend(facecolor="#161b22", labelcolor="#e6edf3")
    ax.spines[:].set_color("#30363d")

    plt.tight_layout()
    fname = f"{PLOTS_DIR}/pca_scree.png"
    plt.savefig(fname, dpi=FIGURE_DPI, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved → {fname}")


def main():
    """Load data, run PCA, visualise, print component summary."""
    df              = load_and_clean(DATA_FILE)
    X, y, feat_cols = to_arrays(df)

    print("=" * 52)
    print("MODULE 14 — PCA from Scratch")
    print("=" * 52)
    print(f"Original data shape: {X.shape}")

    # ── Full PCA (all components) for scree plot ───────────────────────────
    pca_full = PCA(n_components=X.shape[1])
    pca_full.fit(X)
    pca_full.print_summary(feature_names=feat_cols)
    plot_scree(pca_full)

    # ── 2-component PCA for visualisation ─────────────────────────────────
    pca2 = PCA(n_components=N_COMPONENTS)
    X_2d = pca2.fit_transform(X)
    print(f"\nProjected shape: {X_2d.shape}  (6D → 2D)")

    plot_pca_2d(
        X_2d, y,
        title="PCA — coloured by true label",
        fname=f"{PLOTS_DIR}/pca_true_labels.png",
        cmap_labels={0: "Unsolved", 1: "Solved"},
    )

    # ── Overlay k-Means clusters on PCA projection ─────────────────────────
    # Import KMeans from module_13 — shows PCA + clustering working together
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "module_13"))
    from kmeans import KMeans
    km = KMeans(k=3)
    km.fit(X)   # cluster in original 6D space
    plot_pca_2d(
        X_2d, km.labels_,
        title="PCA — coloured by k-Means cluster (k=3)",
        fname=f"{PLOTS_DIR}/pca_kmeans.png",
    )

    # ── Variance check ─────────────────────────────────────────────────────
    total_ratio = pca2.explained_variance_ratio_.sum()
    print(f"\nPC1 + PC2 capture {total_ratio*100:.1f}% of total variance")
    print("Remaining variance is lost when projecting to 2D.")

    print("\nDone.")


if __name__ == "__main__":
    main()
