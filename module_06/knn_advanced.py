"""module_06/knn_advanced.py — Advanced K-Nearest Neighbours.

Covers:
  - KD-tree: spatial indexing for fast nearest-neighbour search
  - Ball tree: works better in high dimensions than KD-tree
  - Weighted KNN (distance-weighted voting)
  - KNN Regression
  - Curse of dimensionality — why KNN breaks in high-d
  - Efficient search complexity: brute O(nd) vs KD-tree O(n log n)

Run standalone: python knn_advanced.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, BallTree, KDTree
from sklearn.datasets import make_moons, make_regression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error

RANDOM_SEED = 42
PLOTS_DIR   = "plots"
FIGURE_DPI  = 120
rng = np.random.default_rng(RANDOM_SEED)
os.makedirs(PLOTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. KD-TREE CONCEPT
# ─────────────────────────────────────────────────────────────────────────────

def demo_kdtree():
    """Show KD-tree construction and nearest-neighbour query."""
    print("── 1. KD-Tree for Nearest Neighbour Search ─────────")
    print("""
  BRUTE-FORCE KNN: for each query, scan all n training points.
    Time complexity: O(n * d) per query
    Acceptable for small n (< 10k) or small d (< 20)

  KD-TREE (K-Dimensional Tree):
    Binary tree that partitions space by splitting along alternating axes.
    Build: recursively split at median of the widest dimension.
    Query: walk the tree, prune branches that can't improve current best.
    Time complexity: O(log n) per query (approximately)
    Falls apart: when d > 20 (curse of dimensionality — pruning becomes useless)

  BALL-TREE:
    Groups points in hyperspheres instead of axis-aligned boxes.
    Better than KD-tree for d > 10-20 (sphere bounds tighter than box bounds).
  """)

    n = 2000
    X_train = rng.standard_normal((n, 2))   # 2D for easy visualisation
    X_query = rng.standard_normal((10, 2))

    # KD-tree query
    kdt  = KDTree(X_train, leaf_size=30)
    dist, idx = kdt.query(X_query, k=5)

    print(f"  Training set: {X_train.shape}")
    print(f"  Query set:    {X_query.shape}")
    print(f"\n  KD-tree: 5 nearest neighbours for query[0]:")
    for i, (d, j) in enumerate(zip(dist[0], idx[0])):
        print(f"    Neighbour {i+1}: point #{j}  distance={d:.4f}")

    # Speed comparison
    X_big = rng.standard_normal((5000, 10))
    q_big = rng.standard_normal((100, 10))

    t0 = time.perf_counter()
    kdt_big = KDTree(X_big)
    kdt_big.query(q_big, k=5)
    t_kdt = time.perf_counter() - t0

    t0 = time.perf_counter()
    # Brute force: compute all pairwise distances
    dists_brute = np.sqrt(((q_big[:, None] - X_big[None, :]) ** 2).sum(-1))
    _ = np.argpartition(dists_brute, 5, axis=1)[:, :5]
    t_brute = time.perf_counter() - t0

    print(f"\n  Speed comparison (n=5000, d=10, 100 queries):")
    print(f"  KD-tree:    {t_kdt*1000:.2f} ms")
    print(f"  Brute force: {t_brute*1000:.2f} ms")
    print(f"  Speedup: {t_brute/t_kdt:.1f}×")


# ─────────────────────────────────────────────────────────────────────────────
# 2. BALL TREE
# ─────────────────────────────────────────────────────────────────────────────

def demo_ball_tree():
    """Show Ball tree works better at higher dimensions than KD-tree."""
    print("\n── 2. Ball Tree vs KD-Tree at Higher Dimensions ────")
    print("""
  KD-tree partition: axis-aligned boxes
  Ball-tree partition: hyperspheres (balls)

  The ball bound is tighter in high dimensions because a hypersphere
  fills more of the enclosed hypercube than an axis-aligned box does
  relative to the actual point distribution.

  Practical rule of thumb:
  d < 10:   KD-tree is fine
  10 < d < 30: try both (Ball-tree often faster)
  d > 30:   both degrade toward brute-force; consider dimensionality reduction
  """)

    n = 1000
    q = rng.standard_normal((50, 20))

    for d in [2, 5, 10, 20]:
        X = rng.standard_normal((n, d))
        q_d = rng.standard_normal((50, d))

        t0 = time.perf_counter()
        KDTree(X, leaf_size=30).query(q_d, k=5)
        t_kdt = time.perf_counter() - t0

        t0 = time.perf_counter()
        BallTree(X, leaf_size=30).query(q_d, k=5)
        t_ball = time.perf_counter() - t0

        t0 = time.perf_counter()
        dists = np.sqrt(((q_d[:, None] - X[None, :]) ** 2).sum(-1))
        np.argpartition(dists, 5, axis=1)[:, :5]
        t_brute = time.perf_counter() - t0

        print(f"  d={d:>3}: KD-tree={t_kdt*1000:>6.1f}ms  "
              f"Ball-tree={t_ball*1000:>6.1f}ms  Brute={t_brute*1000:>6.1f}ms")

    print("\n  At d=20, all methods converge toward brute-force cost.")


# ─────────────────────────────────────────────────────────────────────────────
# 3. WEIGHTED KNN
# ─────────────────────────────────────────────────────────────────────────────

def demo_weighted_knn():
    """Compare uniform vs distance-weighted voting."""
    print("\n── 3. Weighted KNN (distance-weighted voting) ───────")
    print("""
  Standard KNN: each of the k neighbours gets 1 vote.
  Weighted KNN: closer neighbours get more weight.
    weight = 1 / distance (or 1 / distance²)

  Why weighted?
  Consider k=5 and a query point very close to a class-A neighbour
  but the 4 other (distant) neighbours are class-B.
  Uniform: B wins (4 vs 1).
  Weighted: A might win if the closest distance is much smaller.

  sklearn parameter: weights='uniform' or 'distance'
  """)

    X, y = make_moons(n_samples=300, noise=0.2, random_state=RANDOM_SEED)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_SEED)

    print(f"  {'K':>5}  {'Uniform Acc':>13}  {'Weighted Acc':>13}")
    for k in [1, 3, 5, 9, 15]:
        uniform  = KNeighborsClassifier(k, weights="uniform").fit(X_tr, y_tr)
        weighted = KNeighborsClassifier(k, weights="distance").fit(X_tr, y_tr)
        print(f"  {k:>5}  {uniform.score(X_te, y_te):>13.4f}  {weighted.score(X_te, y_te):>13.4f}")

    print("\n  Weighted KNN generally more accurate, especially at large k.")


# ─────────────────────────────────────────────────────────────────────────────
# 4. KNN REGRESSION
# ─────────────────────────────────────────────────────────────────────────────

def demo_knn_regression():
    """Show KNN regression on a noisy sine wave."""
    print("\n── 4. KNN Regression ────────────────────────────────")
    print("""
  KNN regression: predict the MEAN of k nearest neighbours' y values.
  Weighted: predict weighted mean, closer points contribute more.

  Properties:
  - Non-parametric (no assumption on f(x) shape)
  - Piecewise constant for uniform weighting (step function)
  - Piecewise linear in limit (lots of data)
  - No training time; all work at predict time (lazy learner)
  """)

    x_np = np.linspace(0, 2 * np.pi, 200)
    y_np = np.sin(x_np) + 0.2 * rng.standard_normal(200)
    X    = x_np.reshape(-1, 1)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_np, test_size=0.3, random_state=RANDOM_SEED)

    x_plot = np.linspace(0, 2 * np.pi, 300).reshape(-1, 1)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), facecolor="#0d1117")

    for ax, k in zip(axes, [1, 5, 20]):
        model = KNeighborsRegressor(n_neighbors=k, weights="distance").fit(X_tr, y_tr)
        y_hat = model.predict(x_plot)
        mse   = mean_squared_error(y_te, model.predict(X_te))

        ax.set_facecolor("#161b22")
        ax.scatter(x_np, y_np, s=6, color="#8b949e", alpha=0.4)
        ax.plot(x_plot, y_hat, color="#f85149", linewidth=2)
        ax.set_title(f"k={k}  (Test MSE={mse:.4f})", color="#e6edf3", fontsize=10)
        ax.tick_params(colors="#8b949e")
        ax.spines[:].set_color("#30363d")

    plt.suptitle("KNN Regression: Small k=overfit, Large k=underfit", color="#e6edf3")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/knn_regression.png", dpi=FIGURE_DPI,
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved → {PLOTS_DIR}/knn_regression.png")


# ─────────────────────────────────────────────────────────────────────────────
# 5. CURSE OF DIMENSIONALITY
# ─────────────────────────────────────────────────────────────────────────────

def demo_curse_of_dimensionality():
    """Show that distance concentrates in high dimensions — KNN breaks down."""
    print("\n── 5. Curse of Dimensionality ───────────────────────")
    print("""
  In d dimensions, the volume of a hypersphere scales as r^d.
  Most volume is concentrated in a thin shell near the surface.

  Consequence for KNN:
  - In high-d, the nearest neighbour is almost as far as the farthest.
  - All points become "equidistant" — meaningless concept of "nearest".
  - KNN accuracy degrades without exponentially more data.

  Rule: need O(e^d) samples to maintain same neighbourhood density.
  For d=100 with 1000 samples, k=10 neighbours may be on the other
  side of the space — barely "neighbours" at all.
  """)

    n_samples = 500
    rng2 = np.random.default_rng(RANDOM_SEED)

    print(f"  {'Dimensions':>12}  {'Mean dist':>12}  {'Min dist':>12}  "
          f"{'Max dist':>12}  {'Ratio min/max':>15}")
    for d in [1, 2, 5, 10, 20, 50, 100]:
        X = rng2.standard_normal((n_samples, d))
        q = rng2.standard_normal((1, d))

        dists = np.sqrt(((X - q) ** 2).sum(axis=1))
        print(f"  {d:>12}  {dists.mean():>12.4f}  {dists.min():>12.4f}  "
              f"{dists.max():>12.4f}  {dists.min()/dists.max():>15.4f}")

    print("\n  Ratio min/max → 1 as d increases: 'nearest' ≈ 'farthest'.")
    print("  Remedy: dimensionality reduction (PCA, UMAP) before KNN in high-d.")


def main():
    print("=" * 54)
    print("MODULE 6 — Advanced K-Nearest Neighbours")
    print("=" * 54)
    demo_kdtree()
    demo_ball_tree()
    demo_weighted_knn()
    demo_knn_regression()
    demo_curse_of_dimensionality()
    print("\nDone.")


if __name__ == "__main__":
    main()
