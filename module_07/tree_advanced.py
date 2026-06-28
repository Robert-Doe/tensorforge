"""module_07/tree_advanced.py — Advanced Decision Tree Topics.

Covers:
  - CART vs ID3/C4.5 (splitting criteria compared)
  - Regression trees (predict mean of leaf samples)
  - Continuous feature splitting (all possible thresholds)
  - Pre-pruning (max_depth, min_samples_leaf) vs post-pruning (CCP)
  - Instability of single trees (sensitivity to data changes)

Run standalone: python tree_advanced.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text
from sklearn.datasets import make_moons, make_regression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error

RANDOM_SEED = 42
PLOTS_DIR   = "plots"
FIGURE_DPI  = 120
rng = np.random.default_rng(RANDOM_SEED)
os.makedirs(PLOTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. SPLITTING CRITERIA: CART vs ID3 vs C4.5
# ─────────────────────────────────────────────────────────────────────────────

def gini(y: np.ndarray) -> float:
    """Gini impurity: 1 - Σ p_c²  (CART criterion)."""
    n = len(y)
    if n == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probs = counts / n
    return 1.0 - np.sum(probs ** 2)


def entropy(y: np.ndarray) -> float:
    """Shannon entropy: -Σ p_c * log2(p_c)  (ID3/C4.5 criterion)."""
    n = len(y)
    if n == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probs = counts / n
    return -np.sum(probs * np.log2(probs + 1e-12))


def information_gain(y_parent: np.ndarray, y_left: np.ndarray,
                     y_right: np.ndarray, criterion_fn) -> float:
    """IG = H(parent) - weighted_avg(H(left), H(right))."""
    n = len(y_parent)
    n_l, n_r = len(y_left), len(y_right)
    return (criterion_fn(y_parent)
            - (n_l / n) * criterion_fn(y_left)
            - (n_r / n) * criterion_fn(y_right))


def demo_splitting_criteria():
    """Compare Gini vs Entropy on a concrete split."""
    print("── 1. Splitting Criteria: CART vs ID3 ──────────────")
    print("""
  Algorithm comparison:
  ────────────────────────────────────────────────────────────────
  ID3  (Quinlan, 1986): uses Information Gain (entropy reduction)
    - Only categorical features
    - Biased toward features with many values

  C4.5 (Quinlan, 1993): uses Gain Ratio (normalise IG by split info)
    - Handles continuous features (finds threshold)
    - Less biased toward high-cardinality features

  CART (Breiman et al., 1984): uses Gini impurity
    - Binary splits only (unlike ID3/C4.5 which can do multi-way)
    - Works for classification and regression
    - sklearn's DecisionTreeClassifier uses CART
  """)

    # Example split: 100 samples, 60 positive, 40 negative
    # Split A: 50/50 pure left, 50 right (25+/25-)
    # Split B: 40 left (30+/10-), 60 right (30+/30-)
    parent = np.array([1]*60 + [0]*40)

    y_left_A  = np.array([1]*50)             # pure positive
    y_right_A = np.array([1]*10 + [0]*40)   # mixed

    y_left_B  = np.array([1]*30 + [0]*10)   # mostly positive
    y_right_B = np.array([1]*30 + [0]*30)   # 50-50

    print(f"  {'Split':<10} {'IG (Entropy)':>15} {'IG (Gini)':>12}")
    for name, left, right in [("Split A", y_left_A, y_right_A),
                               ("Split B", y_left_B, y_right_B)]:
        ig_entropy = information_gain(parent, left, right, entropy)
        ig_gini    = information_gain(parent, left, right, gini)
        print(f"  {name:<10} {ig_entropy:>15.4f} {ig_gini:>12.4f}")

    print("\n  Both criteria generally agree on the best split.")
    print("  Gini is faster (no log computation); Entropy is slightly more principled.")
    print("  In practice: difference in tree quality is minimal.")


# ─────────────────────────────────────────────────────────────────────────────
# 2. CONTINUOUS FEATURE SPLITTING
# ─────────────────────────────────────────────────────────────────────────────

def find_best_threshold(x: np.ndarray, y: np.ndarray) -> tuple:
    """Find the threshold for feature x that maximises Gini gain.

    CART checks every midpoint between adjacent sorted values.
    For n samples: at most n-1 candidate thresholds.

    Args:
        x: (n,) feature values
        y: (n,) binary labels

    Returns:
        (best_threshold, best_gain)
    """
    sorted_idx      = np.argsort(x)
    x_sorted, y_sorted = x[sorted_idx], y[sorted_idx]
    unique_x        = np.unique(x_sorted)
    thresholds      = (unique_x[:-1] + unique_x[1:]) / 2.0

    best_gain  = -np.inf
    best_threshold = None

    for t in thresholds:
        left  = y_sorted[x_sorted <= t]
        right = y_sorted[x_sorted >  t]
        gain  = information_gain(y_sorted, left, right, gini)
        if gain > best_gain:
            best_gain      = gain
            best_threshold = t

    return best_threshold, best_gain


def demo_continuous_splits():
    """Show threshold search on a 1D feature."""
    print("\n── 2. Continuous Feature Splitting ─────────────────")
    x = np.array([1.0, 2.0, 3.0, 5.0, 6.0, 8.0, 9.0, 10.0])
    y = np.array([0,   0,   0,   1,   1,   1,   0,    1   ])

    thresh, gain = find_best_threshold(x, y)
    print(f"  Feature values: {x}")
    print(f"  Labels:         {y}")
    print(f"  Best threshold: x <= {thresh:.2f}   (Gini gain={gain:.4f})")
    print(f"  Left  (x<={thresh:.1f}): {y[x <= thresh]}")
    print(f"  Right (x> {thresh:.1f}): {y[x >  thresh]}")
    print(f"  CART does this search for EVERY feature, picks the best.")
    print(f"  For p features, n samples: O(p * n * log n) per node split.")


# ─────────────────────────────────────────────────────────────────────────────
# 3. REGRESSION TREE
# ─────────────────────────────────────────────────────────────────────────────

def demo_regression_tree():
    """Fit a regression tree and compare to polynomial regression."""
    print("\n── 3. Regression Trees ──────────────────────────────")
    print("""
  Regression tree splits criterion: minimise MSE at each node.
  Leaf prediction: mean(y_i) for samples in that leaf.
  Optimal split: reduces Σ_leaf (MSE_leaf * n_leaf) most.

  Effect of max_depth:
  - Small: high bias (linear, step-function), low variance
  - Large: low bias (can fit any pattern), HIGH variance (overfit)
  """)
    x_np = np.linspace(0, 10, 200)
    y_np = np.sin(x_np) + 0.3 * rng.standard_normal(200)
    X    = x_np.reshape(-1, 1)
    y    = y_np

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_SEED)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), facecolor="#0d1117")
    depths = [1, 3, 10]
    x_plot = np.linspace(0, 10, 300).reshape(-1, 1)

    for ax, depth in zip(axes, depths):
        model = DecisionTreeRegressor(max_depth=depth, random_state=RANDOM_SEED)
        model.fit(X_tr, y_tr)
        y_pred_plot = model.predict(x_plot)
        y_pred_te   = model.predict(X_te)
        mse         = mean_squared_error(y_te, y_pred_te)

        ax.set_facecolor("#161b22")
        ax.scatter(x_np, y_np, s=8, color="#8b949e", alpha=0.4)
        ax.plot(x_plot, y_pred_plot, color="#f85149", linewidth=2)
        ax.set_title(f"max_depth={depth}\nTest MSE={mse:.4f}", color="#e6edf3", fontsize=10)
        ax.tick_params(colors="#8b949e")
        ax.spines[:].set_color("#30363d")

    plt.suptitle("Regression Tree: Bias-Variance vs Depth", color="#e6edf3")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/regression_tree.png", dpi=FIGURE_DPI,
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved → {PLOTS_DIR}/regression_tree.png")

    print(f"\n  {'max_depth':>10} {'Test MSE':>12} {'n_leaves':>10}")
    for d in depths:
        m = DecisionTreeRegressor(max_depth=d, random_state=RANDOM_SEED).fit(X_tr, y_tr)
        mse = mean_squared_error(y_te, m.predict(X_te))
        print(f"  {str(d):>10} {mse:>12.4f} {m.get_n_leaves():>10}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. PRUNING — preventing overfitting
# ─────────────────────────────────────────────────────────────────────────────

def demo_pruning():
    """Compare pre-pruning and cost-complexity post-pruning."""
    print("\n── 4. Tree Pruning ──────────────────────────────────")
    print("""
  PRE-PRUNING (stop early):
    max_depth        — cap tree depth
    min_samples_leaf — require at least k samples per leaf
    min_impurity_decrease — only split if gain > threshold
    Pros: fast, simple
    Cons: greedy — can miss beneficial splits later

  POST-PRUNING (CCP — Cost Complexity Pruning):
    Build full tree, then remove subtrees that don't help generalisation.
    Controlled by ccp_alpha: higher = more aggressive pruning.
    sklearn: tree.cost_complexity_pruning_path() gives all valid alphas.
    Pros: globally optimal pruning (not greedy)
    Cons: more expensive (build then prune)
  """)

    X, y = make_moons(n_samples=400, noise=0.25, random_state=RANDOM_SEED)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_SEED)

    # Cost-complexity pruning path
    full_tree = DecisionTreeClassifier(random_state=RANDOM_SEED)
    path = full_tree.cost_complexity_pruning_path(X_tr, y_tr)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities

    # Evaluate accuracy at each alpha
    train_accs, test_accs, n_leaves = [], [], []
    for alpha in ccp_alphas:
        dt = DecisionTreeClassifier(ccp_alpha=alpha, random_state=RANDOM_SEED)
        dt.fit(X_tr, y_tr)
        train_accs.append(dt.score(X_tr, y_tr))
        test_accs.append(dt.score(X_te, y_te))
        n_leaves.append(dt.get_n_leaves())

    best_idx   = np.argmax(test_accs)
    best_alpha = ccp_alphas[best_idx]
    print(f"  Best ccp_alpha: {best_alpha:.5f}  "
          f"(test acc={test_accs[best_idx]:.4f},  n_leaves={n_leaves[best_idx]})")
    print(f"  Full tree (alpha=0): leaves={n_leaves[0]}, test acc={test_accs[0]:.4f}")
    print(f"  Stump    (alpha={ccp_alphas[-1]:.4f}): leaves={n_leaves[-1]}, "
          f"test acc={test_accs[-1]:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. TREE INSTABILITY DEMONSTRATION
# ─────────────────────────────────────────────────────────────────────────────

def demo_instability():
    """Show that small data changes cause very different tree structures."""
    print("\n── 5. Decision Tree Instability ────────────────────")
    X, y = make_moons(n_samples=100, noise=0.2, random_state=RANDOM_SEED)

    print(f"  Tree structures after bootstrap resampling (same data, different sample):")
    print(f"  {'Sample':>8} {'n_leaves':>10} {'Depth':>8} {'Features used'}")
    for seed in range(5):
        idx   = np.random.default_rng(seed).integers(0, len(X), len(X))
        X_b, y_b = X[idx], y[idx]
        dt = DecisionTreeClassifier(max_depth=4, random_state=RANDOM_SEED).fit(X_b, y_b)
        features_used = sorted(set(dt.tree_.feature[dt.tree_.feature >= 0]))
        print(f"  {seed:>8} {dt.get_n_leaves():>10} {dt.get_depth():>8} {features_used}")

    print("\n  Same learning algorithm + nearly same data → very different trees!")
    print("  This high variance is why ENSEMBLE methods (Random Forest, Gradient Boosting)")
    print("  aggregate many trees: averaging cancels out individual tree variance.")


def main():
    print("=" * 54)
    print("MODULE 7 — Advanced Decision Trees")
    print("=" * 54)
    demo_splitting_criteria()
    demo_continuous_splits()
    demo_regression_tree()
    demo_pruning()
    demo_instability()
    print("\nDone.")


if __name__ == "__main__":
    main()
