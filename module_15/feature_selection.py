"""module_15/feature_selection.py — Feature Selection techniques.

Covers:
  - Filter methods: variance threshold, correlation, mutual information
  - Wrapper methods: Recursive Feature Elimination (RFE)
  - Embedded methods: L1-based selection, tree feature importances
  - Target encoding (for high-cardinality categoricals)
  - Interaction features and polynomial expansion

Run standalone: python feature_selection.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, mutual_info_classif,
    RFE, SelectFromModel,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures

RANDOM_SEED = 42
PLOTS_DIR   = "plots"
FIGURE_DPI  = 120
rng = np.random.default_rng(RANDOM_SEED)
os.makedirs(PLOTS_DIR, exist_ok=True)


def make_data_with_noise():
    """Create classification data with 10 real features + 10 noise features."""
    X, y = make_classification(
        n_samples=500, n_features=20, n_informative=10,
        n_redundant=0, n_repeated=0, random_state=RANDOM_SEED
    )
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# 1. FILTER METHODS
# ─────────────────────────────────────────────────────────────────────────────

def demo_filter_methods(X, y):
    """Variance threshold, correlation filter, mutual information."""
    print("── 1. Filter Methods ───────────────────────────────")

    # Variance threshold — remove near-constant features
    vt = VarianceThreshold(threshold=0.1)
    X_vt = vt.fit_transform(X)
    print(f"  Variance threshold (0.1): {X.shape[1]} → {X_vt.shape[1]} features")

    # Correlation filter — remove features correlated with each other
    # Keeps only one from each highly-correlated pair
    df = pd.DataFrame(X)
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
    print(f"  Correlation filter (>0.95): dropped {len(to_drop)} features")

    # Mutual information — measures non-linear dependence between feature and target
    mi_scores = mutual_info_classif(X, y, random_state=RANDOM_SEED)
    ranked    = np.argsort(mi_scores)[::-1]
    print(f"\n  Top 5 features by mutual information:")
    for i, feat_idx in enumerate(ranked[:5]):
        print(f"    feature_{feat_idx}: MI={mi_scores[feat_idx]:.4f}")
    print(f"  Bottom 5 (likely noise):")
    for feat_idx in ranked[-5:]:
        print(f"    feature_{feat_idx}: MI={mi_scores[feat_idx]:.4f}")

    # SelectKBest wraps any filter metric
    selector = SelectKBest(mutual_info_classif, k=10)
    X_best   = selector.fit_transform(X, y)
    print(f"\n  SelectKBest(k=10): {X.shape[1]} → {X_best.shape[1]} features")
    return X_best


# ─────────────────────────────────────────────────────────────────────────────
# 2. RECURSIVE FEATURE ELIMINATION (wrapper)
# ─────────────────────────────────────────────────────────────────────────────

def demo_rfe(X, y):
    """RFE trains model, removes weakest feature, repeats until k features remain."""
    print("\n── 2. Recursive Feature Elimination (RFE) ──────────")
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    rfe   = RFE(estimator=model, n_features_to_select=10, step=1)
    rfe.fit(X, y)

    selected = np.where(rfe.support_)[0]
    ranking  = rfe.ranking_
    print(f"  Selected features: {sorted(selected.tolist())}")
    print(f"  Feature rankings (1 = selected, higher = eliminated earlier):")
    for i, rank in enumerate(ranking):
        marker = " ← selected" if rank == 1 else ""
        print(f"    feature_{i}: rank={rank}{marker}")

    # CV accuracy with selected vs all features
    acc_all = cross_val_score(model, X, y, cv=5).mean()
    acc_rfe = cross_val_score(model, rfe.transform(X), y, cv=5).mean()
    print(f"\n  CV accuracy — all features: {acc_all:.4f}")
    print(f"  CV accuracy — RFE features: {acc_rfe:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. EMBEDDED METHODS — L1 and tree importances
# ─────────────────────────────────────────────────────────────────────────────

def demo_embedded(X, y):
    """L1 logistic regression zeroes out irrelevant features; tree importances rank them."""
    print("\n── 3. Embedded Feature Selection ───────────────────")

    # L1 Logistic Regression — sparse coefficients
    l1_model = LogisticRegression(penalty="l1", C=0.5, solver="saga",
                                   max_iter=2000, random_state=RANDOM_SEED)
    l1_model.fit(X, y)
    n_nonzero = (l1_model.coef_[0] != 0).sum()
    print(f"  L1 logistic (C=0.5): {n_nonzero}/{X.shape[1]} non-zero coefficients")

    # SelectFromModel with L1
    sfm = SelectFromModel(l1_model, prefit=True)
    X_l1 = sfm.transform(X)
    print(f"  SelectFromModel (L1): {X.shape[1]} → {X_l1.shape[1]} features")

    # Random Forest feature importances (built-in)
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
    rf.fit(X, y)
    importances = rf.feature_importances_
    top5 = np.argsort(importances)[::-1][:5]
    print(f"\n  Random Forest top 5 feature importances:")
    for idx in top5:
        print(f"    feature_{idx}: {importances[idx]:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. TARGET ENCODING — for high-cardinality categoricals
# ─────────────────────────────────────────────────────────────────────────────

def target_encode(df: pd.DataFrame, col: str, target: str,
                  smoothing: float = 10.0) -> pd.Series:
    """Smoothed target encoding to avoid overfitting on rare categories.

    Replaces category with a weighted average of:
      - Category mean (noisy for rare categories)
      - Global mean (conservative fallback)

    Weight = category_count / (category_count + smoothing)
    More samples in category → closer to category mean.

    Args:
        df:        DataFrame with column and target
        col:       categorical column name
        target:    binary target column name
        smoothing: smoothing strength (higher = more conservative)

    Returns:
        encoded Series
    """
    global_mean   = df[target].mean()
    stats         = df.groupby(col)[target].agg(["mean", "count"])
    smoothed_mean = (stats["count"] * stats["mean"] + smoothing * global_mean) / \
                    (stats["count"] + smoothing)
    return df[col].map(smoothed_mean)


def demo_target_encoding():
    """Compare raw label encoding vs smoothed target encoding."""
    print("\n── 4. Target Encoding ──────────────────────────────")
    # Simulate a high-cardinality categorical (e.g. 50 zip codes)
    n = 500
    n_cats = 50
    df = pd.DataFrame({
        "zip_code": rng.integers(0, n_cats, size=n).astype(str),
        "feature1": rng.standard_normal(n),
        "target":   rng.integers(0, 2, size=n),
    })
    # Make some zips genuinely predictive
    predictive_zips = [str(i) for i in range(0, 10)]
    mask = df["zip_code"].isin(predictive_zips)
    df.loc[mask, "target"] = (rng.uniform(size=mask.sum()) > 0.3).astype(int)

    df["zip_encoded"]     = target_encode(df, "zip_code", "target", smoothing=10)
    target_means          = df.groupby("zip_code")["target"].mean()
    df["zip_raw_mean"]    = df["zip_code"].map(target_means)

    print(f"  Raw target mean for '0': {target_means.get('0', 'N/A'):.3f}")
    print(f"  Smoothed encoding for  '0': {df[df.zip_code=='0']['zip_encoded'].iloc[0]:.3f}")
    print("  Smoothing pulls rare-category estimates toward the global mean,")
    print("  reducing overfit when a category has few samples.")


# ─────────────────────────────────────────────────────────────────────────────
# 5. POLYNOMIAL AND INTERACTION FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def demo_polynomial():
    """Show when polynomial features help and the feature explosion problem."""
    print("\n── 5. Polynomial & Interaction Features ────────────")
    X, y = make_classification(n_samples=300, n_features=5, n_informative=3,
                               random_state=RANDOM_SEED)

    for degree in [1, 2, 3]:
        poly    = PolynomialFeatures(degree=degree, include_bias=False,
                                      interaction_only=False)
        X_poly  = poly.fit_transform(X)
        model   = LogisticRegression(max_iter=2000, random_state=RANDOM_SEED)
        cv_acc  = cross_val_score(model, X_poly, y, cv=5).mean()
        print(f"  Degree {degree}: {X_poly.shape[1]:>4} features  CV acc={cv_acc:.4f}")

    print("\n  Feature count grows as O(d^k) — 5 features → 21 (deg-2) → 56 (deg-3).")
    print("  With 100 original features, degree-2 gives 5,151 features.")
    print("  Use with L1/L2 regularisation or feature selection to prevent overfit.")


def main():
    print("=" * 54)
    print("MODULE 15 — Feature Selection")
    print("=" * 54)
    X, y = make_data_with_noise()
    demo_filter_methods(X, y)
    demo_rfe(X, y)
    demo_embedded(X, y)
    demo_target_encoding()
    demo_polynomial()
    print("\nDone.")


if __name__ == "__main__":
    main()
