"""module_10/sklearn_advanced.py — Advanced Scikit-learn Patterns.

Covers:
  - Custom transformers (BaseEstimator + TransformerMixin)
  - FeatureUnion (apply multiple transforms in parallel, concatenate results)
  - Pipeline caching (Memory) — avoid re-computing expensive steps
  - set_output API (pandas-aware transformers)
  - Custom cross-validation strategies

Run standalone: python sklearn_advanced.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import tempfile
import time
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.joblib import Memory

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# ─────────────────────────────────────────────────────────────────────────────
# 1. CUSTOM TRANSFORMERS
# ─────────────────────────────────────────────────────────────────────────────

class OutlierClipper(BaseEstimator, TransformerMixin):
    """Clip feature values to the [q_low, q_high] percentile range.

    Why custom transformer?
    - sklearn's standard scalers don't have percentile clipping built in.
    - By implementing BaseEstimator + TransformerMixin, this works in Pipelines,
      participates in clone(), get_params(), set_params(), and GridSearchCV.

    Args:
        q_low:  lower percentile to clip at (0-50)
        q_high: upper percentile to clip at (50-100)
    """

    def __init__(self, q_low: float = 1.0, q_high: float = 99.0):
        self.q_low  = q_low
        self.q_high = q_high

    def fit(self, X: np.ndarray, y=None):
        """Learn clip boundaries from training data."""
        self.low_  = np.percentile(X, self.q_low,  axis=0)
        self.high_ = np.percentile(X, self.q_high, axis=0)
        return self   # always return self from fit

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Clip values to [low_, high_] learned at fit time."""
        return np.clip(X, self.low_, self.high_)


class LogTransformer(BaseEstimator, TransformerMixin):
    """Apply log1p to features with positive skew.

    log1p(x) = log(1 + x) — handles zero values without NaN.
    Commonly used for count features (click counts, purchase quantities).

    Args:
        base: logarithm base ('e' for natural log, '10', '2')
    """

    def __init__(self, base: str = "e"):
        self.base = base

    def fit(self, X, y=None):
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.maximum(X, 0)   # avoid log of negative (shouldn't happen but safety)
        if self.base == "10":
            return np.log10(X + 1)
        elif self.base == "2":
            return np.log2(X + 1)
        else:
            return np.log1p(X)


class InteractionFeatures(BaseEstimator, TransformerMixin):
    """Create pairwise interaction terms: xi * xj for all feature pairs.

    Why not PolynomialFeatures(degree=2)?
    This version creates ONLY cross-terms (xi*xj, i≠j), no squared terms.
    Useful when squared features are meaningless but interactions matter.
    """

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[1]
        interactions = []
        for i in range(n):
            for j in range(i + 1, n):
                interactions.append((X[:, i] * X[:, j]).reshape(-1, 1))
        return np.hstack(interactions) if interactions else np.empty((X.shape[0], 0))

    def get_feature_names_out(self, input_features=None):
        n = self.n_features_in_
        if input_features is None:
            input_features = [f"x{i}" for i in range(n)]
        return [f"{input_features[i]}*{input_features[j]}"
                for i in range(n) for j in range(i+1, n)]


def demo_custom_transformers():
    """Show custom transformers plugging into a Pipeline."""
    print("── 1. Custom Transformers ──────────────────────────")
    X, y = make_classification(n_samples=500, n_features=8, random_state=RANDOM_SEED)
    # Add artificial skew and outliers
    X[:, :4] = np.exp(X[:, :4]) * 5   # make positively skewed
    X[0, 0]  = 1000.0                  # outlier

    pipe = Pipeline([
        ("clip",    OutlierClipper(q_low=1, q_high=99)),
        ("log",     LogTransformer(base="e")),
        ("scale",   StandardScaler()),
        ("clf",     LogisticRegression(max_iter=500, random_state=RANDOM_SEED)),
    ])
    scores = cross_val_score(pipe, X, y, cv=5, scoring="accuracy")
    print(f"  Custom pipeline CV accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

    # Verify get_params works (required for GridSearchCV)
    params = pipe.get_params()
    print(f"  Accessible params (sample): clip__q_low={params['clip__q_low']}, "
          f"log__base={params['log__base']}")
    print("  → All custom params accessible via 'step__param' notation in GridSearchCV.")

    # Verify InteractionFeatures
    X_small = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    inter = InteractionFeatures().fit(X_small)
    X_int = inter.transform(X_small)
    print(f"\n  InteractionFeatures on (n=2, p=3) input:")
    print(f"  Input:  {X_small}")
    print(f"  Output: {X_int}   (feature names: {inter.get_feature_names_out()})")


# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE UNION — parallel transforms
# ─────────────────────────────────────────────────────────────────────────────

def demo_feature_union():
    """Apply PCA and polynomial features in parallel, then concatenate."""
    print("\n── 2. FeatureUnion (parallel branches) ─────────────")
    X, y = make_classification(n_samples=400, n_features=10, random_state=RANDOM_SEED)

    # Branch 1: PCA on original features (captures global structure)
    # Branch 2: Polynomial degree-2 features (captures non-linear interactions)
    # Concatenate both branches, then feed to classifier

    union = FeatureUnion([
        ("pca",   PCA(n_components=5)),
        ("poly",  PolynomialFeatures(degree=2, include_bias=False)),
    ])

    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("union", union),
        ("clf",   LogisticRegression(max_iter=500, random_state=RANDOM_SEED)),
    ])

    scores = cross_val_score(pipe, X, y, cv=5, scoring="accuracy")
    n_pca_out  = 5
    n_poly_out = PolynomialFeatures(degree=2, include_bias=False).fit(X).n_output_features_

    print(f"  PCA branch output: {n_pca_out} features")
    print(f"  Poly branch output: {n_poly_out} features")
    print(f"  Total features after union: {n_pca_out + n_poly_out}")
    print(f"  CV accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
    print("  FeatureUnion is the parallel counterpart to Pipeline's serial steps.")


# ─────────────────────────────────────────────────────────────────────────────
# 3. PIPELINE CACHING — avoid re-computing expensive transforms
# ─────────────────────────────────────────────────────────────────────────────

def demo_pipeline_caching():
    """Show how Memory caches fitted transformers to avoid redundant recomputation."""
    print("\n── 3. Pipeline Caching (Memory) ─────────────────────")
    X, y = make_classification(n_samples=300, n_features=10, random_state=RANDOM_SEED)

    # Use a temporary directory for the cache
    cache_dir = tempfile.mkdtemp()
    memory    = Memory(location=cache_dir, verbose=0)

    # Pipeline WITH caching
    pipe_cached = Pipeline([
        ("scale", StandardScaler()),
        ("pca",   PCA(n_components=5)),
        ("clf",   LogisticRegression(max_iter=500)),
    ], memory=memory)

    t0 = time.perf_counter()
    pipe_cached.fit(X, y)
    t1 = time.perf_counter()
    first_fit_ms = (t1 - t0) * 1000

    t0 = time.perf_counter()
    pipe_cached.fit(X, y)   # re-fit: cache hit for scale+pca
    t1 = time.perf_counter()
    second_fit_ms = (t1 - t0) * 1000

    print(f"  First  fit:   {first_fit_ms:.1f} ms")
    print(f"  Second fit:   {second_fit_ms:.1f} ms  (cached transforms)")
    print(f"  The caching benefit is larger for slow transforms (e.g. TFIDF, PCA on big data).")
    print(f"  Cache stored in: {cache_dir}")
    print("  Use case: GridSearchCV with many C/gamma values — shared transforms are cached.")


# ─────────────────────────────────────────────────────────────────────────────
# 4. PANDAS-AWARE PIPELINES (set_output API)
# ─────────────────────────────────────────────────────────────────────────────

def demo_set_output():
    """Show that transformers can be configured to output DataFrames."""
    print("\n── 4. Pandas-Aware Pipelines (set_output) ───────────")
    X_df = pd.DataFrame({
        "age":    [25, 35, 45, 55],
        "income": [40000, 60000, 80000, 100000],
        "score":  [0.3, 0.6, 0.7, 0.9],
    })

    # set_output(transform="pandas") → transform() returns a DataFrame, not ndarray
    scaler = StandardScaler().set_output(transform="pandas")
    X_scaled = scaler.fit_transform(X_df)

    print(f"  Input type:  {type(X_df).__name__}")
    print(f"  Output type: {type(X_scaled).__name__}")
    print(f"  Columns preserved: {list(X_scaled.columns)}")
    print(f"\n  Scaled values:")
    print(X_scaled.round(3).to_string(index=False))
    print("\n  Benefit: column names flow through the pipeline — easier debugging,")
    print("  feature importance mapping, and DataFrame-aware downstream processing.")


# ─────────────────────────────────────────────────────────────────────────────
# 5. CUSTOM CROSS-VALIDATION SPLITTERS
# ─────────────────────────────────────────────────────────────────────────────

class TimeSeriesSplitByDate(BaseEstimator):
    """Custom cross-validator: walk-forward (expanding window) time-series split.

    Unlike sklearn's TimeSeriesSplit, this splits by date boundary rather than
    equal-sized folds, which better mirrors real deployment conditions.

    Args:
        n_splits: number of train/test folds
        test_size: fraction of data in each test window
    """

    def __init__(self, n_splits: int = 5, test_size: float = 0.1):
        self.n_splits  = n_splits
        self.test_size = test_size

    def split(self, X, y=None, groups=None):
        n     = len(X)
        step  = int(n * self.test_size)
        start = n - self.n_splits * step

        for i in range(self.n_splits):
            train_end = start + i * step
            test_end  = train_end + step
            if test_end > n:
                break
            yield (np.arange(train_end),
                   np.arange(train_end, test_end))

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


def demo_custom_splitter():
    """Show walk-forward cross-validation split boundaries."""
    print("\n── 5. Custom Cross-Validation Splitter ─────────────")
    n = 100
    X = np.arange(n).reshape(-1, 1)
    y = np.random.randint(0, 2, n)

    splitter = TimeSeriesSplitByDate(n_splits=5, test_size=0.1)
    print(f"  Walk-forward splits (n={n}, test_size=10%):")
    print(f"  {'Fold':>5} {'Train range':>20} {'Test range':>15}")
    for fold, (tr, te) in enumerate(splitter.split(X), start=1):
        print(f"  {fold:>5} [{tr[0]:>3},{tr[-1]:>3}] ({len(tr):>3} samples)   "
              f"[{te[0]:>3},{te[-1]:>3}]")

    # Plug into cross_val_score
    from sklearn.linear_model import Ridge
    X_reg, y_reg = make_classification(n_samples=n, n_features=5, random_state=RANDOM_SEED)
    scores = cross_val_score(
        Ridge(), X_reg, y_reg,
        cv=splitter, scoring="accuracy"
    )
    print(f"\n  Ridge CV accuracy (walk-forward): {scores.mean():.4f} ± {scores.std():.4f}")
    print("  Custom splitters plug directly into cross_val_score, GridSearchCV, etc.")


def main():
    print("=" * 54)
    print("MODULE 10 — Advanced Scikit-learn Patterns")
    print("=" * 54)
    demo_custom_transformers()
    demo_feature_union()
    demo_pipeline_caching()
    demo_set_output()
    demo_custom_splitter()
    print("\nDone.")


if __name__ == "__main__":
    main()
