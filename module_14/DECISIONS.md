# Module 14 — DECISIONS.md

## Decision 1: Use np.linalg.eigh not np.linalg.eig
**Decision:** Use `eigh` (symmetric eigendecomposition) instead of generic `eig`.
**Why:** The covariance matrix is always real and symmetric (C = C.T). `eigh` exploits this: it's numerically more stable, runs faster, and guarantees real-valued eigenvalues. Generic `eig` may return complex numbers due to floating point noise on symmetric matrices.
**Trade-off:** `eigh` is only valid for symmetric matrices — using it elsewhere would give wrong results without warning.

## Decision 2: Implement fit/transform as separate methods (sklearn convention)
**Decision:** Separate `fit()` from `transform()` rather than a single `fit_transform()` that does both.
**Why:** The sklearn convention of separate fit/transform enables the critical safety property: you fit on training data only, then transform both train and test using the training mean. A single `fit_transform(all_data)` would leak test statistics into the projection.
**Trade-off:** Slightly more verbose API; mitigated by providing `fit_transform()` as a convenience wrapper.

## Decision 3: Print component loadings alongside explained variance
**Decision:** The `print_summary()` method shows both the scree table AND the loading matrix.
**Why:** Explained variance tells you how much each PC captures. Loadings tell you what each PC means — which original features it combines and in which direction. Both are needed to interpret PCA results.
**Trade-off:** More output; verbose for large feature sets. Acceptable here (6 features).

## Decision 4: Cluster in original 6D space, visualise in 2D PCA space
**Decision:** k-Means from M13 runs on the raw 6D data; PCA is applied only for visualisation.
**Why:** Clustering in the full feature space uses all available information. Projecting the cluster labels onto the 2D PCA plot lets you visually inspect whether the clusters make geometric sense — without distorting the cluster boundaries by pre-compressing the data.
**Trade-off:** Cluster boundaries may look non-spherical or overlapping in 2D even if they're well-separated in 6D.
