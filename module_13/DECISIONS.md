# Module 13 — DECISIONS.md

## Decision 1: Random initialisation, not k-means++
**Decision:** Use random centroid initialisation (pick k random rows from X).
**Why:** k-means++ is strictly better (smarter spread-out initialisation, fewer iterations, lower chance of poor local minima). But understanding random init first makes the algorithm clearer — you can see it sometimes converge to suboptimal clusters, which motivates why k-means++ was invented. Module notes this as a known limitation.
**Trade-off:** Risk of poor local minima on unlucky random seeds. Mitigated by fixed RANDOM_SEED=42 which happens to work well on our dataset.

## Decision 2: Compute inertia in a Python loop, not vectorised
**Decision:** The `inertia()` function loops over samples explicitly.
**Why:** Clarity over performance. The vectorised equivalent (`np.sum((X - centroids[labels])**2)`) is compact but requires numpy fancy indexing that beginners haven't seen yet. The loop version makes the "sum of squared distances" formula directly visible.
**Trade-off:** Slow on large datasets. For production use sklearn's KMeans which is fully vectorised and runs in C.

## Decision 3: Compare clusters to true labels via purity, not accuracy
**Decision:** Use "cluster purity" rather than labelling clusters and computing accuracy.
**Why:** k-Means is unsupervised — it produces cluster indices (0, 1, 2) with no inherent meaning. Assigning cluster 0="solved" is arbitrary. Purity sidesteps this by asking "what fraction of each cluster's members share the same true label?" regardless of which label that is.
**Trade-off:** Purity can't distinguish between a model that found two solved-clusters and one unsolved-cluster vs. one cluster per class. More nuanced measures (NMI, ARI) exist but are beyond this module's scope.

## Decision 4: Plot clusters on first two features only
**Decision:** The 2D scatter plot projects 6-dimensional data onto features 0 and 1.
**Why:** 6D data can't be plotted directly. PCA-based projection would be more principled (that's Module 14), but at this point PCA hasn't been introduced. First-two-features projection is honest about being a simplification and motivates why PCA exists.
**Trade-off:** Clusters may look messier than they truly are in 6D space, since we discard 4 dimensions of information.
