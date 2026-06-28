# Module 06 — DECISIONS.md

## Decision 1: Odd values of K only
**Why:** With binary classification, even K can produce ties (e.g. 3 Class-0
vs 3 Class-1 when K=6). Odd K prevents ties with 2 classes.
**Trade-off:** With >2 classes, odd K doesn't guarantee no ties. Tie-breaking
is handled by sklearn's `weights="distance"` parameter — closer points win.

## Decision 2: `weights="distance"` over `weights="uniform"`
**Why:** Distance-weighted voting gives more influence to closer neighbours —
intuitively correct. A neighbour at distance 0.1 should matter more than one
at distance 10. In practice it often improves accuracy slightly at the cost
of slightly slower prediction.
**Trade-off:** Uniform weighting is more interpretable ("K votes, majority wins")
but distance weighting is almost always better. We use the better default.

## Decision 3: Cross-validation to choose K (not test set)
**Why:** If we chose K by looking at test-set accuracy, we'd be tuning a
hyperparameter to the test set — information leaks. 5-fold CV on the training
set gives an honest estimate without touching the test set.
**Trade-off:** 5-fold CV × 15 K values = 75 model fits. For KNN on small
datasets this is fast. For large datasets, CV cost matters — see Module 10.

## Decision 4: 2D visualisation of decision boundaries for 3 different K values
**Why:** The decision boundary is the most powerful intuition for K's effect.
K=1 produces jagged, memorised boundaries. K=21 produces smooth but potentially
wrong boundaries. Seeing all three at once makes the bias-variance trade-off
physically visible instead of abstract.
**Trade-off:** Only uses 2 of 6 features for the visual — the 2D boundary is
a projection, not the actual 6D boundary. Noted as a limitation in the figure.

## Decision 5: Lazy learning explanation
**Why:** KNN has no "training" phase — it just stores data. This is
fundamentally different from all other algorithms in the curriculum.
Highlighting it here prevents students from thinking all ML models "learn"
in the same way and prepares them for online learning and memory-based methods.
**Trade-off:** None — this is a key conceptual distinction worth spending a
paragraph on.
