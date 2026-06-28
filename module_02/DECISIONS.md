# Module 02 — DECISIONS.md

## Decision 1: Synthetic dataset instead of a CSV file
**Why:** A CSV file adds a dependency on a specific file path. The synthetic
dataset is reproducible (fixed seed), self-documenting, and exercises
the same Pandas operations as real data would.
**Trade-off:** Students need to see CSV I/O eventually. Covered in Module 10
(sklearn pipelines) where `pd.read_csv` is used in the full end-to-end flow.

## Decision 2: Introduce `pd.cut` for bucketing
**Why:** Binning continuous variables (days_open → Fresh/Stale/Cold) is one
of the most common data-preparation steps and teaches the difference between
continuous and categorical thinking — foundational for feature engineering.
**Trade-off:** `qcut` (quantile-based) is arguably more useful but introduces
a second concept. Covered in pandas_advanced.py.

## Decision 3: Show both `loc` and `iloc` side by side
**Why:** Confusing `.loc` (label-based) with `.iloc` (position-based) is the
single most common Pandas beginner error. Showing them on the same row makes
the distinction immediately concrete.
**Trade-off:** Adds a bit of redundancy in the demo, but the confusion cost
is high enough that the repetition is worthwhile.

## Decision 4: `fillna(median)` for missing value demo
**Why:** Median is the standard imputer for skewed numerical data — it's
robust to outliers, unlike mean. Introducing the right default immediately
prevents students from using mean imputation as a habit.
**Trade-off:** More sophisticated imputers (IterativeImputer, KNNImputer)
exist; those appear in Module 28's full pipeline.

## Decision 5: `crosstab` for district × status breakdown
**Why:** Cross-tabulation is both a Pandas primitive and a mental model
(contingency table) that appears in chi-square tests, model evaluation, and
feature analysis. Showing it early builds the right mental habit.
**Trade-off:** `pivot_table` is more flexible; demonstrated in pandas_advanced.py.
