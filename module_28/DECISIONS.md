# Module 28 — DECISIONS.md

## Decision 1: Train/test split before any other processing step
**Decision:** Call `train_test_split` before imputation, feature engineering, and pipeline building.
**Why:** This is the most critical data hygiene rule in ML: the test set must be invisible during all preprocessing decisions. Imputing missing values on the full dataset before splitting would leak test-set statistics (median of the full data) into the training process — artificially inflating evaluation metrics. Module 15 demonstrated this explicitly; Module 28 shows the correct pattern in a complete workflow.
**Trade-off:** The split is done at the raw DataFrame level, so we pass raw arrays into sklearn Pipelines which apply preprocessing internally on training data only. Slightly more bookkeeping, but correctly implements no-leakage.

## Decision 2: Synthetic dataset with injected missing values
**Decision:** Generate a synthetic "case outcomes" dataset and manually introduce 3% NaN in one column.
**Why:** Module 28 must demonstrate imputation without relying on an external dataset download. The synthetic data is designed with a realistic structure: a known ground-truth logistic function generates labels, making the problem tractable (AUC >0.85 expected). The 3% missing rate is realistic for sensor or survey data.
**Trade-off:** The synthetic data is simpler than real-world data — no class imbalance, no multicollinearity, no distribution shift. Acceptable for a capstone that demonstrates process rather than a hard problem.

## Decision 3: Three-model comparison then grid-search on winner
**Decision:** Run CV on Logistic Regression, Random Forest, and Gradient Boosting; then tune only the winner.
**Why:** Tuning all three candidates would triple the grid-search runtime. Tuning only the CV winner is the standard practice: use cheap CV to shortlist, then invest compute in tuning the best. This mirrors real project workflow and is computationally tractable on CPU.
**Trade-off:** The CV winner might not be the tuning winner — a different model might have more tunable headroom. Accepted for simplicity; the tutorial acknowledges this.

## Decision 4: Save pipeline (not just model) with pickle
**Decision:** Serialize the entire fitted Pipeline (preprocessor + model) rather than just the estimator.
**Why:** A raw model requires the caller to apply the exact same preprocessing before calling predict(). Saving the Pipeline bundles preprocessing and model together — the caller passes raw data identical in format to training data, and the Pipeline handles everything. This is how production ML systems work: the serialized artifact is the full inference graph, not a naked model.
**Trade-off:** Pickle is not version-safe (unpickling a model saved with scikit-learn 1.5.0 in scikit-learn 2.x may fail). For production, joblib or ONNX is preferred. The tutorial points this out explicitly.
