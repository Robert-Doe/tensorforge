# Module 05 — DECISIONS.md

## Decision 1: `stratify=y` in train_test_split
**Why:** With a fraud rate of ~15%, a random split could produce a test set
with 5% or 25% fraud by chance. Stratification forces both splits to have
the same fraud rate as the full dataset, making evaluation reliable.
**Trade-off:** Students must remember to stratify for imbalanced classification.
We highlight it explicitly here to build the habit.

## Decision 2: StandardScaler before LogisticRegression
**Why:** Logistic regression uses gradient descent internally. Features on
different scales (amount: 0-2000, hour: 0-23) cause the loss surface to be
elongated — gradient descent takes many more steps to converge.
Scaling makes all features contribute equally.
**Trade-off:** Must fit scaler on train only, then transform test.
This distinction is emphasised because it's the most common leakage mistake.

## Decision 3: Show confusion matrix with both counts and percentages
**Why:** Raw counts are unintuitive without context (is 10 FN bad for
n=100 or n=10000?). Row-normalised percentages show the detection rate
(recall) and false alarm rate directly. Both views answer different questions.
**Trade-off:** The 2×2 table with two numbers per cell is slightly crowded.
Acceptable because both numbers are essential for fraud use cases.

## Decision 4: Threshold tuning table (6 thresholds)
**Why:** The default threshold of 0.5 is almost never optimal for imbalanced
classification. Showing the precision-recall trade-off across thresholds is
the most important practical takeaway from logistic regression.
**Trade-off:** Adds ~10 lines but prevents the biggest practical mistake:
trusting default threshold on imbalanced data.

## Decision 5: C=1.0 (inverse regularisation strength) explained implicitly
**Why:** The `C` parameter is the inverse of regularisation strength (1/λ).
Larger C = less regularisation. We leave it at the default (C=1) and
defer regularisation-path analysis to logistic_advanced.py where
cross-validation is used to select C properly.
**Trade-off:** Students may wonder what C does. A brief note in the code
comment is enough here.
