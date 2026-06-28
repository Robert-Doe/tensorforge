# Module 10 — DECISIONS.md

## Decision 1: Use sklearn instead of re-implementing
**Decision:** Switch from scratch implementations to scikit-learn for M10 onward.
**Why:** The scratch modules (M04-M09) established understanding of the math. sklearn provides battle-tested, optimised versions of the same algorithms. Real ML work uses sklearn, not hand-rolled code.
**Trade-off:** Less control; black-box risk for beginners who skipped M04-M09. That's why we built scratch versions first.

## Decision 2: Keep the same data pipeline (data_loader.py)
**Decision:** Reuse `data_loader.py` unchanged from M03.
**Why:** Consistency across modules lets students focus on what changed (the model API) rather than the data loading boilerplate.
**Trade-off:** Tighter coupling between modules; any change to data_loader.py ripples forward.

## Decision 3: Use stratify=y in train_test_split
**Decision:** Pass `stratify=y` to sklearn's `train_test_split`.
**Why:** Our dataset has ~60/40 class imbalance. Without stratification, random splits can accidentally create a 70/30 or 50/50 split, making train/test comparison unfair.
**Trade-off:** Slightly more complex call signature; not available in our hand-rolled splitter from M04.

## Decision 4: Report all four metrics (P, R, F1, AUC) in one table
**Decision:** Print precision, recall, F1, and AUC for every model side-by-side.
**Why:** Forces comparison of models on multiple axes simultaneously, reinforcing M09's lesson that accuracy alone is insufficient.
**Trade-off:** Output is wider and harder to scan; acceptable given pedagogical goal.
