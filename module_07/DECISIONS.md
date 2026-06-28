# Module 07 — DECISIONS.md

## Decision 1: Gini impurity over entropy as the splitting criterion
**Why:** sklearn's CART implementation defaults to Gini. It's faster (no log
computation) and produces nearly identical results to entropy in practice.
Introducing both here would add complexity without meaningful insight.
**Trade-off:** Students may see entropy in other sources and wonder why we use
Gini. We note the difference in tree_advanced.py with a benchmark.

## Decision 2: Show the depth=None (fully grown) tree explicitly
**Why:** A fully grown tree achieves 100% train accuracy — a dramatic
illustration of overfitting that makes the concept concrete. Without seeing
this extreme case, the motivation for pruning feels theoretical.
**Trade-off:** None. The table row for depth=None is the most important
row pedagogically.

## Decision 3: export_text output for tree inspection
**Why:** `plot_tree` gives a visual, but `export_text` gives a readable
text representation that students can copy, study, and trace by hand.
Understanding the split logic at depth=3 is more educational than staring
at a complex visual.
**Trade-off:** Both are saved. The visual (`plot_tree`) helps with intuition;
the text helps with tracing logic.

## Decision 4: No feature scaling
**Why:** Decision trees split on thresholds (feature_i < value). Whether
temperature is in Celsius (22°C) or some scaled form (0.73), the threshold
adapts. Trees are invariant to monotonic transformations of individual features.
This contrast with KNN (which needs scaling) is an important conceptual point.
**Trade-off:** None — this is a true property of decision trees, not a shortcut.

## Decision 5: Feature importance via Gini gain, not permutation importance
**Why:** Gini-based importance is fast and built into sklearn. It teaches
the connection between tree structure and feature relevance directly.
Permutation importance (more reliable but 10-50× slower) is shown in
evaluation_advanced.py once students understand the concept.
**Trade-off:** Gini importance is biased towards high-cardinality features.
This limitation is stated explicitly in evaluation_advanced.py.
