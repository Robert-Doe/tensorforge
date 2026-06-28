"""module_07/main.py — Decision Trees: Following the Clues.

A decision tree is a flowchart of questions. At each node the detective
asks one question ("Is the amount > $500?"). The answer sends them
left or right, until they reach a leaf — the verdict.

Run: python main.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.inspection import permutation_importance

RANDOM_SEED = 42
PLOTS_DIR   = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)
rng = np.random.default_rng(RANDOM_SEED)

print("=" * 56)
print("MODULE 07 — Decision Trees: Following the Clues")
print("=" * 56)


# ── 1. DATASET ───────────────────────────────────────────────
print("\n── 1. Dataset: Suspect Classification ──────────────────")
X, y = make_classification(
    n_samples=500, n_features=8, n_informative=5,
    n_redundant=2, n_classes=2, random_state=RANDOM_SEED)
feature_names = [f"feature_{i}" for i in range(X.shape[1])]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED)
print(f"Train: {X_train.shape[0]}  Test: {X_test.shape[0]}  Features: {X_train.shape[1]}")


# ── 2. GINI IMPURITY INTUITION ───────────────────────────────
print("\n── 2. Gini Impurity: How CART Chooses Splits ───────────")
print("""
  Gini impurity = 1 - Σ p_c²
  (probability that a random sample would be misclassified if randomly labelled)

  Pure node  (all one class):  Gini = 1 - (1² + 0²) = 0.0   ← best
  50-50 node (equal mix):      Gini = 1 - (0.5² + 0.5²) = 0.5  ← worst
""")
for split_name, p_pos in [("Pure (all positive)", 1.0),
                            ("90-10 split",         0.9),
                            ("70-30 split",         0.7),
                            ("50-50 split",         0.5)]:
    gini = 1 - (p_pos**2 + (1-p_pos)**2)
    print(f"  {split_name:<26}: Gini = {gini:.4f}")


# ── 3. DEPTH AND OVERFITTING ─────────────────────────────────
print("\n── 3. Depth vs Overfitting ─────────────────────────────")
print(f"  {'max_depth':>10} {'Train Acc':>12} {'Test Acc':>11} {'n_leaves':>10}")
for depth in [1, 2, 3, 5, 7, None]:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_SEED)
    dt.fit(X_train, y_train)
    tr_acc = dt.score(X_train, y_train)
    te_acc = dt.score(X_test,  y_test)
    leaves = dt.get_n_leaves()
    depth_str = str(depth) if depth else "None"
    print(f"  {depth_str:>10} {tr_acc:>12.4f} {te_acc:>11.4f} {leaves:>10}")


# ── 4. CHOOSE DEPTH VIA CROSS-VALIDATION ─────────────────────
print("\n── 4. Choosing max_depth via Cross-Validation ───────────")
depths    = list(range(1, 12))
cv_scores = []
for d in depths:
    dt = DecisionTreeClassifier(max_depth=d, random_state=RANDOM_SEED)
    sc = cross_val_score(dt, X_train, y_train, cv=5, scoring="accuracy")
    cv_scores.append(sc.mean())
best_depth = depths[np.argmax(cv_scores)]
print(f"  Best max_depth = {best_depth}  (5-fold CV accuracy = {max(cv_scores):.4f})")


# ── 5. FINAL MODEL ───────────────────────────────────────────
print("\n── 5. Final Model Evaluation ───────────────────────────")
final_tree = DecisionTreeClassifier(max_depth=best_depth, random_state=RANDOM_SEED)
final_tree.fit(X_train, y_train)
y_pred = final_tree.predict(X_test)

print(f"Test accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Tree depth: {final_tree.get_depth()}   Leaves: {final_tree.get_n_leaves()}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Class-0","Class-1"]))


# ── 6. FEATURE IMPORTANCE ────────────────────────────────────
print("── 6. Feature Importance ───────────────────────────────")
print("(Gini-based importance = weighted reduction in impurity)")
importances = final_tree.feature_importances_
ranked = sorted(zip(feature_names, importances), key=lambda x: -x[1])
for name, imp in ranked:
    bar = "█" * int(imp * 40)
    print(f"  {name:<15}: {imp:.4f}  {bar}")


# ── 7. PRINT TREE STRUCTURE ──────────────────────────────────
print("\n── 7. Tree Structure (text) ─────────────────────────────")
tree_text = export_text(final_tree, feature_names=feature_names, max_depth=3)
print(tree_text)


# ── 8. PLOTS ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0d1117")

# CV curve
ax = axes[0]
ax.set_facecolor("#161b22")
ax.plot(depths, cv_scores, color="#58a6ff", linewidth=2, marker="o", markersize=5)
ax.axvline(best_depth, color="#3fb950", linestyle="--", linewidth=1.5,
           label=f"Best depth={best_depth}")
ax.set_xlabel("max_depth",     color="#e6edf3")
ax.set_ylabel("CV Accuracy",   color="#e6edf3")
ax.set_title("Depth vs CV Accuracy", color="#e6edf3")
ax.tick_params(colors="#8b949e")
ax.spines[:].set_color("#30363d")
ax.legend(facecolor="#0d1117", labelcolor="#e6edf3")

# Feature importances
ax = axes[1]
ax.set_facecolor("#161b22")
names, imps = zip(*ranked)
colours = ["#58a6ff" if i == 0 else "#3fb950" if i == 1 else "#8b949e"
           for i in range(len(names))]
ax.barh(names[::-1], imps[::-1], color=colours[::-1], edgecolor="#0d1117")
ax.set_xlabel("Importance (Gini)", color="#e6edf3")
ax.set_title("Feature Importances",   color="#e6edf3")
ax.tick_params(colors="#8b949e")
ax.spines[:].set_color("#30363d")

plt.suptitle("Decision Tree: Depth Selection & Feature Importance",
             color="#e6edf3", fontsize=13)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/module07_dtree.png", dpi=120, facecolor=fig.get_facecolor())
plt.close()

# Also save a visual of the tree itself
fig_tree, ax_tree = plt.subplots(figsize=(14, 6), facecolor="#0d1117")
ax_tree.set_facecolor("#0d1117")
plot_tree(final_tree, feature_names=feature_names,
          class_names=["Class-0","Class-1"], filled=True,
          impurity=True, max_depth=3, ax=ax_tree,
          fontsize=7)
plt.savefig(f"{PLOTS_DIR}/module07_tree_visual.png", dpi=120,
            facecolor=fig_tree.get_facecolor(), bbox_inches="tight")
plt.close()
print(f"\nPlots saved → {PLOTS_DIR}/module07_*.png")

print("""
── 9. Strengths & Weaknesses ──────────────────────────────
  STRENGTHS:
  ✓ Fully interpretable — you can explain every prediction
  ✓ Handles both numerical and categorical features
  ✓ No feature scaling needed
  ✓ Captures non-linear relationships and interactions

  WEAKNESSES:
  ✗ High variance — small data changes produce very different trees
  ✗ Prone to overfitting without depth control or pruning
  ✗ Biased towards features with more possible split points
  → Solution: Random Forest / Gradient Boosting (Modules 12-13)
""")
print("✓ Module 07 complete. Your decision tree is ready to interrogate data.")
