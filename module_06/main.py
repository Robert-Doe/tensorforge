"""module_06/main.py — K-Nearest Neighbours: Guilt by Association.

The oldest detective trick: look at a suspect's associates.
KNN does the same with data — classify a new point by what class
its K nearest training points belong to.

Run: python main.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

RANDOM_SEED = 42
PLOTS_DIR   = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)
rng = np.random.default_rng(RANDOM_SEED)

print("=" * 56)
print("MODULE 06 — K-Nearest Neighbours: Guilt by Association")
print("=" * 56)


# ── 1. DATASET ───────────────────────────────────────────────
print("\n── 1. Dataset: Suspect Classification ──────────────────")
X, y = make_classification(
    n_samples=600, n_features=6, n_informative=4,
    n_redundant=2, n_classes=2, random_state=RANDOM_SEED)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=RANDOM_SEED)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print(f"Train: {X_train.shape[0]}  Test: {X_test.shape[0]}")
print(f"Class balance: {np.bincount(y_train)}")
print("Note: KNN is VERY sensitive to feature scales — always StandardScale first.")


# ── 2. KNN INTUITION ─────────────────────────────────────────
print("\n── 2. KNN Intuition ────────────────────────────────────")
print("""
  How KNN works:
  1. Store ALL training examples (no "training" computation)
  2. For a new point x:
     a. Compute distance to every training point
     b. Select the K closest neighbours
     c. Majority vote → predicted class
        (or mean → for regression)

  Key decisions:
  • K: number of neighbours
    - K=1: very flexible, memorises training data (high variance)
    - K=large: very smooth boundary, may miss patterns (high bias)
  • Distance metric: Euclidean (default), Manhattan, Cosine, ...
  • Weighting: uniform (all K vote equally) or distance-weighted
""")


# ── 3. CHOOSING K ────────────────────────────────────────────
print("── 3. Choosing K via Cross-Validation ──────────────────")
ks          = list(range(1, 31, 2))   # odd Ks to avoid ties
cv_scores   = []
train_scores = []

for k in ks:
    knn = KNeighborsClassifier(n_neighbors=k, weights="distance")
    cv  = cross_val_score(knn, X_train_s, y_train, cv=5, scoring="accuracy")
    cv_scores.append(cv.mean())
    knn.fit(X_train_s, y_train)
    train_scores.append(knn.score(X_train_s, y_train))

best_k   = ks[np.argmax(cv_scores)]
best_acc = max(cv_scores)
print(f"\n  {'K':>5} {'Train Acc':>12} {'CV Acc':>10}")
for k, tr, cv in zip(ks, train_scores, cv_scores):
    marker = " ← best" if k == best_k else ""
    print(f"  {k:>5} {tr:>12.4f} {cv:>10.4f}{marker}")
print(f"\nBest K = {best_k}  (CV accuracy = {best_acc:.4f})")


# ── 4. TRAIN AND EVALUATE BEST MODEL ─────────────────────────
print("\n── 4. Final Model Evaluation ───────────────────────────")
best_knn = KNeighborsClassifier(n_neighbors=best_k, weights="distance")
best_knn.fit(X_train_s, y_train)
y_pred = best_knn.predict(X_test_s)

print(f"Test accuracy (K={best_k}): {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Class-0","Class-1"]))


# ── 5. EFFECT OF K ON DECISION BOUNDARY (2D demo) ────────────
print("── 5. Decision Boundary Visualisation (2D) ─────────────")
# Use first 2 features only for visualisation
X2 = X_train_s[:, :2]
y2 = y_train

h = 0.05
x_min, x_max = X2[:,0].min()-1, X2[:,0].max()+1
y_min, y_max = X2[:,1].min()-1, X2[:,1].max()+1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), facecolor="#0d1117")
for ax, k in zip(axes, [1, best_k, 21]):
    knn_vis = KNeighborsClassifier(n_neighbors=k, weights="distance").fit(X2, y2)
    Z = knn_vis.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.set_facecolor("#161b22")
    ax.contourf(xx, yy, Z, alpha=0.35, cmap="RdBu")
    ax.scatter(X2[y2==0,0], X2[y2==0,1], c="#58a6ff", s=12, edgecolors="none", alpha=0.6)
    ax.scatter(X2[y2==1,0], X2[y2==1,1], c="#f85149", s=12, edgecolors="none", alpha=0.6)
    tr_acc = knn_vis.score(X2, y2)
    ax.set_title(f"K={k}  (train={tr_acc:.2f})", color="#e6edf3")
    ax.tick_params(colors="#8b949e")
    ax.spines[:].set_color("#30363d")

plt.suptitle("KNN Decision Boundaries: Underfitting → Optimal → Overfitting",
             color="#e6edf3", fontsize=12)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/module06_knn_boundary.png", dpi=120,
            facecolor=fig.get_facecolor())
plt.close()

# K vs accuracy curve
fig, ax = plt.subplots(figsize=(8, 4), facecolor="#0d1117")
ax.set_facecolor("#161b22")
ax.plot(ks, train_scores, color="#f85149", linewidth=2, label="Train accuracy")
ax.plot(ks, cv_scores,    color="#58a6ff", linewidth=2, label="CV accuracy")
ax.axvline(best_k, color="#3fb950", linestyle="--", linewidth=1.5, label=f"Best K={best_k}")
ax.set_xlabel("K (neighbours)", color="#e6edf3")
ax.set_ylabel("Accuracy",       color="#e6edf3")
ax.set_title("Choosing K: Bias-Variance Trade-off", color="#e6edf3")
ax.legend(facecolor="#0d1117", labelcolor="#e6edf3")
ax.tick_params(colors="#8b949e")
ax.spines[:].set_color("#30363d")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/module06_knn_k_selection.png", dpi=120,
            facecolor=fig.get_facecolor())
plt.close()
print(f"  Plots saved → {PLOTS_DIR}/module06_*.png")


# ── 6. STRENGTHS & WEAKNESSES SUMMARY ───────────────────────
print("""
── 6. KNN Summary ──────────────────────────────────────────
  STRENGTHS:
  ✓ No training time (lazy learner — stores training data)
  ✓ No assumption about data distribution
  ✓ Naturally handles multi-class
  ✓ Prediction is interpretable ("these are your 5 nearest cases")

  WEAKNESSES:
  ✗ Slow prediction: O(n × d) per query for brute-force
  ✗ Memory: must store all training data
  ✗ Breaks down in high dimensions (curse of dimensionality)
  ✗ Must scale features — very sensitive to different scales
""")

print("✓ Module 06 complete. You now classify by association.")
