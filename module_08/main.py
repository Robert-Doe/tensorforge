"""module_08/main.py — Naive Bayes: The Probabilistic Detective.

Sherlock Holmes updated his beliefs about a suspect with each new clue.
Naive Bayes does the same mathematically: start with a prior probability,
then update it with each observed feature (evidence) using Bayes' theorem.

Run: python main.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler

RANDOM_SEED = 42
PLOTS_DIR   = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)
rng = np.random.default_rng(RANDOM_SEED)

print("=" * 56)
print("MODULE 08 — Naive Bayes: The Probabilistic Detective")
print("=" * 56)


# ── 1. BAYES' THEOREM ────────────────────────────────────────
print("\n── 1. Bayes' Theorem (the math) ────────────────────────")
print("""
  Bayes' theorem:
    P(class | features) = P(features | class) * P(class)
                          ─────────────────────────────────
                                   P(features)

  In words for fraud detection:
    P(fraud | overseas, large_amount)
        ∝ P(overseas | fraud) * P(large_amount | fraud) * P(fraud)

  The "naive" assumption:
    P(feature_1, feature_2 | class) ≈ P(feature_1 | class) * P(feature_2 | class)
    (assume features are INDEPENDENT given the class — often false, usually OK)

  Three variants based on the assumed distribution of P(feature | class):
    Gaussian NB:     feature ~ Normal(μ_c, σ_c)  — for continuous features
    Multinomial NB:  feature ~ count data          — for word counts, TF-IDF
    Bernoulli NB:    feature ~ Bernoulli(p_c)     — for binary features
""")


# ── 2. GAUSSIAN NAIVE BAYES ──────────────────────────────────
print("── 2. Gaussian Naive Bayes ─────────────────────────────")
print("(assumes features are normally distributed within each class)")

X, y = make_classification(
    n_samples=600, n_features=8, n_informative=5,
    n_redundant=2, n_classes=2, random_state=RANDOM_SEED)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED)

gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)

print(f"  Train size: {X_train.shape[0]}   Test size: {X_test.shape[0]}")
print(f"  Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Show learned per-class means and stds for first 3 features
print("\n  Learned per-class Gaussian parameters (first 3 features):")
print(f"  {'Feature':<12} {'Class-0 Mean':>14} {'Class-0 Std':>13} "
      f"{'Class-1 Mean':>14} {'Class-1 Std':>13}")
for i in range(3):
    print(f"  {'feature_'+str(i):<12} {gnb.theta_[0,i]:>14.4f} "
          f"{np.sqrt(gnb.var_[0,i]):>13.4f} "
          f"{gnb.theta_[1,i]:>14.4f} "
          f"{np.sqrt(gnb.var_[1,i]):>13.4f}")

print("\n  Class priors (log):")
print(f"  Class 0: P = {np.exp(gnb.class_log_prior_[0]):.4f}")
print(f"  Class 1: P = {np.exp(gnb.class_log_prior_[1]):.4f}")

print("\nFull classification report:")
print(classification_report(y_test, y_pred, target_names=["Class-0","Class-1"]))


# ── 3. INCREMENTAL / ONLINE LEARNING ─────────────────────────
print("── 3. Online Learning with partial_fit ──────────────────")
print("""
  GaussianNB supports partial_fit() — update the model with new batches
  WITHOUT retraining from scratch. Useful for:
    • Streaming data (new cases arriving daily)
    • Data too large to fit in RAM (process in chunks)
    • Concept drift adaptation (update as patterns change)
""")
gnb_online = GaussianNB()
# Train in 4 mini-batches of 100
batch_size = 100
classes    = np.array([0, 1])
for start in range(0, X_train.shape[0], batch_size):
    end = min(start + batch_size, X_train.shape[0])
    gnb_online.partial_fit(X_train[start:end], y_train[start:end], classes=classes)

acc_online = accuracy_score(y_test, gnb_online.predict(X_test))
acc_full   = accuracy_score(y_test, gnb.predict(X_test))
print(f"  Full fit accuracy:   {acc_full:.4f}")
print(f"  Online (4 batches):  {acc_online:.4f}")
print("  (Should match — same data, same model, different training order)")


# ── 4. MULTINOMIAL NAIVE BAYES (word count features) ─────────
print("\n── 4. Multinomial Naive Bayes (text-style features) ─────")
print("(for non-negative integer features like word counts)")

# Create synthetic count-like features via MinMaxScaler + rounding
X_mnb = np.abs(X * 10).astype(int)   # make counts non-negative
X_tr_m, X_te_m, y_tr_m, y_te_m = train_test_split(
    X_mnb, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED)

mnb = MultinomialNB(alpha=1.0)
mnb.fit(X_tr_m, y_tr_m)
acc_mnb = accuracy_score(y_te_m, mnb.predict(X_te_m))
print(f"  MultinomialNB accuracy: {acc_mnb:.4f}  (alpha=1.0 = Laplace smoothing)")


# ── 5. PROBABILITY CALIBRATION ───────────────────────────────
print("\n── 5. Probability Outputs ───────────────────────────────")
sample_probs = gnb.predict_proba(X_test[:5])
print("  First 5 test samples — predicted probabilities:")
print(f"  {'Sample':>8} {'P(Class-0)':>12} {'P(Class-1)':>12} {'True Label':>12}")
for i, (probs, true) in enumerate(zip(sample_probs, y_test[:5])):
    print(f"  {i:>8} {probs[0]:>12.4f} {probs[1]:>12.4f} {true:>12}")

print("""
  NB probabilities are often miscalibrated (too extreme or too uniform)
  due to the independence assumption. For well-calibrated probabilities,
  use CalibratedClassifierCV (see logistic_advanced.py).
  For RANKING tasks (AUC), miscalibration is usually acceptable.
""")


# ── 6. COMPARISON TABLE ──────────────────────────────────────
print("── 6. NB Variant Comparison ────────────────────────────")
print(f"""
  {'Variant':<18} {'Feature type':<25} {'Assumption':<30}
  {'-'*75}
  {'GaussianNB':<18} {'Continuous (real-valued)':<25} {'P(xi|c) ~ N(μ,σ)':<30}
  {'MultinomialNB':<18} {'Counts (non-negative int)':<25} {'P(xi|c) ~ Multinomial':<30}
  {'BernoulliNB':<18} {'Binary (0/1)':<25} {'P(xi|c) ~ Bernoulli':<30}
""")


# ── 7. PLOTS ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), facecolor="#0d1117")
for ax in axes:
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="#8b949e")
    ax.spines[:].set_color("#30363d")

# Probability score distribution
y_prob = gnb.predict_proba(X_test)[:, 1]
axes[0].hist(y_prob[y_test==0], bins=20, alpha=0.7, color="#58a6ff",
             label="Class-0", density=True)
axes[0].hist(y_prob[y_test==1], bins=20, alpha=0.7, color="#f85149",
             label="Class-1", density=True)
axes[0].axvline(0.5, color="#3fb950", linestyle="--", linewidth=1.5)
axes[0].set_xlabel("P(Class-1)", color="#e6edf3")
axes[0].set_title("Naive Bayes Score Distribution", color="#e6edf3")
axes[0].legend(facecolor="#0d1117", labelcolor="#e6edf3")

# Cross-val accuracy: GNB vs other classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

clfs = {
    "Gaussian NB":    GaussianNB(),
    "Logistic Reg":   LogisticRegression(max_iter=500),
    "Decision Tree":  DecisionTreeClassifier(max_depth=5, random_state=RANDOM_SEED),
    "KNN (k=7)":      KNeighborsClassifier(n_neighbors=7),
}
names, means, stds = [], [], []
for name, clf in clfs.items():
    cv = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    names.append(name)
    means.append(cv.mean())
    stds.append(cv.std())

x_pos = np.arange(len(names))
colours = ["#bc8cff","#58a6ff","#3fb950","#f0883e"]
axes[1].barh(x_pos, means, xerr=stds, color=colours, edgecolor="#0d1117",
             height=0.6, error_kw=dict(ecolor="#e6edf3", linewidth=1.5))
axes[1].set_yticks(x_pos)
axes[1].set_yticklabels(names, color="#e6edf3")
axes[1].set_xlabel("5-Fold CV Accuracy", color="#e6edf3")
axes[1].set_title("Classifier Comparison", color="#e6edf3")
axes[1].set_xlim(0.5, 1.0)

plt.suptitle("Naive Bayes: Probability Scores & Comparison", color="#e6edf3", fontsize=13)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/module08_nb.png", dpi=120, facecolor=fig.get_facecolor())
plt.close()
print(f"\nPlot saved → {PLOTS_DIR}/module08_nb.png")

print("\n✓ Module 08 complete. Your probabilistic detective is on duty.")
