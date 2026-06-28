"""module_05/main.py — Logistic Regression: Classifying Suspects.

Linear regression predicts a number. But sometimes we need a YES/NO answer:
Is this case a fraud? Is this suspect likely to reoffend?
Logistic regression gives us a PROBABILITY between 0 and 1.

Run: python main.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, roc_curve,
                              confusion_matrix, classification_report)

RANDOM_SEED = 42
PLOTS_DIR   = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)
rng = np.random.default_rng(RANDOM_SEED)

print("=" * 56)
print("MODULE 05 — Logistic Regression: Classifying Suspects")
print("=" * 56)


# ── 1. DATASET: FRAUD DETECTION ──────────────────────────────
print("\n── 1. Dataset: Fraud Detection ─────────────────────────")
n = 500
amount      = rng.exponential(scale=200, size=n)
hour        = rng.integers(0, 24, n)
n_prev_txn  = rng.integers(1, 50, n)
overseas    = rng.choice([0, 1], n, p=[0.85, 0.15])

# Ground truth: fraud more likely for large/unusual transactions
log_odds = (-3.0
            + 0.005  * amount
            + 0.1    * (hour < 6)       # suspicious late-night hours
            + 0.8    * overseas
            - 0.02   * n_prev_txn)      # more history → less suspicious
prob_fraud = 1 / (1 + np.exp(-log_odds))
fraud = rng.binomial(1, prob_fraud)

df = pd.DataFrame({
    "amount":     amount.round(2),
    "hour":       hour,
    "overseas":   overseas,
    "n_prev_txn": n_prev_txn,
    "fraud":      fraud,
})
print(f"Dataset: {df.shape}  |  Fraud rate: {fraud.mean():.2%}")
print(df.head())


# ── 2. TRAIN / TEST SPLIT ────────────────────────────────────
print("\n── 2. Train / Test Split ───────────────────────────────")
FEATURES = ["amount", "hour", "overseas", "n_prev_txn"]
X = df[FEATURES].values
y = df["fraud"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED)
print(f"Train: {X_train.shape[0]}  Test: {X_test.shape[0]}")
print(f"Train fraud rate: {y_train.mean():.2%}   Test: {y_test.mean():.2%}")
print("(stratify=y ensures both sets have the same fraud proportion)")


# ── 3. FEATURE SCALING ───────────────────────────────────────
print("\n── 3. Feature Scaling ──────────────────────────────────")
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)   # fit only on train
X_test_s  = scaler.transform(X_test)        # transform test with train params

print("StandardScaler: subtracts mean, divides by std for each feature")
print(f"  amount before: mean={X_train[:,0].mean():.1f}  std={X_train[:,0].std():.1f}")
print(f"  amount after:  mean={X_train_s[:,0].mean():.4f}  std={X_train_s[:,0].std():.4f}")


# ── 4. FIT LOGISTIC REGRESSION ───────────────────────────────
print("\n── 4. Fitting Logistic Regression ──────────────────────")
model = LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_SEED)
model.fit(X_train_s, y_train)

print(f"Intercept: {model.intercept_[0]:.4f}")
print("Coefficients (after scaling):")
for name, coef in zip(FEATURES, model.coef_[0]):
    direction = "↑ fraud" if coef > 0 else "↓ fraud"
    print(f"  {name:<15}: {coef:>+.4f}  ({direction})")

print("""
  Logistic regression model:
    log_odds = β₀ + β₁*amount + β₂*hour + ...
    P(fraud) = σ(log_odds) = 1 / (1 + e^(-log_odds))
  σ is the sigmoid function — squashes any real number into [0, 1].
""")


# ── 5. CLASSIFICATION METRICS ────────────────────────────────
print("── 5. Classification Metrics ───────────────────────────")
y_pred       = model.predict(X_test_s)
y_pred_proba = model.predict_proba(X_test_s)[:, 1]

acc  = accuracy_score(y_test,  y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec  = recall_score(y_test,    y_pred, zero_division=0)
f1   = f1_score(y_test,        y_pred, zero_division=0)
auc  = roc_auc_score(y_test,   y_pred_proba)

print(f"  Accuracy:  {acc:.4f}  (fraction correct)")
print(f"  Precision: {prec:.4f} (of predicted fraud: how many were real?)")
print(f"  Recall:    {rec:.4f}  (of real fraud: how many did we catch?)")
print(f"  F1-score:  {f1:.4f}  (harmonic mean of precision & recall)")
print(f"  ROC-AUC:   {auc:.4f}  (area under ROC curve; 1.0 = perfect)")

print("\nFull classification report:")
print(classification_report(y_test, y_pred, target_names=["Legit","Fraud"]))

print("Confusion matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"  [[TN={cm[0,0]}  FP={cm[0,1]}]\n   [FN={cm[1,0]}  TP={cm[1,1]}]]")
print(f"  FP = legitimate flagged as fraud (false alarms)")
print(f"  FN = real fraud missed (most dangerous)")


# ── 6. THRESHOLD TUNING ──────────────────────────────────────
print("\n── 6. Threshold Tuning ─────────────────────────────────")
print("Default threshold = 0.5. In fraud detection, we prefer to catch more fraud")
print("(higher recall) at the cost of more false alarms (lower precision).")
print(f"\n  {'Threshold':>10} {'Precision':>12} {'Recall':>10} {'F1':>8}")
for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
    yp = (y_pred_proba >= t).astype(int)
    p  = precision_score(y_test, yp, zero_division=0)
    r  = recall_score(y_test,    yp, zero_division=0)
    f  = f1_score(y_test,        yp, zero_division=0)
    print(f"  {t:>10.1f} {p:>12.4f} {r:>10.4f} {f:>8.4f}")


# ── 7. PLOTS ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), facecolor="#0d1117")
for ax in axes:
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="#8b949e")
    ax.spines[:].set_color("#30363d")

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
axes[0].plot(fpr, tpr, color="#58a6ff", linewidth=2, label=f"AUC={auc:.3f}")
axes[0].plot([0,1],[0,1], color="#30363d", linestyle="--")
axes[0].set_xlabel("FPR (False Alarm Rate)", color="#e6edf3")
axes[0].set_ylabel("TPR (Recall)",           color="#e6edf3")
axes[0].set_title("ROC Curve",               color="#e6edf3")
axes[0].legend(facecolor="#0d1117", labelcolor="#e6edf3")

# Confusion matrix heatmap
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
axes[1].imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
for i in range(2):
    for j in range(2):
        axes[1].text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]:.0%})",
                     ha="center", va="center", color="#e6edf3", fontsize=11)
axes[1].set_xticks([0,1]); axes[1].set_xticklabels(["Legit","Fraud"], color="#e6edf3")
axes[1].set_yticks([0,1]); axes[1].set_yticklabels(["Legit","Fraud"], color="#e6edf3")
axes[1].set_title("Confusion Matrix", color="#e6edf3")

# Probability distribution for fraud vs legit
legit_probs = y_pred_proba[y_test == 0]
fraud_probs = y_pred_proba[y_test == 1]
axes[2].hist(legit_probs, bins=20, alpha=0.7, color="#3fb950", label="Legit", density=True)
axes[2].hist(fraud_probs, bins=20, alpha=0.7, color="#f85149", label="Fraud", density=True)
axes[2].axvline(0.5, color="#f0883e", linewidth=1.5, linestyle="--", label="threshold=0.5")
axes[2].set_xlabel("P(fraud)", color="#e6edf3")
axes[2].set_title("Score Distribution", color="#e6edf3")
axes[2].legend(facecolor="#0d1117", labelcolor="#e6edf3", fontsize=8)

plt.suptitle("Logistic Regression — Fraud Classifier", color="#e6edf3", fontsize=13)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/module05_logreg.png", dpi=120, facecolor=fig.get_facecolor())
plt.close()
print(f"\nPlot saved → {PLOTS_DIR}/module05_logreg.png")

print("\n✓ Module 05 complete. Your first classifier is on the case.")
