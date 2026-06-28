"""module_09/evaluation_advanced.py — Advanced Model Evaluation.

Covers:
  - Bootstrap confidence intervals on accuracy
  - Learning curves (train size vs accuracy)
  - Validation curves (hyperparameter vs accuracy)
  - Cohen's Kappa (agreement metric for imbalanced classes)
  - Permutation importance (model-agnostic feature importance)
  - McNemar's test (statistical comparison of two classifiers)

Run standalone: python evaluation_advanced.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    train_test_split, learning_curve, validation_curve, cross_val_score
)
from sklearn.metrics import cohen_kappa_score
from sklearn.inspection import permutation_importance
from scipy.stats import chi2

RANDOM_SEED = 42
PLOTS_DIR   = "plots"
FIGURE_DPI  = 120
rng = np.random.default_rng(RANDOM_SEED)
os.makedirs(PLOTS_DIR, exist_ok=True)


def make_data():
    X, y = make_classification(n_samples=1000, n_features=15, n_informative=8,
                               random_state=RANDOM_SEED)
    return train_test_split(X, y, test_size=0.25, random_state=RANDOM_SEED)


# ─────────────────────────────────────────────────────────────────────────────
# 1. BOOTSTRAP CONFIDENCE INTERVALS
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_ci(model, X_test: np.ndarray, y_test: np.ndarray,
                 n_bootstrap: int = 1000, ci: float = 0.95) -> tuple:
    """Estimate confidence interval on test accuracy via bootstrap.

    Resamples the test set with replacement n_bootstrap times and
    measures accuracy on each resample. The percentile method gives
    an empirical CI without assuming a normal distribution.

    Args:
        model:       fitted sklearn estimator
        X_test:      test features
        y_test:      test labels
        n_bootstrap: number of bootstrap resamples
        ci:          desired confidence level (0.95 = 95% CI)

    Returns:
        (mean_acc, lower, upper)
    """
    n = len(y_test)
    accs = []
    for _ in range(n_bootstrap):
        idx  = rng.integers(0, n, size=n)         # resample with replacement
        y_pred = model.predict(X_test[idx])
        accs.append((y_pred == y_test[idx]).mean())
    accs = np.array(accs)
    alpha = (1 - ci) / 2
    return accs.mean(), np.percentile(accs, alpha*100), np.percentile(accs, (1-alpha)*100)


def demo_bootstrap(X_tr, X_te, y_tr, y_te):
    """Bootstrap CI for Random Forest test accuracy."""
    print("── 1. Bootstrap Confidence Intervals ───────────────")
    model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
    model.fit(X_tr, y_tr)
    mean_acc, lo, hi = bootstrap_ci(model, X_te, y_te, n_bootstrap=2000)
    print(f"Test accuracy: {mean_acc:.4f}  95% CI: [{lo:.4f}, {hi:.4f}]")
    print("  If two models' CIs don't overlap → statistically different performance.")


# ─────────────────────────────────────────────────────────────────────────────
# 2. LEARNING CURVES
# ─────────────────────────────────────────────────────────────────────────────

def demo_learning_curves(X, y):
    """Plot how accuracy varies with training set size."""
    print("\n── 2. Learning Curves ──────────────────────────────")
    model = RandomForestClassifier(n_estimators=50, random_state=RANDOM_SEED)
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )

    fig, ax = plt.subplots(figsize=(7, 4), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")

    mean_tr, std_tr = train_scores.mean(1), train_scores.std(1)
    mean_va, std_va = val_scores.mean(1),   val_scores.std(1)

    ax.plot(train_sizes, mean_tr, "o-", color="#58a6ff", label="Train")
    ax.fill_between(train_sizes, mean_tr-std_tr, mean_tr+std_tr,
                    alpha=0.2, color="#58a6ff")
    ax.plot(train_sizes, mean_va, "s-", color="#3fb950", label="Validation")
    ax.fill_between(train_sizes, mean_va-std_va, mean_va+std_va,
                    alpha=0.2, color="#3fb950")

    ax.set_xlabel("Training set size", color="#e6edf3")
    ax.set_ylabel("Accuracy", color="#e6edf3")
    ax.set_title("Learning Curve", color="#e6edf3")
    ax.legend(facecolor="#161b22", labelcolor="#e6edf3")
    ax.tick_params(colors="#e6edf3"); ax.spines[:].set_color("#30363d")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/learning_curve.png", dpi=FIGURE_DPI,
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved → {PLOTS_DIR}/learning_curve.png")
    print(f"  Final validation: {mean_va[-1]:.4f} ± {std_va[-1]:.4f}")
    print("  Converging train/val curves → more data won't help much — need better features.")
    print("  Large train-val gap → overfitting — need regularisation or more data.")


# ─────────────────────────────────────────────────────────────────────────────
# 3. VALIDATION CURVES
# ─────────────────────────────────────────────────────────────────────────────

def demo_validation_curves(X, y):
    """Plot accuracy vs max_depth — the bias-variance tradeoff in one plot."""
    print("\n── 3. Validation Curves ────────────────────────────")
    depths = [1, 2, 3, 5, 7, 10, 15, 20, None]
    model  = RandomForestClassifier(n_estimators=50, random_state=RANDOM_SEED)
    train_scores, val_scores = validation_curve(
        model, X, y,
        param_name="max_depth",
        param_range=depths,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )
    depth_labels = [str(d) if d else "∞" for d in depths]
    fig, ax = plt.subplots(figsize=(8, 4), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    ax.plot(range(len(depths)), train_scores.mean(1), "o-",
            color="#58a6ff", label="Train")
    ax.plot(range(len(depths)), val_scores.mean(1),   "s-",
            color="#3fb950", label="Validation")
    ax.set_xticks(range(len(depths)))
    ax.set_xticklabels(depth_labels, color="#e6edf3")
    ax.set_xlabel("max_depth", color="#e6edf3")
    ax.set_ylabel("Accuracy", color="#e6edf3")
    ax.set_title("Validation Curve — max_depth", color="#e6edf3")
    ax.legend(facecolor="#161b22", labelcolor="#e6edf3")
    ax.tick_params(colors="#e6edf3"); ax.spines[:].set_color("#30363d")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/validation_curve.png", dpi=FIGURE_DPI,
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved → {PLOTS_DIR}/validation_curve.png")
    best_idx = val_scores.mean(1).argmax()
    print(f"  Best max_depth: {depth_labels[best_idx]}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. COHEN'S KAPPA
# ─────────────────────────────────────────────────────────────────────────────

def demo_cohens_kappa(X_tr, X_te, y_tr, y_te):
    """Show why Kappa is better than accuracy for imbalanced classes."""
    print("\n── 4. Cohen's Kappa ────────────────────────────────")
    model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    acc   = (y_pred == y_te).mean()
    kappa = cohen_kappa_score(y_te, y_pred)

    # A majority-class predictor — predicts most common class for everything
    majority_class = np.bincount(y_tr).argmax()
    y_majority     = np.full_like(y_te, majority_class)
    acc_majority   = (y_majority == y_te).mean()
    kappa_majority = cohen_kappa_score(y_te, y_majority)

    print(f"  {'Model':<30} {'Accuracy':>10}  {'Kappa':>8}")
    print(f"  {'Random Forest':<30} {acc:>10.4f}  {kappa:>8.4f}")
    print(f"  {'Majority class predictor':<30} {acc_majority:>10.4f}  {kappa_majority:>8.4f}")
    print("\n  Kappa = (accuracy - chance accuracy) / (1 - chance accuracy)")
    print("  Kappa=0 means no better than chance; Kappa=1 is perfect.")
    print("  The majority predictor has high accuracy but Kappa≈0 — correctly exposed.")


# ─────────────────────────────────────────────────────────────────────────────
# 5. PERMUTATION IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────

def demo_permutation_importance(X_tr, X_te, y_tr, y_te):
    """Model-agnostic feature importance via random permutation."""
    print("\n── 5. Permutation Importance ───────────────────────")
    print("  How much does accuracy drop when we randomly shuffle feature j?")
    print("  Large drop → feature j is important. Near-zero → irrelevant.")

    model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
    model.fit(X_tr, y_tr)
    result = permutation_importance(model, X_te, y_te, n_repeats=20,
                                    random_state=RANDOM_SEED, n_jobs=-1)

    idx = result.importances_mean.argsort()[::-1][:8]   # top 8
    print(f"\n  {'Feature':<12} {'Mean drop':>10}  {'Std':>8}")
    for i in idx:
        print(f"  feat_{i:<8} {result.importances_mean[i]:>10.4f}  "
              f"{result.importances_std[i]:>8.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. McNEMAR'S TEST — statistical comparison of two classifiers
# ─────────────────────────────────────────────────────────────────────────────

def demo_mcnemar(X_tr, X_te, y_tr, y_te):
    """Test whether two classifiers differ significantly on the same test set."""
    print("\n── 6. McNemar's Test (classifier comparison) ───────")
    rf  = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
    lr  = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    rf.fit(X_tr, y_tr);  lr.fit(X_tr, y_tr)

    pred_rf = rf.predict(X_te)
    pred_lr = lr.predict(X_te)

    # Contingency table: how often do the models agree/disagree on the same sample?
    b = ((pred_rf == y_te) & (pred_lr != y_te)).sum()   # RF correct, LR wrong
    c = ((pred_rf != y_te) & (pred_lr == y_te)).sum()   # RF wrong, LR correct

    # McNemar statistic with continuity correction
    chi2_stat = (abs(b - c) - 1) ** 2 / (b + c + 1e-9)
    p_value   = 1 - chi2.cdf(chi2_stat, df=1)

    print(f"  RF correct, LR wrong (b): {b}")
    print(f"  RF wrong, LR correct (c): {c}")
    print(f"  McNemar χ²={chi2_stat:.3f}  p={p_value:.4f}")
    if p_value < 0.05:
        better = "Random Forest" if b < c else "Logistic Regression"
        print(f"  p < 0.05 → significant difference. {better} is better.")
    else:
        print("  p ≥ 0.05 → no statistically significant difference.")


def main():
    print("=" * 56)
    print("MODULE 09 — Advanced Model Evaluation")
    print("=" * 56)
    X_tr, X_te, y_tr, y_te = make_data()
    X = np.vstack([X_tr, X_te])
    y = np.concatenate([y_tr, y_te])

    demo_bootstrap(X_tr, X_te, y_tr, y_te)
    demo_learning_curves(X, y)
    demo_validation_curves(X, y)
    demo_cohens_kappa(X_tr, X_te, y_tr, y_te)
    demo_permutation_importance(X_tr, X_te, y_tr, y_te)
    demo_mcnemar(X_tr, X_te, y_tr, y_te)
    print("\nDone.")


if __name__ == "__main__":
    main()
