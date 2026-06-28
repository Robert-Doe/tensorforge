"""module_12/ensemble_models.py — Random Forests and Gradient Boosting.

Demonstrates bagging (RandomForest) vs boosting (GradientBoostingClassifier),
feature importance extraction, and n_estimators sweep.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# ── Constants ────────────────────────────────────────────────────────────────
RANDOM_SEED       = 42
CV_FOLDS          = 5
N_ESTIMATORS_LIST = [1, 5, 10, 25, 50, 100, 200]   # for the sweep plot
PLOTS_DIR         = "plots"
FIGURE_DPI        = 120


def make_random_forest(n_estimators=100, max_depth=None):
    """Return a RandomForestClassifier with fixed random state.

    Bagging strategy: each tree sees a bootstrap sample of rows
    AND a random subset of features — double randomness prevents
    all trees from learning the same patterns.

    Args:
        n_estimators: number of trees to grow
        max_depth:    max depth per tree (None = grow until pure leaves)

    Returns:
        sklearn RandomForestClassifier
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=RANDOM_SEED,
        n_jobs=-1,          # use all CPU cores — trees are independent
    )


def make_gradient_boosting(n_estimators=100, learning_rate=0.1, max_depth=3):
    """Return a GradientBoostingClassifier.

    Boosting strategy: trees are sequential, not parallel. Each tree
    fits the *residual errors* of the previous ensemble. Learning rate
    scales each tree's contribution — smaller = slower but more robust.

    Args:
        n_estimators:  number of sequential boosting rounds
        learning_rate: shrinkage factor per round (0 < lr <= 1)
        max_depth:     depth of each weak learner (kept shallow on purpose)

    Returns:
        sklearn GradientBoostingClassifier
    """
    return GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=RANDOM_SEED,
    )


def print_feature_importances(model, feature_names):
    """Print sorted feature importances from a tree ensemble.

    Args:
        model:         fitted RF or GB model
        feature_names: list of column names matching X columns
    """
    importances = model.feature_importances_    # mean decrease in impurity per feature
    order       = np.argsort(importances)[::-1] # sort descending

    print("\nFeature Importances (higher = more useful to the ensemble):")
    print(f"  {'Feature':<22} {'Importance':>10}")
    print("  " + "-" * 34)
    for idx in order:
        print(f"  {feature_names[idx]:<22} {importances[idx]:>10.4f}")


def plot_n_estimators_sweep(X, y):
    """Save accuracy vs. n_estimators for RF and GB side by side.

    Args:
        X: feature matrix
        y: binary labels
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    rf_scores, gb_scores = [], []
    for n in N_ESTIMATORS_LIST:
        rf = make_random_forest(n_estimators=n)
        gb = make_gradient_boosting(n_estimators=n)
        rf_scores.append(cross_val_score(rf, X, y, cv=CV_FOLDS).mean())
        gb_scores.append(cross_val_score(gb, X, y, cv=CV_FOLDS).mean())

    fig, ax = plt.subplots(figsize=(7, 4), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    ax.plot(N_ESTIMATORS_LIST, rf_scores, "o-", color="#58a6ff", label="Random Forest")
    ax.plot(N_ESTIMATORS_LIST, gb_scores, "s-", color="#3fb950", label="Gradient Boosting")
    ax.set_xlabel("n_estimators",   color="#e6edf3")
    ax.set_ylabel("CV Accuracy",    color="#e6edf3")
    ax.set_title("Ensemble Size vs. Accuracy", color="#e6edf3")
    ax.tick_params(colors="#e6edf3")
    ax.legend(facecolor="#161b22", labelcolor="#e6edf3")
    ax.spines[:].set_color("#30363d")

    fname = f"{PLOTS_DIR}/ensemble_sweep.png"
    plt.tight_layout()
    plt.savefig(fname, dpi=FIGURE_DPI, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved → {fname}")


def plot_feature_importances(model, feature_names, title):
    """Save a horizontal bar chart of feature importances.

    Args:
        model:         fitted ensemble model
        feature_names: list of feature name strings
        title:         string used in plot title and filename
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    importances = model.feature_importances_
    order       = np.argsort(importances)    # ascending for horizontal bar

    fig, ax = plt.subplots(figsize=(7, 4), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    bars = ax.barh(
        [feature_names[i] for i in order],
        importances[order],
        color="#58a6ff",
    )
    ax.set_xlabel("Importance", color="#e6edf3")
    ax.set_title(f"Feature Importances — {title}", color="#e6edf3")
    ax.tick_params(colors="#e6edf3")
    ax.spines[:].set_color("#30363d")

    fname = f"{PLOTS_DIR}/feat_importance_{title.lower().replace(' ','_')}.png"
    plt.tight_layout()
    plt.savefig(fname, dpi=FIGURE_DPI, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved → {fname}")
