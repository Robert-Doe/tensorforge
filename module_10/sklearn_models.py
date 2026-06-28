"""module_10/sklearn_models.py — All five classifiers via scikit-learn.

Each function mirrors a scratch implementation from M04-M08,
proving the math is identical while the API is far more concise.
"""

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib
matplotlib.use("Agg")   # no GUI — save to file only
import matplotlib.pyplot as plt

# ── Constants ────────────────────────────────────────────────────────────────
RANDOM_SEED   = 42
CV_FOLDS      = 5
TEST_RATIO    = 0.2
FIGURE_DPI    = 120
PLOTS_DIR     = "plots"


def build_models():
    """Return a dict of {name: sklearn_model_instance}.

    All hyperparameters match the scratch implementations from M04-M08
    so the accuracy numbers are directly comparable.
    """
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
        "KNN-5":              KNeighborsClassifier(n_neighbors=5),
        "DecisionTree-3":     DecisionTreeClassifier(max_depth=3, random_state=RANDOM_SEED),
        "NaiveBayes":         GaussianNB(),
    }


def run_cross_validation(models, X, y):
    """Run k-fold cross-validation for every model and print a table.

    Args:
        models: dict from build_models()
        X: feature matrix (n_samples, n_features)
        y: binary labels (n_samples,)
    """
    print(f"\n=== K-Fold Cross-Validation (k={CV_FOLDS}) via sklearn ===")
    print(f"{'Model':<22} {'Mean Acc':>9} {'± Std':>7}")
    print("-" * 42)
    for name, model in models.items():
        # cv=CV_FOLDS automatically shuffles and stratifies by default
        scores = cross_val_score(model, X, y, cv=CV_FOLDS, scoring="accuracy")
        print(f"{name:<22} {scores.mean():.3f}     ± {scores.std():.3f}")


def run_full_eval(models, X_train, X_test, y_train, y_test):
    """Fit each model and report precision, recall, F1, ROC-AUC.

    Args:
        models:  dict from build_models()
        X_train, X_test: feature splits
        y_train, y_test: label splits

    Returns:
        trained: dict of {name: fitted_model}
    """
    trained = {}
    print(f"\n=== Full Evaluation on Test Split ===")
    print(f"{'Model':<22} {'P':>6} {'R':>6} {'F1':>6} {'AUC':>6}")
    print("-" * 52)
    for name, model in models.items():
        model.fit(X_train, y_train)           # one line replaces our entire fit() loop
        preds  = model.predict(X_test)
        probs  = model.predict_proba(X_test)[:, 1]   # probability of positive class

        p   = precision_score(y_test, preds,  zero_division=0)
        r   = recall_score(y_test, preds,     zero_division=0)
        f1  = f1_score(y_test, preds,         zero_division=0)
        auc = roc_auc_score(y_test, probs)
        print(f"{name:<22} {p:>6.3f} {r:>6.3f} {f1:>6.3f} {auc:>6.3f}")
        trained[name] = model
    return trained


def save_confusion_matrix(model, X_test, y_test, name):
    """Save a styled confusion matrix PNG for one model.

    Args:
        model:  fitted sklearn model
        X_test: test features
        y_test: test labels
        name:   string used in filename
    """
    import os
    os.makedirs(PLOTS_DIR, exist_ok=True)

    preds = model.predict(X_test)
    cm    = confusion_matrix(y_test, preds)
    disp  = ConfusionMatrixDisplay(cm, display_labels=["Unsolved", "Solved"])

    fig, ax = plt.subplots(figsize=(5, 4), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {name}", color="#e6edf3")
    for text in ax.texts:              # make cell numbers visible on dark bg
        text.set_color("#e6edf3")
    ax.xaxis.label.set_color("#e6edf3")
    ax.yaxis.label.set_color("#e6edf3")
    ax.tick_params(colors="#e6edf3")

    fname = f"{PLOTS_DIR}/cm_sklearn_{name.lower().replace(' ', '_')}.png"
    plt.tight_layout()
    plt.savefig(fname, dpi=FIGURE_DPI, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved → {fname}")
