"""module_11/svm_models.py — Support Vector Machines via sklearn.

Demonstrates linear SVM, RBF-kernel SVM, and a C-sweep to show
the margin/accuracy trade-off.
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Constants ────────────────────────────────────────────────────────────────
RANDOM_SEED  = 42
CV_FOLDS     = 5
C_VALUES     = [0.01, 0.1, 1, 10, 100]   # regularisation sweep
FIGURE_DPI   = 120
PLOTS_DIR    = "plots"


def make_linear_svm(C=1.0):
    """Return a Pipeline: StandardScaler → linear SVM.

    SVM is sensitive to feature scale, so scaling is mandatory.
    Wrapping in a Pipeline ensures the scaler is fit only on
    training data (no data leakage).

    Args:
        C: regularisation parameter — smaller C = wider margin, more errors allowed

    Returns:
        sklearn Pipeline
    """
    return Pipeline([
        ("scaler", StandardScaler()),          # zero-mean, unit-variance per feature
        ("svm",    SVC(kernel="linear", C=C,
                       probability=True,       # needed for predict_proba / ROC-AUC
                       random_state=RANDOM_SEED)),
    ])


def make_rbf_svm(C=1.0, gamma="scale"):
    """Return a Pipeline: StandardScaler → RBF-kernel SVM.

    RBF (Radial Basis Function) kernel maps data into infinite-dimensional
    space so a linear hyperplane there becomes a curved boundary here.

    Args:
        C:     regularisation — same meaning as linear SVM
        gamma: controls how far a single training example's influence reaches.
               'scale' = 1/(n_features * X.var()) — a good default.

    Returns:
        sklearn Pipeline
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    SVC(kernel="rbf", C=C, gamma=gamma,
                       probability=True, random_state=RANDOM_SEED)),
    ])


def c_sweep(X, y):
    """Try each C value with both linear and RBF kernels; print accuracy table.

    Args:
        X: feature matrix
        y: binary labels
    """
    print("\n=== C-Sweep: margin width vs. accuracy ===")
    print(f"{'C':>7}  {'Linear CV':>10}  {'RBF CV':>10}")
    print("-" * 34)
    for c in C_VALUES:
        lin = cross_val_score(make_linear_svm(C=c), X, y,
                              cv=CV_FOLDS, scoring="accuracy")
        rbf = cross_val_score(make_rbf_svm(C=c),    X, y,
                              cv=CV_FOLDS, scoring="accuracy")
        print(f"{c:>7}  {lin.mean():.3f}±{lin.std():.2f}  {rbf.mean():.3f}±{rbf.std():.2f}")


def plot_c_sweep(X, y):
    """Save a line plot showing CV accuracy vs C for both kernels.

    Args:
        X: feature matrix
        y: binary labels
    """
    import os
    os.makedirs(PLOTS_DIR, exist_ok=True)

    lin_means, rbf_means = [], []
    for c in C_VALUES:
        lin_means.append(cross_val_score(make_linear_svm(C=c), X, y,
                                         cv=CV_FOLDS).mean())
        rbf_means.append(cross_val_score(make_rbf_svm(C=c),    X, y,
                                         cv=CV_FOLDS).mean())

    fig, ax = plt.subplots(figsize=(7, 4), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    ax.semilogx(C_VALUES, lin_means, "o-", color="#58a6ff", label="Linear SVM")
    ax.semilogx(C_VALUES, rbf_means, "s-", color="#3fb950", label="RBF SVM")
    ax.set_xlabel("C (log scale)",  color="#e6edf3")
    ax.set_ylabel("CV Accuracy",    color="#e6edf3")
    ax.set_title("SVM: C vs. Accuracy", color="#e6edf3")
    ax.tick_params(colors="#e6edf3")
    ax.legend(facecolor="#161b22", labelcolor="#e6edf3")
    ax.spines[:].set_color("#30363d")

    fname = f"{PLOTS_DIR}/svm_c_sweep.png"
    plt.tight_layout()
    plt.savefig(fname, dpi=FIGURE_DPI, facecolor=fig.get_facecolor())
    plt.close()
    print(f"\nSaved → {fname}")
