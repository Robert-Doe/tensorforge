"""module_11/main.py — SVMs with Scikit-learn.

Demonstrates linear SVM, RBF SVM, and C-regularisation sweep.
Run: python main.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

from data_loader import load_and_clean, to_arrays
from svm_models import (
    make_linear_svm, make_rbf_svm, c_sweep, plot_c_sweep,
    RANDOM_SEED, CV_FOLDS
)

DATA_FILE  = "cases.csv"
TEST_RATIO = 0.2


def main():
    """Load data, train linear and RBF SVMs, run C-sweep, save plot."""
    df       = load_and_clean(DATA_FILE)
    X, y, _ = to_arrays(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_RATIO, random_state=RANDOM_SEED, stratify=y
    )

    print("=" * 52)
    print("MODULE 11 — Support Vector Machines")
    print("=" * 52)

    # ── Linear SVM ──────────────────────────────────────────────────────────
    lin = make_linear_svm(C=1.0)
    lin.fit(X_train, y_train)
    lin_preds = lin.predict(X_test)
    lin_probs = lin.predict_proba(X_test)[:, 1]
    print(f"\nLinear SVM (C=1.0)")
    print(f"  Accuracy : {accuracy_score(y_test, lin_preds):.3f}")
    print(f"  ROC-AUC  : {roc_auc_score(y_test, lin_probs):.3f}")

    # ── RBF SVM ─────────────────────────────────────────────────────────────
    rbf = make_rbf_svm(C=1.0, gamma="scale")
    rbf.fit(X_train, y_train)
    rbf_preds = rbf.predict(X_test)
    rbf_probs = rbf.predict_proba(X_test)[:, 1]
    print(f"\nRBF SVM (C=1.0, gamma=scale)")
    print(f"  Accuracy : {accuracy_score(y_test, rbf_preds):.3f}")
    print(f"  ROC-AUC  : {roc_auc_score(y_test, rbf_probs):.3f}")

    # ── Support vectors ──────────────────────────────────────────────────────
    n_sv = rbf.named_steps["svm"].n_support_
    print(f"\nRBF support vectors per class: {n_sv}  (total {n_sv.sum()})")
    print(f"  Only these {n_sv.sum()} training samples define the decision boundary.")

    # ── C sweep ─────────────────────────────────────────────────────────────
    c_sweep(X, y)
    plot_c_sweep(X, y)

    print("\nDone.")


if __name__ == "__main__":
    main()
