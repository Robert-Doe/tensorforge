"""module_12/main.py — Random Forests & Gradient Boosting.

Run: python main.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score

from data_loader import load_and_clean, to_arrays
from ensemble_models import (
    make_random_forest, make_gradient_boosting,
    print_feature_importances, plot_n_estimators_sweep, plot_feature_importances,
    RANDOM_SEED, CV_FOLDS,
)

DATA_FILE  = "cases.csv"
TEST_RATIO = 0.2


def main():
    """Load data, compare RF and GB, show feature importances."""
    df              = load_and_clean(DATA_FILE)
    X, y, feat_cols = to_arrays(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_RATIO, random_state=RANDOM_SEED, stratify=y
    )

    print("=" * 52)
    print("MODULE 12 — Random Forests & Gradient Boosting")
    print("=" * 52)

    # ── Random Forest ────────────────────────────────────────────────────────
    rf = make_random_forest(n_estimators=100)
    rf.fit(X_train, y_train)
    rf_cv  = cross_val_score(rf, X, y, cv=CV_FOLDS).mean()
    rf_acc = accuracy_score(y_test, rf.predict(X_test))
    rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])

    print(f"\nRandom Forest (100 trees)")
    print(f"  CV Accuracy : {rf_cv:.3f}")
    print(f"  Test Acc    : {rf_acc:.3f}")
    print(f"  ROC-AUC     : {rf_auc:.3f}")
    print_feature_importances(rf, feat_cols)

    # ── Gradient Boosting ────────────────────────────────────────────────────
    gb = make_gradient_boosting(n_estimators=100, learning_rate=0.1, max_depth=3)
    gb.fit(X_train, y_train)
    gb_cv  = cross_val_score(gb, X, y, cv=CV_FOLDS).mean()
    gb_acc = accuracy_score(y_test, gb.predict(X_test))
    gb_auc = roc_auc_score(y_test, gb.predict_proba(X_test)[:, 1])

    print(f"\nGradient Boosting (100 rounds, lr=0.1)")
    print(f"  CV Accuracy : {gb_cv:.3f}")
    print(f"  Test Acc    : {gb_acc:.3f}")
    print(f"  ROC-AUC     : {gb_auc:.3f}")
    print_feature_importances(gb, feat_cols)

    # ── Head-to-head summary ─────────────────────────────────────────────────
    print("\n=== Head-to-Head ===")
    print(f"{'Model':<28} {'CV Acc':>7} {'AUC':>7}")
    print("-" * 44)
    print(f"{'Random Forest (100)':<28} {rf_cv:>7.3f} {rf_auc:>7.3f}")
    print(f"{'Gradient Boosting (100)':<28} {gb_cv:>7.3f} {gb_auc:>7.3f}")

    # ── Plots ────────────────────────────────────────────────────────────────
    print("\nGenerating plots (this takes ~30s for the n_estimators sweep)...")
    plot_n_estimators_sweep(X, y)
    plot_feature_importances(rf, feat_cols, "RandomForest")
    plot_feature_importances(gb, feat_cols, "GradientBoosting")

    print("\nDone — plots saved to plots/")


if __name__ == "__main__":
    main()
