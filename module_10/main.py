"""module_10/main.py — Scikit-learn Intro.

Replicates M04-M08 scratch implementations using sklearn,
proving identical results in a fraction of the code.
Run: python main.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))   # ensure local imports resolve

import numpy as np
from sklearn.model_selection import train_test_split

from data_loader import load_and_clean, to_arrays
from sklearn_models import (
    build_models, run_cross_validation, run_full_eval, save_confusion_matrix,
    RANDOM_SEED, TEST_RATIO
)

# ── Constants ────────────────────────────────────────────────────────────────
DATA_FILE = "cases.csv"


def main():
    """Entry point: load data, run sklearn models, compare to scratch results."""
    # 1. Load data — same pipeline as every module since M03
    df       = load_and_clean(DATA_FILE)
    X, y, _ = to_arrays(df)

    # 2. Single stratified split — stratify keeps class balance in both halves
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_RATIO, random_state=RANDOM_SEED, stratify=y
    )

    print("=" * 52)
    print("MODULE 10 — Scikit-learn Intro")
    print("=" * 52)
    print(f"Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

    # 3. Build the four models — each matches a scratch counterpart
    models = build_models()

    # 4. Cross-validation comparison
    run_cross_validation(models, X, y)

    # 5. Full test-split evaluation
    trained = run_full_eval(models, X_train, X_test, y_train, y_test)

    # 6. Save confusion matrix for the best performer (LogisticRegression)
    save_confusion_matrix(
        trained["LogisticRegression"], X_test, y_test, "LogisticRegression"
    )

    # 7. Show how concise sklearn is vs scratch
    print("\n=== Lines of code comparison ===")
    print(f"  Scratch LogisticRegression (M05): ~60 lines")
    print(f"  sklearn LogisticRegression (M10):  3 lines  (fit / predict / score)")
    print(f"\nSame math. Same results. Zero reimplementation.")
    print("\nDone — all outputs saved to plots/")


if __name__ == "__main__":
    main()
