"""module_15/main.py — Feature Engineering & Pipelines.

Run: python main.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from feature_engineering import (
    load_raw_dataframe, build_preprocessor, build_pipeline,
    compare_scaling_strategies, plot_scaling_comparison,
    RANDOM_SEED, CV_FOLDS,
)

DATA_FILE  = "cases.csv"
TEST_RATIO = 0.2


def main():
    """Load raw data, engineer features, build pipelines, compare models."""
    X_df, y = load_raw_dataframe(DATA_FILE)

    print("=" * 52)
    print("MODULE 15 — Feature Engineering & Pipelines")
    print("=" * 52)
    print(f"Feature columns: {list(X_df.columns)}")
    print(f"Location unique values: {sorted(X_df['location'].unique())}")

    # ── 1. Show why scaling matters ──────────────────────────────────────────
    raw, std, mm = compare_scaling_strategies(X_df, y)
    plot_scaling_comparison(raw, std, mm)

    # ── 2. Full pipeline including one-hot encoded location ──────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=TEST_RATIO, random_state=RANDOM_SEED, stratify=y
    )

    print("\n=== Full Pipeline (numeric + categorical) ===")
    for name, model in [
        ("LogisticRegression", LogisticRegression(max_iter=1000,
                                                   random_state=RANDOM_SEED)),
        ("RandomForest",       RandomForestClassifier(n_estimators=100,
                                                       random_state=RANDOM_SEED)),
    ]:
        pipe   = build_pipeline(model)
        cv_acc = cross_val_score(pipe, X_df, y, cv=CV_FOLDS).mean()
        pipe.fit(X_train, y_train)
        probs  = pipe.predict_proba(X_test)[:, 1]
        auc    = roc_auc_score(y_test, probs)
        print(f"  {name:<22} CV={cv_acc:.3f}  AUC={auc:.3f}")

    # ── 3. Show what the preprocessor actually produces ──────────────────────
    prep = build_preprocessor()
    X_transformed = prep.fit_transform(X_train)
    print(f"\nPreprocessor output shape: {X_transformed.shape}")
    print(f"  Input:  {X_train.shape[1]} columns (6 numeric + 1 categorical)")
    print(f"  Output: {X_transformed.shape[1]} columns "
          f"(6 scaled + {X_transformed.shape[1]-6} one-hot dummies)")

    # ── 4. Demonstrate leakage danger ────────────────────────────────────────
    print("\n=== Pipeline vs. Manual (leakage demo) ===")
    from sklearn.preprocessing import StandardScaler
    # WRONG: fit scaler on ALL data before splitting
    scaler_bad = StandardScaler()
    X_num      = X_df[["suspect_age","evidence_score","witness_count",
                        "crime_severity","motive_strength","days_open"]].to_numpy()
    X_scaled_bad = scaler_bad.fit_transform(X_num)   # test data leaked into scaler!
    from sklearn.linear_model import LogisticRegression as LR
    lr_bad = LR(max_iter=1000, random_state=RANDOM_SEED)
    bad_cv = cross_val_score(lr_bad, X_scaled_bad, y, cv=CV_FOLDS).mean()

    # RIGHT: Pipeline fits scaler only on training fold
    pipe_good = build_pipeline(LR(max_iter=1000, random_state=RANDOM_SEED))
    good_cv   = cross_val_score(pipe_good, X_df, y, cv=CV_FOLDS).mean()

    print(f"  Leaky manual scaling CV  : {bad_cv:.3f}  ← optimistic (wrong)")
    print(f"  Pipeline (no leakage) CV : {good_cv:.3f}  ← trustworthy (correct)")

    print("\nDone.")


if __name__ == "__main__":
    main()
