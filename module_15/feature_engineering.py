"""module_15/feature_engineering.py — Feature Engineering with sklearn Pipelines.

Demonstrates:
  - OrdinalEncoder / OneHotEncoder for categorical features
  - StandardScaler and MinMaxScaler for numerical features
  - ColumnTransformer to apply different transforms to different columns
  - Pipeline to chain preprocessing + model with zero data leakage
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, OrdinalEncoder, OneHotEncoder
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# ── Constants ────────────────────────────────────────────────────────────────
RANDOM_SEED    = 42
CV_FOLDS       = 5
FIGURE_DPI     = 120
PLOTS_DIR      = "plots"

# Column groups — location is categorical; everything else is numeric
CATEGORICAL_COLS = ["location"]
NUMERICAL_COLS   = ["suspect_age", "evidence_score", "witness_count",
                    "crime_severity", "motive_strength", "days_open"]


def load_raw_dataframe(filepath):
    """Load cases CSV and return a clean DataFrame including the location column.

    Unlike data_loader.to_arrays() which drops location and converts to numpy,
    this function keeps all columns as a DataFrame for Pipeline compatibility.

    Args:
        filepath: path to cases.csv

    Returns:
        df: cleaned DataFrame with all feature columns
        y:  (n_samples,) int array of labels
    """
    df = pd.read_csv(filepath)
    df["days_open"]       = df["days_open"].fillna(df["days_open"].median())
    df["motive_strength"] = df["motive_strength"].fillna(df["motive_strength"].median())

    y = df["solved"].to_numpy(dtype=int)
    X = df.drop(columns=["case_id", "solved"])   # keep location as a column
    return X, y


def build_preprocessor():
    """Return a ColumnTransformer that scales numbers and encodes categories.

    Numerical columns: StandardScaler (zero mean, unit variance)
    Categorical columns: OneHotEncoder (creates binary dummy columns per category)

    Returns:
        sklearn ColumnTransformer
    """
    numerical_transformer = StandardScaler()

    # handle_unknown='ignore' silently creates an all-zeros row for unseen categories
    # sparse_output=False returns a dense array instead of a scipy sparse matrix
    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )

    return ColumnTransformer(
        transformers=[
            ("num", numerical_transformer,   NUMERICAL_COLS),
            ("cat", categorical_transformer, CATEGORICAL_COLS),
        ],
        remainder="drop",   # drop any column not listed above
    )


def build_pipeline(model):
    """Return a full Pipeline: preprocessor → model.

    Using a Pipeline guarantees that the preprocessor is fit ONLY on
    the training fold during cross-validation — no data leakage.

    Args:
        model: any unfitted sklearn estimator

    Returns:
        sklearn Pipeline
    """
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model",        model),
    ])


def compare_scaling_strategies(X_df, y):
    """Compare raw features vs. scaled features for Logistic Regression.

    Shows that LR accuracy improves significantly after proper scaling.

    Args:
        X_df: DataFrame of features (including location)
        y:    binary label array
    """
    X_numeric = X_df[NUMERICAL_COLS].to_numpy()   # drop location for fair comparison

    # Raw features — no scaling
    lr_raw = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    raw_cv = cross_val_score(lr_raw, X_numeric, y, cv=CV_FOLDS).mean()

    # StandardScaler
    from sklearn.pipeline import make_pipeline
    lr_std = make_pipeline(StandardScaler(),
                           LogisticRegression(max_iter=1000, random_state=RANDOM_SEED))
    std_cv = cross_val_score(lr_std, X_numeric, y, cv=CV_FOLDS).mean()

    # MinMaxScaler
    lr_mm  = make_pipeline(MinMaxScaler(),
                           LogisticRegression(max_iter=1000, random_state=RANDOM_SEED))
    mm_cv  = cross_val_score(lr_mm, X_numeric, y, cv=CV_FOLDS).mean()

    print("\n=== Effect of Scaling on Logistic Regression ===")
    print(f"  No scaling      : {raw_cv:.3f}")
    print(f"  StandardScaler  : {std_cv:.3f}")
    print(f"  MinMaxScaler    : {mm_cv:.3f}")
    return raw_cv, std_cv, mm_cv


def plot_scaling_comparison(raw, std, mm):
    """Save a bar chart comparing the three scaling strategies.

    Args:
        raw, std, mm: CV accuracy floats from compare_scaling_strategies
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)
    labels  = ["No Scaling", "StandardScaler", "MinMaxScaler"]
    values  = [raw, std, mm]
    colours = ["#f85149", "#58a6ff", "#3fb950"]

    fig, ax = plt.subplots(figsize=(6, 4), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    bars = ax.bar(labels, values, color=colours)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("CV Accuracy", color="#e6edf3")
    ax.set_title("Scaling Strategy vs. LR Accuracy", color="#e6edf3")
    ax.tick_params(colors="#e6edf3")
    ax.spines[:].set_color("#30363d")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.005,
                f"{val:.3f}", ha="center", color="#e6edf3", fontsize=10)
    plt.tight_layout()
    fname = f"{PLOTS_DIR}/scaling_comparison.png"
    plt.savefig(fname, dpi=FIGURE_DPI, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved → {fname}")
