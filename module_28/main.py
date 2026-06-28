"""module_28/main.py — Complete end-to-end ML pipeline demonstration.

Generates a synthetic "Detective Agency Case Outcome" dataset and runs the
full pipeline: load → inspect → feature engineer → model selection →
hyperparameter tuning → final evaluation → save → load → inference.

Every step uses tools introduced in earlier modules, wired together here.
Run: python main.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble  import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline import (
    load_and_inspect, build_full_pipeline, run_cross_validation,
    tune_hyperparameters, final_evaluation, save_pipeline, load_pipeline,
    predict_new,
)

# ── Constants ────────────────────────────────────────────────────────────────
RANDOM_SEED      = 42
TEST_SIZE        = 0.2
N_SAMPLES        = 1200
DATA_PATH        = "data/case_outcomes.csv"
MODEL_PATH       = "saved_models/best_pipeline.pkl"
PLOTS_DIR        = "plots"
FIGURE_DPI       = 120

CATEGORICAL_COLS = ["district", "case_type"]


# ── Synthetic dataset ─────────────────────────────────────────────────────────

def generate_dataset(n: int, path: str) -> None:
    """Generate a synthetic detective agency case outcome dataset.

    Features:
      - district:          categorical (North/South/East/West)
      - case_type:         categorical (Theft/Fraud/Assault/Missing)
      - num_witnesses:     integer 0–8
      - evidence_strength: float 0–1
      - suspect_age:       integer 18–70
      - days_open:         integer 1–120
      - prior_convictions: integer 0–5
      - tip_received:      binary 0/1

    Target (solved): 1 if case was solved, 0 otherwise.
    Label generated from a noisy logistic function of the features.

    Args:
        n:    number of samples
        path: output CSV path
    """
    rng = np.random.default_rng(RANDOM_SEED)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    districts  = rng.choice(["North", "South", "East", "West"], size=n)
    case_types = rng.choice(["Theft", "Fraud", "Assault", "Missing"], size=n)
    num_witnesses   = rng.integers(0, 9, size=n)
    evidence        = rng.uniform(0, 1, size=n)
    suspect_age     = rng.integers(18, 71, size=n)
    days_open       = rng.integers(1, 121, size=n)
    prior           = rng.integers(0, 6, size=n)
    tip_received    = rng.integers(0, 2, size=n)

    # Logit: more evidence and witnesses → higher solve rate; longer open → lower
    logit  = (-1.0
              + 3.0 * evidence
              + 0.25 * num_witnesses
              + 0.4  * tip_received
              + 0.15 * prior
              - 0.01 * days_open
              + rng.normal(0, 0.5, size=n))   # noise
    prob   = 1 / (1 + np.exp(-logit))
    solved = (rng.uniform(size=n) < prob).astype(int)

    df = pd.DataFrame({
        "district":          districts,
        "case_type":         case_types,
        "num_witnesses":     num_witnesses,
        "evidence_strength": evidence,
        "suspect_age":       suspect_age,
        "days_open":         days_open,
        "prior_convictions": prior,
        "tip_received":      tip_received,
        "solved":            solved,
    })

    # Introduce 3% missing values in evidence_strength to make it realistic
    missing_idx = rng.choice(n, size=int(0.03 * n), replace=False)
    df.loc[missing_idx, "evidence_strength"] = np.nan

    df.to_csv(path, index=False)
    print(f"Generated dataset → {path}  ({n} rows)")


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing numerical values with column medians.

    Median is preferred over mean for skewed features (evidence_strength
    is right-skewed — many weak cases, few with strong evidence).

    Args:
        df: DataFrame possibly containing NaN

    Returns:
        df with NaN replaced by column medians
    """
    for col in df.select_dtypes(include=[float, int]).columns:
        if df[col].isnull().any():
            median = df[col].median()
            df[col] = df[col].fillna(median)
            print(f"  Imputed {col}: {df[col].isnull().sum()} remaining NaN (median={median:.3f})")
    return df


def plot_feature_importances(model_pipeline, feature_names: list):
    """Save feature importance bar chart from the best model.

    Args:
        model_pipeline: fitted Pipeline whose last step is a tree ensemble
        feature_names:  list of feature names after one-hot encoding
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)
    model  = model_pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return
    importances = model.feature_importances_
    idx         = np.argsort(importances)[-15:]   # top 15

    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    ax.barh(range(len(idx)),
            importances[idx],
            color="#58a6ff")
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([feature_names[i] if i < len(feature_names) else f"feat_{i}"
                        for i in idx], color="#e6edf3", fontsize=8)
    ax.set_xlabel("Importance", color="#e6edf3")
    ax.set_title("Top Feature Importances", color="#e6edf3")
    ax.tick_params(colors="#e6edf3")
    ax.spines[:].set_color("#30363d")
    plt.tight_layout()
    fname = f"{PLOTS_DIR}/feature_importances.png"
    plt.savefig(fname, dpi=FIGURE_DPI, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved → {fname}")


def main():
    np.random.seed(RANDOM_SEED)

    print("=" * 58)
    print("MODULE 28 — End-to-End ML Pipeline")
    print("=" * 58)

    # ── Step 1: Generate and inspect data ─────────────────────────────────────
    print("\n── Step 1: Data Generation & Inspection ─────────────")
    generate_dataset(N_SAMPLES, DATA_PATH)
    df = load_and_inspect(DATA_PATH)

    # ── Step 2: Impute missing values ─────────────────────────────────────────
    print("\n── Step 2: Missing Value Imputation ─────────────────")
    df = impute_missing(df)

    # ── Step 3: Train / test split BEFORE anything else ───────────────────────
    print("\n── Step 3: Train/Test Split ─────────────────────────")
    TARGET_COL = "solved"
    X_df = df.drop(columns=[TARGET_COL])
    y    = df[TARGET_COL].values

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    # Convert to numpy for sklearn — pipeline accepts both but array is simpler
    X_train = X_train_df.values
    X_test  = X_test_df.values
    print(f"Train: {len(y_train)} samples  Test: {len(y_test)} samples")
    print(f"Class balance (train): {y_train.mean():.2%} solved")

    # ── Step 4: Model selection via cross-validation ───────────────────────────
    print("\n── Step 4: Model Selection (5-fold CV) ──────────────")
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_SEED),
    }

    cv_results = {}
    best_name, best_auc = None, -1

    for name, model in candidates.items():
        pipeline, _, _ = build_full_pipeline(
            pd.concat([X_train_df, pd.Series(y_train, name=TARGET_COL, index=X_train_df.index)], axis=1),
            TARGET_COL, CATEGORICAL_COLS, model
        )
        result = run_cross_validation(pipeline, X_train, y_train, name)
        cv_results[name] = result
        if result["mean_auc"] > best_auc:
            best_auc  = result["mean_auc"]
            best_name = name

    print(f"\nBest model by CV AUC: {best_name} ({best_auc:.4f})")

    # ── Step 5: Hyperparameter tuning ─────────────────────────────────────────
    print("\n── Step 5: Hyperparameter Tuning ────────────────────")
    best_model = candidates[best_name]
    pipeline_for_tuning, _, _ = build_full_pipeline(
        pd.concat([X_train_df, pd.Series(y_train, name=TARGET_COL, index=X_train_df.index)], axis=1),
        TARGET_COL, CATEGORICAL_COLS, best_model
    )

    if best_name == "Random Forest":
        param_grid = {
            "model__n_estimators": [100, 200],
            "model__max_depth":    [None, 10, 20],
        }
    elif best_name == "Gradient Boosting":
        param_grid = {
            "model__n_estimators":  [100, 200],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth":     [3, 5],
        }
    else:
        param_grid = {"model__C": [0.1, 1.0, 10.0]}

    best_pipeline = tune_hyperparameters(
        pipeline_for_tuning, X_train, y_train, param_grid, best_name
    )

    # ── Step 6: Final evaluation on held-out test set ─────────────────────────
    print("\n── Step 6: Final Evaluation ─────────────────────────")
    # Re-fit tuned pipeline on full training data
    best_pipeline.fit(X_train, y_train)
    final_evaluation(best_pipeline, X_train, y_train, X_test, y_test)

    # ── Step 7: Save model ────────────────────────────────────────────────────
    print("\n── Step 7: Save & Load Pipeline ─────────────────────")
    save_pipeline(best_pipeline, MODEL_PATH)
    loaded_pipeline = load_pipeline(MODEL_PATH)

    # ── Step 8: Inference on new data ─────────────────────────────────────────
    print("\n── Step 8: Inference on New Cases ───────────────────")
    new_cases = pd.DataFrame({
        "district":          ["North", "South", "East"],
        "case_type":         ["Theft", "Fraud", "Assault"],
        "num_witnesses":     [5, 0, 3],
        "evidence_strength": [0.92, 0.10, 0.55],
        "suspect_age":       [34, 55, 28],
        "days_open":         [7, 90, 45],
        "prior_convictions": [2, 0, 1],
        "tip_received":      [1, 0, 1],
    })

    predictions = predict_new(loaded_pipeline, new_cases.values)
    probas      = loaded_pipeline.predict_proba(new_cases.values)[:, 1]

    print("\nNew case predictions:")
    for i, (pred, prob) in enumerate(zip(predictions, probas)):
        outcome = "SOLVED" if pred == 1 else "UNSOLVED"
        print(f"  Case {i+1}: {outcome} (confidence={prob:.3f})")

    # ── Feature importances ────────────────────────────────────────────────────
    numerical_cols = [c for c in X_train_df.columns if c not in CATEGORICAL_COLS]
    cat_features   = (best_pipeline.named_steps["preprocessor"]
                      .named_transformers_["cat"]
                      .get_feature_names_out(CATEGORICAL_COLS).tolist())
    all_features   = numerical_cols + cat_features
    plot_feature_importances(best_pipeline, all_features)

    print("\n" + "=" * 58)
    print("CURRICULUM COMPLETE — All 28 modules finished!")
    print("=" * 58)
    print("\nModules completed:")
    modules = [
        "01 NumPy Arrays",      "02 Pandas",              "03 Matplotlib/Seaborn",
        "04 Linear Regression", "05 Logistic Regression", "06 k-NN",
        "07 Decision Trees",    "08 Naive Bayes",         "09 Model Evaluation",
        "10 Scikit-learn API",  "11 SVMs",                "12 Ensembles",
        "13 k-Means",           "14 PCA",                 "15 Feature Engineering",
        "16 Single Neuron",     "17 Backprop",            "18 MLP",
        "19 Activations",       "20 Training Loops",      "21 Regularisation",
        "22 PyTorch Intro",     "23 Conv Layer",          "24 Full CNN",
        "25 BatchNorm+Dropout", "26 Transfer Learning",   "27 RNNs & LSTMs",
        "28 E2E Pipeline",
    ]
    for i, name in enumerate(modules, 1):
        print(f"  ✓ Module {i:02d}: {name}")


if __name__ == "__main__":
    main()
