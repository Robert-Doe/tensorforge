"""module_28/pipeline.py — End-to-end ML pipeline utilities.

Ties together: raw data → feature engineering → model selection →
cross-validation → hyperparameter tuning → final evaluation → model saving.

Imports from earlier modules to demonstrate the cumulative curriculum.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "module_01"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "module_02"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "module_15"))

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import pickle

from sklearn.pipeline      import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose       import ColumnTransformer
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold, GridSearchCV
)
from sklearn.metrics import (
    accuracy_score, classification_report, roc_auc_score
)
from sklearn.ensemble  import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm           import SVC

# ── Constants ────────────────────────────────────────────────────────────────
RANDOM_SEED  = 42
CV_FOLDS     = 5
MODELS_DIR   = "saved_models"


def load_and_inspect(csv_path: str) -> pd.DataFrame:
    """Load raw CSV and print a data quality report.

    Args:
        csv_path: path to the CSV file

    Returns:
        DataFrame with original data
    """
    df = pd.read_csv(csv_path)
    print(f"\nData shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nMissing values:")
    missing = df.isnull().sum()
    for col, n in missing[missing > 0].items():
        print(f"  {col}: {n} ({n/len(df)*100:.1f}%)")
    if missing.sum() == 0:
        print("  None — clean dataset")
    print(f"\nClass distribution (target):")
    print(df.iloc[:, -1].value_counts().to_string())
    return df


def build_full_pipeline(df: pd.DataFrame, target_col: str,
                        categorical_cols: List[str],
                        model) -> Tuple[Pipeline, np.ndarray, np.ndarray]:
    """Construct sklearn Pipeline with ColumnTransformer + model.

    Separates feature types automatically:
    - categorical → OneHotEncoder
    - numerical   → StandardScaler

    Args:
        df:               full DataFrame (features + target)
        target_col:       name of the target column
        categorical_cols: list of categorical feature column names
        model:            sklearn estimator

    Returns:
        (pipeline, X_array, y_array) where X_array is raw (unprocessed)
    """
    y = df[target_col].values
    X = df.drop(columns=[target_col])

    numerical_cols = [c for c in X.columns if c not in categorical_cols]

    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(),                                      numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
    ], remainder="drop")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model",        model),
    ])

    return pipeline, X.values, y


def run_cross_validation(pipeline: Pipeline, X: np.ndarray, y: np.ndarray,
                         name: str) -> Dict[str, float]:
    """Run stratified k-fold cross-validation and report metrics.

    Args:
        pipeline: sklearn Pipeline (preprocessor + model)
        X:        raw feature array (unprocessed — pipeline handles it)
        y:        label array
        name:     model name for printing

    Returns:
        dict with mean_acc, std_acc, mean_auc, std_auc
    """
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    # Pass X as DataFrame or raw array — pipeline accepts both
    acc_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
    auc_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc")

    result = {
        "mean_acc": acc_scores.mean(),
        "std_acc":  acc_scores.std(),
        "mean_auc": auc_scores.mean(),
        "std_auc":  auc_scores.std(),
    }
    print(f"  {name:<30}  "
          f"Acc={result['mean_acc']:.4f}±{result['std_acc']:.4f}  "
          f"AUC={result['mean_auc']:.4f}±{result['std_auc']:.4f}")
    return result


def tune_hyperparameters(pipeline: Pipeline, X: np.ndarray, y: np.ndarray,
                         param_grid: Dict[str, List], name: str) -> Pipeline:
    """Grid-search hyperparameter tuning with cross-validation.

    Args:
        pipeline:   sklearn Pipeline
        X:          feature array
        y:          label array
        param_grid: dict mapping "model__param_name" → list of values
        name:       model name for printing

    Returns:
        best_estimator: Pipeline fitted on full X with best params
    """
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    gs = GridSearchCV(pipeline, param_grid, cv=cv, scoring="roc_auc",
                      n_jobs=-1, verbose=0)
    gs.fit(X, y)
    print(f"\nBest params for {name}: {gs.best_params_}")
    print(f"Best CV AUC: {gs.best_score_:.4f}")
    return gs.best_estimator_


def final_evaluation(pipeline: Pipeline, X_train: np.ndarray, y_train: np.ndarray,
                     X_test: np.ndarray, y_test: np.ndarray) -> None:
    """Fit on train, evaluate on held-out test set, print full report.

    Args:
        pipeline:  sklearn Pipeline
        X_train:   training features (raw)
        y_train:   training labels
        X_test:    test features (raw)
        y_test:    test labels
    """
    pipeline.fit(X_train, y_train)
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print(f"\nFinal evaluation on held-out test set:")
    print(f"  Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  ROC-AUC  : {roc_auc_score(y_test, y_proba):.4f}")
    print(f"\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4))


def save_pipeline(pipeline: Pipeline, path: str) -> None:
    """Serialise pipeline to disk with pickle.

    Args:
        pipeline: fitted sklearn Pipeline
        path:     file path (e.g. "saved_models/best_model.pkl")
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"Saved pipeline → {path}")


def load_pipeline(path: str) -> Pipeline:
    """Load a previously saved pipeline.

    Args:
        path: file path to the .pkl file

    Returns:
        fitted sklearn Pipeline ready for predict()
    """
    with open(path, "rb") as f:
        pipeline = pickle.load(f)
    print(f"Loaded pipeline ← {path}")
    return pipeline


def predict_new(pipeline: Pipeline, raw_rows: pd.DataFrame) -> np.ndarray:
    """Run inference on new, unseen data rows.

    The pipeline handles all preprocessing automatically — caller passes
    the same raw format as the training data.

    Args:
        pipeline: fitted sklearn Pipeline
        raw_rows: DataFrame with same columns as training features

    Returns:
        (n_rows,) array of predicted labels
    """
    return pipeline.predict(raw_rows)
