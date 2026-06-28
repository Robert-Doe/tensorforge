"""module_12/ensemble_advanced.py — Advanced Ensemble Methods.

Covers:
  - Stacking (meta-learning with out-of-fold predictions)
  - Voting classifier (hard and soft voting)
  - AdaBoost from scratch (weight-based boosting)
  - Extra Trees vs Random Forest (randomness spectrum)

Run standalone: python ensemble_advanced.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, AdaBoostClassifier, VotingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

RANDOM_SEED = 42
PLOTS_DIR   = "plots"
FIGURE_DPI  = 120
rng = np.random.default_rng(RANDOM_SEED)
os.makedirs(PLOTS_DIR, exist_ok=True)


def make_data():
    X, y = make_classification(n_samples=1200, n_features=20, n_informative=10,
                               random_state=RANDOM_SEED)
    return train_test_split(X, y, test_size=0.25, random_state=RANDOM_SEED)


# ─────────────────────────────────────────────────────────────────────────────
# 1. STACKING from scratch — out-of-fold meta-features
# ─────────────────────────────────────────────────────────────────────────────

class StackingClassifier:
    """Two-level stacking ensemble.

    Level 0: base learners trained on folds; their out-of-fold predictions
             form the meta-feature matrix.
    Level 1: meta-learner trained on the meta-feature matrix.

    Using out-of-fold (OOF) predictions prevents leakage: each training sample's
    meta-feature is predicted by a model that never saw that sample.

    Attributes:
        base_models: list of sklearn estimators
        meta_model:  sklearn estimator for second level
        n_folds:     cross-validation folds for OOF generation
    """

    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models  = base_models
        self.meta_model   = meta_model
        self.n_folds      = n_folds
        self._fitted_base = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Generate OOF predictions and train meta-learner.

        Args:
            X: training features
            y: training labels
        """
        n = len(y)
        # Collect OOF probability predictions from each base model
        meta_features = np.zeros((n, len(self.base_models)))
        kf = StratifiedKFold(n_splits=self.n_folds, shuffle=True,
                              random_state=RANDOM_SEED)

        for i, model in enumerate(self.base_models):
            for tr_idx, va_idx in kf.split(X, y):
                clone = type(model)(**model.get_params())
                clone.fit(X[tr_idx], y[tr_idx])
                meta_features[va_idx, i] = clone.predict_proba(X[va_idx])[:, 1]

        # Train meta-learner on the OOF meta-features
        self.meta_model.fit(meta_features, y)

        # Refit all base models on the full training set for test-time inference
        self._fitted_base = []
        for model in self.base_models:
            m = type(model)(**model.get_params())
            m.fit(X, y)
            self._fitted_base.append(m)

    def predict(self, X: np.ndarray) -> np.ndarray:
        meta_features = np.column_stack([
            m.predict_proba(X)[:, 1] for m in self._fitted_base
        ])
        return self.meta_model.predict(meta_features)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        meta_features = np.column_stack([
            m.predict_proba(X)[:, 1] for m in self._fitted_base
        ])
        return self.meta_model.predict_proba(meta_features)


def demo_stacking(X_tr, X_te, y_tr, y_te):
    """Compare stacking vs individual models and simple majority vote."""
    print("── 1. Stacking Ensemble ────────────────────────────")
    base_models = [
        RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED),
        GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_SEED),
        LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
    ]
    meta = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    stacker = StackingClassifier(base_models, meta, n_folds=5)
    stacker.fit(X_tr, y_tr)
    stack_acc = accuracy_score(y_te, stacker.predict(X_te))

    # Individual baselines
    print(f"  {'Model':<35} {'Test Acc':>10}")
    for model in base_models:
        model.fit(X_tr, y_tr)
        acc = accuracy_score(y_te, model.predict(X_te))
        print(f"  {type(model).__name__:<35} {acc:.4f}")
    print(f"  {'Stacking (LR meta-learner)':<35} {stack_acc:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. VOTING CLASSIFIERS — hard and soft
# ─────────────────────────────────────────────────────────────────────────────

def demo_voting(X_tr, X_te, y_tr, y_te):
    """Hard voting (majority label) vs soft voting (average probability)."""
    print("\n── 2. Voting Classifiers ───────────────────────────")
    estimators = [
        ("rf",  RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)),
        ("lr",  LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)),
        ("svc", SVC(probability=True, random_state=RANDOM_SEED)),
    ]

    for voting in ["hard", "soft"]:
        vc = VotingClassifier(estimators=estimators, voting=voting)
        vc.fit(X_tr, y_tr)
        acc = accuracy_score(y_te, vc.predict(X_te))
        print(f"  {voting.capitalize()} voting: {acc:.4f}")

    print("  Soft voting usually outperforms hard: it uses probability confidence,")
    print("  not just the discrete vote. A model that's 99% sure counts more than one")
    print("  that's 51% sure — soft voting respects this.")


# ─────────────────────────────────────────────────────────────────────────────
# 3. ADABOOST FROM SCRATCH
# ─────────────────────────────────────────────────────────────────────────────

class AdaBoostScratch:
    """AdaBoost with decision stumps (depth-1 trees).

    At each round t:
      1. Train a weak learner h_t on the current sample weights.
      2. Compute error_t = sum(w_i * (h_t(x_i) != y_i)).
      3. Compute learner weight alpha_t = 0.5 * log((1-err)/err).
      4. Update sample weights: up-weight misclassified, down-weight correct.
      5. Normalise weights.
    Final prediction: sign(sum_t alpha_t * h_t(x)).
    """

    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train AdaBoost.

        Args:
            X: features
            y: labels in {-1, +1}
        """
        n = len(y)
        w = np.ones(n) / n          # uniform initial weights
        self.alphas    = []
        self.stumps    = []
        self.train_errors = []

        for _ in range(self.n_estimators):
            stump = DecisionTreeClassifier(max_depth=1, random_state=RANDOM_SEED)
            stump.fit(X, y, sample_weight=w)
            pred = stump.predict(X)

            err   = np.sum(w * (pred != y))
            err   = np.clip(err, 1e-10, 1 - 1e-10)   # avoid log(0)
            alpha = 0.5 * np.log((1 - err) / err)

            # Update weights: e^{-alpha} for correct, e^{alpha} for wrong
            w = w * np.exp(-alpha * y * pred)
            w = w / w.sum()

            self.alphas.append(alpha)
            self.stumps.append(stump)
            self.train_errors.append(err)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Weighted majority vote of all stumps."""
        agg = sum(alpha * stump.predict(X)
                  for alpha, stump in zip(self.alphas, self.stumps))
        return np.sign(agg)


def demo_adaboost(X_tr, X_te, y_tr, y_te):
    """Scratch AdaBoost vs sklearn implementation."""
    print("\n── 3. AdaBoost from Scratch ────────────────────────")
    # Convert to {-1, +1} labels for our scratch implementation
    y_tr_pm = 2 * y_tr - 1
    y_te_pm = 2 * y_te - 1

    ada = AdaBoostScratch(n_estimators=100)
    ada.fit(X_tr, y_tr_pm)
    pred_scratch = ada.predict(X_te)
    acc_scratch  = (pred_scratch == y_te_pm).mean()

    ada_sk = AdaBoostClassifier(n_estimators=100, random_state=RANDOM_SEED)
    ada_sk.fit(X_tr, y_tr)
    acc_sk = ada_sk.score(X_te, y_te)

    print(f"  AdaBoost (scratch): {acc_scratch:.4f}")
    print(f"  AdaBoost (sklearn): {acc_sk:.4f}")

    # Plot training error over rounds
    fig, ax = plt.subplots(figsize=(7, 4), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    ax.plot(ada.train_errors, color="#f85149", linewidth=1.5)
    ax.set_xlabel("Boosting round", color="#e6edf3")
    ax.set_ylabel("Weighted error", color="#e6edf3")
    ax.set_title("AdaBoost — Weighted Error per Round", color="#e6edf3")
    ax.tick_params(colors="#e6edf3"); ax.spines[:].set_color("#30363d")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/adaboost_error.png", dpi=FIGURE_DPI,
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved → {PLOTS_DIR}/adaboost_error.png")


# ─────────────────────────────────────────────────────────────────────────────
# 4. EXTRA TREES vs RANDOM FOREST — randomness spectrum
# ─────────────────────────────────────────────────────────────────────────────

def demo_extra_trees(X_tr, X_te, y_tr, y_te):
    """Compare RF vs ExtraTrees on bias-variance."""
    print("\n── 4. Extra Trees vs Random Forest ─────────────────")
    results = {}
    for name, Model, kwargs in [
        ("Decision Tree",  DecisionTreeClassifier, {"random_state": RANDOM_SEED}),
        ("Random Forest",  RandomForestClassifier, {"n_estimators": 100, "random_state": RANDOM_SEED}),
        ("Extra Trees",    ExtraTreesClassifier,   {"n_estimators": 100, "random_state": RANDOM_SEED}),
    ]:
        m = Model(**kwargs)
        m.fit(X_tr, y_tr)
        results[name] = m.score(X_te, y_te)
        print(f"  {name:<20}: {results[name]:.4f}")

    print("\n  Extra Trees splits on random thresholds (not the best threshold).")
    print("  This adds more randomness than RF, reducing variance further,")
    print("  but increasing bias slightly. Often matches RF in accuracy,")
    print("  trains faster (no threshold optimisation per split).")


def main():
    print("=" * 54)
    print("MODULE 12 — Advanced Ensemble Methods")
    print("=" * 54)
    X_tr, X_te, y_tr, y_te = make_data()
    demo_stacking(X_tr, X_te, y_tr, y_te)
    demo_voting(X_tr, X_te, y_tr, y_te)
    demo_adaboost(X_tr, X_te, y_tr, y_te)
    demo_extra_trees(X_tr, X_te, y_tr, y_te)
    print("\nDone.")


if __name__ == "__main__":
    main()
