"""module_28/serving.py — Model Serving, Versioning & Drift Detection.

Covers:
  - FastAPI-style inference server (simulated without requiring FastAPI)
  - Model versioning with metadata registry
  - Data drift detection (PSI — Population Stability Index)
  - Prediction monitoring / logging
  - A/B testing setup for model comparisons

Run standalone: python serving.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import json
import time
import pickle
import hashlib
import numpy as np
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RANDOM_SEED = 42
MODELS_DIR  = "model_registry"
LOGS_DIR    = "prediction_logs"
PLOTS_DIR   = "plots"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR,   exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)
rng = np.random.default_rng(RANDOM_SEED)


# ─────────────────────────────────────────────────────────────────────────────
# 1. FASTAPI-STYLE INFERENCE PATTERN (no actual HTTP server)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PredictRequest:
    """Schema for an inference request.

    In production this would be validated by Pydantic + FastAPI.
    Here we simulate the same interface in plain Python.
    """
    features: List[float]
    request_id: Optional[str] = None


@dataclass
class PredictResponse:
    """Schema for an inference response."""
    request_id:  str
    prediction:  int
    probability: float
    model_version: str
    latency_ms:  float


class InferenceServer:
    """Minimal inference server — simulates what FastAPI /predict does.

    In production:
      POST /predict HTTP/1.1
      Body: {"features": [...], "request_id": "abc123"}

    Here we call .predict() directly, mirroring the same logic.
    """

    def __init__(self, model, model_version: str):
        self.model         = model
        self.model_version = model_version
        self._log: List[Dict] = []

    def predict(self, req: PredictRequest) -> PredictResponse:
        """Run inference on a single request, log result.

        Args:
            req: PredictRequest with feature vector

        Returns:
            PredictResponse with prediction, probability, latency
        """
        t0      = time.perf_counter()
        X       = np.array(req.features).reshape(1, -1)
        prob    = float(self.model.predict_proba(X)[0, 1])
        pred    = int(prob >= 0.5)
        latency = (time.perf_counter() - t0) * 1000   # ms

        rid  = req.request_id or hashlib.md5(
            str(time.time()).encode()).hexdigest()[:8]

        resp = PredictResponse(
            request_id    = rid,
            prediction    = pred,
            probability   = round(prob, 4),
            model_version = self.model_version,
            latency_ms    = round(latency, 3),
        )
        self._log.append({
            "timestamp":     datetime.utcnow().isoformat(),
            "features":      req.features,
            **asdict(resp),
        })
        return resp

    def batch_predict(self, requests: List[PredictRequest]) -> List[PredictResponse]:
        """Run inference on a batch of requests."""
        return [self.predict(r) for r in requests]

    def save_log(self, path: str):
        with open(path, "w") as f:
            json.dump(self._log, f, indent=2)
        print(f"  Prediction log saved → {path}")


def demo_inference_server():
    """Build a model and simulate inference requests."""
    print("── 1. FastAPI-Style Inference Server ───────────────")
    from sklearn.ensemble import GradientBoostingClassifier

    # Train a simple model
    X = rng.standard_normal((200, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    model = GradientBoostingClassifier(random_state=RANDOM_SEED, n_estimators=50)
    model.fit(X, y)

    server = InferenceServer(model, model_version="v1.2.0")

    # Simulate requests
    reqs = [
        PredictRequest(features=[0.5, 1.2, -0.3, 0.8], request_id="req_001"),
        PredictRequest(features=[-1.0, -0.5, 0.1, 0.2], request_id="req_002"),
        PredictRequest(features=[2.0, 0.9, 0.0, -1.5], request_id="req_003"),
    ]

    print(f"  {'Request ID':<12} {'Pred':>6} {'Prob':>8} {'Latency (ms)':>14}")
    for req in reqs:
        resp = server.predict(req)
        print(f"  {resp.request_id:<12} {resp.prediction:>6} "
              f"{resp.probability:>8.4f} {resp.latency_ms:>14.3f}")

    log_path = f"{LOGS_DIR}/predictions.json"
    server.save_log(log_path)
    return server


# ─────────────────────────────────────────────────────────────────────────────
# 2. MODEL VERSION REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelMetadata:
    """Metadata for a registered model version."""
    version:       str
    model_class:   str
    trained_at:    str
    dataset_rows:  int
    val_accuracy:  float
    val_roc_auc:   float
    file_path:     str
    is_production: bool = False


class ModelRegistry:
    """Simple file-based model registry.

    Tracks all trained model versions with their performance metrics.
    Production systems use MLflow, Weights & Biases, or SageMaker Model Registry.
    The interface here mirrors the same concepts.
    """

    def __init__(self, registry_dir: str = MODELS_DIR):
        self.registry_dir  = registry_dir
        self.index_path    = os.path.join(registry_dir, "registry.json")
        self._versions: Dict[str, ModelMetadata] = {}
        self._load_index()

    def _load_index(self):
        if os.path.exists(self.index_path):
            with open(self.index_path) as f:
                raw = json.load(f)
            self._versions = {k: ModelMetadata(**v) for k, v in raw.items()}

    def _save_index(self):
        with open(self.index_path, "w") as f:
            json.dump({k: asdict(v) for k, v in self._versions.items()}, f, indent=2)

    def register(self, model, metadata: ModelMetadata):
        """Save model artifact and register its metadata.

        Args:
            model:    fitted sklearn model
            metadata: ModelMetadata with version, metrics, etc.
        """
        # Persist model artifact
        with open(metadata.file_path, "wb") as f:
            pickle.dump(model, f)

        self._versions[metadata.version] = metadata
        self._save_index()
        print(f"  Registered model {metadata.version} "
              f"(val_acc={metadata.val_accuracy:.4f})")

    def promote_to_production(self, version: str):
        """Mark a version as the production model (demotes previous)."""
        for v, meta in self._versions.items():
            meta.is_production = (v == version)
        self._save_index()
        print(f"  Promoted {version} to PRODUCTION")

    def get_production_model(self):
        """Load and return the current production model."""
        for meta in self._versions.values():
            if meta.is_production:
                with open(meta.file_path, "rb") as f:
                    return pickle.load(f), meta
        raise RuntimeError("No production model registered")

    def list_versions(self):
        print(f"\n  {'Version':<12} {'Val Acc':>10} {'ROC AUC':>10} {'Prod':>6}")
        for meta in sorted(self._versions.values(), key=lambda m: m.version):
            prod = "★" if meta.is_production else ""
            print(f"  {meta.version:<12} {meta.val_accuracy:>10.4f} "
                  f"{meta.val_roc_auc:>10.4f} {prod:>6}")


def demo_model_registry():
    """Register two model versions and promote the better one."""
    print("\n── 2. Model Version Registry ───────────────────────")
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, roc_auc_score

    X, y = make_classification(n_samples=500, n_features=10, random_state=RANDOM_SEED)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED)

    registry = ModelRegistry()

    for version, model_cls, kwargs in [
        ("v1.0.0", LogisticRegression,    {"max_iter": 500}),
        ("v1.1.0", RandomForestClassifier, {"n_estimators": 100, "random_state": RANDOM_SEED}),
    ]:
        model = model_cls(**kwargs).fit(X_tr, y_tr)
        acc   = accuracy_score(y_te, model.predict(X_te))
        auc   = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])
        path  = os.path.join(MODELS_DIR, f"model_{version}.pkl")

        registry.register(model, ModelMetadata(
            version      = version,
            model_class  = model_cls.__name__,
            trained_at   = datetime.utcnow().isoformat(),
            dataset_rows = len(X_tr),
            val_accuracy = acc,
            val_roc_auc  = auc,
            file_path    = path,
        ))

    # Promote better model to production
    best = max(registry._versions.values(), key=lambda m: m.val_roc_auc)
    registry.promote_to_production(best.version)
    registry.list_versions()

    prod_model, prod_meta = registry.get_production_model()
    print(f"\n  Production model: {prod_meta.version} ({prod_meta.model_class})")
    return registry


# ─────────────────────────────────────────────────────────────────────────────
# 3. DATA DRIFT DETECTION (PSI — Population Stability Index)
# ─────────────────────────────────────────────────────────────────────────────

def population_stability_index(reference: np.ndarray,
                                 current: np.ndarray,
                                 bins: int = 10) -> float:
    """Compute PSI to detect feature distribution drift.

    PSI = sum( (actual% - expected%) * ln(actual% / expected%) )

    Interpretation:
      PSI < 0.1  → no significant drift (model is stable)
      PSI 0.1-0.2 → moderate drift (monitor)
      PSI > 0.2  → major drift (retrain model)

    Args:
        reference: distribution at training time
        current:   distribution in production now
        bins:      number of histogram bins

    Returns:
        PSI score
    """
    eps   = 1e-9   # prevent log(0)
    edges = np.percentile(reference, np.linspace(0, 100, bins + 1))
    edges = np.unique(edges)   # deduplicate for concentrated distributions

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current,   bins=edges)

    ref_pct = ref_counts / ref_counts.sum() + eps
    cur_pct = cur_counts / cur_counts.sum() + eps

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def demo_drift_detection():
    """Simulate training data vs drifted production data."""
    print("\n── 3. Data Drift Detection (PSI) ───────────────────")
    n = 1000

    # Training distribution (clean)
    train_feat = rng.normal(loc=0.0, scale=1.0, size=n)

    # Production scenarios
    scenarios = {
        "No drift":       rng.normal(loc=0.0, scale=1.0, size=n),
        "Mild drift":     rng.normal(loc=0.5, scale=1.2, size=n),
        "Major drift":    rng.normal(loc=2.0, scale=2.0, size=n),
        "Distribution shift": rng.exponential(scale=1.0, size=n),
    }

    print(f"  {'Scenario':<22} {'PSI':>8}  Verdict")
    for name, prod_feat in scenarios.items():
        psi = population_stability_index(train_feat, prod_feat)
        verdict = ("✓ stable" if psi < 0.1 else
                   "⚠ monitor" if psi < 0.2 else
                   "✗ RETRAIN")
        print(f"  {name:<22} {psi:>8.4f}  {verdict}")

    # Visualise distributions
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), facecolor="#0d1117")
    colours = ["#3fb950", "#f0883e", "#f85149", "#bc8cff"]

    for ax, (name, prod_feat), colour in zip(axes.flat, scenarios.items(), colours):
        ax.set_facecolor("#161b22")
        ax.hist(train_feat, bins=30, alpha=0.6, color="#58a6ff",
                label="Training", density=True)
        ax.hist(prod_feat,  bins=30, alpha=0.6, color=colour,
                label="Production", density=True)
        psi = population_stability_index(train_feat, prod_feat)
        ax.set_title(f"{name}  (PSI={psi:.3f})", color="#e6edf3", fontsize=9)
        ax.legend(fontsize=8, facecolor="#0d1117", labelcolor="#e6edf3")
        ax.tick_params(colors="#8b949e")
        ax.spines[:].set_color("#30363d")

    plt.suptitle("Data Drift Detection — Training vs Production", color="#e6edf3")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/drift_detection.png", dpi=120,
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  Saved → {PLOTS_DIR}/drift_detection.png")


# ─────────────────────────────────────────────────────────────────────────────
# 4. A/B TEST SETUP FOR MODEL COMPARISONS
# ─────────────────────────────────────────────────────────────────────────────

class ABTestRouter:
    """Route inference requests between model A and model B.

    Traffic split: alpha% → model_a, (1-alpha)% → model_b.
    Collects outcomes to determine which model is better.

    In production:
    - Route by user_id hash for deterministic assignment (same user
      always sees the same model — avoids confusion).
    - Log predictions and business outcomes (click, purchase, etc.).
    - Run a statistical test (chi-square, t-test) on accumulated outcomes.
    """

    def __init__(self, model_a, model_b, alpha: float = 0.5, seed: int = RANDOM_SEED):
        self.model_a = model_a
        self.model_b = model_b
        self.alpha   = alpha
        self._rng    = np.random.default_rng(seed)
        self._logs   = {"A": [], "B": []}

    def route(self, X: np.ndarray, true_label: int = None) -> dict:
        """Route a request, log prediction and optional outcome."""
        arm  = "A" if self._rng.random() < self.alpha else "B"
        model = self.model_a if arm == "A" else self.model_b

        prob = float(model.predict_proba(X.reshape(1, -1))[0, 1])
        pred = int(prob >= 0.5)

        result = {"arm": arm, "pred": pred, "prob": prob}
        if true_label is not None:
            result["correct"] = int(pred == true_label)
        self._logs[arm].append(result)
        return result

    def summary(self):
        """Print accuracy comparison between model A and B."""
        print("\n  A/B Test Results:")
        for arm, logs in self._logs.items():
            n       = len(logs)
            correct = [l.get("correct", 0) for l in logs]
            acc     = sum(correct) / n if n > 0 else 0.0
            avg_p   = np.mean([l["prob"] for l in logs])
            print(f"    Model {arm}: n={n:>5}  accuracy={acc:.4f}  avg_prob={avg_p:.4f}")


def demo_ab_test():
    """Simulate A/B test between two models."""
    print("\n── 4. A/B Test Router ──────────────────────────────")
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=800, n_features=8, random_state=RANDOM_SEED)
    X_tr, X_te, y_tr, y_te = X[:600], X[600:], y[:600], y[600:]

    model_a = LogisticRegression(max_iter=500).fit(X_tr, y_tr)
    model_b = GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_SEED
                                          ).fit(X_tr, y_tr)

    router = ABTestRouter(model_a, model_b, alpha=0.5)
    for xi, yi in zip(X_te, y_te):
        router.route(xi, true_label=yi)

    router.summary()
    print("\n  In production: continue routing until a chi-square test")
    print("  confirms statistical significance (p < 0.05), then promote winner.")


def main():
    print("=" * 54)
    print("MODULE 28 — Model Serving, Versioning & Drift")
    print("=" * 54)
    demo_inference_server()
    demo_model_registry()
    demo_drift_detection()
    demo_ab_test()
    print("\nDone.")


if __name__ == "__main__":
    main()
