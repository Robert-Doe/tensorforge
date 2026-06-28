"""module_21/regularisation_advanced.py — Advanced Regularisation Techniques.

Covers:
  - L1 weight sparsity (why L1 drives weights to exactly zero)
  - Elastic net (L1 + L2 combined)
  - Label smoothing (regularise the targets, not the weights)
  - Data augmentation as implicit regularisation
  - Weight decay vs L2 regularisation (they're the same for SGD but not Adam)

Run standalone: python regularisation_advanced.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
PLOTS_DIR   = "plots"
FIGURE_DPI  = 120
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
os.makedirs(PLOTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. L1 SPARSITY — why L1 zeros out weights
# ─────────────────────────────────────────────────────────────────────────────

def demo_l1_sparsity():
    """Show that L1 penalty creates sparse weight vectors while L2 does not."""
    print("── 1. L1 Sparsity vs L2 Shrinkage ─────────────────")

    # Build an MLP and apply L1 vs L2 penalty manually to see the effect
    class TinyMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(20, 32)
            self.fc2 = nn.Linear(32, 1)
            self.relu = nn.ReLU()
        def forward(self, x):
            return torch.sigmoid(self.fc2(self.relu(self.fc1(x))))

    X, y = make_classification(n_samples=500, n_features=20, random_state=RANDOM_SEED)
    X    = torch.tensor(X, dtype=torch.float32)
    y    = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    results = {}
    for name, reg_fn in [
        ("No reg",     lambda m: 0.0),
        ("L2 (λ=1e-2)", lambda m: 1e-2 * sum(p.pow(2).sum() for p in m.parameters())),
        ("L1 (λ=1e-2)", lambda m: 1e-2 * sum(p.abs().sum() for p in m.parameters())),
    ]:
        model = TinyMLP()
        opt   = optim.Adam(model.parameters(), lr=1e-3)
        bce   = nn.BCELoss()

        for _ in range(200):
            opt.zero_grad()
            loss = bce(model(X), y) + reg_fn(model)
            loss.backward()
            opt.step()

        # Collect all fc1 weights
        w = model.fc1.weight.detach().numpy().flatten()
        results[name] = w

    # Sparsity = fraction of weights with |w| < threshold
    thresh = 0.01
    print(f"  Sparsity threshold: |w| < {thresh}")
    print(f"  {'Regulariser':<20} {'Mean |w|':>10} {'Sparsity%':>12}")
    for name, w in results.items():
        sparse = (np.abs(w) < thresh).mean() * 100
        print(f"  {name:<20} {np.abs(w).mean():>10.4f} {sparse:>11.1f}%")

    # Plot weight distributions
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), facecolor="#0d1117")
    colours = ["#58a6ff", "#3fb950", "#f85149"]
    for ax, (name, w), colour in zip(axes, results.items(), colours):
        ax.set_facecolor("#161b22")
        ax.hist(w, bins=30, color=colour, edgecolor="#30363d", alpha=0.85)
        ax.set_title(name, color="#e6edf3", fontsize=10)
        ax.axvline(0, color="#ffffff", linewidth=1, linestyle="--")
        ax.tick_params(colors="#8b949e")
        ax.spines[:].set_color("#30363d")
    plt.suptitle("Weight Distributions: L1 vs L2 vs No Reg", color="#e6edf3")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/l1_l2_weights.png", dpi=FIGURE_DPI,
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved → {PLOTS_DIR}/l1_l2_weights.png")
    print("\n  WHY L1 creates exact zeros:")
    print("  L2 gradient: 2w → becomes 0 only as w→0 (asymptotic)")
    print("  L1 gradient: sign(w) → constant pull regardless of magnitude")
    print("  → Sub-differential at w=0 allows optimal point exactly at 0")


# ─────────────────────────────────────────────────────────────────────────────
# 2. ELASTIC NET — L1 + L2 combined
# ─────────────────────────────────────────────────────────────────────────────

def demo_elastic_net():
    """Compare Lasso, Ridge, and Elastic Net on a regression task."""
    print("\n── 2. Elastic Net (L1 + L2) ─────────────────────────")
    from sklearn.linear_model import Lasso, Ridge, ElasticNet
    from sklearn.metrics import mean_squared_error

    # 100 features but only 10 are truly predictive
    rng = np.random.default_rng(RANDOM_SEED)
    n, p = 300, 100
    X    = rng.standard_normal((n, p))
    true_coef = np.zeros(p)
    true_coef[:10] = rng.uniform(1, 3, 10)   # first 10 features matter
    y     = X @ true_coef + rng.standard_normal(n) * 0.5

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                               random_state=RANDOM_SEED)

    print(f"  {'Model':<20} {'Test MSE':>10} {'Nonzero coefs':>15}")
    for name, model in [
        ("Lasso (L1)",      Lasso(alpha=0.1, max_iter=5000)),
        ("Ridge (L2)",      Ridge(alpha=1.0)),
        ("ElasticNet (L1+L2)", ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000)),
    ]:
        model.fit(X_tr, y_tr)
        mse  = mean_squared_error(y_te, model.predict(X_te))
        nnz  = (np.abs(model.coef_) > 1e-6).sum()
        print(f"  {name:<20} {mse:>10.4f} {nnz:>15}")

    print("\n  ElasticNet advantages over pure Lasso:")
    print("  - When p >> n, Lasso selects at most n features (ElasticNet doesn't)")
    print("  - Handles correlated groups better (tends to select all or none)")
    print("  - l1_ratio=1 → Lasso; l1_ratio=0 → Ridge; 0<l1_ratio<1 → ElasticNet")


# ─────────────────────────────────────────────────────────────────────────────
# 3. LABEL SMOOTHING
# ─────────────────────────────────────────────────────────────────────────────

class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy with label smoothing regularisation.

    Instead of training targets [0, 1], use [ε/K, 1-ε+ε/K]
    where K = number of classes and ε = smoothing factor.

    Why it works:
    - Hard labels push logits to ±∞ — the model is incentivised to be
      overconfident even on wrong examples.
    - Soft labels cap the log-probability at a finite value, preventing
      the model from becoming too certain.
    - Improves calibration (predicted probabilities match actual frequencies).
    - Used in: Inception-v4, BERT, ViT, many modern classifiers.

    Args:
        smoothing: ε value (typical: 0.05–0.2)
        classes:   number of output classes
    """

    def __init__(self, smoothing: float = 0.1, classes: int = 10):
        super().__init__()
        self.smoothing = smoothing
        self.classes   = classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute smoothed cross-entropy loss.

        Args:
            logits:  (B, C) raw logits
            targets: (B,) integer class indices

        Returns:
            scalar loss
        """
        log_probs = torch.log_softmax(logits, dim=-1)

        # Smooth labels: ε/K for all classes, (1-ε+ε/K) for the true class
        smooth = self.smoothing / self.classes
        with torch.no_grad():
            soft_targets = torch.full_like(log_probs, smooth)
            soft_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing + smooth)

        # NLL loss against soft targets
        return -(soft_targets * log_probs).sum(dim=-1).mean()


def demo_label_smoothing():
    """Show effect of label smoothing on loss value and gradient magnitude."""
    print("\n── 3. Label Smoothing ───────────────────────────────")
    B, C = 16, 10
    logits  = torch.randn(B, C)
    targets = torch.randint(0, C, (B,))

    standard  = nn.CrossEntropyLoss()(logits, targets)
    smoothed  = LabelSmoothingCrossEntropy(smoothing=0.1, classes=C)(logits, targets)

    print(f"  Standard CE loss:     {standard.item():.4f}")
    print(f"  Label-smoothed loss:  {smoothed.item():.4f}")
    print(f"  (Loss is higher with smoothing — harder target to achieve)")

    # Compare gradient norms
    logits_a = logits.detach().requires_grad_(True)
    logits_b = logits.detach().requires_grad_(True)
    nn.CrossEntropyLoss()(logits_a, targets).backward()
    LabelSmoothingCrossEntropy(0.1, C)(logits_b, targets).backward()

    print(f"  Gradient norm (standard):  {logits_a.grad.norm().item():.4f}")
    print(f"  Gradient norm (smoothed):  {logits_b.grad.norm().item():.4f}")
    print("  Smaller gradient with smoothing → slower but more stable training.")
    print("  Typical ε: 0.1 (standard), 0.05 (gentle), 0.2 (aggressive).")


# ─────────────────────────────────────────────────────────────────────────────
# 4. DATA AUGMENTATION AS REGULARISATION
# ─────────────────────────────────────────────────────────────────────────────

def demo_data_augmentation():
    """Explain augmentation as implicit regularisation with examples."""
    print("\n── 4. Data Augmentation — Implicit Regularisation ───")
    print("""
  Data augmentation is the most powerful regulariser for image models.
  It creates new training examples from existing ones by applying
  label-preserving transformations.

  WHY IT REGULARISES:
  ────────────────────────────────────────────────────────────────
  - Forces the model to learn invariances (rotation, scale, colour)
    rather than memorising exact pixel patterns.
  - Effectively multiplies dataset size without new data collection.
  - Reduces the gap between training distribution and real-world
    test distribution.

  COMMON AUGMENTATIONS (torchvision.transforms):
  ────────────────────────────────────────────────────────────────
  transforms.RandomHorizontalFlip(p=0.5)     # free for most tasks
  transforms.RandomCrop(32, padding=4)        # commonly used for CIFAR
  transforms.ColorJitter(brightness, contrast, saturation, hue)
  transforms.RandomRotation(degrees=15)
  transforms.RandomGrayscale(p=0.1)
  transforms.RandomErasing(p=0.5, scale=(0.02, 0.33))  # Cutout

  ADVANCED (torchvision >= 0.12):
  ────────────────────────────────────────────────────────────────
  transforms.AutoAugment(policy=AutoAugmentPolicy.CIFAR10)
  transforms.TrivialAugmentWide()   # used in modern ViT training
  transforms.RandAugment(num_ops=2, magnitude=9)

  TASK-SPECIFIC CHOICES:
  ────────────────────────────────────────────────────────────────
  Medical imaging:  mild rotation + brightness only (no mirror for asymmetric organs)
  Text detection:   no horizontal flip (reverses text)
  Satellite:        heavy rotation + flip (no canonical orientation)
  Faces:            horizontal flip OK, no vertical (face upside-down = wrong class)

  MIXUP (Zhang et al. 2018):
  ────────────────────────────────────────────────────────────────
  Mix two examples: x_mix = λ*x_i + (1-λ)*x_j
  Mix their labels: y_mix = λ*y_i + (1-λ)*y_j
  Forces model to learn linear interpolations of class representations.
  """)

    # Minimal Mixup demonstration
    rng = np.random.default_rng(RANDOM_SEED)
    B, C_in = 4, 3
    X = torch.randn(B, C_in, 8, 8)
    y = torch.eye(10)[[0, 1, 2, 3]]   # one-hot labels (10 classes)

    lam   = rng.beta(0.2, 0.2)   # λ ~ Beta(α, α), α=0.2 typical
    idx   = torch.randperm(B)
    X_mix = lam * X + (1 - lam) * X[idx]
    y_mix = lam * y + (1 - lam) * y[idx]
    print(f"  Mixup demo: λ={lam:.3f}")
    print(f"  X_mix shape: {X_mix.shape}")
    print(f"  y_mix sample (two classes blended):")
    print(f"    {y_mix[0].numpy().round(3)}")
    print("  Non-zero entries: soft targets between two classes.")


# ─────────────────────────────────────────────────────────────────────────────
# 5. WEIGHT DECAY vs L2 REGULARISATION
# ─────────────────────────────────────────────────────────────────────────────

def demo_weight_decay_vs_l2():
    """Explain why weight_decay in Adam ≠ L2 regularisation."""
    print("\n── 5. Weight Decay vs L2 Regularisation ─────────────")
    print("""
  For SGD:  weight_decay and L2 penalty are EQUIVALENT
    L2 gradient: ∂L/∂w += λ*w
    SGD update:  w ← w - η*(∂L/∂w + λ*w) = (1-ηλ)*w - η*∂L/∂w
    weight_decay in SGD does: w ← (1-ηλ)*w - η*∂L/∂w
    → Same thing.

  For Adam:  they are DIFFERENT (Loshchilov & Hutter, 2017 — "Decoupled Weight Decay")
    Adam L2 penalty: adds λ*w to gradient BEFORE Adam's m/v moment estimation
      → λ*w gets absorbed into m_t and v_t, diluted by the adaptive scaling
      → Regularisation strength depends on gradient statistics

    Adam weight_decay: subtracts λ*w from weights AFTER Adam's update
      → Regularisation is always exactly λ*w, independent of m_t/v_t
      → Stronger and more predictable regularisation

  AdamW = Adam + decoupled weight decay
  ─────────────────────────────────────────────────────────────────
  optim.AdamW(params, lr=1e-3, weight_decay=0.01)   ← CORRECT for Adam
  optim.Adam(params, lr=1e-3, weight_decay=0.01)    ← L2 (diluted, less effective)

  Modern recommendation: always use AdamW, not Adam, when you want regularisation.
  Hugging Face Transformers uses AdamW with weight_decay=0.01 by default.
  """)

    D = 32
    linear_adam  = nn.Linear(D, 1)
    linear_adamw = nn.Linear(D, 1)

    # Same initial weights
    with torch.no_grad():
        linear_adamw.weight.copy_(linear_adam.weight)
        linear_adamw.bias.copy_(linear_adam.bias)

    opt_adam  = optim.Adam( linear_adam.parameters(),  lr=1e-3, weight_decay=1e-2)
    opt_adamw = optim.AdamW(linear_adamw.parameters(), lr=1e-3, weight_decay=1e-2)

    X = torch.randn(16, D)
    y = torch.zeros(16, 1)
    loss_fn = nn.MSELoss()

    for _ in range(50):
        for opt, model in [(opt_adam, linear_adam), (opt_adamw, linear_adamw)]:
            opt.zero_grad()
            loss_fn(model(X), y).backward()
            opt.step()

    w_adam  = linear_adam.weight.data.norm().item()
    w_adamw = linear_adamw.weight.data.norm().item()
    print(f"  After 50 steps with weight_decay=0.01:")
    print(f"  Adam  weight norm: {w_adam:.4f}")
    print(f"  AdamW weight norm: {w_adamw:.4f}")
    print(f"  AdamW applies stronger, cleaner regularisation.")


def main():
    print("=" * 54)
    print("MODULE 21 — Advanced Regularisation Techniques")
    print("=" * 54)
    demo_l1_sparsity()
    demo_elastic_net()
    demo_label_smoothing()
    demo_data_augmentation()
    demo_weight_decay_vs_l2()
    print("\nDone.")


if __name__ == "__main__":
    main()
