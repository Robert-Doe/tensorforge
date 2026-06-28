"""module_19/activations_advanced.py — Advanced Activation Functions.

Covers:
  - GELU (Gaussian Error Linear Unit) — used in BERT, GPT
  - Swish / SiLU — used in EfficientNet, LLaMA
  - Mish — smooth self-regularised activation
  - Dying ReLU problem and solutions (Leaky ReLU, PReLU, ELU)
  - Softmax: temperature, numerical stability, log-sum-exp trick

Run standalone: python activations_advanced.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RANDOM_SEED = 42
PLOTS_DIR   = "plots"
FIGURE_DPI  = 120
torch.manual_seed(RANDOM_SEED)
os.makedirs(PLOTS_DIR, exist_ok=True)

X_RANGE = np.linspace(-4, 4, 400)


# ─────────────────────────────────────────────────────────────────────────────
# 1. GELU — BERT/GPT activation
# ─────────────────────────────────────────────────────────────────────────────

def gelu(x: np.ndarray) -> np.ndarray:
    """GELU(x) = x * Φ(x)  where Φ is the standard normal CDF.

    Approximation used in practice:
    GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))

    Why it works:
    - Soft gate: x is multiplied by the probability that a Gaussian input
      would be positive. Large positive x → passes almost entirely.
      Large negative x → almost zeroed.
    - Differentiable everywhere (no hard kink at 0 like ReLU).
    - Allows small negative values through (unlike ReLU which kills them all).
    """
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


# ─────────────────────────────────────────────────────────────────────────────
# 2. SWISH / SiLU
# ─────────────────────────────────────────────────────────────────────────────

def swish(x: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """Swish(x) = x * σ(β*x)  where σ is sigmoid.

    β=1 gives SiLU (Sigmoid Linear Unit) — nn.SiLU() in PyTorch.
    Discovered via neural architecture search (Ramachandran et al. 2017).
    - Smooth (no kink)
    - Non-monotonic: small dip below 0 for slightly negative x
    - Self-gated: sigmoid(x) acts as a soft gate on x itself
    """
    return x / (1 + np.exp(-beta * x))


# ─────────────────────────────────────────────────────────────────────────────
# 3. MISH
# ─────────────────────────────────────────────────────────────────────────────

def mish(x: np.ndarray) -> np.ndarray:
    """Mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + eˣ)).

    Proposed by Misra (2019). Similar properties to Swish but smoother.
    Used in YOLOv4, some modern CNNs.
    """
    return x * np.tanh(np.log1p(np.exp(x)))


# ─────────────────────────────────────────────────────────────────────────────
# 4. DYING RELU SOLUTIONS
# ─────────────────────────────────────────────────────────────────────────────

def leaky_relu(x: np.ndarray, negative_slope: float = 0.01) -> np.ndarray:
    """LeakyReLU: small negative slope instead of 0."""
    return np.where(x > 0, x, negative_slope * x)


def elu(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """ELU: exponential for negative values — smooth and has negative saturation."""
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))


def demo_activations():
    """Plot all activation functions for comparison."""
    print("── 1-4. Activation Function Comparison ─────────────")
    x = X_RANGE

    funcs = {
        "ReLU":          lambda x: np.maximum(0, x),
        "GELU":          gelu,
        "Swish/SiLU":    swish,
        "Mish":          mish,
        "Leaky ReLU":    leaky_relu,
        "ELU":           elu,
    }
    colours = ["#f85149", "#58a6ff", "#3fb950", "#f0883e", "#bc8cff", "#79c0ff"]

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), facecolor="#0d1117")
    for ax, (name, fn), colour in zip(axes.flat, funcs.items(), colours):
        ax.set_facecolor("#161b22")
        ax.plot(x, fn(x), color=colour, linewidth=2.2)
        ax.axhline(0, color="#30363d", linewidth=0.8)
        ax.axvline(0, color="#30363d", linewidth=0.8)
        ax.set_title(name, color="#e6edf3", fontsize=11)
        ax.set_xlim(-4, 4)
        ax.set_ylim(-2, 4)
        ax.tick_params(colors="#8b949e")
        ax.spines[:].set_color("#30363d")

    plt.suptitle("Activation Functions: Shape Comparison", color="#e6edf3", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/activation_functions.png", dpi=FIGURE_DPI,
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved → {PLOTS_DIR}/activation_functions.png")

    # Print key properties table
    print(f"\n  {'Function':<15} {'Monotonic':>10} {'Smooth':>8} {'Neg vals':>10} {'Use case'}")
    rows = [
        ("ReLU",       "Yes",  "No",  "No",   "CNN default, fast"),
        ("GELU",       "No",   "Yes", "Yes",  "BERT, GPT, ViT"),
        ("Swish/SiLU", "No",   "Yes", "Yes",  "EfficientNet, LLaMA"),
        ("Mish",       "No",   "Yes", "Yes",  "YOLOv4, detectors"),
        ("Leaky ReLU", "Yes",  "No",  "Yes",  "GANs, dying ReLU fix"),
        ("ELU",        "Yes",  "Yes", "Yes",  "Better than ReLU in some NLP"),
    ]
    for row in rows:
        print(f"  {row[0]:<15} {row[1]:>10} {row[2]:>8} {row[3]:>10}  {row[4]}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. DYING RELU — demonstration
# ─────────────────────────────────────────────────────────────────────────────

def demo_dying_relu():
    """Demonstrate the dying ReLU problem and how to fix it."""
    print("\n── 5. Dying ReLU Problem ────────────────────────────")
    print("""
  DYING RELU:
  A ReLU neuron "dies" when its pre-activation is always negative.
  Once dead: output = 0, gradient = 0 → no more learning for that neuron.

  Causes:
  1. Large negative bias initialisation
  2. Large learning rate → weight update pushes all inputs negative
  3. Very negative input features (without normalisation)

  Solutions:
  ─────────────────────────────────────────────────────────────
  LeakyReLU  : constant small slope (0.01) for x < 0
               → always has a gradient, neuron can recover

  PReLU      : LeakyReLU with LEARNABLE slope α
               → can adapt slope to each neuron's needs

  ELU        : smooth exponential for x < 0
               → negative saturation helps push mean activation to 0

  Batch Norm : normalises pre-activations → reduces extreme negatives

  He init    : initialise weights from N(0, √(2/fan_in))
               → keeps variance consistent through layers
  """)

    # Demonstrate dying neuron
    torch.manual_seed(42)
    dying_neuron = nn.Linear(10, 1)
    # Force dying: large negative bias
    dying_neuron.bias.data.fill_(-10.0)

    X = torch.randn(100, 10)
    pre_act = dying_neuron(X)
    after_relu = F.relu(pre_act)
    after_leaky = F.leaky_relu(pre_act, 0.1)

    print(f"  Pre-activation range: [{pre_act.min().item():.2f}, {pre_act.max().item():.2f}]")
    print(f"  After ReLU:     mean={after_relu.mean().item():.4f}   "
          f"fraction dead: {(after_relu == 0).float().mean().item():.1%}")
    print(f"  After LeakyReLU: mean={after_leaky.mean().item():.4f}  "
          f"fraction dead: {(after_leaky == 0).float().mean().item():.1%}")
    print("  → LeakyReLU keeps all neurons alive even with extreme bias.")


# ─────────────────────────────────────────────────────────────────────────────
# 6. SOFTMAX — temperature and numerical stability
# ─────────────────────────────────────────────────────────────────────────────

def softmax_with_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Softmax with temperature scaling.

    σ(z/T) : dividing logits by T before softmax
    T < 1: sharper distribution (more confident, towards argmax)
    T > 1: flatter distribution (more uniform, more exploration)
    T → 0: one-hot (pure greedy)
    T → ∞: uniform distribution
    """
    return F.softmax(logits / temperature, dim=-1)


def log_sum_exp_trick(logits: torch.Tensor) -> torch.Tensor:
    """Numerically stable log-softmax via the log-sum-exp trick.

    log(softmax(z_i)) = z_i - log(sum_j exp(z_j))

    Problem: exp(z_j) overflows for large z_j (e.g. z=1000 → exp(1000)=∞).
    Fix: subtract max(z) before exponentiating.

    log(softmax(z_i)) = (z_i - max_z) - log(sum_j exp(z_j - max_z))
    The subtracted max cancels out, giving the same result without overflow.
    """
    max_z   = logits.max(dim=-1, keepdim=True).values
    log_sum = torch.log(torch.exp(logits - max_z).sum(dim=-1, keepdim=True))
    return logits - max_z - log_sum   # = log_softmax


def demo_softmax():
    """Show temperature effect and numerical stability."""
    print("\n── 6. Softmax: Temperature & Stability ─────────────")
    logits = torch.tensor([2.0, 1.0, 0.5, 0.1])

    print("  Temperature effect on distribution:")
    print(f"  {'Temperature':>12}  {'Distribution'}")
    for T in [0.1, 0.5, 1.0, 2.0, 10.0]:
        probs = softmax_with_temperature(logits, T)
        print(f"  {T:>12.1f}  {probs.numpy().round(3)}")

    print("\n  Numerical stability (log-sum-exp trick):")
    large_logits = torch.tensor([1000.0, 999.0, 998.0])

    # Naive: overflows
    try:
        naive = torch.exp(large_logits) / torch.exp(large_logits).sum()
        print(f"  Naive softmax: {naive}")
    except Exception:
        print(f"  Naive softmax: OVERFLOW!")

    if torch.exp(large_logits).sum().isinf():
        print("  Naive: OVERFLOW (exp(1000) = inf)")
    else:
        naive = torch.exp(large_logits) / torch.exp(large_logits).sum()
        print(f"  Naive: {naive}")

    stable = torch.exp(log_sum_exp_trick(large_logits))
    print(f"  Stable: {stable.numpy().round(4)}")
    print("  → Log-sum-exp trick: subtract max before exp to avoid overflow.")
    print("  PyTorch's F.softmax and F.log_softmax always use this trick internally.")


def main():
    print("=" * 54)
    print("MODULE 19 — Advanced Activation Functions")
    print("=" * 54)
    demo_activations()
    demo_dying_relu()
    demo_softmax()
    print("\nDone.")


if __name__ == "__main__":
    main()
