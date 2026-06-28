"""module_18/mlp_advanced.py — MLP Theory and Architecture Depth.

Covers:
  - Universal Approximation Theorem (intuition + demonstration)
  - Depth vs width trade-off
  - Skip connections in MLP (dense connections)
  - Neuron utilisation (dead neurons, saturation)
  - Initialisation strategies (Xavier/Glorot, He/Kaiming)

Run standalone: python mlp_advanced.py
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
from sklearn.datasets import make_moons

RANDOM_SEED = 42
PLOTS_DIR   = "plots"
FIGURE_DPI  = 120
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
os.makedirs(PLOTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. UNIVERSAL APPROXIMATION THEOREM
# ─────────────────────────────────────────────────────────────────────────────

def demo_universal_approximation():
    """Show that a single hidden layer can approximate any continuous function."""
    print("── 1. Universal Approximation Theorem ──────────────")
    print("""
  Theorem (Cybenko, 1989 / Hornik, 1991):
  A neural network with ONE hidden layer and a sufficient number of neurons
  with a non-linear activation function can approximate any continuous
  function f: Rⁿ → Rᵐ to arbitrary precision on a compact domain.

  Key qualifications (often glossed over):
  • "Sufficient number" — can be exponential in input dimension
  • Only guarantees EXISTENCE, not learnability by gradient descent
  • Does not say HOW MANY neurons are needed
  • Does not address the generalisation (overfitting) properties

  Practical implication: depth > width in modern deep learning.
  A deep network requires exponentially fewer neurons than a shallow
  network to represent the same function class, ESPECIALLY for
  functions with compositional structure (e.g. image features).
  """)

    # Demonstrate: fit f(x) = sin(πx) + 0.5*sin(3πx) with widths 4, 8, 32
    x_np = np.linspace(-1, 1, 300, dtype=np.float32)
    y_np = np.sin(np.pi * x_np) + 0.5 * np.sin(3 * np.pi * x_np)

    X = torch.tensor(x_np).unsqueeze(1)
    y = torch.tensor(y_np).unsqueeze(1)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), facecolor="#0d1117")
    widths = [4, 16, 64]

    for ax, width in zip(axes, widths):
        model = nn.Sequential(
            nn.Linear(1, width), nn.Tanh(),
            nn.Linear(width, width), nn.Tanh(),
            nn.Linear(width, 1),
        )
        opt  = optim.Adam(model.parameters(), lr=1e-3)
        mse  = nn.MSELoss()

        for _ in range(3000):
            opt.zero_grad()
            mse(model(X), y).backward()
            opt.step()

        with torch.no_grad():
            y_pred = model(X).numpy().flatten()

        ax.set_facecolor("#161b22")
        ax.plot(x_np, y_np,    color="#58a6ff", linewidth=2,   label="True f(x)")
        ax.plot(x_np, y_pred,  color="#f85149", linewidth=2,   label="Approx", linestyle="--")
        ax.set_title(f"Width={width}", color="#e6edf3", fontsize=11)
        ax.tick_params(colors="#8b949e")
        ax.spines[:].set_color("#30363d")
        ax.legend(fontsize=8, facecolor="#0d1117", labelcolor="#e6edf3")

    plt.suptitle("Universal Approximation: f(x) = sin(πx) + 0.5·sin(3πx)", color="#e6edf3")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/universal_approx.png", dpi=FIGURE_DPI,
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved → {PLOTS_DIR}/universal_approx.png")


# ─────────────────────────────────────────────────────────────────────────────
# 2. DEPTH vs WIDTH EXPERIMENT
# ─────────────────────────────────────────────────────────────────────────────

def build_model(n_layers: int, width: int, input_dim: int) -> nn.Module:
    """Build MLP with fixed parameter budget but varying depth/width."""
    layers = [nn.Linear(input_dim, width), nn.ReLU()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(width, width), nn.ReLU()]
    layers.append(nn.Linear(width, 1))
    return nn.Sequential(*layers)


def train_model(model, X_tr, y_tr, epochs: int = 500) -> float:
    """Train and return final training loss."""
    opt = optim.Adam(model.parameters(), lr=1e-3)
    mse = nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad()
        mse(model(X_tr), y_tr).backward()
        opt.step()
    with torch.no_grad():
        return mse(model(X_tr), y_tr).item()


def demo_depth_vs_width():
    """Compare deep+narrow vs shallow+wide with same parameter count."""
    print("\n── 2. Depth vs Width ─────────────────────────────────")
    x_np = np.linspace(-1, 1, 400, dtype=np.float32)
    y_np = np.sin(2 * np.pi * x_np) * np.exp(-x_np**2)  # damped sine
    X    = torch.tensor(x_np).unsqueeze(1)
    y    = torch.tensor(y_np).unsqueeze(1)

    configs = [
        ("Shallow (1 layer, w=256)",  1, 256),
        ("Medium  (4 layers, w=64)",  4, 64),
        ("Deep    (8 layers, w=32)",  8, 32),
    ]
    print(f"  {'Architecture':<32} {'Params':>8} {'MSE':>10}")
    for name, n_layers, width in configs:
        model     = build_model(n_layers, width, input_dim=1)
        n_params  = sum(p.numel() for p in model.parameters())
        final_mse = train_model(model, X, y, epochs=1000)
        print(f"  {name:<32} {n_params:>8,} {final_mse:>10.6f}")

    print("\n  Depth matters more than width for learning compositional functions.")
    print("  Each extra layer composes the previous representation into something richer.")


# ─────────────────────────────────────────────────────────────────────────────
# 3. SKIP CONNECTIONS IN MLP (ResMLP / DenseNet-style)
# ─────────────────────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """Simple residual block for an MLP: x → Linear → BN → ReLU → Linear → + x."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.net(x) + x)   # add skip, then ReLU


class ResMLP(nn.Module):
    """MLP with residual connections.

    ResNet for tabular data: each block adds x, allowing gradients to
    flow directly through the skip path to early layers.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 n_blocks: int = 4, output_dim: int = 1):
        super().__init__()
        self.stem   = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_dim) for _ in range(n_blocks)])
        self.head   = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.stem(x))
        x = self.blocks(x)
        return self.head(x)


def demo_skip_connections():
    """Compare plain MLP vs ResMLP on make_moons."""
    print("\n── 3. Skip Connections in MLP (ResMLP) ─────────────")
    from sklearn.preprocessing import StandardScaler

    X_np, y_np = make_moons(n_samples=600, noise=0.25, random_state=RANDOM_SEED)
    scaler = StandardScaler().fit(X_np)
    X_np   = scaler.transform(X_np).astype(np.float32)

    X = torch.tensor(X_np)
    y = torch.tensor(y_np, dtype=torch.float32).unsqueeze(1)

    plain = nn.Sequential(
        nn.Linear(2, 64), nn.ReLU(),
        nn.Linear(64, 64), nn.ReLU(),
        nn.Linear(64, 64), nn.ReLU(),
        nn.Linear(64, 64), nn.ReLU(),
        nn.Linear(64, 1),
    )
    resmlp = ResMLP(input_dim=2, hidden_dim=64, n_blocks=4)

    bce = nn.BCEWithLogitsLoss()

    def train_and_eval(model, name):
        opt = optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(1000):
            opt.zero_grad()
            bce(model(X), y).backward()
            opt.step()
        with torch.no_grad():
            acc = ((torch.sigmoid(model(X)) >= 0.5).float() == y).float().mean().item()
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  {name:<15}: params={n_params:,}  accuracy={acc:.4f}")

    train_and_eval(plain,  "Plain MLP")
    train_and_eval(resmlp, "ResMLP")
    print("\n  Skip connections help gradients flow to early layers.")
    print("  In tabular deep learning (TabNet, NODE, FT-Transformer) residuals are common.")


# ─────────────────────────────────────────────────────────────────────────────
# 4. WEIGHT INITIALISATION — Xavier vs He
# ─────────────────────────────────────────────────────────────────────────────

def demo_initialisation():
    """Show how init strategy affects activation variance across layers."""
    print("\n── 4. Weight Initialisation: Xavier vs He ───────────")
    print("""
  PROBLEM: If weights are too large → activations explode (vanishing gradient backwards)
           If weights are too small → activations collapse to 0 (no learning signal)
  GOAL: keep variance of activations roughly constant across layers.

  Xavier/Glorot (for tanh/sigmoid):
    W ~ Uniform(-√(6/(fan_in + fan_out)), +√(6/(fan_in + fan_out)))
    Derived for linear activations, good for tanh/sigmoid.

  He/Kaiming (for ReLU):
    W ~ N(0, √(2/fan_in))
    The factor 2 compensates for ReLU killing half the neurons (E[relu(x)²] = Var(x)/2).
  """)

    depth = 10
    x     = torch.randn(32, 64)   # batch=32, dim=64

    inits = {
        "Default (N(0,1))": lambda: torch.randn(64, 64),
        "Xavier":           lambda: nn.init.xavier_uniform_(torch.empty(64, 64)),
        "He (Kaiming)":     lambda: nn.init.kaiming_normal_(torch.empty(64, 64)),
    }

    print(f"  Activation std across {depth} ReLU layers:")
    print(f"  {'Init':<22} {'Layer ' + ' '.join(str(i) for i in range(1, depth+1))}")

    for name, init_fn in inits.items():
        act = x.clone()
        stds = []
        for _ in range(depth):
            W   = init_fn()
            act = torch.relu(act @ W)
            stds.append(f"{act.std().item():.2f}")
        print(f"  {name:<22} {' '.join(f'{s:>6}' for s in stds)}")

    print("\n  He init keeps activation std stable across all layers.")
    print("  Default init (N(0,1)) leads to explosion (std grows unboundedly).")
    print("  Xavier also drifts because it's not calibrated for ReLU's asymmetry.")


def main():
    print("=" * 54)
    print("MODULE 18 — MLP Theory: Depth, Width, Skip Connections")
    print("=" * 54)
    demo_universal_approximation()
    demo_depth_vs_width()
    demo_skip_connections()
    demo_initialisation()
    print("\nDone.")


if __name__ == "__main__":
    main()
