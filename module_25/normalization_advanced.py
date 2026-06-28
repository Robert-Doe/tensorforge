"""module_25/normalization_advanced.py — Normalization Layer Variants.

Covers:
  - Layer Normalization (normalise across features, not batch)
  - Instance Normalization (normalise each sample's feature map independently)
  - Group Normalization (split channels into groups, normalise per group)
  - When to use each: batch size matters
  - Spectral Normalization (for GAN training stability)

Run standalone: python normalization_advanced.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RANDOM_SEED = 42
PLOTS_DIR   = "plots"
FIGURE_DPI  = 120
torch.manual_seed(RANDOM_SEED)
os.makedirs(PLOTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. LAYER NORM FROM SCRATCH
# ─────────────────────────────────────────────────────────────────────────────

def layer_norm_scratch(x: torch.Tensor, gamma: torch.Tensor,
                        beta: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Layer Normalisation applied to the last dimension.

    Normalises across the feature dimension for each sample independently.
    Unlike BatchNorm, statistics are computed per-sample (not per-batch),
    so behaviour is identical at train and test time.

    LayerNorm(x) = gamma * (x - mean) / std + beta
    where mean/std are computed over the last dim (features), per sample.

    Args:
        x:     (batch, seq_len, features) or (batch, features)
        gamma: learnable scale parameter, same shape as last dim
        beta:  learnable shift parameter, same shape as last dim
        eps:   small constant for numerical stability

    Returns:
        normalised tensor
    """
    mean = x.mean(dim=-1, keepdim=True)
    std  = x.std( dim=-1, keepdim=True, unbiased=False)
    x_hat = (x - mean) / (std + eps)
    return gamma * x_hat + beta


def demo_layer_norm():
    """Verify scratch LayerNorm matches nn.LayerNorm."""
    print("── 1. Layer Normalization ──────────────────────────")
    B, T, D = 4, 10, 32   # batch, seq_len, features (Transformer typical)
    x = torch.randn(B, T, D)

    ln_torch = nn.LayerNorm(D)
    out_torch = ln_torch(x)

    gamma = ln_torch.weight.data
    beta  = ln_torch.bias.data
    out_scratch = layer_norm_scratch(x, gamma, beta)

    print(f"  Match: {torch.allclose(out_torch, out_scratch, atol=1e-5)}")
    print(f"  Input mean per sample: {x.mean(-1)[0].mean().item():.4f}")
    print(f"  Output mean per sample: {out_scratch.mean(-1)[0].mean().item():.6f}")
    print(f"  Output std  per sample: {out_scratch.std(-1)[0].mean().item():.6f}")
    print("  → After LayerNorm: mean≈0, std≈1 per sample per sequence position.")
    print("  LayerNorm is used in EVERY Transformer (BERT, GPT, T5, LLaMA).")


# ─────────────────────────────────────────────────────────────────────────────
# 2. INSTANCE NORM — per-sample, per-channel
# ─────────────────────────────────────────────────────────────────────────────

def demo_instance_norm():
    """Show InstanceNorm and its use in style transfer."""
    print("\n── 2. Instance Normalization ───────────────────────")
    B, C, H, W = 4, 16, 28, 28
    x = torch.randn(B, C, H, W)

    in_norm = nn.InstanceNorm2d(C, affine=True)
    out     = in_norm(x)

    print(f"  Input:  {x.shape}  mean={x.mean():.4f}  std={x.std():.4f}")
    print(f"  Output: {out.shape}")
    # InstanceNorm normalises each sample × channel independently
    # After: mean≈0, std≈1 per (sample, channel) pair
    per_sc_mean = out.mean(dim=[2,3]).abs().mean().item()
    per_sc_std  = out.std( dim=[2,3]).mean().item()
    print(f"  Per-(sample,channel) mean≈0: {per_sc_mean:.4f}")
    print(f"  Per-(sample,channel) std ≈1: {per_sc_std:.4f}")
    print("  Used in style transfer (neural style, CycleGAN):")
    print("  Normalises the style (statistics) of each feature map independently,")
    print("  allowing the model to apply any artistic style without batch coupling.")


# ─────────────────────────────────────────────────────────────────────────────
# 3. GROUP NORM — split channels into groups
# ─────────────────────────────────────────────────────────────────────────────

def demo_group_norm():
    """Show GroupNorm and its advantage at small batch sizes."""
    print("\n── 3. Group Normalization ───────────────────────────")
    B, C, H, W = 2, 32, 28, 28   # small batch (like detection)
    x = torch.randn(B, C, H, W)

    # GroupNorm(num_groups, num_channels): divide C channels into G groups
    # Each group of channels is normalised independently
    gn_g8  = nn.GroupNorm(num_groups=8,  num_channels=C)
    gn_g16 = nn.GroupNorm(num_groups=16, num_channels=C)
    gn_g32 = nn.GroupNorm(num_groups=32, num_channels=C)   # = InstanceNorm

    for name, gn in [("GroupNorm(G=8)",  gn_g8),
                     ("GroupNorm(G=16)", gn_g16),
                     ("GroupNorm(G=32)", gn_g32)]:
        out = gn(x)
        print(f"  {name}: output mean={out.mean():.4f}  std={out.std():.4f}")

    print("\n  Comparison at different batch sizes:")
    print("  ┌─────────────────┬────────────────────────────────────────────────┐")
    print("  │ Norm type       │ Statistics computed over                       │")
    print("  ├─────────────────┼────────────────────────────────────────────────┤")
    print("  │ BatchNorm       │ (batch, H, W) per channel — degrades at B<8   │")
    print("  │ LayerNorm       │ (C, H, W) per sample — common in Transformers │")
    print("  │ InstanceNorm    │ (H, W) per sample×channel — style transfer    │")
    print("  │ GroupNorm       │ (G_channels, H, W) — good for detection/B<8  │")
    print("  └─────────────────┴────────────────────────────────────────────────┘")


# ─────────────────────────────────────────────────────────────────────────────
# 4. VISUALISE ALL NORM TYPES
# ─────────────────────────────────────────────────────────────────────────────

def plot_norm_comparison():
    """Visualise which dimensions each norm type operates over."""
    fig, axes = plt.subplots(1, 4, figsize=(12, 4), facecolor="#0d1117")

    B, C = 4, 4   # small for visualisation: batch=4, channels=4
    titles  = ["BatchNorm", "LayerNorm", "InstanceNorm", "GroupNorm(G=2)"]
    colours = ["#58a6ff", "#3fb950", "#f85149", "#bc8cff"]

    # Grid: rows = batch dim, cols = channel dim
    # Shade which (batch, channel) pairs are normalised together
    norm_masks = [
        # BatchNorm: across batch, per channel → shade column-wise
        np.array([[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0]]).T,   # channel 0
        # LayerNorm: across channels, per sample → shade row-wise
        np.eye(B),
        # InstanceNorm: per (sample, channel) → shade each cell
        np.eye(B),
        # GroupNorm: (batch, group of channels) → groups of 2
        np.eye(B),
    ]

    for ax, title, colour in zip(axes, titles, colours):
        ax.set_facecolor("#0d1117")
        # Draw B×C grid
        for b in range(B):
            for c in range(C):
                is_highlighted = False
                if title == "BatchNorm"      and b == 0:      is_highlighted = True
                if title == "LayerNorm"      and b == 0:      is_highlighted = True
                if title == "InstanceNorm"   and b==0 and c==0: is_highlighted = True
                if title.startswith("Group") and b==0 and c<2: is_highlighted = True

                rect = plt.Rectangle([c, b], 1, 1,
                    facecolor=colour if is_highlighted else "#161b22",
                    edgecolor="#30363d", linewidth=1.5)
                ax.add_patch(rect)

        ax.set_xlim(0, C); ax.set_ylim(0, B)
        ax.set_xlabel("Channel", color="#e6edf3", fontsize=9)
        ax.set_ylabel("Batch",   color="#e6edf3", fontsize=9)
        ax.set_title(title,      color=colour,    fontsize=10)
        ax.tick_params(colors="#e6edf3")
        ax.spines[:].set_color("#30363d")
        xt = np.arange(0.5, C); yt = np.arange(0.5, B)
        ax.set_xticks(xt); ax.set_xticklabels(range(C), color="#8b949e", fontsize=8)
        ax.set_yticks(yt); ax.set_yticklabels(range(B), color="#8b949e", fontsize=8)

    plt.suptitle("Normalization Types — highlighted = statistics computed jointly",
                 color="#e6edf3", fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/norm_comparison.png", dpi=FIGURE_DPI,
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"\nSaved → {PLOTS_DIR}/norm_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# 5. SPECTRAL NORMALIZATION (GAN stability)
# ─────────────────────────────────────────────────────────────────────────────

def demo_spectral_norm():
    """Show how spectral norm constrains weight matrices."""
    print("\n── 5. Spectral Normalization ────────────────────────")
    linear = nn.Linear(64, 64)
    sn_linear = nn.utils.spectral_norm(linear)

    x = torch.randn(16, 64)
    out = sn_linear(x)

    # After spectral norm, the weight matrix has spectral norm = 1
    # (largest singular value = 1)
    W  = sn_linear.weight.data
    sv = torch.linalg.svdvals(W)
    print(f"  Largest singular value of W (after SN): {sv.max().item():.4f}")
    print(f"  (should be ≈ 1.0)")
    print("\n  Spectral norm controls the Lipschitz constant of the layer.")
    print("  Lipschitz=1 means ||f(x) - f(y)|| ≤ ||x - y|| — bounded sensitivity.")
    print("  Used in: SNGAN, BigGAN to stabilise discriminator training.")
    print("  Without it, GAN discriminators can diverge (mode collapse).")


def main():
    print("=" * 54)
    print("MODULE 25 — Normalization Layer Variants")
    print("=" * 54)
    demo_layer_norm()
    demo_instance_norm()
    demo_group_norm()
    plot_norm_comparison()
    demo_spectral_norm()
    print("\nDone.")


if __name__ == "__main__":
    main()
