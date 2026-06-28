"""module_23/conv_advanced.py — Advanced Convolution Operations.

Covers:
  - Dilated (atrous) convolution — expands receptive field without more params
  - Depthwise separable convolution — MobileNet's efficiency trick
  - Transposed convolution (deconvolution) — used in decoders/GANs
  - Receptive field calculation
  - 1×1 convolution — channel mixing without spatial computation

Run standalone: python conv_advanced.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

RANDOM_SEED = 42
PLOTS_DIR   = "plots"
FIGURE_DPI  = 120
torch.manual_seed(RANDOM_SEED)
os.makedirs(PLOTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. DILATED CONVOLUTION
# ─────────────────────────────────────────────────────────────────────────────

def dilated_convolve2d(image: np.ndarray, kernel: np.ndarray,
                       dilation: int = 1) -> np.ndarray:
    """2D convolution with dilation (atrous convolution).

    Inserts (dilation-1) zeros between kernel elements.
    Effective kernel size: k_eff = k + (k-1)*(dilation-1)
    Receptive field grows as dilation increases WITHOUT extra parameters.

    Args:
        image:   (H, W) input
        kernel:  (kH, kW) filter
        dilation: gap between kernel elements

    Returns:
        feature map
    """
    H, W   = image.shape
    kH, kW = kernel.shape
    # Effective kernel size
    kH_eff = kH + (kH - 1) * (dilation - 1)
    kW_eff = kW + (kW - 1) * (dilation - 1)

    H_out = H - kH_eff + 1
    W_out = W - kW_eff + 1
    out   = np.zeros((H_out, W_out))

    for i in range(H_out):
        for j in range(W_out):
            for ki in range(kH):
                for kj in range(kW):
                    # Skip according to dilation
                    out[i, j] += (image[i + ki*dilation, j + kj*dilation]
                                  * kernel[ki, kj])
    return out


def demo_dilated():
    """Show how dilation expands receptive field."""
    print("── 1. Dilated Convolution ───────────────────────────")
    image  = np.zeros((20, 20))
    image[8:12, 8:12] = 1.0   # bright square in centre
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])   # Laplacian

    for dilation in [1, 2, 4]:
        feat = dilated_convolve2d(image, kernel, dilation)
        kH, kW = kernel.shape
        k_eff  = kH + (kH - 1) * (dilation - 1)
        print(f"  Dilation={dilation}: effective kernel={k_eff}×{k_eff}  "
              f"output shape={feat.shape}  "
              f"receptive field covers {k_eff}×{k_eff} pixels with 3×3 kernel")

    # PyTorch nn.Conv2d with dilation parameter
    conv_d1 = nn.Conv2d(1, 1, kernel_size=3, dilation=1, padding=1)
    conv_d4 = nn.Conv2d(1, 1, kernel_size=3, dilation=4, padding=4)
    x = torch.randn(1, 1, 28, 28)
    print(f"\n  Same 3×3 kernel, same output shape:")
    print(f"  dilation=1: output {conv_d1(x).shape}  (receptive field 3×3)")
    print(f"  dilation=4: output {conv_d4(x).shape}  (receptive field 11×11)")
    print("  Used in WaveNet (audio), DeepLab (segmentation) for large context.")


# ─────────────────────────────────────────────────────────────────────────────
# 2. DEPTHWISE SEPARABLE CONVOLUTION
# ─────────────────────────────────────────────────────────────────────────────

class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution: depthwise + pointwise.

    Standard Conv2d(C_in, C_out, k):  params = C_in * C_out * k²
    Depthwise Sep:                     params = C_in * k² + C_in * C_out

    For C_in=C_out=64, k=3:
      Standard:  64 * 64 * 9 = 36,864
      Sep:       64 * 9 + 64 * 64 = 576 + 4096 = 4,672  (8× fewer params)

    Used in: MobileNet, Xception, EfficientNet.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, padding: int = 1):
        super().__init__()
        # Depthwise: one filter per channel (groups=in_channels)
        self.depthwise  = nn.Conv2d(in_channels, in_channels,
                                     kernel_size=kernel_size,
                                     padding=padding,
                                     groups=in_channels)   # groups=in_ch = depthwise
        # Pointwise: 1×1 conv to mix channels
        self.pointwise  = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


def demo_depthwise_separable():
    """Compare parameter counts: standard vs depthwise separable."""
    print("\n── 2. Depthwise Separable Convolution ───────────────")
    C_in, C_out, k = 64, 64, 3

    standard = nn.Conv2d(C_in, C_out, kernel_size=k, padding=1)
    sep      = DepthwiseSeparableConv(C_in, C_out, kernel_size=k, padding=1)

    std_params = sum(p.numel() for p in standard.parameters())
    sep_params = sum(p.numel() for p in sep.parameters())

    x = torch.randn(1, C_in, 28, 28)
    assert standard(x).shape == sep(x).shape

    print(f"  Standard Conv2d({C_in},{C_out},{k}): {std_params:,} params")
    print(f"  Depthwise Separable:              {sep_params:,} params")
    print(f"  Reduction: {std_params/sep_params:.1f}× fewer parameters")
    print(f"  Output shape preserved: {standard(x).shape}")
    print("  Accuracy trade-off is small (MobileNetV2 matches ResNet50 at 1/10 params).")


# ─────────────────────────────────────────────────────────────────────────────
# 3. TRANSPOSED CONVOLUTION
# ─────────────────────────────────────────────────────────────────────────────

def demo_transposed_conv():
    """Show how transposed conv upsamples spatial dimensions."""
    print("\n── 3. Transposed Convolution (Upsampling) ───────────")
    # Downsampled feature map → want to recover spatial resolution
    x = torch.randn(1, 32, 7, 7)   # e.g. after CNN encoder

    # stride=2: output is 2× larger in each spatial dim
    t_conv = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
    x_up   = t_conv(x)
    print(f"  Input:  {x.shape}")
    print(f"  After ConvTranspose2d(stride=2): {x_up.shape}")

    # Full encoder-decoder (U-Net style) example
    encoder = nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1),    # 28×28
        nn.ReLU(),
        nn.MaxPool2d(2),                   # 14×14
        nn.Conv2d(16, 32, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),                   # 7×7
    )
    decoder = nn.Sequential(
        nn.ConvTranspose2d(32, 16, 2, stride=2),  # 14×14
        nn.ReLU(),
        nn.ConvTranspose2d(16, 1, 2, stride=2),   # 28×28
    )

    inp   = torch.randn(1, 1, 28, 28)
    enc   = encoder(inp)
    dec   = decoder(enc)
    print(f"\n  Encoder: {inp.shape} → {enc.shape}")
    print(f"  Decoder: {enc.shape} → {dec.shape}")
    print("  Transposed conv is used in: image segmentation (FCN, U-Net),")
    print("  GANs (generator network), super-resolution, depth estimation.")


# ─────────────────────────────────────────────────────────────────────────────
# 4. RECEPTIVE FIELD CALCULATION
# ─────────────────────────────────────────────────────────────────────────────

def receptive_field(layer_configs: list) -> int:
    """Calculate effective receptive field of a stack of conv/pool layers.

    For each layer: RF grows by (kernel_size - 1) * stride_product_so_far.

    Args:
        layer_configs: list of (kernel_size, stride) tuples

    Returns:
        receptive field in input pixels
    """
    rf             = 1
    stride_product = 1
    print(f"\n  {'Layer':<8} {'k':>4} {'s':>4} {'RF':>8} {'Stride prod':>12}")
    for i, (k, s) in enumerate(layer_configs):
        rf             = rf + (k - 1) * stride_product
        stride_product *= s
        print(f"  Layer {i+1:<3} {k:>4} {s:>4} {rf:>8} {stride_product:>12}")
    return rf


def demo_receptive_field():
    """Calculate RF for VGG-style and ResNet-style architectures."""
    print("\n── 4. Receptive Field Calculation ──────────────────")
    print("  VGG-like (5 conv blocks, 2 max-pool):")
    vgg_layers = [
        (3, 1), (3, 1),             # block 1: 2× conv, no stride
        (2, 2),                     # pool
        (3, 1), (3, 1),             # block 2
        (2, 2),                     # pool
        (3, 1), (3, 1),             # block 3
    ]
    rf_vgg = receptive_field(vgg_layers)
    print(f"  Final RF: {rf_vgg} pixels")

    print("\n  LeNet-like (2 conv + 2 pool):")
    lenet_layers = [(5, 1), (2, 2), (5, 1), (2, 2)]
    rf_lenet = receptive_field(lenet_layers)
    print(f"  Final RF: {rf_lenet} pixels")

    print("\n  Deeper network → larger RF → captures larger patterns.")
    print("  Dilated convolutions increase RF without adding layers.")


# ─────────────────────────────────────────────────────────────────────────────
# 5. 1×1 CONVOLUTION — channel mixing
# ─────────────────────────────────────────────────────────────────────────────

def demo_pointwise():
    """Show what 1×1 conv does and why it's used."""
    print("\n── 5. 1×1 Convolution (Pointwise) ──────────────────")
    x = torch.randn(1, 256, 14, 14)   # 256 feature maps

    # 1×1 conv: applies a different linear combination per spatial position
    # (same weights at every position — weight sharing in spatial dims)
    conv1x1 = nn.Conv2d(256, 64, kernel_size=1)   # reduce 256 → 64 channels
    x_reduced = conv1x1(x)
    print(f"  Input:  {x.shape}")
    print(f"  After 1×1 conv (256→64): {x_reduced.shape}")
    print(f"  Parameters: {sum(p.numel() for p in conv1x1.parameters()):,}")
    print("  Used for:")
    print("    - Channel reduction (bottleneck in ResNet, Inception)")
    print("    - Projection to a different number of channels between depthwise convs")
    print("    - Applying a per-pixel fully-connected layer (no spatial context)")
    print("    - Final 1×1 conv in FCN for pixel-wise classification")


def main():
    print("=" * 54)
    print("MODULE 23 — Advanced Convolution Operations")
    print("=" * 54)
    demo_dilated()
    demo_depthwise_separable()
    demo_transposed_conv()
    demo_receptive_field()
    demo_pointwise()
    print("\nDone.")


if __name__ == "__main__":
    main()
