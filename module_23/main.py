"""module_23/main.py — Convolutional Layer from Scratch.

Generates a synthetic image, applies hand-crafted filters,
and visualises the resulting feature maps.
Run: python main.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from conv_layer import (
    convolve2d, make_synthetic_image,
    edge_detect_kernel, vertical_edge_kernel,
    sharpen_kernel, blur_kernel,
)

PLOTS_DIR  = "plots"
FIGURE_DPI = 120


def plot_feature_maps(image, kernels_and_names):
    """Save the original image and one feature map per filter side by side."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    n = len(kernels_and_names)

    fig, axes = plt.subplots(1, n + 1, figsize=(3 * (n + 1), 3.5),
                             facecolor="#0d1117")
    fig.suptitle("Convolution Feature Maps", color="#e6edf3", fontsize=12)

    # Original image
    axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Original", color="#e6edf3")
    axes[0].axis("off")

    for ax, (kernel, name) in zip(axes[1:], kernels_and_names):
        fmap = convolve2d(image, kernel, padding=1)   # padding=1 keeps size same
        ax.imshow(fmap, cmap="gray")
        ax.set_title(name, color="#e6edf3")
        ax.axis("off")

    plt.tight_layout()
    fname = f"{PLOTS_DIR}/conv_feature_maps.png"
    plt.savefig(fname, dpi=FIGURE_DPI, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved → {fname}")


def print_kernel_demo(image, kernel, name):
    """Show a 5×5 patch of input, the kernel, and the output value."""
    patch  = image[3:6, 3:6]    # 3×3 patch from near top-left
    kernel = kernel[:3, :3]     # ensure 3×3 for display
    result = np.sum(patch * kernel)
    print(f"\n  {name} — one output pixel:")
    print(f"  Input patch:\n{patch.round(2)}")
    print(f"  Kernel:\n{kernel}")
    print(f"  Dot product (sum of element-wise mult): {result:.3f}")


def main():
    print("=" * 52)
    print("MODULE 23 — Convolutional Layer from Scratch")
    print("=" * 52)

    image = make_synthetic_image(H=28, W=28)
    print(f"\nSynthetic image shape: {image.shape}")
    print(f"Pixel value range: [{image.min():.2f}, {image.max():.2f}]")

    # ── Single-filter convolution demo ───────────────────────────────────────
    h_kernel = edge_detect_kernel()
    fmap_h   = convolve2d(image, h_kernel, padding=1)
    v_kernel = vertical_edge_kernel()
    fmap_v   = convolve2d(image, v_kernel, padding=1)

    print(f"\nHorizontal edge feature map shape: {fmap_h.shape}")
    print(f"Strong response range: [{fmap_h.min():.2f}, {fmap_h.max():.2f}]")
    print("  High values = horizontal edges found in the image")

    # ── Stride demo ──────────────────────────────────────────────────────────
    print("\n=== Stride effect on output size ===")
    for stride in [1, 2, 4]:
        fmap = convolve2d(image, h_kernel, stride=stride)
        print(f"  stride={stride}  output shape: {fmap.shape}")

    # ── Kernel dot product walkthrough ───────────────────────────────────────
    print_kernel_demo(image, h_kernel, "Horizontal edge detector")

    # ── Learnable filter comparison ───────────────────────────────────────────
    print("\n=== Random kernel vs. hand-crafted kernel ===")
    rng = np.random.default_rng(42)
    rand_kernel = rng.normal(0, 1, (3, 3))
    rand_fmap   = convolve2d(image, rand_kernel, padding=1)
    print(f"  Random kernel response range  : [{rand_fmap.min():.2f}, {rand_fmap.max():.2f}]")
    print(f"  Edge detector response range  : [{fmap_h.min():.2f}, {fmap_h.max():.2f}]")
    print("  Hand-crafted kernels are interpretable;")
    print("  CNNs learn kernels from data that are often not.")

    # ── Visualise all filters ─────────────────────────────────────────────────
    kernels_and_names = [
        (edge_detect_kernel(),    "Horiz Edge"),
        (vertical_edge_kernel(),  "Vert Edge"),
        (sharpen_kernel(),        "Sharpen"),
        (blur_kernel(3),          "Blur"),
    ]
    plot_feature_maps(image, kernels_and_names)

    print("\nDone.")


if __name__ == "__main__":
    main()
