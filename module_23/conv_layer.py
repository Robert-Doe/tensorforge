"""module_23/conv_layer.py — 2D Convolution from scratch with NumPy.

Implements a single convolutional layer:
  - Forward: slide a kernel over an image, compute dot products
  - Demonstrates padding, stride, and multiple filters
  - Shows feature maps and what filters detect

No backward pass implemented here — the point is visual intuition.
Module 24 uses PyTorch's autograd-enabled Conv2d for training.
"""

import numpy as np

# ── Constants ────────────────────────────────────────────────────────────────
RANDOM_SEED = 42


def convolve2d(image, kernel, stride=1, padding=0):
    """Apply a single 2-D kernel to a single-channel image.

    Slides the kernel over every valid position in the (padded) image,
    computing the dot product at each location.

    Args:
        image:   (H, W) NumPy array — single-channel image
        kernel:  (kH, kW) NumPy array — filter weights
        stride:  int — step size between kernel positions (default 1)
        padding: int — zero-padding added to each side (default 0)

    Returns:
        feature_map: (H_out, W_out) NumPy array
            where H_out = (H + 2*padding - kH) // stride + 1
    """
    H, W   = image.shape
    kH, kW = kernel.shape

    # Add zero-padding to all four sides
    if padding > 0:
        image = np.pad(image, pad_width=padding, mode="constant", constant_values=0)

    H_pad, W_pad = image.shape
    H_out = (H_pad - kH) // stride + 1
    W_out = (W_pad - kW) // stride + 1

    feature_map = np.zeros((H_out, W_out))

    for i in range(H_out):
        for j in range(W_out):
            row_start = i * stride
            col_start = j * stride
            # Extract the patch the kernel currently sits over
            patch = image[row_start:row_start + kH,
                          col_start:col_start + kW]
            # Dot product: element-wise multiply then sum
            feature_map[i, j] = np.sum(patch * kernel)

    return feature_map


def convolve2d_multichannel(volume, kernels, stride=1, padding=0):
    """Apply multiple kernels to a multi-channel input volume.

    Args:
        volume:  (C_in, H, W) — input with C_in channels (e.g. RGB = 3)
        kernels: (C_out, C_in, kH, kW) — C_out filters, each covering all C_in channels
        stride:  int
        padding: int

    Returns:
        output: (C_out, H_out, W_out) — one feature map per filter
    """
    C_out, C_in, kH, kW = kernels.shape
    C_in_vol, H, W = volume.shape
    assert C_in == C_in_vol, "Kernel channels must match input channels"

    H_out = (H + 2 * padding - kH) // stride + 1
    W_out = (W + 2 * padding - kW) // stride + 1
    output = np.zeros((C_out, H_out, W_out))

    for f in range(C_out):                      # for each filter
        for c in range(C_in):                   # accumulate over input channels
            output[f] += convolve2d(
                volume[c], kernels[f, c], stride=stride, padding=padding
            )
    return output


# ── Classic hand-crafted kernels ─────────────────────────────────────────────

def edge_detect_kernel():
    """Sobel-style horizontal edge detector.

    Responds strongly where pixel intensity changes vertically.
    """
    return np.array([[-1, -2, -1],
                     [ 0,  0,  0],
                     [ 1,  2,  1]], dtype=float)


def vertical_edge_kernel():
    """Sobel-style vertical edge detector."""
    return np.array([[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]], dtype=float)


def sharpen_kernel():
    """Sharpening filter — amplifies high-frequency details."""
    return np.array([[ 0, -1,  0],
                     [-1,  5, -1],
                     [ 0, -1,  0]], dtype=float)


def blur_kernel(size=3):
    """Box blur — averages neighbouring pixels."""
    return np.ones((size, size), dtype=float) / (size * size)


def make_synthetic_image(H=28, W=28, seed=RANDOM_SEED):
    """Generate a synthetic grayscale image with geometric shapes.

    Creates a simple 28×28 image with a bright rectangle and a dark circle,
    simulating the kind of structured input a CNN would process.

    Args:
        H, W: image dimensions
        seed: random seed

    Returns:
        image: (H, W) float array with values in [0, 1]
    """
    rng   = np.random.default_rng(seed)
    image = rng.uniform(0.0, 0.15, (H, W))   # low-level noise background

    # Bright rectangle (simulating a document region)
    image[6:14, 5:20]  = 0.85

    # Dark circle (simulating a stamp or seal)
    cy, cx, r = 20, 20, 5
    for y in range(H):
        for x in range(W):
            if (y - cy) ** 2 + (x - cx) ** 2 <= r ** 2:
                image[y, x] = 0.1

    return image
