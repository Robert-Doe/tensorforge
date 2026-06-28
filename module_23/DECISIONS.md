# Module 23 — DECISIONS.md

## Decision 1: No backward pass — forward and intuition only
**Decision:** Implement only the forward convolution; no gradient derivation.
**Why:** The convolution backward pass is the most algebraically complex operation in this curriculum. Deriving it here would overwhelm the primary lesson (what convolution IS and what filters detect). PyTorch's autograd handles the backward pass in M24. Keeping M23 purely forward lets students build visual intuition without symbol manipulation.
**Trade-off:** Students don't see how conv gradients work. Offset by a clear explanation in the tutorial that autograd handles it and a pointer to resources for the curious.

## Decision 2: Hand-crafted kernels (Sobel, blur, sharpen)
**Decision:** Use classic image-processing kernels rather than randomly initialised ones.
**Why:** Hand-crafted kernels have known, interpretable effects — edge detection produces sharp lines at boundaries, blur produces smoothed output. Showing these first establishes that a kernel IS a feature detector, motivating why CNNs learning kernels from data is so powerful (they learn what to detect, not just how to detect something pre-specified).
**Trade-off:** Classic kernels are not what CNNs actually learn — they're idealised. The random kernel comparison in main.py bridges the gap.

## Decision 3: padding=1 for "same" convolution in visualisation
**Decision:** Default to padding=1 with 3×3 kernels so output size = input size.
**Why:** Students can compare input and output images directly at the same resolution. Without padding, output is 2 pixels smaller per side — distracting from the feature-detection story.
**Trade-off:** Padding introduces border artefacts (zero-padded rows/columns produce slightly different responses near edges). Acknowledged in tutorial.

## Decision 4: Synthetic image over real dataset download
**Decision:** Generate a 28×28 synthetic image with geometric shapes (rectangle + circle) instead of downloading MNIST.
**Why:** Keeps the module self-contained (no internet required). The geometric shapes produce crisp, predictable responses to the hand-crafted kernels, making the feature maps easier to interpret than noisy handwritten digits. MNIST appears in M24 via torchvision.
**Trade-off:** Synthetic images don't capture the texture complexity of real images. Acceptable for a geometry-of-convolution lesson.
