"""module_19/activations.py — Activation functions and their gradients.

Compares: Sigmoid, Tanh, ReLU, Leaky ReLU, ELU.
Demonstrates the vanishing gradient problem with sigmoid in deep networks.
"""

import numpy as np

# ── Activation functions ────────────────────────────────────────────────────

def sigmoid(z):
    """Sigmoid: output in (0, 1). Saturates for large |z| → vanishing gradient."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_grad(z):
    """d(sigmoid)/dz = s * (1-s). Max gradient = 0.25 at z=0."""
    s = sigmoid(z)
    return s * (1.0 - s)

def tanh(z):
    """Tanh: output in (-1, 1). Zero-centred (better than sigmoid for hidden layers)."""
    return np.tanh(z)

def tanh_grad(z):
    """d(tanh)/dz = 1 - tanh(z)^2. Max gradient = 1.0 at z=0."""
    return 1.0 - np.tanh(z) ** 2

def relu(z):
    """ReLU: max(0, z). Gradient = 1 for z>0, 0 for z<=0."""
    return np.maximum(0.0, z)

def relu_grad(z):
    """d(ReLU)/dz — a step function, undefined at z=0 (return 0 there)."""
    return (z > 0).astype(float)

def leaky_relu(z, alpha=0.01):
    """Leaky ReLU: max(alpha*z, z). Small negative gradient for z<0.

    Fixes the 'dead neuron' problem: ReLU neurons that receive
    only negative inputs never fire and their weights never update.
    Leaky ReLU keeps a tiny gradient (alpha) for z<0.

    Args:
        z:     input array
        alpha: leak coefficient (default 0.01 — 1% of the signal passes)
    """
    return np.where(z > 0, z, alpha * z)

def leaky_relu_grad(z, alpha=0.01):
    """d(Leaky ReLU)/dz."""
    return np.where(z > 0, 1.0, alpha)

def elu(z, alpha=1.0):
    """Exponential Linear Unit: smooth negative side, zero-centred mean.

    For z>0:  z          (like ReLU)
    For z<=0: alpha*(e^z - 1)  (smooth, approaches -alpha)

    Args:
        z:     input array
        alpha: scale for negative side (default 1.0)
    """
    return np.where(z > 0, z, alpha * (np.exp(np.clip(z, -500, 0)) - 1.0))

def elu_grad(z, alpha=1.0):
    """d(ELU)/dz."""
    return np.where(z > 0, 1.0, elu(z, alpha) + alpha)


# ── Vanishing gradient demonstration ─────────────────────────────────────────

def vanishing_gradient_demo(n_layers=10):
    """Show how sigmoid gradient shrinks as it passes through many layers.

    Simulates backpropagation through n_layers sigmoid layers with zero input.
    At z=0, sigmoid derivative = 0.25. After n layers, gradient ≈ 0.25^n.

    Args:
        n_layers: number of layers to simulate

    Returns:
        gradient magnitudes: list of floats, one per layer (from output backward)
    """
    INIT_GRADIENT = 1.0           # gradient at the output layer
    gradient      = INIT_GRADIENT
    magnitudes    = [gradient]

    for layer in range(1, n_layers + 1):
        # At z=0, sigmoid derivative = 0.25 — a conservative (best-case) estimate
        gradient *= 0.25          # each sigmoid layer multiplies gradient by <=0.25
        magnitudes.append(gradient)

    return magnitudes


def relu_gradient_demo(n_layers=10):
    """Show that ReLU gradients don't shrink (for active neurons).

    At z>0, ReLU derivative = 1.0. Gradient is unchanged across layers.

    Args:
        n_layers: number of layers to simulate

    Returns:
        gradient magnitudes: list of floats (stays near 1.0)
    """
    INIT_GRADIENT = 1.0
    gradient      = INIT_GRADIENT
    magnitudes    = [gradient]

    for layer in range(1, n_layers + 1):
        gradient *= 1.0           # ReLU derivative is 1 for active neurons
        magnitudes.append(gradient)

    return magnitudes
