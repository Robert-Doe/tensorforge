"""module_17/backprop.py — Backpropagation through a 2-layer network from scratch.

Network architecture:
    Input (d features)
      → Hidden layer (H neurons, ReLU activation)
      → Output layer (1 neuron, sigmoid activation)
      → Binary prediction

Forward pass:  Z1 = X@W1+b1  →  A1 = relu(Z1)  →  Z2 = A1@W2+b2  →  A2 = sigmoid(Z2)
Backward pass: dL/dA2  →  dL/dZ2  →  dL/dW2, dL/db2
                        →  dL/dA1  →  dL/dZ1  →  dL/dW1, dL/db1

The chain rule connects each layer's gradient to the next layer's.
"""

import numpy as np

# ── Constants ────────────────────────────────────────────────────────────────
RANDOM_SEED   = 42
LEARNING_RATE = 0.05
EPOCHS        = 1000
HIDDEN_SIZE   = 8     # number of neurons in the hidden layer


# ── Activations and their derivatives ────────────────────────────────────────

def sigmoid(z):
    """Sigmoid activation: maps z to (0,1)."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def relu(z):
    """ReLU activation: max(0, z)."""
    return np.maximum(0.0, z)


def relu_derivative(z):
    """Derivative of ReLU with respect to z.

    Gradient is 1 where z > 0, 0 where z <= 0.
    This is the step function — not differentiable at z=0,
    but in practice z==0 never occurs for random initialisation.

    Args:
        z: pre-activation values (NOT relu output)

    Returns:
        binary mask, same shape as z
    """
    return (z > 0).astype(float)


# ── Two-layer network ─────────────────────────────────────────────────────────

class TwoLayerNet:
    """A two-layer neural network trained with backpropagation.

    Layer 1: H hidden neurons with ReLU activation
    Layer 2: 1 output neuron with sigmoid activation

    Attributes:
        W1: (n_features, H) — hidden layer weights
        b1: (H,)            — hidden layer biases
        W2: (H, 1)          — output layer weights
        b2: (1,)            — output layer bias
        history_: list of loss per epoch
    """

    def __init__(self, n_features, hidden_size=HIDDEN_SIZE,
                 lr=LEARNING_RATE, epochs=EPOCHS):
        """Initialise with He-style weight initialisation for ReLU layers.

        Args:
            n_features:  number of input features
            hidden_size: H — number of hidden neurons
            lr:          learning rate
            epochs:      training iterations
        """
        self.lr      = lr
        self.epochs  = epochs
        self.history_ = []

        rng = np.random.default_rng(RANDOM_SEED)

        # He initialisation: std = sqrt(2 / n_in) — optimal for ReLU layers
        # Prevents activations from exploding or vanishing at initialisation
        self.W1 = rng.normal(0, np.sqrt(2.0 / n_features), (n_features, hidden_size))
        self.b1 = np.zeros(hidden_size)

        # Xavier initialisation for sigmoid output layer
        self.W2 = rng.normal(0, np.sqrt(1.0 / hidden_size), (hidden_size, 1))
        self.b2 = np.zeros(1)

    # ── Forward pass ─────────────────────────────────────────────────────────

    def forward(self, X):
        """Run input X through both layers and return all intermediate values.

        We save Z1 and A1 because the backward pass needs them.

        Args:
            X: (n_samples, n_features)

        Returns:
            Z1: (n, H)  — pre-activation hidden layer
            A1: (n, H)  — post-activation hidden layer (relu output)
            Z2: (n, 1)  — pre-activation output layer
            A2: (n, 1)  — post-activation output (sigmoid = probability)
        """
        Z1 = X  @ self.W1 + self.b1     # (n, H) — hidden pre-activation
        A1 = relu(Z1)                    # (n, H) — hidden activation

        Z2 = A1 @ self.W2 + self.b2     # (n, 1) — output pre-activation
        A2 = sigmoid(Z2)                 # (n, 1) — output probability

        return Z1, A1, Z2, A2

    # ── Backward pass ─────────────────────────────────────────────────────────

    def backward(self, X, y, Z1, A1, A2):
        """Compute gradients for all weights using the chain rule.

        Chain rule notation:
            dL/dW2 = dL/dA2 · dA2/dZ2 · dZ2/dW2

        Each layer's gradient depends on the layer above it —
        this is why it's called *back*propagation.

        Args:
            X:  (n, d) — input data
            y:  (n,)   — binary labels
            Z1: (n, H) — hidden pre-activation (saved from forward)
            A1: (n, H) — hidden activation     (saved from forward)
            A2: (n, 1) — output probability    (saved from forward)

        Returns:
            dict of gradients: dW1, db1, dW2, db2
        """
        n = X.shape[0]
        y_col = y.reshape(-1, 1)    # make y a column vector for broadcasting

        # ── Output layer gradients ────────────────────────────────────────────
        # dL/dZ2 = A2 - y  (simplified: cross-entropy loss + sigmoid output)
        dZ2 = A2 - y_col            # (n, 1)
        dW2 = (A1.T @ dZ2) / n     # (H, 1) — chain: dZ2 * dZ2/dW2=A1
        db2 = dZ2.mean(axis=0)     # (1,)

        # ── Hidden layer gradients ────────────────────────────────────────────
        # dL/dA1 = dZ2 @ W2.T   — propagate error back through output weights
        dA1 = dZ2 @ self.W2.T      # (n, H)
        # dL/dZ1 = dA1 * ReLU'(Z1)  — ReLU gate: zero gradient where Z1<=0
        dZ1 = dA1 * relu_derivative(Z1)   # (n, H) — element-wise
        dW1 = (X.T @ dZ1) / n     # (d, H)
        db1 = dZ1.mean(axis=0)    # (H,)

        return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    # ── Training loop ─────────────────────────────────────────────────────────

    def fit(self, X, y):
        """Train the network using gradient descent + backpropagation.

        Args:
            X: (n_samples, n_features)
            y: (n_samples,) binary labels

        Returns:
            self
        """
        for epoch in range(self.epochs):
            # Forward
            Z1, A1, Z2, A2 = self.forward(X)

            # Loss
            a_clipped = np.clip(A2, 1e-9, 1 - 1e-9)
            y_col     = y.reshape(-1, 1)
            loss      = -np.mean(y_col * np.log(a_clipped) +
                                 (1 - y_col) * np.log(1 - a_clipped))
            self.history_.append(float(loss))

            # Backward
            grads = self.backward(X, y, Z1, A1, A2)

            # Update weights
            self.W1 -= self.lr * grads["dW1"]
            self.b1 -= self.lr * grads["db1"]
            self.W2 -= self.lr * grads["dW2"]
            self.b2 -= self.lr * grads["db2"]

        return self

    def predict_proba(self, X):
        """Return output probabilities."""
        _, _, _, A2 = self.forward(X)
        return A2.ravel()

    def predict(self, X):
        """Return binary class predictions."""
        return (self.predict_proba(X) >= 0.5).astype(int)
