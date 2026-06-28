"""module_16/neuron.py — A single artificial neuron from scratch.

A neuron computes:
    z = w · x + b          (linear combination — "weighted evidence sum")
    a = activation(z)      (non-linear squash — "decision")

For binary classification we use sigmoid as the activation,
making this neuron mathematically identical to Logistic Regression (M05)
— which is exactly the point.
"""

import numpy as np

# ── Constants ────────────────────────────────────────────────────────────────
RANDOM_SEED     = 42
LEARNING_RATE   = 0.1
EPOCHS          = 500
DECISION_BOUNDARY = 0.5


# ── Activation functions ────────────────────────────────────────────────────

def sigmoid(z):
    """Map any real number to (0, 1).

    Used as the output activation for binary classification.
    The neuron's "confidence" that the answer is 1.

    Args:
        z: scalar or ndarray — pre-activation value

    Returns:
        float or ndarray in (0, 1)
    """
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
    # clip prevents overflow in exp for very large |z|


def relu(z):
    """Rectified Linear Unit — max(0, z).

    The most common hidden-layer activation.
    Gradient is 1 for z>0, 0 for z<0 — simple and fast.

    Args:
        z: scalar or ndarray

    Returns:
        same shape as z, all negative values replaced with 0
    """
    return np.maximum(0.0, z)


def sigmoid_derivative(a):
    """Derivative of sigmoid with respect to z, expressed in terms of output a.

    d(sigmoid)/dz = a * (1 - a)

    Used during backpropagation to compute how much to adjust weights.

    Args:
        a: sigmoid output (not z!) — value in (0, 1)

    Returns:
        float or ndarray — same shape as a
    """
    return a * (1.0 - a)


# ── Single Neuron ────────────────────────────────────────────────────────────

class Neuron:
    """A single artificial neuron with sigmoid activation.

    This is deliberately identical to LogisticRegression from M05.
    The goal is to show that a neuron IS a logistic regression unit —
    the difference only appears when you stack multiple neurons in layers.

    Attributes:
        w:        (n_features,) weight vector — learnt during fit()
        b:        float — bias term — learnt during fit()
        lr:       float — learning rate (step size per gradient update)
        epochs:   int   — number of full passes over training data
        history_: list of float — binary cross-entropy loss per epoch
    """

    def __init__(self, lr=LEARNING_RATE, epochs=EPOCHS):
        """Initialise neuron with random weights.

        Args:
            lr:     learning rate
            epochs: training iterations
        """
        self.lr      = lr
        self.epochs  = epochs
        self.w       = None
        self.b       = 0.0
        self.history_ = []

    def _init_weights(self, n_features):
        """Initialise weights to small random values.

        Zero initialisation would cause all neurons in a layer to learn
        identically — symmetry breaking requires random init.
        Small values keep the initial sigmoid output near 0.5 (uncertain).
        """
        rng    = np.random.default_rng(RANDOM_SEED)
        self.w = rng.normal(0, 0.01, size=n_features)  # tiny random weights
        self.b = 0.0

    def forward(self, X):
        """Compute the neuron's output for input X.

        Args:
            X: (n_samples, n_features)

        Returns:
            a: (n_samples,) — sigmoid output, probability of class 1
        """
        z = X @ self.w + self.b    # linear combination: weighted sum + bias
        return sigmoid(z)          # squash to (0,1)

    def fit(self, X, y):
        """Train using gradient descent on binary cross-entropy loss.

        Gradient derivation (same as M05 LogisticRegression):
            loss   = -mean(y*log(a) + (1-y)*log(1-a))
            dL/dw  = (1/n) * X.T @ (a - y)
            dL/db  = (1/n) * sum(a - y)

        Args:
            X: (n_samples, n_features)
            y: (n_samples,) binary labels {0, 1}

        Returns:
            self
        """
        n, d = X.shape
        self._init_weights(d)

        for epoch in range(self.epochs):
            # ── Forward pass ────────────────────────────────────────────────
            a = self.forward(X)                     # (n,) predicted probabilities

            # ── Loss (binary cross-entropy) ──────────────────────────────────
            a_clipped = np.clip(a, 1e-9, 1 - 1e-9) # prevent log(0)
            loss = -np.mean(y * np.log(a_clipped) +
                            (1 - y) * np.log(1 - a_clipped))
            self.history_.append(float(loss))

            # ── Gradients ───────────────────────────────────────────────────
            error  = a - y                          # (n,) residuals
            grad_w = (X.T @ error) / n             # (d,) weight gradient
            grad_b = error.mean()                  # scalar bias gradient

            # ── Weight update (gradient descent) ────────────────────────────
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b

        return self

    def predict_proba(self, X):
        """Return probability of class 1 for each sample.

        Args:
            X: (n_samples, n_features)

        Returns:
            (n_samples,) floats in (0, 1)
        """
        return self.forward(X)

    def predict(self, X):
        """Return class prediction (0 or 1) using DECISION_BOUNDARY threshold.

        Args:
            X: (n_samples, n_features)

        Returns:
            (n_samples,) int array of {0, 1}
        """
        return (self.predict_proba(X) >= DECISION_BOUNDARY).astype(int)
