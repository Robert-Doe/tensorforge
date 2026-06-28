"""module_18/layers.py — Reusable layer abstraction for a from-scratch MLP.

Each layer has:
  forward(X)  — compute output, cache inputs for backward
  backward(dOut) — receive upstream gradient, return downstream gradient
  params / grads — weights and their gradients for the optimizer

This pattern mirrors how PyTorch nn.Module works internally.
"""

import numpy as np

RANDOM_SEED = 42


class DenseLayer:
    """Fully-connected (dense) linear layer: out = X @ W + b.

    Stores the input during forward() so backward() can compute dW = X.T @ dOut.

    Attributes:
        W:      (n_in, n_out) weight matrix
        b:      (n_out,) bias vector
        dW:     gradient of loss w.r.t. W (computed in backward)
        db:     gradient of loss w.r.t. b (computed in backward)
        _cache: saved input from the last forward() call
    """

    def __init__(self, n_in, n_out, init="he"):
        """Initialise weights.

        Args:
            n_in:  number of input features
            n_out: number of output neurons
            init:  'he' for ReLU layers, 'xavier' for sigmoid/tanh layers
        """
        rng = np.random.default_rng(RANDOM_SEED)
        if init == "he":
            std = np.sqrt(2.0 / n_in)        # He — optimal for ReLU
        else:
            std = np.sqrt(1.0 / n_in)        # Xavier — optimal for sigmoid

        self.W  = rng.normal(0, std, (n_in, n_out))
        self.b  = np.zeros(n_out)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._cache = None

    def forward(self, X):
        """Linear transform: out = X @ W + b.

        Args:
            X: (n_samples, n_in)

        Returns:
            out: (n_samples, n_out)
        """
        self._cache = X                      # save for backward
        return X @ self.W + self.b

    def backward(self, dOut):
        """Compute gradients and propagate error to previous layer.

        Args:
            dOut: (n_samples, n_out) — gradient from the layer above

        Returns:
            dX: (n_samples, n_in) — gradient to pass to the layer below
        """
        X  = self._cache
        n  = X.shape[0]
        self.dW = (X.T @ dOut) / n          # gradient for weights
        self.db = dOut.mean(axis=0)         # gradient for biases
        return dOut @ self.W.T              # gradient for inputs (pass backward)


class ReLULayer:
    """ReLU activation layer: out = max(0, z).

    Stateless during training — only needs to remember which inputs were positive.
    """

    def __init__(self):
        self._cache = None

    def forward(self, X):
        """Apply ReLU element-wise.

        Args:
            X: any shape

        Returns:
            out: same shape, negatives zeroed
        """
        self._cache = X                      # save pre-activation for backward
        return np.maximum(0.0, X)

    def backward(self, dOut):
        """Pass gradient only through neurons that were active (Z > 0).

        Args:
            dOut: gradient from above, same shape as X

        Returns:
            dX: gradient masked by ReLU derivative
        """
        return dOut * (self._cache > 0)      # zero out where ReLU was inactive


class SigmoidLayer:
    """Sigmoid output activation: out = 1 / (1 + e^-z)."""

    def __init__(self):
        self._cache = None

    def forward(self, X):
        """Apply sigmoid element-wise."""
        out          = 1.0 / (1.0 + np.exp(-np.clip(X, -500, 500)))
        self._cache  = out                   # save output (not input) — deriv uses a
        return out

    def backward(self, dOut):
        """Sigmoid derivative: a * (1 - a)."""
        a = self._cache
        return dOut * a * (1.0 - a)


class BinaryCrossEntropyLoss:
    """Binary cross-entropy loss with sigmoid output.

    The combined sigmoid + BCE backward pass simplifies to (a - y),
    avoiding the numerically unstable product of two small numbers.
    """

    def forward(self, probs, y):
        """Compute mean binary cross-entropy.

        Args:
            probs: (n_samples, 1) predicted probabilities
            y:     (n_samples,)   binary labels

        Returns:
            float loss value
        """
        p = np.clip(probs, 1e-9, 1 - 1e-9)
        y_col = y.reshape(-1, 1)
        return float(-np.mean(y_col * np.log(p) + (1 - y_col) * np.log(1 - p)))

    def backward(self, probs, y):
        """Gradient of BCE w.r.t. the sigmoid input (combined simplification).

        Returns:
            dZ_out: (n_samples, 1)
        """
        return (probs - y.reshape(-1, 1)) / len(y)


class MLP:
    """Multi-Layer Perceptron built from DenseLayer + activation layers.

    Architecture is defined by hidden_sizes list:
        hidden_sizes=[16, 8] → input → Dense(16) → ReLU → Dense(8) → ReLU → Dense(1) → Sigmoid

    Attributes:
        layers:   ordered list of layer objects (Dense + activation, alternating)
        history_: list of loss values per epoch
    """

    def __init__(self, n_in, hidden_sizes, lr=0.05, epochs=1000):
        """Build layer stack.

        Args:
            n_in:         number of input features
            hidden_sizes: list of hidden layer widths, e.g. [16, 8]
            lr:           learning rate
            epochs:       training iterations
        """
        self.lr       = lr
        self.epochs   = epochs
        self.history_ = []
        self.layers   = []

        sizes = [n_in] + hidden_sizes
        for i in range(len(sizes) - 1):
            self.layers.append(DenseLayer(sizes[i], sizes[i+1], init="he"))
            self.layers.append(ReLULayer())

        # output: single neuron with sigmoid for binary classification
        self.layers.append(DenseLayer(sizes[-1], 1, init="xavier"))
        self.layers.append(SigmoidLayer())

        self.loss_fn = BinaryCrossEntropyLoss()

    def forward(self, X):
        """Run X through all layers sequentially.

        Args:
            X: (n_samples, n_features)

        Returns:
            out: (n_samples, 1) predicted probabilities
        """
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, probs, y):
        """Backpropagate loss through all layers in reverse order.

        Args:
            probs: (n_samples, 1) output of forward()
            y:     (n_samples,)   binary labels
        """
        grad = self.loss_fn.backward(probs, y)
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def _update_weights(self):
        """Apply gradient descent update to all DenseLayer weights."""
        for layer in self.layers:
            if isinstance(layer, DenseLayer):
                layer.W -= self.lr * layer.dW
                layer.b -= self.lr * layer.db

    def fit(self, X, y):
        """Train the MLP with full-batch gradient descent.

        Args:
            X: (n_samples, n_features)
            y: (n_samples,) binary labels

        Returns:
            self
        """
        for epoch in range(self.epochs):
            probs = self.forward(X)
            loss  = self.loss_fn.forward(probs, y)
            self.history_.append(loss)
            self.backward(probs, y)
            self._update_weights()
        return self

    def predict_proba(self, X):
        """Return predicted probabilities."""
        return self.forward(X).ravel()

    def predict(self, X):
        """Return binary class predictions."""
        return (self.predict_proba(X) >= 0.5).astype(int)
