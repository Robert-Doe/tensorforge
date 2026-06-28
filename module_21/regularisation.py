"""module_21/regularisation.py — L2 weight decay and Dropout from scratch.

Extends the M18 MLP layer abstraction with two regularisation techniques:
  - L2 penalty: adds lambda * ||W||^2 to the loss, shrinking weights toward zero
  - Dropout: randomly zeroes a fraction of neurons during training only
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "module_18"))
from layers import DenseLayer, ReLULayer, SigmoidLayer, BinaryCrossEntropyLoss

RANDOM_SEED = 42


class DropoutLayer:
    """Dropout: randomly zero out neurons during training.

    During training:  each neuron is kept with probability (1-p), zeroed otherwise.
                      Kept values are scaled up by 1/(1-p) so the expected output
                      stays the same — this is called 'inverted dropout'.
    During inference: no dropout — all neurons active.

    Attributes:
        p:         dropout probability (fraction of neurons to zero out)
        training:  bool — if False, forward() is a no-op passthrough
        _mask:     binary mask from last forward() — needed for backward()
    """

    def __init__(self, p=0.5):
        """Initialise Dropout.

        Args:
            p: probability of zeroing a neuron (0 = no dropout, 0.9 = very aggressive)
        """
        assert 0.0 <= p < 1.0, "Dropout probability must be in [0, 1)"
        self.p        = p
        self.training = True    # set to False during evaluation
        self._mask    = None
        self._rng     = np.random.default_rng(RANDOM_SEED)

    def forward(self, X):
        """Apply dropout mask (training only).

        Args:
            X: (n_samples, n_features) — input activations

        Returns:
            out: same shape as X, with p fraction zeroed and kept values scaled
        """
        if not self.training:
            return X             # inference: no dropout, no scaling

        # Bernoulli mask: 1 with probability (1-p), 0 with probability p
        self._mask = (self._rng.random(X.shape) >= self.p).astype(float)
        # Inverted dropout: divide by (1-p) so expected value is unchanged
        return X * self._mask / (1.0 - self.p)

    def backward(self, dOut):
        """Apply the same mask to the incoming gradient.

        Neurons that were zeroed in the forward pass also get zeroed gradients.

        Args:
            dOut: upstream gradient, same shape as X

        Returns:
            dX: masked gradient
        """
        if not self.training:
            return dOut
        return dOut * self._mask / (1.0 - self.p)


class RegularisedMLP:
    """MLP with optional L2 weight decay and Dropout.

    Architecture: Dense → ReLU → [Dropout] → Dense → ReLU → [Dropout] → Dense → Sigmoid

    Attributes:
        l2_lambda:   L2 regularisation strength (0 = no regularisation)
        dropout_p:   Dropout probability per hidden layer (0 = no dropout)
        layers:      ordered list of layer objects
        history_:    dict with keys 'train', 'val' — lists of epoch losses
    """

    def __init__(self, n_features, hidden_sizes, lr=0.05, epochs=200,
                 l2_lambda=0.0, dropout_p=0.0):
        """Build regularised MLP.

        Args:
            n_features:   input dimensionality
            hidden_sizes: list of hidden layer widths
            lr:           learning rate
            epochs:       training iterations
            l2_lambda:    L2 penalty coefficient (try 0.001 – 0.01)
            dropout_p:    fraction of neurons to drop per hidden layer (try 0.2 – 0.5)
        """
        self.lr         = lr
        self.epochs     = epochs
        self.l2_lambda  = l2_lambda
        self.dropout_p  = dropout_p
        self.history_   = {"train": [], "val": []}
        self.layers     = []
        self._dense_layers = []    # keep reference for L2 gradient penalty

        sizes = [n_features] + hidden_sizes
        for i in range(len(sizes) - 1):
            dl = DenseLayer(sizes[i], sizes[i+1], init="he")
            self.layers.append(dl)
            self._dense_layers.append(dl)
            self.layers.append(ReLULayer())
            if dropout_p > 0.0:
                self.layers.append(DropoutLayer(p=dropout_p))

        out_layer = DenseLayer(sizes[-1], 1, init="xavier")
        self.layers.append(out_layer)
        self._dense_layers.append(out_layer)
        self.layers.append(SigmoidLayer())

        self.loss_fn = BinaryCrossEntropyLoss()

    def _set_training(self, mode: bool):
        """Switch all DropoutLayers between training and eval mode."""
        for layer in self.layers:
            if isinstance(layer, DropoutLayer):
                layer.training = mode

    def forward(self, X):
        """Run X through all layers."""
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, probs, y):
        """Backpropagate and add L2 gradient to dense layers."""
        grad = self.loss_fn.backward(probs, y)
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

        # L2 regularisation: add lambda * W to each dense layer's dW
        # Derived from d/dW [loss + lambda/2 * ||W||^2] = dL/dW + lambda * W
        for dl in self._dense_layers:
            dl.dW += self.l2_lambda * dl.W   # gradient of the penalty term

    def _update_weights(self):
        """Gradient descent update for all DenseLayer weights."""
        for layer in self.layers:
            if isinstance(layer, DenseLayer):
                layer.W -= self.lr * layer.dW
                layer.b -= self.lr * layer.db

    def _l2_penalty(self):
        """Compute L2 regularisation term: lambda/2 * sum(W^2).

        Added to the loss for logging — weights are NOT penalised via this
        path (the gradient is added in backward() directly).

        Returns:
            float: regularisation penalty
        """
        penalty = 0.0
        for dl in self._dense_layers:
            penalty += float(np.sum(dl.W ** 2))
        return 0.5 * self.l2_lambda * penalty

    def fit(self, X_train, y_train, X_val=None, y_val=None, batch_size=32):
        """Train with mini-batch SGD, L2 decay, and dropout.

        Args:
            X_train, y_train: training data
            X_val,   y_val:   optional validation data
            batch_size:       samples per gradient step

        Returns:
            self
        """
        rng = np.random.default_rng(RANDOM_SEED)

        for epoch in range(self.epochs):
            self._set_training(True)
            idx  = rng.permutation(len(X_train))
            Xs   = X_train[idx]
            ys   = y_train[idx]

            batch_losses = []
            for start in range(0, len(X_train), batch_size):
                Xb    = Xs[start:start + batch_size]
                yb    = ys[start:start + batch_size]
                probs = self.forward(Xb)
                loss  = self.loss_fn.forward(probs, yb) + self._l2_penalty()
                batch_losses.append(loss)
                self.backward(probs, yb)
                self._update_weights()

            self.history_["train"].append(float(np.mean(batch_losses)))

            if X_val is not None:
                self._set_training(False)          # no dropout during validation
                vp = self.forward(X_val)
                vl = self.loss_fn.forward(vp, y_val)
                self.history_["val"].append(float(vl))
                self._set_training(True)

        return self

    def predict(self, X):
        """Return binary class predictions (eval mode — no dropout)."""
        self._set_training(False)
        return (self.forward(X).ravel() >= 0.5).astype(int)

    def predict_proba(self, X):
        """Return predicted probabilities (eval mode)."""
        self._set_training(False)
        return self.forward(X).ravel()
