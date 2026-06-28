"""module_20/training_loop.py — Mini-batch SGD training loop.

Wraps the M18 MLP with a proper mini-batch training loop:
  - Shuffles data each epoch
  - Splits into batches of BATCH_SIZE
  - Records loss per step AND per epoch
  - Tracks validation loss alongside training loss
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "module_18"))
from layers import MLP, BinaryCrossEntropyLoss

# ── Constants ────────────────────────────────────────────────────────────────
RANDOM_SEED = 42
BATCH_SIZE  = 32     # samples per gradient update
EPOCHS      = 100
LEARNING_RATE = 0.05


def make_batches(X, y, batch_size, rng):
    """Shuffle data and yield (X_batch, y_batch) tuples.

    Shuffling every epoch prevents the network from memorising
    the order of training examples.

    Args:
        X:          (n_samples, n_features)
        y:          (n_samples,)
        batch_size: number of samples per batch
        rng:        np.random.Generator for reproducible shuffling

    Yields:
        (X_batch, y_batch): tuple of arrays
    """
    idx      = rng.permutation(len(X))   # shuffle indices, not the data array
    X_shuf   = X[idx]
    y_shuf   = y[idx]

    n_batches = int(np.ceil(len(X) / batch_size))  # round up to include last batch
    for b in range(n_batches):
        start = b * batch_size
        end   = start + batch_size         # slice beyond end is fine — numpy clips it
        yield X_shuf[start:end], y_shuf[start:end]


class MiniBatchTrainer:
    """Trains an MLP using mini-batch stochastic gradient descent.

    Records:
        train_loss_per_epoch: list — one loss value per epoch (mean over batches)
        val_loss_per_epoch:   list — validation loss per epoch (full-batch, no shuffle)
        step_losses:          list — loss after every mini-batch update

    Attributes:
        model:                 MLP instance (from module_18/layers.py)
        batch_size:            int
        epochs:                int
        lr:                    float
    """

    def __init__(self, n_features, hidden_sizes, lr=LEARNING_RATE,
                 epochs=EPOCHS, batch_size=BATCH_SIZE):
        """Build the MLP and store training hyperparameters.

        Args:
            n_features:   input dimensionality
            hidden_sizes: list of hidden layer widths
            lr:           learning rate
            epochs:       number of full passes over training data
            batch_size:   samples per gradient update
        """
        self.epochs     = epochs
        self.batch_size = batch_size
        self.lr         = lr
        self.model      = MLP(n_in=n_features, hidden_sizes=hidden_sizes,
                              lr=lr, epochs=1)    # epochs=1: we control the loop here
        self.loss_fn    = BinaryCrossEntropyLoss()

        self.train_loss_per_epoch = []
        self.val_loss_per_epoch   = []
        self.step_losses          = []

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Run mini-batch training for self.epochs passes over the data.

        Args:
            X_train: (n_train, n_features)
            y_train: (n_train,)
            X_val:   optional validation features
            y_val:   optional validation labels

        Returns:
            self
        """
        rng = np.random.default_rng(RANDOM_SEED)

        for epoch in range(self.epochs):
            batch_losses = []

            for X_batch, y_batch in make_batches(X_train, y_train,
                                                  self.batch_size, rng):
                # Forward pass on this mini-batch
                probs = self.model.forward(X_batch)

                # Loss for this batch
                loss  = self.loss_fn.forward(probs, y_batch)
                batch_losses.append(loss)
                self.step_losses.append(loss)

                # Backward pass — gradients computed on batch only
                self.model.backward(probs, y_batch)
                self.model._update_weights()

            # Average loss over all batches in this epoch
            self.train_loss_per_epoch.append(float(np.mean(batch_losses)))

            # Validation loss — full batch, no weight update
            if X_val is not None and y_val is not None:
                val_probs = self.model.forward(X_val)
                val_loss  = self.loss_fn.forward(val_probs, y_val)
                self.val_loss_per_epoch.append(float(val_loss))

            if (epoch + 1) % 20 == 0:
                val_str = (f"  val_loss={self.val_loss_per_epoch[-1]:.4f}"
                           if X_val is not None else "")
                print(f"  Epoch {epoch+1:3d}/{self.epochs}  "
                      f"train_loss={self.train_loss_per_epoch[-1]:.4f}{val_str}")

        return self

    def predict(self, X):
        """Return binary class predictions."""
        return self.model.predict(X)

    def predict_proba(self, X):
        """Return predicted probabilities."""
        return self.model.predict_proba(X)
