# Module 20 — DECISIONS.md

## Decision 1: Separate MiniBatchTrainer from MLP
**Decision:** Wrap the M18 MLP in a new MiniBatchTrainer class rather than adding mini-batch logic inside MLP.
**Why:** Separation of concerns — MLP owns the layer math; MiniBatchTrainer owns the training loop strategy. This mirrors real frameworks: PyTorch's nn.Module defines the model; the training loop lives in user code or a Trainer class. The pattern makes it easy to swap optimisers (SGD → Adam) later without touching the model.
**Trade-off:** Extra class to understand; mitigated because the class is thin (fit / predict only).

## Decision 2: Three-way split (train / val / test), not two-way
**Decision:** Reserve a validation set in addition to the test set.
**Why:** The validation loss curve reveals overfitting in real time during training — you can see it diverge from training loss. The test set remains held-out until the final evaluation. Using test loss to monitor training would constitute data leakage via hyperparameter tuning.
**Trade-off:** With only 120 samples, splitting three ways gives ~78 train / 18 val / 24 test. Each set is small; results are noisy. Acknowledged in tutorial.

## Decision 3: Full-batch as a special case of mini-batch (batch_size=n)
**Decision:** Implement full-batch GD by setting batch_size = len(X_train), not as separate code.
**Why:** Conceptually, full-batch GD and mini-batch SGD are the same algorithm with different batch sizes. Showing this explicitly (rather than two separate codepaths) reinforces the unification. It also means every comparison is code-for-code identical except the batch_size argument.
**Trade-off:** The make_batches generator does unnecessary permutation overhead when batch_size=n (you permute then take the whole dataset). Negligible on 120 samples.

## Decision 4: Print progress every 20 epochs
**Decision:** Log train_loss and val_loss every 20 epochs, not every epoch.
**Why:** Printing 100 lines of loss would flood the terminal and make it hard to see trends. Every 20 epochs gives 5 checkpoints — enough to see convergence without noise.
**Trade-off:** Students who want finer-grained logs can change the modulo condition; documented in code comment.
