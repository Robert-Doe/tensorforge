# Module 21 — DECISIONS.md

## Decision 1: Inverted dropout (scale at training, not inference)
**Decision:** Divide kept activations by (1-p) during training, making inference a no-op.
**Why:** The alternative (scale by (1-p) at inference) requires modifying the model at deployment. Inverted dropout keeps the inference path identical to a standard forward pass — no extra operation needed when shipping the model. This is the approach used by PyTorch, TensorFlow, and every modern framework.
**Trade-off:** The training forward pass does slightly more work (one extra division), but this is negligible compared to the matrix multiplications.

## Decision 2: L2 gradient added in backward(), not as a loss term
**Decision:** Add `lambda * W` directly to `dW` in the backward pass rather than computing L2 loss and differentiating it separately.
**Why:** The L2 penalty's gradient is analytically `lambda * W`. Adding it to dW in backward() is numerically equivalent to differentiating `loss + lambda/2 * ||W||^2`, but avoids computing the penalty in the forward pass for purposes other than logging. This is exactly how weight decay is implemented in PyTorch's SGD optimizer (`weight_decay` parameter).
**Trade-off:** The L2 penalty IS computed in forward() for logging purposes only (so loss curves show regularised loss). Two code paths — clear in comments.

## Decision 3: Use an overfit-prone architecture [64, 32, 16]
**Decision:** Intentionally use a large architecture (3 hidden layers, 64→32→16 neurons) that will overfit on 78 training samples.
**Why:** You cannot demonstrate regularisation without first having a model that overfits. The [64,32,16] architecture has many more parameters than the dataset warrants, making overfitting visible (train_acc >> val_acc with no regularisation). Regularisation then visibly closes the gap.
**Trade-off:** The unregularised baseline may look bad — that's the point.

## Decision 4: Compare four configs in one run
**Decision:** Show None, L2-only, Dropout-only, and L2+Dropout side by side.
**Why:** Each combination adds one more layer of regularisation. The progression (worse → better → good → best) tells a story. If only L2 were shown, students wouldn't know if Dropout was unnecessary.
**Trade-off:** Four training runs is slow (~30 seconds on CPU); mitigated by limiting to 200 epochs.
