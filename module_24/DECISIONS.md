# Module 24 — DECISIONS.md

## Decision 1: NLLLoss + log_softmax instead of CrossEntropyLoss + raw logits
**Decision:** Output `F.log_softmax(x, dim=1)` from the model and use `nn.NLLLoss`.
**Why:** PyTorch's `nn.CrossEntropyLoss` internally applies `log_softmax` + NLL, making the model's output layer opaque. Splitting them makes the pipeline explicit: the model outputs probabilities (via `argmax`), and the loss evaluates those log-probabilities. Students can see the prediction step independently from the loss step.
**Trade-off:** PyTorch docs recommend `CrossEntropyLoss` for slight numerical advantage (numerically fused). The difference is negligible at this scale, and explicitness wins for pedagogy.

## Decision 2: Adam optimizer instead of SGD
**Decision:** Use `optim.Adam(lr=1e-3)` rather than SGD with momentum.
**Why:** Module 22 already introduced SGD with momentum. Adam converges faster on MNIST (typically 98%+ in 3 epochs vs 5+ for SGD), letting students see clear progress within a short run. The principle is the same — gradient-based update — but Adam adapts the learning rate per parameter, removing the need to hand-tune lr and momentum.
**Trade-off:** SGD with careful tuning can match or exceed Adam in final accuracy and generalises better. Covered conceptually in the tutorial; SGD comparison left as an exercise.

## Decision 3: padding=1 in both Conv layers (same-padding)
**Decision:** Both `nn.Conv2d` layers use `padding=1` with a 3×3 kernel.
**Why:** Same-padding keeps spatial dimensions constant after each conv layer (28→28, 14→14). Without it, spatial size shrinks by 2 per conv — hard to reason about. MaxPool is then solely responsible for downsampling, making the architecture easier to explain: conv extracts features, pool reduces spatial size.
**Trade-off:** Slight increase in computation. At 28×28 the cost is negligible.

## Decision 4: Shared MaxPool layer
**Decision:** One `self.pool = nn.MaxPool2d(2, 2)` instance is called twice in `forward()`.
**Why:** MaxPool has no learnable parameters, so reusing the same layer object produces identical behaviour. It avoids defining `pool1` and `pool2` as separate attributes, which would mislead students into thinking they hold separate learned state.
**Trade-off:** Could confuse students expecting one-layer-per-forward-call symmetry. Addressed in the tutorial.
