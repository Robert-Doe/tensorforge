# Module 18 — DECISIONS.md

## Decision 1: Layer abstraction (forward/backward per object)
**Decision:** Each layer is a Python object with forward() and backward() methods rather than inline code.
**Why:** The layer abstraction is the core design pattern of every DL framework (PyTorch nn.Module, Keras Layer). Learning it here in NumPy means students already understand what PyTorch is doing in M22 — they've seen the same pattern without the magic. MLP.forward() is just a loop over layers; MLP.backward() is the same loop reversed.
**Trade-off:** More classes and files than the M17 monolithic approach. Worth it for the abstraction payoff.

## Decision 2: XOR problem as the first benchmark
**Decision:** Demonstrate the MLP's power on the XOR problem before applying it to real data.
**Why:** XOR is the canonical "single neuron fails, MLP succeeds" example. It's provably impossible to linearly separate XOR, making it the perfect motivator for hidden layers. The result is crisp: single neuron ≤75%, 2-layer MLP 100%.
**Trade-off:** XOR is a toy problem with 4 samples — no train/test split possible. Used for illustration only; detective data provides the real benchmark.

## Decision 3: Compare three architectures ([8], [16,8], [32,16,8])
**Decision:** Train three MLP architectures and print their train/test accuracy side by side.
**Why:** Shows that deeper/wider isn't always better on small datasets. The [16,8] architecture may already overfit on 96 training samples. Students see the overfitting signature (train_acc >> test_acc) on the deepest architecture.
**Trade-off:** Three training runs slows the module down slightly; mitigated by using only 1500 epochs.

## Decision 4: BinaryCrossEntropyLoss combined with SigmoidLayer backward
**Decision:** The combined sigmoid+BCE backward simplifies to (probs - y) / n rather than computing sigmoid_derivative then BCE_derivative separately.
**Why:** The algebraic simplification (dL/dZ = a - y) is numerically more stable than multiplying two tiny numbers. This is what PyTorch's BCEWithLogitsLoss uses internally. Showing the simplification teaches a real production trick.
**Trade-off:** Slightly less educational — students don't see the two-step chain explicitly. Offset by DECISIONS.md and tutorial explanation.
