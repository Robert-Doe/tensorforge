# Module 25 — DECISIONS.md

## Decision 1: Conv → BN → ReLU (not Conv → ReLU → BN)
**Decision:** Place BatchNorm2d between Conv2d and ReLU.
**Why:** BN normalises raw conv activations to N(0,1). ReLU then clips only the negative half of this centred distribution, which is the intended effect. If we put ReLU first, BN normalises an already-clipped (non-negative) distribution — the mean is no longer near zero, reducing BN's effectiveness. The original BN paper used this order; it remains the standard despite ongoing debate.
**Trade-off:** Some modern architectures (ResNets in particular) experiment with Pre-Activation (BN → ReLU → Conv). Both work; the classic order is chosen here for simplicity.

## Decision 2: Three-way ablation study (baseline, BN-only, BN+Dropout)
**Decision:** Train all three variants back-to-back and compare accuracy curves rather than just showing the "best" configuration.
**Why:** The pedagogical goal is demonstrating *what BN and Dropout each contribute*. Running a single "best" model would leave students unable to distinguish the effect of each component. The ablation makes the comparison concrete.
**Trade-off:** Three full training runs triples the runtime (~3× longer on CPU). Kept at 5 epochs to stay manageable.

## Decision 3: forward_hook to show BN statistics
**Decision:** Use `register_forward_hook` to capture conv1 output before and after BN, then print mean/std.
**Why:** The claim "BN normalises activations to ~N(0,1)" needs to be verified, not just asserted. Seeing mean≈0 and std≈1 in the terminal makes the abstraction concrete.
**Trade-off:** Hooks are an advanced API; students haven't seen them before. The code is self-contained in `demo_bn_statistics()` and not needed to understand the core lesson.

## Decision 4: Reuse Module 24's data directory
**Decision:** `DATA_DIR` points to `../module_24/data` so MNIST is only downloaded once across both modules.
**Why:** MNIST is ~12 MB; downloading it twice wastes time and bandwidth. Sharing the directory across modules mirrors real project organisation where datasets live in a central location.
**Trade-off:** Module 25 has a hard dependency on the module_24 folder being present. Documented in the tutorial; acceptable given the cumulative curriculum design.
