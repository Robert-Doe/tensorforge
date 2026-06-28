# Module 17 — DECISIONS.md

## Decision 1: 2-layer network (one hidden layer), not deeper
**Decision:** Implement exactly one hidden layer (input → hidden → output).
**Why:** Two layers are the minimum to demonstrate backpropagation through multiple layers. Adding more layers before explaining the vanishing gradient problem (M19) would introduce confusion about why deep networks need special tricks. One hidden layer is sufficient to show the chain rule in action.
**Trade-off:** Doesn't demonstrate why depth helps; that comes in M18-M21.

## Decision 2: He initialisation for ReLU, Xavier for sigmoid
**Decision:** Use He init (std=√(2/n_in)) for the ReLU hidden layer and Xavier init (std=√(1/n_in)) for the sigmoid output layer.
**Why:** Poor weight initialisation causes exploding or vanishing activations at layer 0, before training even starts. He init is provably optimal for ReLU; Xavier is optimal for symmetric activations like sigmoid and tanh. Using the wrong init here would make training unstable and obscure the backprop lesson.
**Trade-off:** Slightly more complex __init__; worth it because weight init is a real-world concern students will immediately encounter.

## Decision 3: Save Z1 in forward(), not just A1
**Decision:** The forward() method returns Z1 (pre-activation) as well as A1 (post-activation).
**Why:** The ReLU derivative is computed on Z1, not A1. Students must see that backpropagation needs the *pre-activation* values. This is why real DL frameworks cache intermediate values in memory during the forward pass — the backward pass needs them all.
**Trade-off:** More return values from forward(); managed by returning a 4-tuple and naming them consistently.

## Decision 4: Print a gradient trace for one sample
**Decision:** print_gradient_trace() shows actual gradient magnitudes flowing backward.
**Why:** Backprop is abstract until you see real numbers flowing. Showing ||dW2|| > ||dW1|| (because dW1 passes through an extra multiplication) builds intuition for why gradients shrink as they travel further from the output — motivating the vanishing gradient discussion in M19.
**Trade-off:** Adds output noise; clearly sectioned and labelled.
