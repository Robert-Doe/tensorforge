# Module 19 — DECISIONS.md

## Decision 1: No data loading — this module is about functions, not data
**Decision:** Module 19 does not load the detective dataset. It only analyses mathematical functions.
**Why:** Activation functions are mathematical objects, not dataset-dependent. Forcing artificial classification tasks would distract from the gradient analysis. The vanishing gradient demonstration is more convincing as a pure numerical simulation.
**Trade-off:** No real-world benchmark this module; compensated by the clear connection to the detective dataset in M18 and M20.

## Decision 2: Log scale for vanishing gradient plot
**Decision:** Use a logarithmic y-axis for the vanishing gradient comparison plot.
**Why:** After 12 sigmoid layers, the gradient is approximately 5.96e-8 — on a linear scale it would appear indistinguishable from zero at layer 2. The log scale reveals the exponential decay, making the problem visually undeniable.
**Trade-off:** Log scales can confuse beginners; labelled clearly and explained in the tutorial.

## Decision 3: Show 5 activations (not just sigmoid and ReLU)
**Decision:** Demonstrate Sigmoid, Tanh, ReLU, Leaky ReLU, and ELU.
**Why:** Each solves a specific limitation of the previous: Tanh is zero-centred (better than Sigmoid), ReLU avoids saturation, Leaky ReLU fixes dead neurons, ELU smooths the negative side. Showing the progression makes each one purposeful.
**Trade-off:** 5 activations × 2 rows (function + gradient) = 10 subplots. Wide figure required.

## Decision 4: Conservative vanishing gradient simulation (z=0 case)
**Decision:** Use derivative = 0.25 (sigmoid at z=0) for all layers in the simulation, not a random z value.
**Why:** z=0 gives the MAXIMUM sigmoid derivative (0.25). In practice, saturated neurons (|z|>>0) have derivatives near zero, making the problem even worse than simulated. The simulation is therefore conservative — reality is worse.
**Trade-off:** Slightly misleading if students think "well, z isn't always 0". Addressed in tutorial text.
