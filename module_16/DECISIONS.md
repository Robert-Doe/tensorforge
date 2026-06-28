# Module 16 — DECISIONS.md

## Decision 1: Neuron = LogisticRegression (intentional)
**Decision:** Make the single neuron mathematically identical to the M05 LogisticRegression.
**Why:** The biggest conceptual hurdle in DL is "what IS a neuron?" The answer is: a logistic regression unit. By making this explicit, students who struggled with M05 can revisit it here; students who got M05 can immediately grasp the neuron. The novelty comes in M18 when neurons are stacked.
**Trade-off:** May feel redundant to students who fully understood M05. Addressed by framing: "same math, different vocabulary — now you know both vocabularies."

## Decision 2: Random weight initialisation, not zero
**Decision:** Initialise weights with small random Gaussian values (mean=0, std=0.01).
**Why:** Zero initialisation causes a symmetry problem in multi-layer networks — all neurons receive identical gradients and learn identical features. Introducing this here for a single neuron (where it doesn't matter technically) prepares the student for M18 where it does matter critically.
**Trade-off:** For a single neuron, zero init would work fine. The small random init is forward-looking.

## Decision 3: Show sigmoid AND ReLU, train with sigmoid only
**Decision:** Demonstrate both activation functions visually but only train the neuron with sigmoid.
**Why:** ReLU is the dominant hidden-layer activation in modern networks, but for a binary output neuron sigmoid is the correct choice (outputs are probabilities). Showing both prepares the student for M19 (Activation Functions) without conflating output and hidden activations yet.
**Trade-off:** Student may wonder "why not ReLU?" — answered in the tutorial.

## Decision 4: Manual scaling in main.py, not a Pipeline
**Decision:** Scale features manually (mean/std from X_train) rather than wrapping in an sklearn Pipeline.
**Why:** In M16-M21 we build neural networks from scratch using only NumPy. sklearn Pipeline won't integrate with our custom Neuron class. Students learn to scale manually here so they understand what the Pipeline was doing for them.
**Trade-off:** More code; risk of leakage if student forgets to use only training stats. Clearly commented.
