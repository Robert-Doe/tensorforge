# Module 27 — DECISIONS.md

## Decision 1: Vanilla RNN in NumPy first, then LSTM in PyTorch
**Decision:** Part 1 implements the bare recurrent equation in NumPy; Part 2 uses nn.LSTM.
**Why:** nn.LSTM is a black box with four gates and two state vectors. Without seeing the simpler `h_t = tanh(W_hh @ h_prev + W_xh @ x_t + b_h)` first, students have no mental model for what "recurrent" means. The NumPy version makes the state update equation concrete. LSTM then appears as "the same idea, but with gating to control what to remember and forget."
**Trade-off:** The NumPy RNN is not trained (training a character-level RNN requires BPTT — backprop through time — which is another module in itself). It only demonstrates the forward pass and hidden state accumulation.

## Decision 2: Synthetic sentiment dataset (no download)
**Decision:** Generate synthetic movie reviews from word banks instead of downloading IMDB (25 MB+).
**Why:** Module 27's primary lesson is the LSTM architecture, not dataset wrangling. A synthetic dataset keeps the module self-contained, removes network dependency, and runs in seconds. The vocabulary size (~100 words) is intentionally tiny so the embedding layer is small and training is fast.
**Trade-off:** Synthetic reviews have an unrealistically simple distribution — accuracy will be high (>90%) because the vocabulary of positive vs negative words barely overlaps. A real IMDB run would show more realistic ~85% accuracy. Documented in the tutorial.

## Decision 3: Gradient clipping (clip_grad_norm_, max_norm=1.0)
**Decision:** Clip gradient norms before each optimizer step.
**Why:** Even LSTMs (which solve vanilla RNN's vanishing gradient problem) can still produce exploding gradients on longer sequences if the loss surface is steep. Clipping at norm=1.0 is a standard default for any RNN/LSTM — it doesn't hurt training when gradients are small and prevents NaN/inf loss when they aren't.
**Trade-off:** Clipping is a blunt tool; it scales down the entire gradient vector uniformly. A per-parameter adaptive optimizer (Adam) partially mitigates this already, but the clip is cheap insurance.

## Decision 4: Take last LSTM layer's final hidden state (h_n[-1])
**Decision:** Use `h_n[-1]` (last layer, last timestep) as the sequence representation for the classifier.
**Why:** `nn.LSTM` returns `h_n` of shape `(num_layers, batch, hidden_dim)`. The last layer's final hidden state has "seen" the entire sequence through all layers of processing — it is the most information-dense single vector. Alternatives: mean-pool all timestep outputs, use the first timestep, or take max over timesteps. Mean-pooling is slightly better empirically but requires using the `output` tensor; h_n is simpler and sufficient here.
**Trade-off:** On very long sequences the final hidden state can forget information from the beginning (even with LSTM gates). Attention mechanisms (Module 28+ territory) address this explicitly.
