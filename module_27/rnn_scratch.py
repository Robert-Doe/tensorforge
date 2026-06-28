"""module_27/rnn_scratch.py — Vanilla RNN implemented in NumPy.

Demonstrates the core recurrent computation before switching to PyTorch's LSTM.
Processes one character at a time; hidden state carries memory across timesteps.

h_t = tanh(W_hh @ h_{t-1} + W_xh @ x_t + b_h)
y_t = W_hy @ h_t + b_y
"""

import numpy as np
from typing import List, Tuple


RANDOM_SEED = 42


class VanillaRNN:
    """Single-layer vanilla RNN for character-level sequence processing.

    Attributes:
        vocab_size:   number of unique characters (input/output dimension)
        hidden_size:  number of hidden units
        W_xh:         input-to-hidden weight matrix
        W_hh:         hidden-to-hidden weight matrix
        W_hy:         hidden-to-output weight matrix
        b_h, b_y:     bias vectors
    """

    def __init__(self, vocab_size: int, hidden_size: int):
        """Initialise weights with Xavier scaling.

        Args:
            vocab_size:  size of character vocabulary
            hidden_size: number of recurrent hidden units
        """
        rng = np.random.default_rng(RANDOM_SEED)
        scale_xh = np.sqrt(2.0 / (vocab_size + hidden_size))
        scale_hh = np.sqrt(2.0 / (hidden_size + hidden_size))
        scale_hy = np.sqrt(2.0 / (hidden_size + vocab_size))

        self.vocab_size  = vocab_size
        self.hidden_size = hidden_size

        self.W_xh = rng.normal(0, scale_xh, (hidden_size, vocab_size))
        self.W_hh = rng.normal(0, scale_hh, (hidden_size, hidden_size))
        self.W_hy = rng.normal(0, scale_hy, (vocab_size,  hidden_size))
        self.b_h  = np.zeros(hidden_size)
        self.b_y  = np.zeros(vocab_size)

    def forward(self, xs: List[np.ndarray], h_prev: np.ndarray) -> Tuple[List, List, np.ndarray]:
        """Forward pass through a sequence of one-hot input vectors.

        Args:
            xs:     list of one-hot vectors, each shape (vocab_size,)
            h_prev: initial hidden state, shape (hidden_size,)

        Returns:
            hs:   list of hidden states (one per timestep), each (hidden_size,)
            ys:   list of raw output scores (logits), each (vocab_size,)
            h_t:  final hidden state — can be passed as h_prev for next chunk
        """
        hs, ys = [h_prev], []

        for x_t in xs:
            # Core recurrent equation: mix input and previous hidden state
            h_t = np.tanh(self.W_hh @ hs[-1] + self.W_xh @ x_t + self.b_h)
            y_t = self.W_hy @ h_t + self.b_y   # raw logit for each character
            hs.append(h_t)
            ys.append(y_t)

        return hs[1:], ys, hs[-1]   # drop initial h_prev from hs list

    def softmax(self, logits: np.ndarray) -> np.ndarray:
        """Numerically stable softmax.

        Args:
            logits: raw scores, shape (vocab_size,)

        Returns:
            probabilities summing to 1
        """
        e = np.exp(logits - logits.max())  # subtract max for numerical stability
        return e / e.sum()

    def sample(self, seed_char_idx: int, n_chars: int,
               temperature: float = 1.0) -> List[int]:
        """Sample a sequence of character indices from the model.

        Args:
            seed_char_idx: index of the first character
            n_chars:       number of characters to generate
            temperature:   >1 = more random, <1 = more peaked (greedy-ish)

        Returns:
            list of character indices (length n_chars)
        """
        rng = np.random.default_rng(RANDOM_SEED)
        h   = np.zeros(self.hidden_size)
        x   = np.zeros(self.vocab_size)
        x[seed_char_idx] = 1.0
        sampled = []

        for _ in range(n_chars):
            h      = np.tanh(self.W_hh @ h + self.W_xh @ x + self.b_h)
            logits = self.W_hy @ h + self.b_y
            probs  = self.softmax(logits / temperature)
            idx    = rng.choice(self.vocab_size, p=probs)
            sampled.append(idx)
            x      = np.zeros(self.vocab_size)
            x[idx] = 1.0

        return sampled


def demonstrate_hidden_state(rnn: VanillaRNN, vocab: dict, text: str) -> None:
    """Show how the hidden state changes character-by-character.

    Runs forward pass and prints h_t norm at each timestep to illustrate
    that information accumulates in the hidden state.

    Args:
        rnn:   VanillaRNN instance
        vocab: char → index mapping
        text:  short string to process
    """
    xs = []
    for ch in text:
        x = np.zeros(rnn.vocab_size)
        x[vocab.get(ch, 0)] = 1.0
        xs.append(x)

    h0 = np.zeros(rnn.hidden_size)
    hs, _, _ = rnn.forward(xs, h0)

    print(f"\nHidden state norms for '{text}':")
    print(f"  {'Step':>5}  {'Char':>6}  {'‖h_t‖':>8}  {'h_t range':>20}")
    for i, (ch, h) in enumerate(zip(text, hs)):
        print(f"  {i:>5}  {repr(ch):>6}  {np.linalg.norm(h):>8.4f}  "
              f"[{h.min():+.4f}, {h.max():+.4f}]")
    print("  Each step's hidden state encodes all characters seen so far.")
