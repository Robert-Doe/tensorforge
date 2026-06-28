"""module_27/lstm_model.py — LSTM sentiment classifier in PyTorch.

Classifies movie review text as positive (1) or negative (0).
Architecture:
  Embedding(vocab_size, embed_dim) → LSTM(embed_dim, hidden_dim, num_layers)
  → take last hidden state → Dropout → Linear(hidden_dim, 1) → Sigmoid

Uses nn.LSTM which handles the four gates internally:
  f_t = sigmoid(W_f @ [h_{t-1}, x_t] + b_f)   # forget gate
  i_t = sigmoid(W_i @ [h_{t-1}, x_t] + b_i)   # input gate
  g_t = tanh   (W_g @ [h_{t-1}, x_t] + b_g)   # cell candidate
  o_t = sigmoid(W_o @ [h_{t-1}, x_t] + b_o)   # output gate
  c_t = f_t * c_{t-1} + i_t * g_t             # cell state update
  h_t = o_t * tanh(c_t)                        # hidden state output
"""

import torch
import torch.nn as nn


class SentimentLSTM(nn.Module):
    """LSTM-based binary sentiment classifier.

    Attributes:
        embedding:  token index → dense vector
        lstm:       multi-layer LSTM
        dropout:    applied after LSTM before linear
        fc:         hidden → 1 logit
        sigmoid:    converts logit to probability
    """

    def __init__(self, vocab_size: int, embed_dim: int = 64,
                 hidden_dim: int = 128, num_layers: int = 2,
                 dropout_p: float = 0.3):
        """Build the model.

        Args:
            vocab_size:  size of the vocabulary (including padding token 0)
            embed_dim:   dimensionality of token embeddings
            hidden_dim:  number of LSTM hidden units per layer
            num_layers:  number of stacked LSTM layers
            dropout_p:   dropout between LSTM layers and before classifier
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Embedding table: maps integer token id → float vector
        # padding_idx=0 ensures the padding token always produces a zero vector
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # nn.LSTM returns (output, (h_n, c_n))
        # batch_first=True expects input shape (batch, seq_len, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_p if num_layers > 1 else 0,  # dropout only between LSTM layers
        )

        self.dropout = nn.Dropout(p=dropout_p)
        self.fc      = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, seq_len) integer token indices

        Returns:
            (batch, 1) probability that review is positive
        """
        # Token ids → dense embeddings
        embedded = self.embedding(x)                    # (B, seq_len, embed_dim)

        # Run LSTM over entire sequence
        # h_n: (num_layers, B, hidden_dim) — hidden state of each layer at final step
        # We use the last layer's final hidden state as the sequence representation
        _, (h_n, _) = self.lstm(embedded)              # h_n: (num_layers, B, hidden_dim)

        # Take the top (last) layer's hidden state
        last_hidden = h_n[-1]                          # (B, hidden_dim)

        out = self.dropout(last_hidden)
        out = self.fc(out)                             # (B, 1) raw logit
        return self.sigmoid(out)                       # (B, 1) probability


def count_parameters(model: nn.Module) -> int:
    """Count and print trainable parameters.

    Args:
        model: nn.Module

    Returns:
        total trainable parameter count
    """
    total = 0
    print("\nParameter count:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            n = param.numel()
            total += n
            print(f"  {name:<35} {str(list(param.shape)):<25} → {n:,}")
    print(f"  {'TOTAL':<35}                           → {total:,}")
    return total
