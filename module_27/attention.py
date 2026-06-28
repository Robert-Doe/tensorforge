"""module_27/attention.py — Attention Mechanisms & Bidirectional LSTM.

Covers:
  - Bahdanau (additive) attention from scratch
  - Scaled dot-product attention (Transformer building block)
  - Multi-head attention
  - Bidirectional LSTM
  - GRU (Gated Recurrent Unit) — simpler alternative to LSTM

Run standalone: python attention.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RANDOM_SEED = 42
PLOTS_DIR   = "plots"
FIGURE_DPI  = 120
torch.manual_seed(RANDOM_SEED)
os.makedirs(PLOTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. SCALED DOT-PRODUCT ATTENTION (Transformer)
# ─────────────────────────────────────────────────────────────────────────────

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor,
                                   V: torch.Tensor,
                                   mask: torch.Tensor = None) -> tuple:
    """Scaled dot-product attention.

    Attention(Q, K, V) = softmax(Q K^T / √d_k) V

    Q: queries — "what am I looking for?"
    K: keys    — "what do I offer?"
    V: values  — "what do I actually contain?"

    The dot product Q·K^T measures compatibility between each query and key.
    Dividing by √d_k prevents dot products from growing too large (saturating softmax).

    Args:
        Q:    (batch, heads, seq_q, d_k) query tensor
        K:    (batch, heads, seq_k, d_k) key tensor
        V:    (batch, heads, seq_k, d_v) value tensor
        mask: optional (batch, 1, 1, seq_k) boolean mask (True = ignore)

    Returns:
        (output, attention_weights) both tensors
    """
    d_k = Q.size(-1)

    # (batch, heads, seq_q, seq_k) — each query scores against every key
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)

    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))

    attn_weights = F.softmax(scores, dim=-1)   # attention distribution over keys
    output       = torch.matmul(attn_weights, V)  # weighted sum of values
    return output, attn_weights


def demo_attention():
    """Show attention weights on a simple token sequence."""
    print("── 1. Scaled Dot-Product Attention ─────────────────")
    B, T, D = 2, 6, 16   # batch=2, seq_len=6, embed_dim=16
    Q = torch.randn(B, 1, T, D)   # single head
    K = torch.randn(B, 1, T, D)
    V = torch.randn(B, 1, T, D)

    out, weights = scaled_dot_product_attention(Q, K, V)
    print(f"  Input shape:    Q/K/V = {Q.shape}")
    print(f"  Output shape:   {out.shape}")
    print(f"  Weights shape:  {weights.shape}")
    print(f"  Weights sum to 1 per query: {weights[0,0].sum(-1)}")

    # Visualise attention map for first sample
    attn_np = weights[0, 0].detach().numpy()   # (T, T)
    fig, ax = plt.subplots(figsize=(5, 4), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    im = ax.imshow(attn_np, cmap="Blues", aspect="auto")
    plt.colorbar(im, ax=ax)
    ax.set_xlabel("Key position (attended to)", color="#e6edf3")
    ax.set_ylabel("Query position",             color="#e6edf3")
    ax.set_title("Attention Weights",           color="#e6edf3")
    ax.tick_params(colors="#e6edf3")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/attention_weights.png", dpi=FIGURE_DPI,
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved → {PLOTS_DIR}/attention_weights.png")


# ─────────────────────────────────────────────────────────────────────────────
# 2. MULTI-HEAD ATTENTION
# ─────────────────────────────────────────────────────────────────────────────

class MultiHeadAttention(nn.Module):
    """Multi-head attention: run h attention heads in parallel.

    Each head projects Q, K, V to a lower-dimensional subspace (d_k = d_model//h),
    runs attention, then concatenates and projects back.

    Why multiple heads?
    Different heads can attend to different types of relationships simultaneously:
    one head might focus on syntactic dependencies, another on semantic similarity.

    Args:
        d_model: total embedding dimension
        n_heads: number of attention heads (d_model must be divisible by n_heads)
        dropout: attention dropout probability
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k     = d_model // n_heads

        # Linear projections for Q, K, V (applied to all heads at once)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)   # output projection

        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor, mask=None) -> torch.Tensor:
        """Multi-head attention forward pass.

        Args:
            query: (B, T_q, d_model)
            key:   (B, T_k, d_model)
            value: (B, T_k, d_model)
            mask:  optional attention mask

        Returns:
            (B, T_q, d_model) attended output
        """
        B = query.size(0)

        def split_heads(x):
            """(B, T, d_model) → (B, n_heads, T, d_k)"""
            return x.view(B, -1, self.n_heads, self.d_k).transpose(1, 2)

        Q = split_heads(self.W_q(query))
        K = split_heads(self.W_k(key))
        V = split_heads(self.W_v(value))

        out, _ = scaled_dot_product_attention(Q, K, V, mask)
        # (B, n_heads, T, d_k) → (B, T, d_model)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        return self.W_o(out)


def demo_multihead():
    """Verify MultiHeadAttention shapes and parameter count."""
    print("\n── 2. Multi-Head Attention ──────────────────────────")
    B, T, D = 4, 10, 64
    n_heads  = 8
    mha      = MultiHeadAttention(d_model=D, n_heads=n_heads)
    x        = torch.randn(B, T, D)
    out      = mha(x, x, x)   # self-attention: query=key=value=x

    total_params = sum(p.numel() for p in mha.parameters())
    print(f"  Input:  {x.shape}")
    print(f"  Output: {out.shape}")
    print(f"  Parameters: {total_params:,}")
    print(f"  d_k per head: {D // n_heads}  ({n_heads} heads × {D//n_heads} = {D})")
    print("  Each head sees full sequence but only 8-dim subspace of features.")
    print("  Heads can specialise: one for local context, one for long-range deps.")


# ─────────────────────────────────────────────────────────────────────────────
# 3. BAHDANAU ATTENTION (seq2seq style)
# ─────────────────────────────────────────────────────────────────────────────

class BahdanauAttention(nn.Module):
    """Additive (Bahdanau) attention for encoder-decoder models.

    score(h_t, h_s) = v^T * tanh(W_h * h_t + W_s * h_s)
    alpha_s = softmax(score(h_t, h_s))
    context = sum_s alpha_s * h_s

    h_t: decoder hidden state at current step
    h_s: encoder hidden states (all of them)

    Args:
        hidden_dim:  LSTM hidden size
        attn_dim:    internal attention scoring dimension
    """

    def __init__(self, hidden_dim: int, attn_dim: int):
        super().__init__()
        self.W_h = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.W_s = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.v   = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, decoder_hidden: torch.Tensor,
                encoder_outputs: torch.Tensor) -> tuple:
        """Compute context vector and attention weights.

        Args:
            decoder_hidden:  (B, hidden_dim) — current decoder state
            encoder_outputs: (B, T, hidden_dim) — all encoder states

        Returns:
            context:      (B, hidden_dim) weighted sum of encoder states
            attn_weights: (B, T) attention distribution
        """
        # Expand decoder hidden to (B, T, hidden_dim) for broadcasting
        dec_expanded = decoder_hidden.unsqueeze(1).expand_as(encoder_outputs)

        energy = torch.tanh(self.W_h(dec_expanded) + self.W_s(encoder_outputs))
        scores = self.v(energy).squeeze(-1)   # (B, T)

        attn_weights = F.softmax(scores, dim=-1)            # (B, T)
        context      = (attn_weights.unsqueeze(-1)          # (B, T, 1)
                        * encoder_outputs).sum(dim=1)       # (B, hidden_dim)
        return context, attn_weights


def demo_bahdanau():
    """Show Bahdanau attention on a short sequence."""
    print("\n── 3. Bahdanau (Additive) Attention ────────────────")
    B, T, H = 2, 8, 32
    encoder_outputs = torch.randn(B, T, H)
    decoder_hidden  = torch.randn(B, H)

    attn = BahdanauAttention(hidden_dim=H, attn_dim=16)
    context, weights = attn(decoder_hidden, encoder_outputs)

    print(f"  Encoder outputs: {encoder_outputs.shape}")
    print(f"  Decoder hidden:  {decoder_hidden.shape}")
    print(f"  Context vector:  {context.shape}")
    print(f"  Attention weights (first sample): {weights[0].detach().numpy().round(3)}")
    print(f"  Weights sum: {weights[0].sum().item():.4f}")
    print("  Context vector is a weighted blend of encoder states.")
    print("  The model LEARNS which input positions to attend to for each output step.")


# ─────────────────────────────────────────────────────────────────────────────
# 4. BIDIRECTIONAL LSTM
# ─────────────────────────────────────────────────────────────────────────────

class BiLSTMClassifier(nn.Module):
    """Bidirectional LSTM: run LSTM forward AND backward over sequence.

    Forward pass: reads left-to-right
    Backward pass: reads right-to-left
    Concatenate both final hidden states → richer representation.

    Particularly useful for classification where context from both
    ends of the sequence matters (e.g. "not at all good" — 'not' is crucial
    and appears at the start, far from 'good').
    """

    def __init__(self, vocab_size: int, embed_dim: int = 64,
                 hidden_dim: int = 64, n_layers: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm      = nn.LSTM(embed_dim, hidden_dim, n_layers,
                                  batch_first=True,
                                  bidirectional=True,     # ← the key change
                                  dropout=0.3)
        # hidden_dim*2 because forward + backward concatenated
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        _, (h_n, _) = self.lstm(embedded)
        # h_n shape: (n_layers*2, B, hidden_dim) — 2× for bidirectional
        # Last layer: h_n[-2] = forward, h_n[-1] = backward
        forward_h  = h_n[-2]   # (B, hidden_dim)
        backward_h = h_n[-1]   # (B, hidden_dim)
        combined   = torch.cat([forward_h, backward_h], dim=1)  # (B, hidden_dim*2)
        return self.sigmoid(self.fc(combined))


def demo_bilstm():
    """Compare BiLSTM vs unidirectional LSTM parameter counts and shapes."""
    print("\n── 4. Bidirectional LSTM ────────────────────────────")
    vocab_size = 1000
    bi_model   = BiLSTMClassifier(vocab_size)
    uni_model  = nn.LSTM(64, 64, 2, batch_first=True, bidirectional=False)

    x   = torch.randint(0, vocab_size, (8, 20))   # batch=8, seq_len=20
    out = bi_model(x)

    bi_params  = sum(p.numel() for p in bi_model.lstm.parameters())
    uni_params = sum(p.numel() for p in uni_model.parameters())

    print(f"  Bidirectional LSTM params: {bi_params:,}")
    print(f"  Unidirectional LSTM params: {uni_params:,}")
    print(f"  Output shape: {out.shape}")
    print("  BiLSTM has ~2× parameters (two direction passes) but sees")
    print("  the full context from both sides before making a prediction.")


# ─────────────────────────────────────────────────────────────────────────────
# 5. GRU — Gated Recurrent Unit
# ─────────────────────────────────────────────────────────────────────────────

def demo_gru():
    """Compare GRU vs LSTM architecture and parameter count."""
    print("\n── 5. GRU vs LSTM ───────────────────────────────────")
    print("""
  LSTM has 4 gates and 2 state vectors (h_t, c_t):
    - Forget gate f_t: control cell memory erasure
    - Input gate i_t:  control new memory writing
    - Output gate o_t: control hidden state exposure
    - Cell candidate g_t: proposed new memory content

  GRU has 2 gates and 1 state vector (h_t):
    - Reset gate r_t:  how much past h to mix with new candidate
    - Update gate z_t: balance between old h and new candidate
    h_t = (1 - z_t) * h_{t-1} + z_t * tanh(W * [r_t * h_{t-1}, x_t])
  """)
    D, H, B, T = 32, 64, 4, 10

    lstm = nn.LSTM(D, H, batch_first=True)
    gru  = nn.GRU( D, H, batch_first=True)

    x = torch.randn(B, T, D)
    lstm_out, (h_lstm, c_lstm) = lstm(x)
    gru_out,   h_gru            = gru(x)

    print(f"  LSTM params: {sum(p.numel() for p in lstm.parameters()):,}")
    print(f"  GRU  params: {sum(p.numel() for p in gru.parameters()):,}")
    print(f"  Ratio: {sum(p.numel() for p in lstm.parameters()) / sum(p.numel() for p in gru.parameters()):.2f}×")
    print(f"\n  LSTM output: {lstm_out.shape}  hidden: {h_lstm.shape}  cell: {c_lstm.shape}")
    print(f"  GRU  output: {gru_out.shape}   hidden: {h_gru.shape}")
    print("\n  GRU is simpler and ~25% fewer parameters.")
    print("  On most tasks they perform similarly; LSTM has slight edge on very long seqs.")
    print("  Modern default: if computing budget is tight, use GRU; otherwise LSTM or Transformer.")


def main():
    print("=" * 54)
    print("MODULE 27 — Attention Mechanisms & Bidirectional LSTM")
    print("=" * 54)
    demo_attention()
    demo_multihead()
    demo_bahdanau()
    demo_bilstm()
    demo_gru()
    print("\nDone.")


if __name__ == "__main__":
    main()
