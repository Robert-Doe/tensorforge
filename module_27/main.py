"""module_27/main.py — RNN from scratch demo + LSTM sentiment classifier.

Part 1: VanillaRNN shows the hidden-state update equation on a short text.
Part 2: SentimentLSTM trains on synthetic movie reviews (no download required).

Run: python main.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rnn_scratch  import VanillaRNN, demonstrate_hidden_state
from lstm_model   import SentimentLSTM, count_parameters
from data_utils   import make_dataset

# ── Constants ────────────────────────────────────────────────────────────────
RANDOM_SEED  = 42
EPOCHS       = 10
LR           = 1e-3
EMBED_DIM    = 64
HIDDEN_DIM   = 128
NUM_LAYERS   = 2
DROPOUT_P    = 0.3
PLOTS_DIR    = "plots"
FIGURE_DPI   = 120


def train_one_epoch(model, loader, criterion, optimizer, device) -> tuple:
    """Train for one epoch.

    Returns:
        (mean_loss, accuracy)
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device).unsqueeze(1)   # (B,) → (B,1)
        optimizer.zero_grad()
        probs = model(X)
        loss  = criterion(probs, y)
        loss.backward()
        # Clip gradients — RNNs/LSTMs can have large gradients over long sequences
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(y)
        preds       = (probs >= 0.5).float()
        correct    += (preds == y).sum().item()
        total      += len(y)
    return total_loss / total, correct / total


def eval_one_epoch(model, loader, criterion, device) -> tuple:
    """Evaluate without gradient computation.

    Returns:
        (mean_loss, accuracy)
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y  = X.to(device), y.to(device).unsqueeze(1)
            probs = model(X)
            total_loss += criterion(probs, y).item() * len(y)
            preds       = (probs >= 0.5).float()
            correct    += (preds == y).sum().item()
            total      += len(y)
    return total_loss / total, correct / total


def predict_review(model, sentence: str, vocab: dict, device: str,
                   max_len: int = 20) -> str:
    """Classify a single review string.

    Args:
        model:    trained SentimentLSTM
        sentence: raw text to classify
        vocab:    word → index mapping
        device:   'cpu' or 'cuda'
        max_len:  sequence length (must match training)

    Returns:
        'POSITIVE' or 'NEGATIVE' with probability
    """
    tokens  = [vocab.get(w.lower(), 0) for w in sentence.split()[:max_len]]
    tokens += [0] * (max_len - len(tokens))
    x       = torch.tensor([tokens], dtype=torch.long).to(device)

    model.eval()
    with torch.no_grad():
        prob = model(x).item()

    label = "POSITIVE" if prob >= 0.5 else "NEGATIVE"
    return f"{label} (p={prob:.3f})"


def plot_training(history: dict):
    """Save loss and accuracy curves."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor="#0d1117")
    for ax, (tr_k, va_k, ylabel) in zip(axes, [
        ("train_loss", "val_loss", "Loss"),
        ("train_acc",  "val_acc",  "Accuracy"),
    ]):
        ax.set_facecolor("#0d1117")
        ax.plot(epochs, history[tr_k], "o-", color="#58a6ff", label="Train")
        ax.plot(epochs, history[va_k], "s-", color="#f85149", label="Val")
        ax.set_xlabel("Epoch", color="#e6edf3")
        ax.set_ylabel(ylabel,  color="#e6edf3")
        ax.set_title(f"LSTM {ylabel}", color="#e6edf3")
        ax.legend(facecolor="#161b22", labelcolor="#e6edf3")
        ax.tick_params(colors="#e6edf3")
        ax.spines[:].set_color("#30363d")

    plt.tight_layout()
    fname = f"{PLOTS_DIR}/lstm_training.png"
    plt.savefig(fname, dpi=FIGURE_DPI, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved → {fname}")


def main():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 54)
    print("MODULE 27 — RNNs & LSTMs")
    print("=" * 54)

    # ── Part 1: Vanilla RNN hidden-state demo ─────────────────────────────────
    print("\n── Part 1: Vanilla RNN (NumPy) ─────────────────────")
    DEMO_VOCAB = {ch: i for i, ch in enumerate("abcdefghijklmnopqrstuvwxyz ")}
    rnn = VanillaRNN(vocab_size=len(DEMO_VOCAB), hidden_size=8)
    demonstrate_hidden_state(rnn, DEMO_VOCAB, "hello world")

    # Sample random output (untrained — just shows the mechanism)
    sampled_idx = rnn.sample(seed_char_idx=DEMO_VOCAB['h'], n_chars=12)
    inv_vocab   = {v: k for k, v in DEMO_VOCAB.items()}
    sampled_str = "".join(inv_vocab.get(i, '?') for i in sampled_idx)
    print(f"\nUntrained RNN sample (seed='h'): '{sampled_str}'")
    print("(Gibberish — expected, weights are random)")

    # ── Part 2: LSTM sentiment classifier ─────────────────────────────────────
    print("\n── Part 2: LSTM Sentiment Classifier (PyTorch) ─────")
    train_loader, test_loader, vocab, vocab_size = make_dataset()
    print(f"Vocab size: {vocab_size}  |  "
          f"Train: {len(train_loader.dataset)}  Test: {len(test_loader.dataset)}")

    model = SentimentLSTM(
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout_p=DROPOUT_P,
    ).to(device)
    count_parameters(model)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    print(f"\nTraining for {EPOCHS} epochs:")
    print(f"{'Epoch':>6} {'Tr Loss':>9} {'Tr Acc':>8} {'Va Loss':>9} {'Va Acc':>8}")
    print("-" * 46)

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = eval_one_epoch (model, test_loader,  criterion,            device)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)
        print(f"{epoch:>6} {tr_loss:>9.4f} {tr_acc:>8.4f} {va_loss:>9.4f} {va_acc:>8.4f}")

    print(f"\nFinal test accuracy: {history['val_acc'][-1]*100:.2f}%")

    # ── Inference demo ────────────────────────────────────────────────────────
    print("\n── Inference demo ────────────────────────────────────")
    test_reviews = [
        "this film was absolutely wonderful and i loved it",
        "terrible boring movie worst i have ever seen",
        "the story was brilliant and the acting superb",
        "awful dreadful and completely forgettable film",
    ]
    for review in test_reviews:
        pred = predict_review(model, review, vocab, device)
        print(f"  '{review[:45]}...' → {pred}")

    plot_training(history)
    print("\nDone.")


if __name__ == "__main__":
    main()
