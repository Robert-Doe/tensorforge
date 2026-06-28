"""module_27/data_utils.py — Synthetic sentiment dataset and tokeniser.

Generates a synthetic movie-review dataset so Module 27 has no external download.
Reviews are short sentences constructed from positive/negative word lists.
Vocabulary: word → integer id. Padding token = 0.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict

RANDOM_SEED  = 42
PAD_TOKEN    = 0           # reserved; embedding layer treats this as zeros
MAX_SEQ_LEN  = 20          # reviews padded/truncated to this length
N_TRAIN      = 800
N_TEST       = 200

# Word banks for synthetic review generation
_POSITIVE_SEEDS = [
    "the film was excellent and i loved every moment",
    "brilliant performances and a gripping story",
    "a masterpiece of modern cinema highly recommended",
    "wonderful characters and outstanding direction",
    "this movie moved me deeply truly unforgettable",
    "a stunning visual experience with great acting",
    "the best film i have seen this year amazing",
    "heartwarming story with superb cinematography",
    "an absolute joy to watch from start to finish",
    "exceptional writing and a fantastic cast overall",
]

_NEGATIVE_SEEDS = [
    "the film was terrible and i hated every moment",
    "awful performances and a boring confusing story",
    "a disaster of modern cinema do not bother",
    "dreadful characters and incompetent direction",
    "this movie bored me deeply truly forgettable",
    "a hideous visual mess with poor acting throughout",
    "the worst film i have seen this year dreadful",
    "depressing story with amateurish cinematography",
    "an absolute chore to watch from start to finish",
    "atrocious writing and a terrible cast overall",
]


def build_vocab(sentences: List[str]) -> Dict[str, int]:
    """Build word-to-index vocabulary from a list of sentences.

    Index 0 is reserved for padding. Indices start at 1.

    Args:
        sentences: list of whitespace-tokenised strings

    Returns:
        dict mapping word → integer index
    """
    words = set()
    for s in sentences:
        words.update(s.lower().split())
    vocab = {word: idx + 1 for idx, word in enumerate(sorted(words))}
    return vocab


def encode(sentence: str, vocab: Dict[str, int], max_len: int) -> List[int]:
    """Convert a sentence to a fixed-length integer sequence.

    Unknown words map to 0 (same as padding — treated as absent).
    Sequences are right-padded with 0 to max_len; truncated if too long.

    Args:
        sentence: raw text
        vocab:    word → index mapping
        max_len:  target sequence length

    Returns:
        list of integer indices, length == max_len
    """
    tokens = [vocab.get(w, 0) for w in sentence.lower().split()[:max_len]]
    tokens += [PAD_TOKEN] * (max_len - len(tokens))   # right-pad
    return tokens


class SentimentDataset(Dataset):
    """Synthetic sentiment dataset.

    Each item is (token_ids, label) where label ∈ {0, 1}.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        """Store pre-encoded arrays.

        Args:
            X: (n_samples, max_len) integer array
            y: (n_samples,) binary label array
        """
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def make_dataset(n_train: int = N_TRAIN, n_test: int = N_TEST,
                 max_len: int = MAX_SEQ_LEN) -> Tuple:
    """Generate train/test split and vocabulary.

    Constructs n_train + n_test reviews by randomly selecting and
    shuffling words from the positive and negative seed sentences.

    Args:
        n_train:  number of training samples
        n_test:   number of test samples
        max_len:  sequence length for padding/truncation

    Returns:
        (train_loader, test_loader, vocab, vocab_size)
    """
    rng = np.random.default_rng(RANDOM_SEED)

    def augment(seeds: List[str], n: int) -> List[str]:
        """Generate n reviews by sampling words from positive or negative seeds."""
        all_words = " ".join(seeds).split()
        reviews = []
        for _ in range(n):
            length   = rng.integers(6, 15)
            chosen   = rng.choice(all_words, size=length, replace=True)
            reviews.append(" ".join(chosen))
        return reviews

    half_train = n_train // 2
    half_test  = n_test  // 2

    pos_train = augment(_POSITIVE_SEEDS, half_train)
    neg_train = augment(_NEGATIVE_SEEDS, half_train)
    pos_test  = augment(_POSITIVE_SEEDS, half_test)
    neg_test  = augment(_NEGATIVE_SEEDS, half_test)

    all_sentences = pos_train + neg_train + pos_test + neg_test
    vocab    = build_vocab(all_sentences)
    vocab_size = len(vocab) + 1   # +1 for padding token at index 0

    def encode_all(sentences):
        return np.array([encode(s, vocab, max_len) for s in sentences])

    X_train = encode_all(pos_train + neg_train)
    y_train = np.array([1] * half_train + [0] * half_train)
    X_test  = encode_all(pos_test  + neg_test)
    y_test  = np.array([1] * half_test  + [0] * half_test)

    # Shuffle training set
    idx     = rng.permutation(n_train)
    X_train = X_train[idx]
    y_train = y_train[idx]

    g = torch.Generator().manual_seed(RANDOM_SEED)
    train_loader = DataLoader(SentimentDataset(X_train, y_train),
                              batch_size=32, shuffle=True, generator=g)
    test_loader  = DataLoader(SentimentDataset(X_test,  y_test),
                              batch_size=32, shuffle=False)

    return train_loader, test_loader, vocab, vocab_size
