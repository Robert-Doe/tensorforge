"""module_21/main.py — Overfitting & Regularisation.

Shows overfitting on a deep network, then compares L2 and Dropout cures.
Run: python main.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_loader import load_and_clean, to_arrays
from regularisation import RegularisedMLP

RANDOM_SEED = 42
VAL_RATIO   = 0.15
TEST_RATIO  = 0.20
PLOTS_DIR   = "plots"
FIGURE_DPI  = 120
DATA_FILE   = "cases.csv"


def three_way_split(X, y, val_ratio, test_ratio, seed):
    rng    = np.random.default_rng(seed)
    idx    = rng.permutation(len(X))
    nte    = int(len(X) * test_ratio)
    nval   = int(len(X) * val_ratio)
    return (X[idx[nte+nval:]], y[idx[nte+nval:]],
            X[idx[nte:nte+nval]], y[idx[nte:nte+nval]],
            X[idx[:nte]], y[idx[:nte]])


def scale(X_tr, X_other):
    mean = X_tr.mean(axis=0);  std = X_tr.std(axis=0) + 1e-8
    return (X_tr-mean)/std, (X_other-mean)/std


def plot_loss(histories, labels, colours, title, fname):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor="#0d1117")
    subtitles = ["Training Loss", "Validation Loss"]
    keys      = ["train", "val"]
    for ax, key, sub in zip(axes, keys, subtitles):
        ax.set_facecolor("#0d1117")
        for hist, label, colour in zip(histories, labels, colours):
            if key in hist and hist[key]:
                ax.plot(hist[key], color=colour, linewidth=1.5, label=label)
        ax.set_title(sub, color="#e6edf3")
        ax.set_xlabel("Epoch", color="#e6edf3")
        ax.set_ylabel("Loss",  color="#e6edf3")
        ax.legend(facecolor="#161b22", labelcolor="#e6edf3", fontsize=9)
        ax.tick_params(colors="#e6edf3")
        ax.spines[:].set_color("#30363d")
    plt.suptitle(title, color="#e6edf3", fontsize=12)
    plt.tight_layout()
    path = f"{PLOTS_DIR}/{fname}"
    plt.savefig(path, dpi=FIGURE_DPI, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved → {path}")


def main():
    df           = load_and_clean(DATA_FILE)
    X, y, _      = to_arrays(df)
    X_tr, y_tr, X_val, y_val, X_te, y_te = three_way_split(
        X, y, VAL_RATIO, TEST_RATIO, RANDOM_SEED)
    X_tr_s,  X_te_s  = scale(X_tr, X_te)
    _,       X_val_s = scale(X_tr, X_val)

    print("=" * 52)
    print("MODULE 21 — Overfitting & Regularisation")
    print("=" * 52)
    print(f"Train: {len(X_tr)}  Val: {len(X_val)}  Test: {len(X_te)}\n")

    configs = [
        ("No regularisation",  dict(l2_lambda=0.0,   dropout_p=0.0), "#f85149"),
        ("L2 λ=0.01",          dict(l2_lambda=0.01,  dropout_p=0.0), "#58a6ff"),
        ("Dropout p=0.3",      dict(l2_lambda=0.0,   dropout_p=0.3), "#3fb950"),
        ("L2 + Dropout",       dict(l2_lambda=0.005, dropout_p=0.2), "#d29922"),
    ]

    print(f"{'Config':<24} {'Train':>7} {'Val':>7} {'Test':>7}")
    print("-" * 48)
    histories, labels, colours = [], [], []

    for name, kwargs, colour in configs:
        model = RegularisedMLP(n_features=6, hidden_sizes=[64, 32, 16],
                               lr=0.05, epochs=200, **kwargs)
        model.fit(X_tr_s, y_tr, X_val_s, y_val, batch_size=32)

        tr_acc = np.mean(model.predict(X_tr_s) == y_tr)
        va_acc = np.mean(model.predict(X_val_s) == y_val)
        te_acc = np.mean(model.predict(X_te_s)  == y_te)
        print(f"  {name:<22} {tr_acc:>7.3f} {va_acc:>7.3f} {te_acc:>7.3f}")

        histories.append(model.history_)
        labels.append(name)
        colours.append(colour)

    plot_loss(histories, labels, colours,
              "Regularisation Comparison", "regularisation_loss.png")
    print("\nDone.")


if __name__ == "__main__":
    main()
