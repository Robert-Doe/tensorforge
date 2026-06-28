"""module_08/naive_bayes_advanced.py — Advanced Naive Bayes Variants.

Covers:
  - Multinomial Naive Bayes from scratch (word count features)
  - Text classification pipeline with TF-IDF
  - Bernoulli Naive Bayes (binary presence/absence)
  - Gaussian NB from scratch (continuous features)
  - Laplace smoothing — why and what it does
  - Naive Bayes assumption test

Run standalone: python naive_bayes_advanced.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from collections import defaultdict
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)


# ─────────────────────────────────────────────────────────────────────────────
# 1. MULTINOMIAL NAIVE BAYES FROM SCRATCH
# ─────────────────────────────────────────────────────────────────────────────

class MultinomialNBScratch:
    """Multinomial Naive Bayes: for word-count (bag-of-words) features.

    Models document as a bag of words: P(word_k | class_c) = count(word_k in c) / count(all words in c)
    P(class_c | doc) ∝ P(class_c) * Π_k P(word_k | class_c)^count_k

    Args:
        alpha: Laplace smoothing constant (add-alpha smoothing)
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Learn class priors and per-class word likelihoods.

        Args:
            X: (n_docs, vocab_size) word-count matrix
            y: (n_docs,) class labels (int)

        Returns:
            self
        """
        n_docs, vocab_size = X.shape
        self.classes_  = np.unique(y)
        n_classes      = len(self.classes_)

        # Log prior: log P(class_c) = log(count_c / n_docs)
        self.log_prior_ = np.array([
            np.log((y == c).sum() / n_docs)
            for c in self.classes_
        ])

        # Log likelihood: log P(word_k | class_c) with Laplace smoothing
        # = log( (count(word_k in c) + alpha) / (total_words_in_c + alpha*vocab_size) )
        self.log_likelihood_ = np.zeros((n_classes, vocab_size))
        for i, c in enumerate(self.classes_):
            X_c             = X[y == c]
            word_counts_c   = X_c.sum(axis=0) + self.alpha
            total_c         = word_counts_c.sum()
            self.log_likelihood_[i] = np.log(word_counts_c / total_c)

        return self

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """Return log-posterior for each class.

        Args:
            X: (n_docs, vocab_size)

        Returns:
            (n_docs, n_classes) log probabilities (unnormalised)
        """
        # log P(c | doc) ∝ log P(c) + Σ_k count_k * log P(word_k | c)
        return X @ self.log_likelihood_.T + self.log_prior_

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_log_proba(X), axis=1)]


# ─────────────────────────────────────────────────────────────────────────────
# 2. LAPLACE SMOOTHING EXPLAINED
# ─────────────────────────────────────────────────────────────────────────────

def demo_laplace_smoothing():
    """Show why Laplace smoothing prevents zero-probability catastrophes."""
    print("── 1. Laplace (Add-Alpha) Smoothing ────────────────")
    print("""
  PROBLEM without smoothing:
  If word "detective" never appears in training spam emails,
  P("detective" | spam) = 0
  Then for any test document containing "detective":
  P(spam | doc) ∝ 0  →  never classified as spam, even if doc is mostly spam.

  LAPLACE SMOOTHING:
  Add alpha to every word count before computing probability.
  P(word_k | class_c) = (count(k, c) + alpha) / (total_c + alpha * |vocab|)

  alpha=1: add-one (Laplace) smoothing — the standard
  alpha<1: Lidstone smoothing — gentler
  alpha=0: no smoothing (dangerous with unseen words)
  """)

    vocab = ["spam", "free", "money", "detective", "case", "agency"]
    spam_counts  = np.array([10, 8, 7, 0, 0, 0])
    ham_counts   = np.array([0,  1, 0, 5, 8, 9])

    print(f"  {'Word':<12} {'Spam (α=0)':>12} {'Spam (α=1)':>12} "
          f"{'Ham (α=0)':>11} {'Ham (α=1)':>12}")
    for i, word in enumerate(vocab):
        for alpha in [0.0, 1.0]:
            spam_total = spam_counts.sum() + alpha * len(vocab)
            ham_total  = ham_counts.sum()  + alpha * len(vocab)
            p_spam = (spam_counts[i] + alpha) / spam_total
            p_ham  = (ham_counts[i]  + alpha) / ham_total
        # Print both
        p_spam_a0 = spam_counts[i] / (spam_counts.sum() + 1e-9)
        p_spam_a1 = (spam_counts[i] + 1) / (spam_counts.sum() + len(vocab))
        p_ham_a0  = ham_counts[i]  / (ham_counts.sum() + 1e-9)
        p_ham_a1  = (ham_counts[i] + 1)  / (ham_counts.sum()  + len(vocab))
        print(f"  {word:<12} {p_spam_a0:>12.4f} {p_spam_a1:>12.4f} "
              f"{p_ham_a0:>11.4f} {p_ham_a1:>12.4f}")

    print("\n  With α=0: P('detective'|spam)=0 → zero-product catastrophe")
    print("  With α=1: P('detective'|spam)=0.0357 → safe, document still classified")


# ─────────────────────────────────────────────────────────────────────────────
# 3. TEXT CLASSIFICATION WITH TFIDF + MULTINOMIAL NB
# ─────────────────────────────────────────────────────────────────────────────

# Mini synthetic dataset (no internet required)
TEXTS = [
    # Spam (class 1)
    "free money win cash prize now",
    "you won a million dollars click here",
    "earn money fast free offer limited time",
    "congratulations you are selected winner prize",
    "free gift card cash reward claim now",
    "urgent wire transfer money today",
    "earn online passive income free sign up",
    "win lottery jackpot prize claim",
    # Ham (class 0)
    "meeting tomorrow at the office conference room",
    "can you review the quarterly report please",
    "please find attached the project schedule",
    "hi team the deployment is scheduled for friday",
    "detective agency case update solved the mystery",
    "call me when you get to the airport",
    "reminder to submit your timesheet by friday",
    "happy birthday hope you have a great day",
]
LABELS = [1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0]


def demo_text_classification():
    """TF-IDF + Multinomial NB pipeline for spam detection."""
    print("\n── 2. Text Classification: TF-IDF + Multinomial NB ─")

    # TF-IDF: TF-IDF(word, doc) = count(word, doc) / len(doc)
    #                              * log(n_docs / df(word))
    # The IDF downweights common words ("the", "a") and upweights rare ones.
    # This often helps more than raw counts.

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
        ("clf",   MultinomialNB(alpha=1.0)),
    ])

    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(pipe, TEXTS, LABELS, cv=4, scoring="accuracy")
    print(f"  CV accuracy (4-fold): {scores.mean():.4f} ± {scores.std():.4f}")

    # Fit on all data and show top spam/ham words
    pipe.fit(TEXTS, LABELS)
    tfidf   = pipe.named_steps["tfidf"]
    clf     = pipe.named_steps["clf"]
    vocab   = np.array(tfidf.get_feature_names_out())

    # Log likelihood difference: positive = more spam-like
    log_diff = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
    top_spam = vocab[np.argsort(log_diff)[-5:][::-1]]
    top_ham  = vocab[np.argsort(log_diff)[:5]]

    print(f"\n  Most spam-like words/bigrams: {list(top_spam)}")
    print(f"  Most ham-like  words/bigrams: {list(top_ham)}")

    # Predict new messages
    new_msgs = [
        "claim your free prize now",
        "see you at the team meeting tomorrow",
    ]
    preds = pipe.predict(new_msgs)
    probs = pipe.predict_proba(new_msgs)
    for msg, pred, prob in zip(new_msgs, preds, probs):
        label = "SPAM" if pred == 1 else "HAM"
        print(f"\n  '{msg}'")
        print(f"  → {label}  (spam_prob={prob[1]:.4f})")


# ─────────────────────────────────────────────────────────────────────────────
# 4. BERNOULLI NB — binary feature version
# ─────────────────────────────────────────────────────────────────────────────

def demo_bernoulli_nb():
    """Bernoulli NB: does the word appear at all (0/1) vs how many times."""
    print("\n── 3. Bernoulli Naive Bayes ─────────────────────────")
    print("""
  Multinomial NB: feature = word COUNT (bag of words)
  Bernoulli NB:   feature = word PRESENCE (1 if word appears, else 0)

  Bernoulli explicitly models the absence of words as a feature.
  If "free" is a spam indicator, Bernoulli NB gives a negative signal
  when "free" is ABSENT (ham signal), while Multinomial NB ignores absence.

  Use Bernoulli when:
  - Documents are short (binary presence matters more than counts)
  - Features are naturally binary (clicked a link, contains attachment, etc.)
  Use Multinomial when:
  - Documents are long (word frequency carries meaningful information)
  """)

    cv   = CountVectorizer(binary=True)   # binary=True: cap all counts at 1
    X_bin = cv.fit_transform(TEXTS).toarray()
    y     = np.array(LABELS)

    from sklearn.model_selection import cross_val_score
    scores_bern = cross_val_score(BernoulliNB(alpha=1.0), X_bin, y, cv=4, scoring="accuracy")
    scores_mult = cross_val_score(
        Pipeline([("cv",  CountVectorizer(binary=False)),
                  ("clf", MultinomialNB(alpha=1.0))]),
        TEXTS, y, cv=4, scoring="accuracy"
    )
    print(f"  Bernoulli NB CV accuracy:    {scores_bern.mean():.4f} ± {scores_bern.std():.4f}")
    print(f"  Multinomial NB CV accuracy:  {scores_mult.mean():.4f} ± {scores_mult.std():.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. NAIVE BAYES INDEPENDENCE ASSUMPTION TEST
# ─────────────────────────────────────────────────────────────────────────────

def demo_independence_test():
    """Show that NB works even when independence assumption is violated."""
    print("\n── 4. Testing the Independence Assumption ───────────")
    print("""
  The "naive" in Naive Bayes: assumes features are CONDITIONALLY INDEPENDENT
  given the class: P(x1, x2, ..., xp | c) = Π P(xi | c)

  In practice this is ALMOST NEVER TRUE.
  "free" and "money" are correlated in spam.
  "project" and "schedule" are correlated in ham.

  Why does it still work despite violated assumptions?
  1. Decision boundary is still often correct even if probabilities are wrong
  2. Correct class doesn't need to have exactly right probability — just HIGHER
     than the wrong class
  3. With balanced violations, errors partially cancel out

  When does it fail?
  - Features are very strongly correlated (duplicated features hurt badly)
  - You need calibrated probabilities, not just class predictions
  """)

    # Create highly correlated features
    from sklearn.datasets import make_classification
    X_indep, y = make_classification(n_samples=500, n_features=10,
                                      n_informative=5, n_redundant=0,
                                      random_state=RANDOM_SEED)
    # Duplicate 5 features → strong correlations
    X_corr = np.hstack([X_indep, X_indep[:, :5]])   # 15 features, 5 duplicated

    gnb = GaussianNB()
    from sklearn.model_selection import cross_val_score
    score_indep = cross_val_score(gnb, X_indep, y, cv=5, scoring="accuracy").mean()
    score_corr  = cross_val_score(gnb, X_corr,  y, cv=5, scoring="accuracy").mean()

    print(f"  Gaussian NB accuracy (independent features): {score_indep:.4f}")
    print(f"  Gaussian NB accuracy (5 features duplicated): {score_corr:.4f}")
    print(f"  Degradation from correlation: {score_indep - score_corr:.4f}")
    print("\n  Strong correlation can hurt but often not catastrophically.")
    print("  The bigger risk is bad calibration, not bad ranking.")


def demo_scratch_vs_sklearn():
    """Verify scratch Multinomial NB matches sklearn's on a small example."""
    print("\n── 5. Scratch vs sklearn: Verification ─────────────")
    cv    = CountVectorizer()
    X_cnt = cv.fit_transform(TEXTS).toarray()
    y     = np.array(LABELS)

    scratch_model = MultinomialNBScratch(alpha=1.0).fit(X_cnt, y)
    sklearn_model = MultinomialNB(alpha=1.0).fit(X_cnt, y)

    acc_scratch = accuracy_score(y, scratch_model.predict(X_cnt))
    acc_sklearn = accuracy_score(y, sklearn_model.predict(X_cnt))

    print(f"  Scratch accuracy: {acc_scratch:.4f}")
    print(f"  Sklearn accuracy: {acc_sklearn:.4f}")
    print(f"  Match: {acc_scratch == acc_sklearn}")


def main():
    print("=" * 54)
    print("MODULE 8 — Advanced Naive Bayes Variants")
    print("=" * 54)
    demo_laplace_smoothing()
    demo_text_classification()
    demo_bernoulli_nb()
    demo_independence_test()
    demo_scratch_vs_sklearn()
    print("\nDone.")


if __name__ == "__main__":
    main()
