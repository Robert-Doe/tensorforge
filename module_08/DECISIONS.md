# Module 08 — DECISIONS.md

## Decision 1: Demonstrate partial_fit for online learning
**Why:** Naive Bayes is one of the very few sklearn classifiers that supports
true incremental learning via partial_fit. This is a major practical advantage
for streaming data or large datasets that don't fit in RAM.
Introducing it here (when the concept is fresh) prepares students for
real-world deployment scenarios.
**Trade-off:** Adds complexity to the module. But the demonstration is short
(4 lines) and the concept is too important for Naive Bayes to skip.

## Decision 2: Show all three NB variants (Gaussian, Multinomial, Bernoulli)
**Why:** Students frequently use GaussianNB on text data or MultinomialNB on
continuous data — wrong choices for the feature type. Showing the correct
variant for each type of data right here prevents this common mistake.
**Trade-off:** More content in one module, but the table comparison at the
end is concise and makes the choice obvious.

## Decision 3: Explain the "naive" assumption explicitly
**Why:** The word "naive" in Naive Bayes is not an insult — it's a mathematical
assumption that simplifies computation from O(2^p) to O(p). Students who
understand WHY the assumption is made will know when it breaks (strongly
correlated features) and won't blindly trust calibrated probabilities.
**Trade-off:** The independence assumption discussion takes space. Worth it
because it's the single most important conceptual point of this module.

## Decision 4: Classifier comparison bar chart
**Why:** Naive Bayes is often dismissed as "too simple." Showing it alongside
Logistic Regression, Decision Tree, and KNN on the same dataset demonstrates
it's competitive despite its simplicity — and much faster to train.
**Trade-off:** This benchmark is informal (one dataset, no statistical test).
Proper comparison with confidence intervals is in Module 09.

## Decision 5: Note miscalibration without over-explaining
**Why:** Naive Bayes probabilities are notoriously poorly calibrated (often
pushed toward 0 and 1 due to the independence assumption multiplying many
small terms). Flagging this prevents students from using NB probabilities
as-is for decision-making without calibration.
**Trade-off:** Full calibration (isotonic regression, Platt scaling) is a
deep topic; we defer to logistic_advanced.py with a note here.
