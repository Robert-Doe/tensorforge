# Module 04 — DECISIONS.md

## Decision 1: Train/test split BEFORE any other step
**Why:** If you fit a scaler or imputer on all data then split, information
from the test set leaks into training — the model will appear better than it
really is. The golden rule: split first, fit everything only on train.
**Trade-off:** Beginners often find this counterintuitive ("but I haven't
touched the data yet!"). Worth the upfront explanation to prevent the habit
forming wrong.

## Decision 2: Synthetic dataset with a known ground truth
**Why:** When we know the true coefficients (temp × 2.5, weekend × 5.0),
we can check that the model recovers roughly the right values. This builds
intuition for what regression is doing — it's finding the data-generating process.
**Trade-off:** Real datasets are messier. The clean recovery of coefficients
may give false confidence. Module 09 introduces learning curves and validation
to calibrate this.

## Decision 3: Both RMSE and R² as metrics
**Why:** R² tells you relative quality (vs. just predicting the mean).
RMSE tells you absolute error in real units (cases). Both are needed:
R²=0.8 sounds great until you see RMSE=20 cases per day is unacceptable.
**Trade-off:** Showing three metrics (RMSE, MAE, R²) may overwhelm. But
the table is concise and each metric answers a different question.

## Decision 4: Residual plot as a required diagnostic
**Why:** Residuals vs predicted values is the single most diagnostic plot
for regression. Patterns in residuals (fan shape = heteroscedasticity,
curves = non-linearity) reveal when the model's assumptions are violated.
**Trade-off:** Interpreting residual plots requires some statistical maturity.
We keep the interpretation simple and defer formal tests to Module 09.

## Decision 5: sklearn's LinearRegression over from-scratch implementation
**Why:** At this stage the goal is to build intuition about WHAT regression
does (fit a line, minimise MSE) not HOW it's solved internally. The
normal equations / gradient descent implementation is in regression_advanced.py.
**Trade-off:** Students don't see the math until the advanced file. Acceptable
because Module 17 (autograd) builds the from-scratch intuition for optimisation.
