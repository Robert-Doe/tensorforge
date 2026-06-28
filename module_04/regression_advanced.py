"""module_04/regression_advanced.py — Advanced Linear Regression.

Covers:
  - Normal equations (closed-form solution)
  - Ridge regression (L2 regularisation)
  - Lasso regression (L1 regularisation, coordinate descent)
  - Polynomial feature expansion
  - Bias-variance tradeoff visualisation

Run standalone: python regression_advanced.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "module_01"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "module_02"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RANDOM_SEED = 42
PLOTS_DIR   = "plots"
FIGURE_DPI  = 120
rng = np.random.default_rng(RANDOM_SEED)

os.makedirs(PLOTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. NORMAL EQUATIONS — closed-form OLS solution
# ─────────────────────────────────────────────────────────────────────────────

def normal_equations(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve linear regression analytically.

    The OLS solution minimises ||y - Xw||² and has exact solution:
        w* = (X^T X)^{-1} X^T y

    This is derived by setting the gradient of the MSE to zero.
    For large n (>10k features) gradient descent is preferred because
    inverting X^T X is O(n³) — prohibitively expensive.

    Args:
        X: (n_samples, n_features) design matrix WITH bias column prepended
        y: (n_samples,) target vector

    Returns:
        w: (n_features,) optimal weight vector
    """
    # (X^T X)^{-1} X^T y  — the Moore-Penrose pseudoinverse handles non-square X
    return np.linalg.pinv(X.T @ X) @ X.T @ y


def demo_normal_equations():
    """Compare gradient descent vs normal equations on the same problem."""
    print("── 1. Normal Equations ─────────────────────────────")
    n = 200
    true_w, true_b = 3.5, -1.2
    X_raw = rng.uniform(-3, 3, n)
    y     = true_w * X_raw + true_b + rng.normal(0, 1.0, n)

    # Add bias column
    X = np.column_stack([np.ones(n), X_raw])   # (n, 2): [1, x]

    w = normal_equations(X, y)
    print(f"Normal equation solution: bias={w[0]:.4f}, slope={w[1]:.4f}")
    print(f"True values:              bias={true_b:.4f}, slope={true_w:.4f}")
    residuals = y - X @ w
    print(f"MSE: {(residuals**2).mean():.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. RIDGE REGRESSION — L2 regularisation
# ─────────────────────────────────────────────────────────────────────────────

def ridge_normal_equations(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Solve Ridge regression analytically.

    Minimises: ||y - Xw||² + alpha * ||w||²
    Closed form: w* = (X^T X + alpha * I)^{-1} X^T y

    Adding alpha*I to X^T X makes the matrix invertible even when X^T X is
    singular (more features than samples), and shrinks weights toward zero.

    Args:
        X:     design matrix with bias column
        y:     target vector
        alpha: regularisation strength (larger = more shrinkage)

    Returns:
        w: regularised weight vector
    """
    n_features = X.shape[1]
    I = np.eye(n_features)
    I[0, 0] = 0   # do NOT regularise the bias term — it has no reason to be small
    return np.linalg.solve(X.T @ X + alpha * I, X.T @ y)


def demo_ridge():
    """Show how Ridge shrinks weights and reduces overfitting on noisy data."""
    print("\n── 2. Ridge Regression ─────────────────────────────")
    n = 30
    X_raw = rng.uniform(-3, 3, n)
    # True relationship: y = 2x — but high noise so OLS will overfit polynomial
    y = 2 * X_raw + rng.normal(0, 3, n)

    # Polynomial features up to degree 8 (way too many for 30 samples)
    degree = 8
    X = np.column_stack([np.ones(n)] + [X_raw**d for d in range(1, degree+1)])

    alphas = [0, 0.1, 1, 10, 100]
    x_line = np.linspace(-3, 3, 200)
    X_line = np.column_stack([np.ones(200)] + [x_line**d for d in range(1, degree+1)])

    fig, ax = plt.subplots(figsize=(9, 5), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    ax.scatter(X_raw, y, color="#e6edf3", s=25, alpha=0.8, label="Data")

    colours = ["#f85149", "#d29922", "#3fb950", "#58a6ff", "#bc8cff"]
    for alpha, colour in zip(alphas, colours):
        w = ridge_normal_equations(X, y, alpha)
        y_pred = X_line @ w
        train_mse = ((y - X @ w)**2).mean()
        ax.plot(x_line, y_pred, color=colour, linewidth=2,
                label=f"α={alpha}  (train MSE={train_mse:.1f})")

    ax.set_ylim(-12, 12)
    ax.set_xlabel("x",  color="#e6edf3"); ax.set_ylabel("y", color="#e6edf3")
    ax.set_title("Ridge Regression — Effect of α on degree-8 polynomial fit",
                 color="#e6edf3")
    ax.legend(facecolor="#161b22", labelcolor="#e6edf3", fontsize=8)
    ax.tick_params(colors="#e6edf3"); ax.spines[:].set_color("#30363d")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/ridge_comparison.png", dpi=FIGURE_DPI,
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved → {PLOTS_DIR}/ridge_comparison.png")

    for alpha in alphas:
        w = ridge_normal_equations(X, y, alpha)
        print(f"  α={alpha:>5}: max|w|={np.abs(w[1:]).max():.2f}  "
              f"(α=0 is OLS, large α shrinks weights)")


# ─────────────────────────────────────────────────────────────────────────────
# 3. LASSO — L1 regularisation via coordinate descent
# ─────────────────────────────────────────────────────────────────────────────

def soft_threshold(z: float, gamma: float) -> float:
    """Soft-thresholding operator — the key step in Lasso coordinate descent.

    Shrinks z toward zero by gamma. Sets exactly to zero if |z| < gamma.
    This is what creates sparsity in Lasso (weights become exactly 0).

    Args:
        z:     raw unnormalised coordinate update
        gamma: threshold = alpha / ||x_j||^2

    Returns:
        thresholded value
    """
    return np.sign(z) * max(abs(z) - gamma, 0)


def lasso_coordinate_descent(X: np.ndarray, y: np.ndarray, alpha: float,
                              max_iter: int = 500, tol: float = 1e-4) -> np.ndarray:
    """Lasso regression via coordinate descent.

    Minimises: (1/2n)||y - Xw||² + alpha * ||w||_1

    Coordinate descent cycles through each weight w_j, holding others fixed,
    and updates w_j to the univariate minimum — which has the soft-threshold
    closed form. The bias term (w[0]) is excluded from L1 penalty.

    Args:
        X:        design matrix (with bias column at index 0)
        y:        target vector
        alpha:    L1 regularisation strength
        max_iter: maximum coordinate descent iterations
        tol:      convergence tolerance on max weight change

    Returns:
        w: sparse weight vector
    """
    n, p = X.shape
    w = np.zeros(p)

    for _ in range(max_iter):
        w_old = w.copy()
        for j in range(p):
            # Partial residual: residual excluding feature j's contribution
            r_j = y - X @ w + X[:, j] * w[j]
            # Raw coordinate update (OLS for this single feature)
            z_j = (X[:, j] @ r_j) / n
            col_norm_sq = (X[:, j] ** 2).sum() / n

            if j == 0:   # bias — no L1 penalty
                w[j] = z_j / col_norm_sq
            else:
                w[j] = soft_threshold(z_j, alpha) / col_norm_sq

        if np.max(np.abs(w - w_old)) < tol:
            break

    return w


def demo_lasso():
    """Show Lasso's feature selection property (drives coefficients to exactly 0)."""
    print("\n── 3. Lasso Regression (coordinate descent) ────────")
    n, p = 100, 20
    # True signal: only features 0, 3, 7 are relevant
    true_w = np.zeros(p)
    true_w[[0, 3, 7]] = [2.0, -1.5, 3.0]

    X_raw = rng.standard_normal((n, p))
    X_raw = (X_raw - X_raw.mean(0)) / X_raw.std(0)   # standardise
    X = np.column_stack([np.ones(n), X_raw])           # add bias
    y = X_raw @ true_w + rng.normal(0, 1, n)

    alphas = [0.0, 0.05, 0.1, 0.3]
    print(f"{'Alpha':>8}  {'Non-zero weights':>18}  {'Features selected'}")
    for alpha in alphas:
        w = lasso_coordinate_descent(X, y, alpha)
        nz = np.where(np.abs(w[1:]) > 1e-4)[0]   # skip bias
        print(f"{alpha:>8.2f}  {len(nz):>18}  {sorted(nz.tolist())}")

    print("\n  Lasso drives irrelevant weights to EXACTLY 0 → automatic feature selection.")
    print("  Ridge shrinks weights toward 0 but rarely sets them exactly to 0.")


# ─────────────────────────────────────────────────────────────────────────────
# 4. BIAS-VARIANCE TRADEOFF
# ─────────────────────────────────────────────────────────────────────────────

def demo_bias_variance():
    """Empirically demonstrate bias-variance tradeoff via repeated train/test splits."""
    print("\n── 4. Bias-Variance Tradeoff ───────────────────────")
    true_fn = lambda x: np.sin(2 * np.pi * x)
    N_TRIALS = 50
    N_TRAIN  = 15
    x_test   = np.linspace(0, 1, 100)
    X_test   = np.column_stack([x_test**d for d in range(1, 10)])

    degrees = [1, 3, 9]
    bias_sq = {}
    variance = {}

    for degree in degrees:
        predictions = []
        for _ in range(N_TRIALS):
            x_tr = rng.uniform(0, 1, N_TRAIN)
            y_tr = true_fn(x_tr) + rng.normal(0, 0.3, N_TRAIN)
            X_tr = np.column_stack([np.ones(N_TRAIN)] +
                                   [x_tr**d for d in range(1, degree+1)])
            X_te = np.column_stack([np.ones(100)] +
                                   [x_test**d for d in range(1, degree+1)])
            w    = np.linalg.pinv(X_tr.T @ X_tr) @ X_tr.T @ y_tr
            predictions.append(X_te @ w)

        preds  = np.array(predictions)          # (N_TRIALS, 100)
        y_true = true_fn(x_test)
        mean_pred = preds.mean(0)
        bias_sq[degree]  = ((mean_pred - y_true)**2).mean()
        variance[degree] = preds.var(0).mean()

    print(f"{'Degree':>8}  {'Bias²':>10}  {'Variance':>10}  {'Bias²+Var':>12}")
    for d in degrees:
        print(f"{d:>8}  {bias_sq[d]:>10.4f}  {variance[d]:>10.4f}  "
              f"{bias_sq[d]+variance[d]:>12.4f}")
    print("\n  Degree 1: high bias, low variance (underfit — can't capture the sine)")
    print("  Degree 3: balanced (close to the true function)")
    print("  Degree 9: low bias, high variance (overfit — memorises training noise)")


def main():
    print("=" * 54)
    print("MODULE 04 — Advanced Linear Regression")
    print("=" * 54)
    demo_normal_equations()
    demo_ridge()
    demo_lasso()
    demo_bias_variance()
    print("\nDone.")


if __name__ == "__main__":
    main()
