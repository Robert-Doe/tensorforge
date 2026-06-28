"""module_04/main.py — Linear Regression: Predicting the Next Crime Wave.

The agency wants to forecast how many cases tomorrow based on today's
temperature, day-of-week, and recent trend. Linear regression draws the
best-fit straight line through past data so we can predict future values.

Run: python main.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42
PLOTS_DIR   = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)
rng = np.random.default_rng(RANDOM_SEED)

print("=" * 56)
print("MODULE 04 — Linear Regression: Predicting Case Volume")
print("=" * 56)


# ── 1. GENERATE DATASET ──────────────────────────────────────
print("\n── 1. Dataset Generation ───────────────────────────────")
n = 250
temp     = rng.uniform(5, 35, n)
weekday  = rng.integers(0, 7, n)            # 0=Mon, 6=Sun
is_hot   = (temp > 25).astype(float)
cases    = (2.5 * temp
            + 5.0 * (weekday >= 5)           # weekends busier
            - 3.0 * is_hot                   # extreme heat → fewer street crimes
            + rng.normal(0, 6, n))
cases    = np.maximum(cases, 1).round(1)

df = pd.DataFrame({"temp_c": temp, "weekday": weekday,
                   "is_weekend": (weekday >= 5).astype(int),
                   "cases": cases})
print(f"Dataset: {df.shape}")
print(df.describe().round(2))


# ── 2. TRAIN / TEST SPLIT (ALWAYS FIRST!) ────────────────────
print("\n── 2. Train / Test Split ───────────────────────────────")
FEATURES = ["temp_c", "is_weekend"]
X = df[FEATURES].values
y = df["cases"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED)
print(f"Train: {X_train.shape[0]} samples   Test: {X_test.shape[0]} samples")
print("Rule: ALWAYS split BEFORE any fitting or preprocessing.")


# ── 3. FIT LINEAR REGRESSION ─────────────────────────────────
print("\n── 3. Fitting Linear Regression ────────────────────────")
model = LinearRegression()
model.fit(X_train, y_train)

print(f"Intercept (β₀): {model.intercept_:.4f}")
for name, coef in zip(FEATURES, model.coef_):
    print(f"  β({name}): {coef:.4f}")
print("\nInterpretation:")
print(f"  Each 1°C rise in temperature → +{model.coef_[0]:.2f} more cases")
print(f"  Weekends vs weekdays → +{model.coef_[1]:.2f} more cases")


# ── 4. EVALUATE ──────────────────────────────────────────────
print("\n── 4. Model Evaluation ─────────────────────────────────")
y_pred_train = model.predict(X_train)
y_pred_test  = model.predict(X_test)

def metrics(y_true, y_pred, split_name):
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"  {split_name:<8}: RMSE={rmse:.2f}  MAE={mae:.2f}  R²={r2:.4f}")

metrics(y_train, y_pred_train, "Train")
metrics(y_test,  y_pred_test,  "Test")

print("""
  MSE:  Mean Squared Error — punishes large errors heavily (outlier-sensitive)
  RMSE: Root MSE — in same units as target (cases), easier to interpret
  MAE:  Mean Absolute Error — average |error|, robust to outliers
  R²:   1 = perfect prediction; 0 = model = mean; <0 = worse than mean
""")


# ── 5. RESIDUAL ANALYSIS ─────────────────────────────────────
print("── 5. Residual Analysis ────────────────────────────────")
residuals = y_test - y_pred_test
print(f"Residual mean: {residuals.mean():.4f}   (should be ≈ 0)")
print(f"Residual std:  {residuals.std():.4f}")
print(f"Max overestimate: {residuals.min():.2f}")
print(f"Max underestimate: {residuals.max():.2f}")


# ── 6. PREDICTIONS ON NEW DATA ───────────────────────────────
print("\n── 6. Making Predictions ───────────────────────────────")
new_days = pd.DataFrame({
    "temp_c":     [10,   28,   22,   15],
    "is_weekend": [0,    1,    0,    1],
    "description":["Cold weekday","Hot weekend","Mild weekday","Cool weekend"],
})
preds = model.predict(new_days[["temp_c","is_weekend"]].values)
print(f"  {'Scenario':<22} {'Predicted Cases':>15}")
for desc, pred in zip(new_days["description"], preds):
    print(f"  {desc:<22} {pred:>15.1f}")


# ── 7. PLOTS ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), facecolor="#0d1117")
for ax in axes:
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="#8b949e")
    ax.spines[:].set_color("#30363d")

# Actual vs Predicted
axes[0].scatter(y_test, y_pred_test, alpha=0.5, s=20, color="#58a6ff")
lims = [min(y_test.min(), y_pred_test.min()), max(y_test.max(), y_pred_test.max())]
axes[0].plot(lims, lims, color="#f85149", linewidth=1.5, linestyle="--")
axes[0].set_xlabel("Actual Cases",    color="#e6edf3")
axes[0].set_ylabel("Predicted Cases", color="#e6edf3")
axes[0].set_title("Actual vs Predicted",  color="#e6edf3")

# Residuals vs Predicted
axes[1].scatter(y_pred_test, residuals, alpha=0.5, s=20, color="#3fb950")
axes[1].axhline(0, color="#f85149", linestyle="--", linewidth=1.5)
axes[1].set_xlabel("Predicted",  color="#e6edf3")
axes[1].set_ylabel("Residual",   color="#e6edf3")
axes[1].set_title("Residual Plot", color="#e6edf3")

# Cases vs Temperature (with regression line)
t_range = np.linspace(temp.min(), temp.max(), 100)
pred_line_weekday = model.predict(np.c_[t_range, np.zeros(100)])
pred_line_weekend = model.predict(np.c_[t_range, np.ones(100)])
mask_wd = df["is_weekend"] == 0
mask_we = df["is_weekend"] == 1
axes[2].scatter(df.loc[mask_wd,"temp_c"], df.loc[mask_wd,"cases"],
                alpha=0.3, s=15, color="#58a6ff", label="Weekday")
axes[2].scatter(df.loc[mask_we,"temp_c"], df.loc[mask_we,"cases"],
                alpha=0.3, s=15, color="#bc8cff", label="Weekend")
axes[2].plot(t_range, pred_line_weekday, color="#58a6ff", linewidth=2)
axes[2].plot(t_range, pred_line_weekend, color="#bc8cff", linewidth=2)
axes[2].set_xlabel("Temperature (°C)", color="#e6edf3")
axes[2].set_ylabel("Cases",            color="#e6edf3")
axes[2].set_title("Regression Lines",  color="#e6edf3")
axes[2].legend(facecolor="#0d1117", labelcolor="#e6edf3", fontsize=8)

plt.suptitle("Linear Regression — Case Volume Prediction", color="#e6edf3", fontsize=13)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/module04_linreg.png", dpi=120, facecolor=fig.get_facecolor())
plt.close()
print(f"\nPlot saved → {PLOTS_DIR}/module04_linreg.png")

print("\n✓ Module 04 complete. Your first predictive model is operational.")
