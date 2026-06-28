"""module_01/main.py — NumPy: The Detective's Data Toolkit.

The Detective Agency runs on data. Before we can spot patterns
in crime reports, we need a fast, memory-efficient way to store
and manipulate numbers. That's NumPy.

Run: python main.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RANDOM_SEED = 42
PLOTS_DIR   = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)
rng = np.random.default_rng(RANDOM_SEED)

print("=" * 56)
print("MODULE 01 — NumPy: The Detective's Data Toolkit")
print("=" * 56)


# ── 1. CREATING ARRAYS ───────────────────────────────────────
print("\n── 1. Creating Arrays ──────────────────────────────────")

cases_per_day = np.array([3, 7, 2, 9, 4, 6, 1])   # 1-D array
print(f"Cases per day:    {cases_per_day}")
print(f"Shape:            {cases_per_day.shape}")
print(f"dtype:            {cases_per_day.dtype}")

# 2-D array: rows = districts, cols = crime types
report_matrix = np.array([
    [10, 5, 3],    # North
    [7,  8, 2],    # South
    [4,  2, 9],    # East
])
print(f"\nReport matrix (3 districts × 3 types):\n{report_matrix}")
print(f"Shape: {report_matrix.shape}")

# Useful constructors
print(f"\nzeros(3):    {np.zeros(3)}")
print(f"ones(4):     {np.ones(4)}")
print(f"arange(5):   {np.arange(5)}")
print(f"linspace:    {np.linspace(0, 1, 5)}")
print(f"eye(3):\n{np.eye(3, dtype=int)}")


# ── 2. INDEXING AND SLICING ───────────────────────────────────
print("\n── 2. Indexing & Slicing ───────────────────────────────")
print(f"cases_per_day[0]      = {cases_per_day[0]}   (first day)")
print(f"cases_per_day[-1]     = {cases_per_day[-1]}  (last day)")
print(f"cases_per_day[2:5]    = {cases_per_day[2:5]}")
print(f"cases_per_day[::2]    = {cases_per_day[::2]}  (every other)")

print(f"\nreport_matrix[1, :]   = {report_matrix[1, :]}  (all South)")
print(f"report_matrix[:, 2]   = {report_matrix[:, 2]}  (all type-2)")
print(f"report_matrix[0:2, 1] = {report_matrix[0:2, 1]}  (North+South, type-1)")

# Boolean indexing
high_crime_days = cases_per_day[cases_per_day > 5]
print(f"\nDays with > 5 cases: {high_crime_days}")


# ── 3. VECTORISED OPERATIONS ──────────────────────────────────
print("\n── 3. Vectorised Operations ────────────────────────────")
# NumPy applies operations to ALL elements — no Python loop needed
solved    = np.array([2, 5, 1, 7, 3, 4, 1])
unsolved  = cases_per_day - solved
rate      = solved / cases_per_day   # element-wise division

print(f"Cases:    {cases_per_day}")
print(f"Solved:   {solved}")
print(f"Unsolved: {unsolved}")
print(f"Rate:     {rate.round(2)}")

# Aggregations
print(f"\nTotal cases:   {cases_per_day.sum()}")
print(f"Mean cases:    {cases_per_day.mean():.2f}")
print(f"Max cases:     {cases_per_day.max()}  (day {cases_per_day.argmax()})")
print(f"Std deviation: {cases_per_day.std():.2f}")


# ── 4. BROADCASTING ──────────────────────────────────────────
print("\n── 4. Broadcasting ─────────────────────────────────────")
# Scale a different bonus to each column WITHOUT a loop
weights = np.array([1.0, 1.5, 2.0])   # type-2 crimes count double
weighted = report_matrix * weights     # (3,3) × (3,) → broadcasts along rows
print(f"Original:\n{report_matrix}")
print(f"Weights: {weights}")
print(f"Weighted:\n{weighted}")
print("Broadcasting applies weights to every row without a Python loop.")


# ── 5. RESHAPING AND STACKING ────────────────────────────────
print("\n── 5. Reshaping & Stacking ─────────────────────────────")
flat = np.arange(12)
grid = flat.reshape(3, 4)
print(f"arange(12).reshape(3,4):\n{grid}")
print(f"ravel back: {grid.ravel()}")

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(f"\nhstack([a,b]): {np.hstack([a, b])}")
print(f"vstack([a,b]):\n{np.vstack([a, b])}")


# ── 6. STATISTICAL OPERATIONS ────────────────────────────────
print("\n── 6. Statistical Operations ───────────────────────────")
data = rng.integers(0, 50, size=(30,))
print(f"Data (30 random case counts): {data}")
print(f"  mean:   {data.mean():.2f}")
print(f"  median: {np.median(data):.2f}")
print(f"  std:    {data.std():.2f}")
print(f"  25th percentile: {np.percentile(data, 25):.2f}")
print(f"  75th percentile: {np.percentile(data, 75):.2f}")


# ── 7. PLOT — weekly case trend ───────────────────────────────
days = np.arange(1, 8)
fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor="#0d1117")

for ax in axes:
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="#8b949e")
    ax.spines[:].set_color("#30363d")

axes[0].bar(days, cases_per_day, color="#58a6ff", edgecolor="#0d1117")
axes[0].bar(days, solved,        color="#3fb950", edgecolor="#0d1117", label="Solved")
axes[0].set_title("Daily Cases vs Solved", color="#e6edf3")
axes[0].set_xlabel("Day",   color="#e6edf3")
axes[0].set_ylabel("Count", color="#e6edf3")
axes[0].legend(facecolor="#0d1117", labelcolor="#e6edf3")

axes[1].hist(data, bins=10, color="#bc8cff", edgecolor="#0d1117")
axes[1].set_title("Case Count Distribution (30 days)", color="#e6edf3")
axes[1].set_xlabel("Cases",  color="#e6edf3")
axes[1].set_ylabel("Frequency", color="#e6edf3")

plt.suptitle("Detective Agency — Case Statistics", color="#e6edf3", fontsize=13)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/module01_numpy.png", dpi=120, facecolor=fig.get_facecolor())
plt.close()
print(f"\nPlot saved → {PLOTS_DIR}/module01_numpy.png")

print("\n✓ Module 01 complete. NumPy is now your case-data engine.")
