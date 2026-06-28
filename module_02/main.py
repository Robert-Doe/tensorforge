"""module_02/main.py — Pandas: The Detective's Case File System.

A detective agency drowns in case files. Pandas is our filing cabinet:
rows are cases, columns are attributes (date, district, type, status).
We can filter, sort, group, and summarise thousands of records instantly.

Run: python main.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RANDOM_SEED = 42
PLOTS_DIR   = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)
rng = np.random.default_rng(RANDOM_SEED)

print("=" * 56)
print("MODULE 02 — Pandas: The Detective's Case File System")
print("=" * 56)


# ── 1. CREATING A DATAFRAME ──────────────────────────────────
print("\n── 1. Creating a DataFrame ─────────────────────────────")

DISTRICTS   = ["North", "South", "East", "West", "Central"]
CRIME_TYPES = ["Theft", "Fraud", "Vandalism", "Assault"]
STATUSES    = ["Open", "Closed", "Pending"]

n = 100
cases = pd.DataFrame({
    "case_id":   [f"CASE-{i:04d}" for i in range(1, n+1)],
    "district":  rng.choice(DISTRICTS, n),
    "type":      rng.choice(CRIME_TYPES, n),
    "status":    rng.choice(STATUSES, n, p=[0.3, 0.5, 0.2]),
    "days_open": rng.integers(1, 180, n),
    "priority":  rng.integers(1, 6, n),
})

print(f"Shape: {cases.shape}   (rows × columns)")
print(f"Columns: {list(cases.columns)}")
print(f"\nFirst 5 rows:")
print(cases.head())
print(f"\ndtypes:\n{cases.dtypes}")


# ── 2. SELECTING DATA ────────────────────────────────────────
print("\n── 2. Selecting Data ───────────────────────────────────")
# Column selection
print(f"All districts (first 5): {cases['district'].head().tolist()}")
print(f"Multi-col select:\n{cases[['case_id','type','status']].head(3)}")

# Row selection with loc/iloc
print(f"\nRow 0 (loc):  {cases.loc[0, ['case_id','district','type']].tolist()}")
print(f"Row 0 (iloc): {cases.iloc[0, :3].tolist()}")

# Boolean filtering
high_priority = cases[cases["priority"] >= 4]
print(f"\nHigh-priority cases (priority >= 4): {len(high_priority)} cases")

open_thefts = cases[(cases["status"] == "Open") & (cases["type"] == "Theft")]
print(f"Open theft cases: {len(open_thefts)}")


# ── 3. SUMMARY STATISTICS ────────────────────────────────────
print("\n── 3. Summary Statistics ───────────────────────────────")
print(cases[["days_open", "priority"]].describe().round(2))

print(f"\nValue counts — Case Status:")
print(cases["status"].value_counts())

print(f"\nValue counts — Districts:")
print(cases["district"].value_counts())


# ── 4. GROUPBY ───────────────────────────────────────────────
print("\n── 4. GroupBy Aggregations ─────────────────────────────")
by_district = cases.groupby("district").agg(
    n_cases    = ("case_id",  "count"),
    avg_days   = ("days_open","mean"),
    avg_priority = ("priority","mean"),
).round(2)
print(by_district)

print("\nStatus breakdown per district:")
cross_tab = pd.crosstab(cases["district"], cases["status"])
print(cross_tab)


# ── 5. ADDING AND MODIFYING COLUMNS ──────────────────────────
print("\n── 5. Adding & Modifying Columns ───────────────────────")
cases["is_urgent"] = (cases["priority"] >= 4) & (cases["days_open"] > 30)
cases["days_bucket"] = pd.cut(cases["days_open"],
                               bins=[0, 30, 90, 180],
                               labels=["Fresh", "Stale", "Cold"])
print(cases[["case_id","days_open","is_urgent","days_bucket"]].head(8))


# ── 6. SORTING ───────────────────────────────────────────────
print("\n── 6. Sorting ──────────────────────────────────────────")
top5 = cases.sort_values(["priority","days_open"], ascending=[False, False]).head(5)
print(top5[["case_id","district","type","priority","days_open"]])


# ── 7. MISSING DATA ──────────────────────────────────────────
print("\n── 7. Handling Missing Data ────────────────────────────")
# Inject some NaN
cases_dirty = cases.copy()
null_idx = rng.integers(0, n, 10)
cases_dirty.loc[null_idx, "days_open"] = np.nan

print(f"Null count before: {cases_dirty['days_open'].isna().sum()}")
cases_dirty["days_open"].fillna(cases_dirty["days_open"].median(), inplace=True)
print(f"Null count after fill with median: {cases_dirty['days_open'].isna().sum()}")


# ── 8. PLOT ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4), facecolor="#0d1117")
for ax in axes:
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="#8b949e")
    ax.spines[:].set_color("#30363d")

# Bar: cases per district
dist_counts = cases["district"].value_counts()
axes[0].bar(dist_counts.index, dist_counts.values, color="#58a6ff", edgecolor="#0d1117")
axes[0].set_title("Cases per District", color="#e6edf3")
axes[0].set_xlabel("District", color="#e6edf3")
axes[0].set_ylabel("Count",    color="#e6edf3")
axes[0].tick_params(axis="x", rotation=30)

# Hist: days open distribution
axes[1].hist(cases["days_open"], bins=20, color="#3fb950", edgecolor="#0d1117")
axes[1].set_title("Days Open Distribution", color="#e6edf3")
axes[1].set_xlabel("Days", color="#e6edf3")

# Bar: status counts
stat_counts = cases["status"].value_counts()
colours = {"Open": "#f85149", "Closed": "#3fb950", "Pending": "#f0883e"}
axes[2].bar(stat_counts.index, stat_counts.values,
            color=[colours[s] for s in stat_counts.index], edgecolor="#0d1117")
axes[2].set_title("Case Status Breakdown", color="#e6edf3")

plt.suptitle("Detective Agency Case File Overview", color="#e6edf3", fontsize=13)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/module02_pandas.png", dpi=120, facecolor=fig.get_facecolor())
plt.close()
print(f"\nPlot saved → {PLOTS_DIR}/module02_pandas.png")

print("\n✓ Module 02 complete. Pandas is now your case management system.")
