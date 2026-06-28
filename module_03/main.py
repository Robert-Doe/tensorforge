"""module_03/main.py — Visualisation: Seeing the Evidence.

A detective doesn't stare at raw numbers — they look at charts.
Patterns that are invisible in a spreadsheet jump out in a good plot.
This module teaches Matplotlib and Seaborn for data exploration.

Run: python main.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_SEED = 42
PLOTS_DIR   = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)
rng = np.random.default_rng(RANDOM_SEED)

DARK_BG  = "#0d1117"
PANEL_BG = "#161b22"
TEXT_COL = "#e6edf3"
GRID_COL = "#30363d"

print("=" * 56)
print("MODULE 03 — Visualisation: Seeing the Evidence")
print("=" * 56)

# ── Shared dataset: 200-day detective agency log ──────────────
n = 200
DISTRICTS = ["North", "South", "East", "West", "Central"]
data = pd.DataFrame({
    "day":      np.arange(1, n+1),
    "district": rng.choice(DISTRICTS, n),
    "cases":    rng.integers(1, 25, n),
    "solved":   rng.integers(0, 20, n),
    "temp_c":   20 + 10 * np.sin(np.linspace(0, 4*np.pi, n)) + rng.normal(0, 2, n),
    "severity": rng.exponential(scale=3, size=n).clip(0, 15),
})
data["solve_rate"] = (data["solved"] / data["cases"]).clip(0, 1)

print(f"Dataset: {data.shape[0]} days, {data.shape[1]} columns")
print(data.head(3).to_string())


# ── 1. LINE PLOT — case trend over time ──────────────────────
print("\n── 1. Line plot ────────────────────────────────────────")
fig, ax = plt.subplots(figsize=(10, 4), facecolor=DARK_BG)
ax.set_facecolor(PANEL_BG)

# 7-day rolling average
rolling = data["cases"].rolling(7, center=True).mean()

ax.plot(data["day"], data["cases"], color="#58a6ff", alpha=0.4,
        linewidth=1, label="Daily cases")
ax.plot(data["day"], rolling,       color="#f0883e", linewidth=2,
        label="7-day rolling mean")
ax.set_title("Case Count Over 200 Days", color=TEXT_COL, fontsize=13)
ax.set_xlabel("Day",    color=TEXT_COL)
ax.set_ylabel("Cases",  color=TEXT_COL)
ax.tick_params(colors="#8b949e")
ax.spines[:].set_color(GRID_COL)
ax.legend(facecolor=DARK_BG, labelcolor=TEXT_COL)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/03_line.png", dpi=120, facecolor=fig.get_facecolor())
plt.close()
print(f"  Saved → {PLOTS_DIR}/03_line.png")


# ── 2. SCATTER PLOT — cases vs severity ──────────────────────
print("── 2. Scatter plot ─────────────────────────────────────")
fig, ax = plt.subplots(figsize=(7, 5), facecolor=DARK_BG)
ax.set_facecolor(PANEL_BG)

sc = ax.scatter(data["cases"], data["severity"],
                c=data["solve_rate"], cmap="RdYlGn",
                s=30, alpha=0.7, edgecolors="none")
plt.colorbar(sc, ax=ax, label="Solve rate")
ax.set_xlabel("Cases",    color=TEXT_COL)
ax.set_ylabel("Severity", color=TEXT_COL)
ax.set_title("Cases vs Severity (colour = solve rate)", color=TEXT_COL)
ax.tick_params(colors="#8b949e")
ax.spines[:].set_color(GRID_COL)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/03_scatter.png", dpi=120, facecolor=fig.get_facecolor())
plt.close()
print(f"  Saved → {PLOTS_DIR}/03_scatter.png")


# ── 3. BAR CHART — mean cases per district ───────────────────
print("── 3. Bar chart ────────────────────────────────────────")
by_dist = data.groupby("district")["cases"].mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(7, 4), facecolor=DARK_BG)
ax.set_facecolor(PANEL_BG)
colours = ["#f85149","#58a6ff","#3fb950","#bc8cff","#f0883e"]
ax.bar(by_dist.index, by_dist.values, color=colours, edgecolor=DARK_BG)
ax.set_title("Mean Cases per District", color=TEXT_COL)
ax.set_xlabel("District", color=TEXT_COL)
ax.set_ylabel("Mean Daily Cases", color=TEXT_COL)
ax.tick_params(colors="#8b949e")
ax.spines[:].set_color(GRID_COL)
for i, v in enumerate(by_dist.values):
    ax.text(i, v + 0.2, f"{v:.1f}", ha="center", color=TEXT_COL, fontsize=9)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/03_bar.png", dpi=120, facecolor=fig.get_facecolor())
plt.close()
print(f"  Saved → {PLOTS_DIR}/03_bar.png")


# ── 4. HISTOGRAM — severity distribution ─────────────────────
print("── 4. Histogram ────────────────────────────────────────")
fig, ax = plt.subplots(figsize=(7, 4), facecolor=DARK_BG)
ax.set_facecolor(PANEL_BG)
ax.hist(data["severity"], bins=25, color="#bc8cff", edgecolor=DARK_BG, alpha=0.85)
ax.axvline(data["severity"].mean(),   color="#f85149", linestyle="--",
           linewidth=1.8, label=f"Mean={data['severity'].mean():.1f}")
ax.axvline(data["severity"].median(), color="#3fb950", linestyle="--",
           linewidth=1.8, label=f"Median={data['severity'].median():.1f}")
ax.set_title("Severity Distribution", color=TEXT_COL)
ax.set_xlabel("Severity", color=TEXT_COL)
ax.set_ylabel("Frequency", color=TEXT_COL)
ax.tick_params(colors="#8b949e")
ax.spines[:].set_color(GRID_COL)
ax.legend(facecolor=DARK_BG, labelcolor=TEXT_COL)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/03_hist.png", dpi=120, facecolor=fig.get_facecolor())
plt.close()
print(f"  Saved → {PLOTS_DIR}/03_hist.png")


# ── 5. BOX PLOT — solve rate per district ────────────────────
print("── 5. Box plot ─────────────────────────────────────────")
fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=DARK_BG)
ax.set_facecolor(PANEL_BG)
groups = [data[data["district"]==d]["solve_rate"].values for d in DISTRICTS]
bp = ax.boxplot(groups, labels=DISTRICTS, patch_artist=True,
                medianprops=dict(color="#f0883e", linewidth=2))
for patch, colour in zip(bp["boxes"], colours):
    patch.set_facecolor(colour)
    patch.set_alpha(0.7)
ax.set_title("Solve Rate Distribution by District", color=TEXT_COL)
ax.set_xlabel("District",    color=TEXT_COL)
ax.set_ylabel("Solve Rate",  color=TEXT_COL)
ax.tick_params(colors="#8b949e")
ax.spines[:].set_color(GRID_COL)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/03_boxplot.png", dpi=120, facecolor=fig.get_facecolor())
plt.close()
print(f"  Saved → {PLOTS_DIR}/03_boxplot.png")


# ── 6. SEABORN CORRELATION HEATMAP ───────────────────────────
print("── 6. Seaborn heatmap ──────────────────────────────────")
numeric_cols = ["cases", "solved", "solve_rate", "severity", "temp_c"]
corr = data[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(6, 5), facecolor=DARK_BG)
ax.set_facecolor(PANEL_BG)
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax,
            linewidths=0.5, linecolor=DARK_BG, annot_kws={"color": TEXT_COL, "size": 9})
ax.set_title("Feature Correlation Heatmap", color=TEXT_COL)
ax.tick_params(colors="#8b949e")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/03_heatmap.png", dpi=120, facecolor=fig.get_facecolor())
plt.close()
print(f"  Saved → {PLOTS_DIR}/03_heatmap.png")


print("""
Key Matplotlib concepts:
  fig, ax = plt.subplots()   → create figure and axes
  ax.plot()                  → line chart
  ax.scatter()               → scatter chart
  ax.bar() / ax.barh()       → bar charts
  ax.hist()                  → histogram
  ax.boxplot()               → box-and-whisker
  plt.savefig("out.png")     → always save, never show (Windows/Agg)
  plt.close()                → free memory after saving
""")

print("✓ Module 03 complete. Your evidence speaks in pictures now.")
