"""module_03/viz_advanced.py — Advanced Matplotlib/Seaborn Visualisation.

Covers:
  - Subplot layouts (GridSpec for unequal cells)
  - Custom colormaps and diverging palettes
  - Annotations and arrows on plots
  - Twin axes (shared x, two y-axes for different scales)
  - Inset axes (zoomed-in subplot within a plot)
  - Publication-quality figure export (DPI, tight_layout, fonts)
  - Seaborn statistical plots (violinplot, pairplot, heatmap)

Run standalone: python viz_advanced.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
import pandas as pd

RANDOM_SEED = 42
PLOTS_DIR   = "plots"
FIGURE_DPI  = 150   # publication quality
rng = np.random.default_rng(RANDOM_SEED)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Dark theme baseline (consistent with module style)
DARK_BG   = "#0d1117"
PANEL_BG  = "#161b22"
TEXT_COL  = "#e6edf3"
GRID_COL  = "#30363d"
ACCENT    = "#58a6ff"


# ─────────────────────────────────────────────────────────────────────────────
# 1. GRIDSPEC — unequal subplot layouts
# ─────────────────────────────────────────────────────────────────────────────

def demo_gridspec():
    """Create a dashboard layout with cells of different sizes."""
    x = np.linspace(0, 10, 300)

    fig = plt.figure(figsize=(12, 8), facecolor=DARK_BG)
    gs  = gridspec.GridSpec(3, 3, figure=fig,
                             hspace=0.5, wspace=0.4)

    # Large plot spanning 2 rows and 2 columns
    ax_main = fig.add_subplot(gs[:2, :2])
    ax_main.set_facecolor(PANEL_BG)
    ax_main.plot(x, np.sin(x), color=ACCENT, linewidth=2, label="sin(x)")
    ax_main.plot(x, np.cos(x), color="#3fb950", linewidth=2, label="cos(x)")
    ax_main.set_title("Main Panel (2×2 span)", color=TEXT_COL)
    ax_main.legend(facecolor=DARK_BG, labelcolor=TEXT_COL)
    ax_main.tick_params(colors="#8b949e")
    ax_main.spines[:].set_color(GRID_COL)

    # Three narrow panels on the right column
    for row, (fn, name, colour) in enumerate([
        (np.sin,     "sin",  "#f85149"),
        (np.cos,     "cos",  "#bc8cff"),
        (np.tan,     "tan",  "#f0883e"),
    ]):
        ax = fig.add_subplot(gs[row, 2])
        ax.set_facecolor(PANEL_BG)
        y  = fn(x)
        y  = np.clip(y, -5, 5)   # clip tan spikes
        ax.plot(x, y, color=colour, linewidth=1.5)
        ax.set_title(name, color=TEXT_COL, fontsize=9)
        ax.tick_params(colors="#8b949e", labelsize=7)
        ax.spines[:].set_color(GRID_COL)

    # Bottom-left: histogram
    ax_hist = fig.add_subplot(gs[2, :2])
    ax_hist.set_facecolor(PANEL_BG)
    data = rng.standard_normal(500)
    ax_hist.hist(data, bins=30, color=ACCENT, edgecolor=DARK_BG, alpha=0.85)
    ax_hist.set_title("Histogram (1×2 span)", color=TEXT_COL)
    ax_hist.tick_params(colors="#8b949e")
    ax_hist.spines[:].set_color(GRID_COL)

    plt.suptitle("GridSpec Dashboard Layout", color=TEXT_COL, fontsize=14, y=1.01)
    plt.savefig(f"{PLOTS_DIR}/gridspec_layout.png", dpi=FIGURE_DPI,
                facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"── 1. GridSpec saved → {PLOTS_DIR}/gridspec_layout.png")


# ─────────────────────────────────────────────────────────────────────────────
# 2. CUSTOM COLORMAPS
# ─────────────────────────────────────────────────────────────────────────────

def demo_custom_colormaps():
    """Create a custom diverging colormap from our dark theme colours."""
    # Diverging: blue (negative) → white → red (positive)
    colours = ["#58a6ff", "#ffffff", "#f85149"]
    detective_cmap = LinearSegmentedColormap.from_list(
        "detective", colours, N=256)

    data = rng.standard_normal((20, 20))   # random correlation-like matrix

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), facecolor=DARK_BG)

    for ax, (cmap, title) in zip(axes, [
        ("RdBu_r",       "Built-in RdBu_r"),
        ("viridis",      "Built-in viridis"),
        (detective_cmap, "Custom diverging"),
    ]):
        ax.set_facecolor(PANEL_BG)
        im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=-2, vmax=2)
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title(title, color=TEXT_COL, fontsize=11)
        ax.tick_params(colors="#8b949e")

    plt.suptitle("Custom vs Built-in Colormaps", color=TEXT_COL, fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/custom_colormaps.png", dpi=FIGURE_DPI,
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"── 2. Custom colormaps → {PLOTS_DIR}/custom_colormaps.png")


# ─────────────────────────────────────────────────────────────────────────────
# 3. ANNOTATIONS AND ARROWS
# ─────────────────────────────────────────────────────────────────────────────

def demo_annotations():
    """Annotate a plot with arrows, text boxes, and spans."""
    x = np.linspace(0, 4 * np.pi, 400)
    y = np.exp(-0.2 * x) * np.sin(x)

    peak_idx   = np.argmax(y)
    trough_idx = np.argmin(y)

    fig, ax = plt.subplots(figsize=(10, 4.5), facecolor=DARK_BG)
    ax.set_facecolor(PANEL_BG)
    ax.plot(x, y, color=ACCENT, linewidth=2)
    ax.axhline(0, color=GRID_COL, linewidth=0.8)

    # Arrow annotation for maximum
    ax.annotate(
        f"Peak ({x[peak_idx]:.2f}, {y[peak_idx]:.2f})",
        xy    = (x[peak_idx],   y[peak_idx]),
        xytext= (x[peak_idx]+1, y[peak_idx]+0.2),
        color = "#3fb950",
        fontsize = 10,
        arrowprops = dict(arrowstyle="->", color="#3fb950", lw=1.5),
    )

    # Arrow annotation for trough
    ax.annotate(
        f"Trough ({x[trough_idx]:.2f}, {y[trough_idx]:.2f})",
        xy    = (x[trough_idx],   y[trough_idx]),
        xytext= (x[trough_idx]+1, y[trough_idx]-0.15),
        color = "#f85149",
        fontsize = 10,
        arrowprops = dict(arrowstyle="->", color="#f85149", lw=1.5),
    )

    # Span annotation for decay region
    ax.axvspan(0, 2, alpha=0.08, color="#bc8cff", label="High energy region")
    ax.text(0.5, 0.5, "High energy\nregion", transform=ax.transAxes,
            color="#bc8cff", fontsize=9, ha="left", va="top")

    # Text box
    textbox = dict(boxstyle="round,pad=0.3", facecolor=DARK_BG, edgecolor=GRID_COL)
    ax.text(0.98, 0.98, "Damped oscillation\ny = e^{-0.2x} sin(x)",
            transform=ax.transAxes, ha="right", va="top",
            color=TEXT_COL, fontsize=9, bbox=textbox)

    ax.set_title("Annotations, Arrows & Text Boxes", color=TEXT_COL, fontsize=12)
    ax.tick_params(colors="#8b949e")
    ax.spines[:].set_color(GRID_COL)
    ax.set_xlabel("x", color=TEXT_COL)
    ax.set_ylabel("y", color=TEXT_COL)

    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/annotations.png", dpi=FIGURE_DPI,
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"── 3. Annotations → {PLOTS_DIR}/annotations.png")


# ─────────────────────────────────────────────────────────────────────────────
# 4. TWIN AXES (two y-axes, shared x)
# ─────────────────────────────────────────────────────────────────────────────

def demo_twin_axes():
    """Plot temperature and precipitation on the same x-axis with two scales."""
    months = np.arange(1, 13)
    temp_c = np.array([3, 5, 9, 13, 18, 22, 25, 24, 20, 14, 8, 4])
    precip_mm = np.array([55, 40, 45, 50, 60, 30, 20, 25, 45, 70, 65, 60])

    fig, ax1 = plt.subplots(figsize=(10, 4.5), facecolor=DARK_BG)
    ax1.set_facecolor(PANEL_BG)

    # Primary y-axis: temperature
    colour_temp = "#f85149"
    ax1.plot(months, temp_c, color=colour_temp, linewidth=2.5,
             marker="o", markersize=5, label="Temperature (°C)")
    ax1.set_xlabel("Month", color=TEXT_COL)
    ax1.set_ylabel("Temperature (°C)", color=colour_temp)
    ax1.tick_params(axis="y", colors=colour_temp)
    ax1.tick_params(axis="x", colors="#8b949e")
    ax1.spines[:].set_color(GRID_COL)

    # Twin y-axis: precipitation
    ax2          = ax1.twinx()   # share the x-axis
    colour_rain  = "#58a6ff"
    ax2.bar(months, precip_mm, alpha=0.5, color=colour_rain,
            width=0.7, label="Precipitation (mm)")
    ax2.set_ylabel("Precipitation (mm)", color=colour_rain)
    ax2.tick_params(axis="y", colors=colour_rain)
    ax2.spines[:].set_color(GRID_COL)
    ax2.set_facecolor(PANEL_BG)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               facecolor=DARK_BG, labelcolor=TEXT_COL, fontsize=9)

    ax1.set_title("Twin Axes: Temperature & Precipitation", color=TEXT_COL, fontsize=12)
    ax1.set_xticks(months)
    ax1.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                          "Jul","Aug","Sep","Oct","Nov","Dec"], color="#8b949e")

    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/twin_axes.png", dpi=FIGURE_DPI,
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"── 4. Twin axes → {PLOTS_DIR}/twin_axes.png")


# ─────────────────────────────────────────────────────────────────────────────
# 5. INSET AXES (zoom window)
# ─────────────────────────────────────────────────────────────────────────────

def demo_inset_axes():
    """Show a detailed zoom inset within the main plot."""
    x = np.linspace(0, 10, 500)
    y = np.sin(5 * x) * np.exp(-0.3 * x) + 0.05 * rng.standard_normal(500)

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=DARK_BG)
    ax.set_facecolor(PANEL_BG)
    ax.plot(x, y, color=ACCENT, linewidth=1.2)
    ax.set_title("Inset Axes: Zoom Into Region of Interest", color=TEXT_COL)
    ax.tick_params(colors="#8b949e")
    ax.spines[:].set_color(GRID_COL)

    # Highlight the zoomed region on the main plot
    zoom_x1, zoom_x2 = 4.0, 5.5
    ax.axvspan(zoom_x1, zoom_x2, alpha=0.12, color="#f0883e")
    ax.annotate("", xy=(0.45, 0.5), xytext=(0.35, 0.3), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color="#f0883e"))

    # Create inset axes in upper-right corner
    ax_inset = inset_axes(ax, width="35%", height="40%",
                           bbox_to_anchor=ax.bbox,
                           bbox_transform=ax.transData,
                           loc="upper right")
    ax_inset.set_facecolor(PANEL_BG)
    mask = (x >= zoom_x1) & (x <= zoom_x2)
    ax_inset.plot(x[mask], y[mask], color="#f0883e", linewidth=2)
    ax_inset.set_xlim(zoom_x1, zoom_x2)
    ax_inset.tick_params(labelsize=7, colors="#8b949e")
    ax_inset.spines[:].set_color("#f0883e")
    ax_inset.set_facecolor("#1c1f26")

    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/inset_axes.png", dpi=FIGURE_DPI,
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"── 5. Inset axes → {PLOTS_DIR}/inset_axes.png")


# ─────────────────────────────────────────────────────────────────────────────
# 6. SEABORN STATISTICAL PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def demo_seaborn_plots():
    """Violin, pair plot, and correlation heatmap with seaborn."""
    sns.set_theme(style="dark", palette="deep")

    # Synthetic crime-by-district dataset
    districts = ["North", "South", "East", "West", "Central"]
    n = 80
    data = pd.DataFrame({
        "District":    rng.choice(districts, n),
        "Cases":       rng.poisson(lam=20, size=n),
        "Resolution%": rng.beta(3, 2, n) * 100,
        "Response_min": rng.exponential(8, n),
        "Staff":       rng.integers(5, 25, n),
    })

    # ── Violin plot
    fig, ax = plt.subplots(figsize=(9, 4), facecolor=DARK_BG)
    ax.set_facecolor(PANEL_BG)
    sns.violinplot(data=data, x="District", y="Cases", hue="District",
                   palette="muted", ax=ax, legend=False)
    ax.set_title("Cases per District (Violin Plot)", color=TEXT_COL)
    ax.tick_params(colors="#8b949e")
    ax.spines[:].set_color(GRID_COL)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/seaborn_violin.png", dpi=FIGURE_DPI,
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"── 6a. Seaborn violin → {PLOTS_DIR}/seaborn_violin.png")

    # ── Correlation heatmap
    numeric_cols = ["Cases", "Resolution%", "Response_min", "Staff"]
    corr = data[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(6, 5), facecolor=DARK_BG)
    ax.set_facecolor(PANEL_BG)
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, ax=ax, linewidths=0.5, linecolor=DARK_BG,
                annot_kws={"color": TEXT_COL, "size": 9})
    ax.set_title("Correlation Heatmap", color=TEXT_COL)
    ax.tick_params(colors="#8b949e")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/seaborn_heatmap.png", dpi=FIGURE_DPI,
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"── 6b. Seaborn heatmap → {PLOTS_DIR}/seaborn_heatmap.png")


# ─────────────────────────────────────────────────────────────────────────────
# 7. PUBLICATION-QUALITY FIGURE TIPS
# ─────────────────────────────────────────────────────────────────────────────

def demo_publication_tips():
    """Demonstrate settings for publication-quality outputs."""
    print("\n── 7. Publication-Quality Figure Tips ──────────────")
    print("""
  Key settings for camera-ready figures:
  ─────────────────────────────────────────────────────────────
  DPI:      300+ for print (journals), 150 for web/slides
            plt.savefig("fig.png", dpi=300)

  Format:   SVG/PDF for vector (scale without pixelation)
            PNG for raster (photos, complex plots)
            plt.savefig("fig.pdf")  ← use in LaTeX

  Fonts:    Use system fonts or serif for journals
            plt.rcParams["font.family"] = "serif"
            plt.rcParams["font.size"]   = 12

  Layout:   Always call tight_layout() or constrained_layout=True
            Avoids text overlap and label clipping
            fig = plt.figure(constrained_layout=True)

  Export:   bbox_inches="tight" includes all labels outside axes area
            plt.savefig("fig.png", dpi=300, bbox_inches="tight")

  Colour:   Consider colourblind accessibility:
            - Avoid red-green combinations
            - Use okabe-ito palette or seaborn colorblind palette
            - sns.set_palette("colorblind")

  Line width: 2.0+ for lines (1px too thin on high-DPI screens/print)
  Marker size: 6+ for scatter (too small = invisible in print)
  """)

    # Demonstrate constrained_layout
    fig, axes = plt.subplots(2, 2, figsize=(8, 6),
                              facecolor=DARK_BG,
                              constrained_layout=True)   # auto layout

    titles = ["Learning Curve", "ROC Curve", "Confusion Matrix", "Feature Importances"]
    for ax, title in zip(axes.flat, titles):
        ax.set_facecolor(PANEL_BG)
        ax.set_title(title, color=TEXT_COL, fontsize=10)
        ax.tick_params(colors="#8b949e")
        ax.spines[:].set_color(GRID_COL)
        ax.text(0.5, 0.5, title, ha="center", va="center",
                color="#8b949e", transform=ax.transAxes, fontsize=8)

    fig.suptitle("constrained_layout=True — auto spacing", color=TEXT_COL, fontsize=12)
    plt.savefig(f"{PLOTS_DIR}/publication_layout.png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved → {PLOTS_DIR}/publication_layout.png")


def main():
    print("=" * 54)
    print("MODULE 3 — Advanced Visualisation")
    print("=" * 54)
    demo_gridspec()
    demo_custom_colormaps()
    demo_annotations()
    demo_twin_axes()
    demo_inset_axes()
    demo_seaborn_plots()
    demo_publication_tips()
    print("\nDone.")


if __name__ == "__main__":
    main()
