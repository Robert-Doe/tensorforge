# Module 03 — DECISIONS.md

## Decision 1: One shared dataset for all 6 chart types
**Why:** Switching datasets between plots forces students to re-orient mentally.
Using the same 200-day agency log for every chart lets them focus on HOW
each chart type reveals a different aspect of the SAME data.
**Trade-off:** Some chart types (e.g. heatmap) work better with more features.
We keep all charts on the same data anyway — the pedagogical benefit outweighs
the slight awkwardness of a 5-column heatmap.

## Decision 2: Rolling average on the line plot
**Why:** Raw daily data is noisy — the rolling mean reveals the underlying trend.
This is the first exposure to the signal/noise distinction that runs through
every future module on model fitting.
**Trade-off:** Introduces `rolling()` early (before Pandas is fully covered).
The line is short and the concept is intuitive enough to work here.

## Decision 3: Colour-encoded scatter plot (solve_rate as hue)
**Why:** Encoding a third variable as colour teaches the principle of aesthetic
mapping — a core grammar-of-graphics concept. It also reveals whether high-case
days tend to have lower solve rates (actionable insight).
**Trade-off:** Colorblinds may misread the RdYlGn palette. Covered in viz_advanced.py
(colorblind-safe palettes).

## Decision 4: Show both mean and median lines on the histogram
**Why:** The difference between mean and median is abstract. Seeing them as
vertical lines on a real skewed distribution (exponential severity scores)
makes the "median is robust to outliers" statement concrete and visual.
**Trade-off:** Adds two lines to the histogram which can confuse beginners.
Worth it because this visual sticks better than any written explanation.

## Decision 5: Seaborn for the heatmap, Matplotlib for everything else
**Why:** Seaborn's `heatmap` with `annot=True` takes 2 lines vs ~30 for
a fully annotated Matplotlib equivalent. Introducing Seaborn here shows
students it exists and saves effort when Seaborn's defaults are better.
**Trade-off:** Mixing two APIs can be confusing. We keep Seaborn to one plot
and explicitly note it sits on top of Matplotlib.
