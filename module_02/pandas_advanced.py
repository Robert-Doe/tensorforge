"""module_02/pandas_advanced.py — Advanced Pandas techniques.

Covers: groupby + agg, merge/join types, pivot tables, time series indexing,
method chaining, categorical dtype, window functions (rolling/ewm).

Run standalone: python pandas_advanced.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "module_02"))

import numpy as np
import pandas as pd

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)


def make_sales_df() -> pd.DataFrame:
    """Generate a synthetic sales DataFrame for demonstrations."""
    n = 200
    regions    = rng.choice(["North", "South", "East", "West"], size=n)
    products   = rng.choice(["Widget", "Gadget", "Doohickey"], size=n)
    months     = rng.integers(1, 13, size=n)
    revenue    = rng.uniform(100, 5000, size=n).round(2)
    units      = rng.integers(1, 50, size=n)
    return pd.DataFrame({
        "region":  regions,
        "product": products,
        "month":   months,
        "revenue": revenue,
        "units":   units,
    })


# ─────────────────────────────────────────────────────────────────────────────
# 1. GROUPBY + AGG
# ─────────────────────────────────────────────────────────────────────────────

def demo_groupby(df: pd.DataFrame):
    """GroupBy with multiple aggregations per column."""
    print("── 1. GroupBy + Aggregation ────────────────────────")

    # Multiple agg functions on multiple columns at once
    summary = df.groupby("region").agg(
        total_revenue=("revenue", "sum"),
        mean_revenue =("revenue", "mean"),
        max_units    =("units",   "max"),
        n_orders     =("revenue", "count"),
    ).round(2)
    print(summary)

    # Multi-level groupby
    by_region_product = (df.groupby(["region", "product"])["revenue"]
                         .sum()
                         .unstack(fill_value=0)   # pivot product → columns
                         .round(0))
    print(f"\nRevenue by region × product:\n{by_region_product}")

    # transform — broadcast group statistic back to original index
    df["region_mean_revenue"] = df.groupby("region")["revenue"].transform("mean")
    df["revenue_vs_region"]   = df["revenue"] / df["region_mean_revenue"]
    print(f"\nSample with region-normalised revenue:\n{df[['region','revenue','revenue_vs_region']].head(4)}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. MERGE / JOIN
# ─────────────────────────────────────────────────────────────────────────────

def demo_merge():
    """Show all four join types with a concrete example."""
    print("\n── 2. Merge / Join Types ───────────────────────────")
    employees = pd.DataFrame({
        "emp_id":   [1, 2, 3, 4],
        "name":     ["Alice", "Bob", "Carol", "Dave"],
        "dept_id":  [10, 20, 10, 30],
    })
    departments = pd.DataFrame({
        "dept_id":  [10, 20, 40],
        "dept_name":["Engineering", "Marketing", "HR"],
    })

    print("Employees:\n", employees)
    print("Departments:\n", departments)

    for how in ["inner", "left", "right", "outer"]:
        merged = pd.merge(employees, departments, on="dept_id", how=how)
        print(f"\n{how.upper()} join — {len(merged)} rows:\n{merged}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. PIVOT TABLES
# ─────────────────────────────────────────────────────────────────────────────

def demo_pivot(df: pd.DataFrame):
    """Pivot table: aggregate with multiple index/column/value combinations."""
    print("\n── 3. Pivot Tables ─────────────────────────────────")
    pivot = pd.pivot_table(
        df,
        values=["revenue", "units"],
        index="region",
        columns="product",
        aggfunc="sum",
        fill_value=0,
    ).round(0)
    print(pivot)

    # Add margin totals
    pivot_margins = pd.pivot_table(
        df, values="revenue", index="region", columns="product",
        aggfunc="sum", fill_value=0, margins=True, margins_name="TOTAL",
    ).round(0)
    print(f"\nWith totals:\n{pivot_margins}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. TIME SERIES INDEXING
# ─────────────────────────────────────────────────────────────────────────────

def demo_time_series():
    """DatetimeIndex, resampling, rolling statistics."""
    print("\n── 4. Time Series ──────────────────────────────────")
    # Generate daily revenue for 2 years
    dates   = pd.date_range("2022-01-01", periods=730, freq="D")
    revenue = (200 + 50 * np.sin(np.arange(730) * 2 * np.pi / 365)   # seasonality
               + rng.normal(0, 20, 730))                               # noise
    ts = pd.Series(revenue, index=dates, name="daily_revenue")

    # Slice by date string
    q1_2023 = ts["2023-01":"2023-03"]
    print(f"Q1 2023: {len(q1_2023)} days  mean={q1_2023.mean():.1f}")

    # Resample: daily → monthly average
    monthly = ts.resample("ME").mean().round(1)
    print(f"\nMonthly averages (last 6):\n{monthly.tail(6)}")

    # Rolling 30-day mean (smooths noise)
    rolling_30 = ts.rolling(window=30).mean()
    print(f"\n30-day rolling mean (first non-NaN at index 29): {rolling_30.iloc[29]:.1f}")

    # Exponentially weighted mean (recent data weighted more)
    ewm_30 = ts.ewm(span=30).mean()
    print(f"EWM span=30 at same index: {ewm_30.iloc[29]:.1f}")

    # Lag features — essential for time series ML
    df_ts = pd.DataFrame({"revenue": ts})
    for lag in [1, 7, 30]:
        df_ts[f"lag_{lag}"] = df_ts["revenue"].shift(lag)
    print(f"\nDataFrame with lag features:\n{df_ts.dropna().head(3)}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. METHOD CHAINING
# ─────────────────────────────────────────────────────────────────────────────

def demo_method_chaining(df: pd.DataFrame):
    """Readable multi-step transformation without intermediate variables."""
    print("\n── 5. Method Chaining ──────────────────────────────")
    result = (
        df
        .query("region in ['North', 'South']")       # filter rows
        .assign(revenue_k=lambda x: x["revenue"] / 1000)   # add column
        .groupby(["region", "product"])["revenue_k"]
        .mean()
        .round(2)
        .reset_index()
        .sort_values("revenue_k", ascending=False)
        .rename(columns={"revenue_k": "mean_revenue_k"})
    )
    print(result.head(8))
    print("  Each step returns a new DataFrame — no mutations, no temp variables.")


# ─────────────────────────────────────────────────────────────────────────────
# 6. CATEGORICAL DTYPE — memory and performance
# ─────────────────────────────────────────────────────────────────────────────

def demo_categorical(df: pd.DataFrame):
    """Show memory savings and operations with Categorical dtype."""
    print("\n── 6. Categorical Dtype ────────────────────────────")
    mem_before = df["region"].memory_usage(deep=True)
    df["region_cat"] = df["region"].astype("category")
    mem_after = df["region_cat"].memory_usage(deep=True)

    print(f"'region' as object   : {mem_before:,} bytes")
    print(f"'region' as category : {mem_after:,} bytes")
    print(f"Memory reduction: {(1 - mem_after/mem_before)*100:.0f}%")
    print(f"Categories: {list(df['region_cat'].cat.categories)}")
    print(f"Codes (integer repr): {df['region_cat'].cat.codes.unique()}")

    # Ordered categorical — for ordinal data like sizes or ratings
    ratings = pd.Categorical(
        ["good", "bad", "excellent", "good", "bad"],
        categories=["bad", "good", "excellent"],
        ordered=True,
    )
    print(f"\nOrdered: {ratings}")
    print(f"Is 'good' > 'bad'? {ratings[0] > ratings[1]}")


def main():
    print("=" * 54)
    print("MODULE 02 — Advanced Pandas")
    print("=" * 54)
    df = make_sales_df()
    demo_groupby(df.copy())
    demo_merge()
    demo_pivot(df.copy())
    demo_time_series()
    demo_method_chaining(df.copy())
    demo_categorical(df.copy())
    print("\nDone.")


if __name__ == "__main__":
    main()
