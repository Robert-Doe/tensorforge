# Module 15 — DECISIONS.md

## Decision 1: OneHotEncoder not OrdinalEncoder for location
**Decision:** Encode the `location` column with OneHotEncoder, not OrdinalEncoder.
**Why:** OrdinalEncoder assigns integer codes (Downtown=0, Suburbs=1, Warehouse=2). This implies an ordering that doesn't exist — "Warehouse" is not numerically greater than "Suburbs". OneHotEncoder creates one binary column per category, making the encoding completely order-free.
**Trade-off:** OneHotEncoder adds k-1 new columns (k=number of categories). For high-cardinality features (hundreds of categories), this can explode the feature space. Use target encoding or embedding layers instead for high-cardinality categoricals.

## Decision 2: Use ColumnTransformer not manual column splitting
**Decision:** Apply StandardScaler and OneHotEncoder via sklearn's ColumnTransformer.
**Why:** ColumnTransformer handles the split/apply/concatenate of different transformations in one object that integrates cleanly with Pipeline. Manual splitting (X_num = X[:, :6]; X_cat = X[:, 6:]) is fragile and breaks if column order changes.
**Trade-off:** ColumnTransformer output column order (numeric first, then one-hot) differs from input column order — can be confusing when inspecting transformed data.

## Decision 3: Demonstrate data leakage explicitly
**Decision:** Show a "wrong" manual approach (fit scaler on all data) alongside the correct Pipeline approach.
**Why:** Data leakage is one of the most common and subtle mistakes in ML. Beginners often fit their scaler before splitting. Showing both approaches and their CV scores (the leaky one is optimistically inflated) makes the danger concrete.
**Trade-off:** More code to show a wrong pattern; carefully labelled in comments so students don't accidentally copy the bad version.

## Decision 4: remainder="drop" in ColumnTransformer
**Decision:** Set `remainder="drop"` to discard any columns not explicitly listed.
**Why:** Explicit is safer than implicit. If a new column appears in the data, it should be consciously added to NUMERICAL_COLS or CATEGORICAL_COLS rather than silently passed through. `remainder="passthrough"` can accidentally include ID columns or leaked target information.
**Trade-off:** Requires updating the column lists if the schema changes; minor maintenance cost.
