# Module 01 — DECISIONS.md

## Decision 1: NumPy over plain Python lists
**Why:** Python lists store pointers to Python objects — slow and memory-heavy.
NumPy arrays store raw C-level numeric data in contiguous memory, enabling
SIMD/vectorised CPU instructions. A 1M-element addition is ~100× faster.
**Trade-off:** NumPy arrays are fixed-type (homogeneous). For mixed-type tabular
data, Pandas (Module 02) wraps NumPy with column-type handling.

## Decision 2: `np.random.default_rng(seed)` over `np.random.seed()`
**Why:** The legacy `np.random.seed()` sets global state — any library that calls
NumPy random will interfere. `default_rng` returns an isolated Generator object
with no shared state. Reproducible without side effects.
**Trade-off:** The new API (Generator) has slightly different function names
(`rng.integers` not `np.random.randint`). Worth learning the new names.

## Decision 3: `matplotlib.use("Agg")` before any pyplot import
**Why:** On Windows without a display backend (or when running headless),
the default backend tries to open a GUI window and can crash.
`Agg` renders to memory and saves to file — always works.
**Trade-off:** `plt.show()` does nothing with Agg. Always use `plt.savefig()`.

## Decision 4: Demonstrate broadcasting with a real weights example
**Why:** Broadcasting is NumPy's single most misunderstood feature.
Showing it on the crime-type weight scaling (something concrete) makes the
"rules" memorable: dimensions are compared right-to-left, size 1 stretches.
**Trade-off:** Only covers the (n,m) × (m,) case. The full rule set
(adding new axes, keepdims) is in numpy_advanced.py.

## Decision 5: Cover both 1-D and 2-D arrays in the same module
**Why:** Jumping straight to matrices without 1-D arrays leaves students
confused about axes and shapes. Building from 1-D makes shape arithmetic obvious.
**Trade-off:** More content in one module, but both are needed for Module 04
(Linear Regression uses matrix-vector products everywhere).
