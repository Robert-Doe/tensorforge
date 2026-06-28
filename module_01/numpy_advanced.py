"""module_01/numpy_advanced.py — Advanced NumPy techniques.

Covers: fancy indexing, einsum, memory layout, structured arrays,
vectorisation vs Python loops (speed comparison), strides.

Run standalone: python numpy_advanced.py
"""

import numpy as np
import time

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

# ─────────────────────────────────────────────────────────────────────────────
# 1. FANCY INDEXING — index with integer arrays, not just slices
# ─────────────────────────────────────────────────────────────────────────────

def demo_fancy_indexing():
    """Show integer-array and boolean indexing."""
    print("── 1. Fancy Indexing ───────────────────────────────")
    X = rng.integers(0, 100, size=(6, 4))
    print(f"X =\n{X}")

    # Select specific rows by index list
    row_idx = [0, 2, 5]
    print(f"\nRows [0,2,5]:\n{X[row_idx]}")

    # Select specific (row, col) pairs — one element per pair
    rows = [0, 1, 2]
    cols = [3, 0, 2]
    print(f"\nElements at (row, col) pairs {list(zip(rows,cols))}: {X[rows, cols]}")

    # Boolean mask — select rows where first column > 50
    mask = X[:, 0] > 50
    print(f"\nRows where col-0 > 50:\n{X[mask]}")

    # np.where — element-wise conditional selection
    A = np.array([1, -2, 3, -4, 5])
    result = np.where(A > 0, A, 0)   # keep positive, zero out negative
    print(f"\nnp.where (clip negatives to 0): {result}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. EINSUM — Einstein summation notation
# ─────────────────────────────────────────────────────────────────────────────

def demo_einsum():
    """Show how einsum replaces dot, transpose, trace, outer product."""
    print("\n── 2. Einstein Summation (einsum) ──────────────────")
    A = rng.standard_normal((3, 4))
    B = rng.standard_normal((4, 5))

    # Matrix multiply: 'ij,jk->ik'  means  C[i,k] = sum_j A[i,j]*B[j,k]
    C_einsum = np.einsum('ij,jk->ik', A, B)
    C_matmul = A @ B
    print(f"Matrix multiply matches @: {np.allclose(C_einsum, C_matmul)}")

    # Transpose: 'ij->ji'
    AT = np.einsum('ij->ji', A)
    print(f"Transpose matches .T: {np.allclose(AT, A.T)}")

    # Trace (sum of diagonal): 'ii->'
    M = rng.standard_normal((4, 4))
    print(f"Trace via einsum: {np.einsum('ii->', M):.4f}  np.trace: {np.trace(M):.4f}")

    # Outer product: 'i,j->ij'
    v1 = np.array([1, 2, 3])
    v2 = np.array([10, 20])
    outer = np.einsum('i,j->ij', v1, v2)
    print(f"Outer product shape: {outer.shape}\n{outer}")

    # Batch matrix multiply (very common in deep learning)
    # (batch, M, K) @ (batch, K, N) -> (batch, M, N)
    batch_A = rng.standard_normal((8, 3, 4))
    batch_B = rng.standard_normal((8, 4, 5))
    batch_C = np.einsum('bik,bkj->bij', batch_A, batch_B)
    print(f"\nBatch matmul via einsum: {batch_A.shape} @ {batch_B.shape} → {batch_C.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. MEMORY LAYOUT — C-order vs F-order, strides
# ─────────────────────────────────────────────────────────────────────────────

def demo_memory_layout():
    """Show row-major vs column-major and how strides affect performance."""
    print("\n── 3. Memory Layout & Strides ──────────────────────")
    A = np.array([[1, 2, 3],
                  [4, 5, 6]])   # default: C-order (row-major)

    print(f"Shape: {A.shape}  Strides: {A.strides} bytes")
    print("  Stride[0]=12 means moving one row = 12 bytes (3 × 4-byte floats)")
    print("  Stride[1]=4  means moving one col =  4 bytes (1 × 4-byte float)")

    A_F = np.asfortranarray(A)   # F-order (column-major, like MATLAB/Fortran)
    print(f"\nF-order strides: {A_F.strides}")
    print("  Stride[0]=4  means moving one row =  4 bytes (contiguous in columns)")

    # Performance: row-wise operations are fast on C-order arrays (cache locality)
    N = 5000
    C_arr = rng.standard_normal((N, N))   # C-order
    F_arr = np.asfortranarray(C_arr)       # F-order

    t0 = time.perf_counter()
    _ = C_arr.sum(axis=1)   # sum each row — fast (C-order is row-contiguous)
    t_c = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = F_arr.sum(axis=1)   # sum each row — slower (F-order is column-contiguous)
    t_f = time.perf_counter() - t0

    print(f"\nRow-sum on {N}×{N}: C-order={t_c*1000:.2f}ms  F-order={t_f*1000:.2f}ms")
    print("  C-order is faster for row operations because each row is contiguous in RAM.")


# ─────────────────────────────────────────────────────────────────────────────
# 4. STRUCTURED ARRAYS — typed records, like a lightweight in-memory database
# ─────────────────────────────────────────────────────────────────────────────

def demo_structured_arrays():
    """Show how structured arrays hold heterogeneous typed data."""
    print("\n── 4. Structured Arrays ────────────────────────────")
    dtype = np.dtype([
        ('name',     'U20'),    # Unicode string, max 20 chars
        ('age',      'i4'),     # 4-byte integer
        ('salary',   'f8'),     # 8-byte float
        ('active',   '?'),      # boolean
    ])

    people = np.array([
        ('Alice',  34, 95_000.0, True),
        ('Bob',    28, 72_000.0, True),
        ('Carol',  45, 120_000.0, False),
        ('Dave',   31, 88_000.0, True),
    ], dtype=dtype)

    print(f"Structured array:\n{people}")
    print(f"\nAll ages: {people['age']}")
    print(f"Active employees: {people[people['active']]['name']}")
    print(f"Average salary (active): £{people[people['active']]['salary'].mean():,.0f}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. VECTORISATION vs PYTHON LOOP — speed comparison
# ─────────────────────────────────────────────────────────────────────────────

def demo_vectorisation():
    """Benchmark Python loop vs NumPy vectorised operation."""
    print("\n── 5. Vectorisation Speed Comparison ───────────────")
    N = 1_000_000
    x = rng.standard_normal(N)

    # Python loop — each iteration has interpreter overhead
    t0 = time.perf_counter()
    result_loop = [xi ** 2 for xi in x]
    t_loop = time.perf_counter() - t0

    # NumPy vectorised — compiled C code, SIMD instructions
    t0 = time.perf_counter()
    result_np = x ** 2
    t_np = time.perf_counter() - t0

    speedup = t_loop / t_np
    print(f"N = {N:,}")
    print(f"  Python loop : {t_loop*1000:.1f} ms")
    print(f"  NumPy       : {t_np*1000:.1f} ms")
    print(f"  Speedup     : {speedup:.0f}×")
    print("  Rule: if you write a Python for-loop over array elements, ask whether")
    print("  a NumPy operation (sum, mean, *, @, np.where) can replace it.")


# ─────────────────────────────────────────────────────────────────────────────
# 6. BROADCASTING RULES — detailed
# ─────────────────────────────────────────────────────────────────────────────

def demo_broadcasting_rules():
    """Demonstrate the three broadcasting rules with shapes."""
    print("\n── 6. Broadcasting Rules ────────────────────────────")
    print("Rule 1: prepend 1s to shape of the shorter array until shapes match.")
    print("Rule 2: dimensions of size 1 are stretched to match the other.")
    print("Rule 3: if sizes differ and neither is 1 → error.\n")

    # (3, 1) + (1, 4) → (3, 4)
    A = np.array([[1], [2], [3]])           # shape (3,1)
    B = np.array([[10, 20, 30, 40]])        # shape (1,4)
    C = A + B                               # shape (3,4)
    print(f"(3,1) + (1,4) → {C.shape}")
    print(C)

    # Normalise a matrix row-wise using broadcasting
    M = rng.standard_normal((5, 4))
    row_means = M.mean(axis=1, keepdims=True)   # (5,1) — keepdims preserves dim
    row_stds  = M.std( axis=1, keepdims=True)
    M_norm    = (M - row_means) / row_stds
    print(f"\nRow-normalised M — row means: {M_norm.mean(axis=1).round(10)}")
    print(f"Row-normalised M — row stds : {M_norm.std(axis=1).round(10)}")


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 54)
    print("MODULE 01 — Advanced NumPy")
    print("=" * 54)
    demo_fancy_indexing()
    demo_einsum()
    demo_memory_layout()
    demo_structured_arrays()
    demo_vectorisation()
    demo_broadcasting_rules()
    print("\nDone.")


if __name__ == "__main__":
    main()
