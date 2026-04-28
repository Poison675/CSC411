import os
import pickle
import random
from itertools import combinations
from typing import Dict, Tuple

import numpy as np
from scipy.optimize import linprog, minimize, minimize_scalar
from concurrent.futures import ProcessPoolExecutor, as_completed


def _solve_lp(args):
    """Solve one LPS,j linear program."""
    G, j, barS = args
    c = -G[:, j]                     # maximize g_j · u  =>  minimize -g_j · u
    barS_matrix = G[:, barS].T
    A_ub = np.vstack([barS_matrix, -barS_matrix])
    b_ub = np.ones(2 * len(barS))
    res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                  bounds=(None, None),
                  method='highs',
                  options={'presolve': True, 'disp': False})
    return -res.fun if res.success else np.inf


def m_height(G: np.ndarray, m: int) -> float:
    """Exact m-height using the LP formulation (parallelized over all LPs)."""
    n = G.shape[1]
    if m == 0:
        return 1.0
    # All tasks: (G, j, barS) for every S of size m and every j in S
    tasks = [(G, j, [t for t in range(n) if t not in S])
             for S in combinations(range(n), m) for j in S]
    # Parallel execution (safe because each LP is independent)
    with ProcessPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
        results = list(executor.map(_solve_lp, tasks))
    return max(max(results), 1.0)

# ============================================================
# === LOCAL IMPROVEMENT HELPERS ==============================
# ============================================================

def create_identity_rows_P(k: int, n_minus_k: int) -> np.ndarray:
    """Candidate where every column is a single 1 in a cycling row."""
    P = np.zeros((k, n_minus_k), dtype=int)
    for col in range(n_minus_k):
        row = col % k
        P[row, col] = 1
    return P


def build_systematic_G(k: int, n: int, P: np.ndarray) -> np.ndarray:
    """Build full generator matrix [I_k | P]."""
    I = np.eye(k, dtype=float)
    return np.concatenate([I, P.astype(float)], axis=1)


def evaluate_neighbor(args: tuple) -> float:
    """Worker for parallel neighbor evaluation."""
    P_new, n, k, m = args
    G_new = build_systematic_G(k, n, P_new)
    return m_height(G_new, m)


def local_improve(P: np.ndarray, n: int, k: int, m: int) -> Tuple[np.ndarray, float]:
    """
    1. First check the "matrix of all identity rows" candidate.
    2. Then run greedy ±1 local search (parallel neighbors).
    """
    P = P.copy().astype(int)
    current_h = m_height(build_systematic_G(k, n, P), m)
    print(f'Best {m}_height at start of local improvement: {current_h}')

    # Parallel greedy ±1 local search
    iteration = 0
    while True:
        iteration += 1
        neighbors = []
        for i in range(k):
            for j in range(n - k):
                for delta in [-3, -2, -1, 1, 2, 3]:
                    P_new = P.copy()
                    P_new[i, j] += delta
                    if np.all(P_new[:, j] == 0):
                        continue
                    neighbors.append((P_new, n, k, m))

        if not neighbors:
            break

        num_workers = min(len(neighbors), os.cpu_count() or 4)
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            h_list = list(executor.map(evaluate_neighbor, neighbors))

        best_h_new = current_h
        best_P_new = None
        for idx, h_new in enumerate(h_list):
            if h_new < best_h_new:
                best_h_new = h_new
                best_P_new = neighbors[idx][0]

        if best_P_new is None:
            break

        P = best_P_new
        current_h = best_h_new
        # print(f"  Local improve iter {iteration} → h_{m} = {current_h:.6g}")

    return P, current_h


def reduced_row_echelon_left_identity(A: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """
    Perform Gaussian elimination with partial pivoting on a matrix A of shape (k, n)
    so that the left k columns become the identity matrix (if full rank),
    and return the full transformed matrix.
    
    This is equivalent to computing [I | B] where the original matrix was [M | C]
    and M is k x k (assuming M is invertible).
    
    Parameters
    ----------
    A : np.ndarray
        Input matrix of shape (k, n) with k <= n.
    tol : float
        Tolerance for detecting zero pivots.
    
    Returns
    -------
    np.ndarray
        The transformed matrix of shape (k, n) where the left k columns are
        the identity matrix (or as close as possible if rank-deficient).
    
    Raises
    ------
    ValueError
        If A is not 2D or has fewer columns than rows.
    """
    if A.ndim != 2:
        raise ValueError("Input must be a 2D matrix")
    
    k, n = A.shape
    if k > n:
        raise ValueError(f"Number of rows (k={k}) cannot exceed number of columns (n={n})")
    
    # Work on a copy to avoid modifying the original
    M = A.astype(float).copy()
    
    # Forward elimination with partial pivoting
    for col in range(k):
        # Find pivot row with largest absolute value in current column
        pivot_row = col + np.argmax(np.abs(M[col:, col]))
        
        if abs(M[pivot_row, col]) < tol:
            # Singular (or nearly singular) case - cannot make full identity
            # We continue but the diagonal will be small/zero
            continue
        
        # Swap rows if necessary
        if pivot_row != col:
            M[[col, pivot_row]] = M[[pivot_row, col]]
        
        # Normalize pivot to 1
        pivot = M[col, col]
        M[col] /= pivot
        
        # Eliminate column below and above the pivot (full reduction)
        for row in range(k):
            if row == col:
                continue
            factor = M[row, col]
            M[row] -= factor * M[col]
    
    # Optional: clean up very small values
    M[np.abs(M) < tol] = 0.0
    
    return M


def approximate_etf(n: int, k: int, tol: float = 1e-9, max_iter: int = 2000,
                    random_state: int = 42) -> np.ndarray:
    """
    Numerically approximate an Equiangular Tight Frame (ETF) with
    n vectors in k dimensions.
    
    Works for any n >= k >= 1. When n = k+1 it usually recovers (numerically)
    the exact regular simplex ETF.
    """
    if n < k or k < 1:
        raise ValueError("n must be >= k and k >= 1")

    np.random.seed(random_state)

    # Good initialization: random orthogonalized matrix
    X = np.random.randn(n, k)
    Q, _ = np.linalg.qr(X)           # Q has shape (n, k) with orthonormal columns
    X_init = Q

    def objective(flat_X):
        X = flat_X.reshape(n, k)
        
        # Row-normalize to unit vectors
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        Xn = X / norms
        
        # 1. Tight frame penalty
        frame_op = Xn.T @ Xn                     # k x k
        target = (n / k) * np.eye(k)
        tightness = np.sum((frame_op - target) ** 2)

        # 2. Equiangular penalty (variance of |<vi,vj>| for i≠j)
        gram = Xn @ Xn.T
        triu_idx = np.triu_indices(n, k=1)
        off_abs = np.abs(gram[triu_idx])
        
        if len(off_abs) > 1:
            equi_var = np.var(off_abs)
        else:
            equi_var = 0.0

        return tightness + 30.0 * equi_var   # higher weight on equiangularity

    # Run optimization
    res = minimize(
        fun=objective,
        x0=X_init.ravel(),
        method='L-BFGS-B',
        tol=tol,
        options={
            'maxiter': max_iter,
            'disp': False,
            # Remove deprecated 'iprint' if it was present
        }
    )

    # Extract and normalize final result
    X_opt = res.x.reshape(n, k)
    norms = np.linalg.norm(X_opt, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    vectors = X_opt / norms

    return vectors


# ====================== Verification ======================
def verify_etf(vectors: np.ndarray, name: str = ""):
    n, k = vectors.shape
    norms = np.linalg.norm(vectors, axis=1)
    gram = vectors @ vectors.T
    
    off_abs = np.abs(gram[np.triu_indices(n, k=1)])
    
    print(f"\n=== {name} (n={n}, k={k}) ===")
    print(f"Norms:     min={norms.min():.6f}  max={norms.max():.6f}")
    print(f"|Inner products| (i≠j): mean={off_abs.mean():.6f}  std={off_abs.std():.6f}")
    tightness_err = np.linalg.norm((vectors.T @ vectors) - (n/k)*np.eye(k), ord='fro')
    print(f"Tightness error (Frobenius): {tightness_err:.2e}   (0 = perfect)")
    print(f"Optimization success: {tightness_err < 1e-4 and off_abs.std() < 0.05}")


def duplicate_last_m_columns(A: np.ndarray, m: int) -> np.ndarray:
    k, n = A.shape
    
    if m < 1 or m > n:
        return A
    
    # Extract the last m columns
    last_m_cols = A[:, -m:]          # shape (k, m)
    
    # Append them to the original matrix
    result = np.hstack((A, last_m_cols))   # shape (k, n + m)
    
    return result


def display(m: np.ndarray, decimals: int = 4, title: str = None) -> None:
    """
    Pretty-print a float matrix with evenly spaced columns and consistent rounding.
    
    Parameters
    ----------
    m : np.ndarray
        The matrix to display (2D numpy array)
    decimals : int
        Number of decimal places to show (default: 4)
    title : str, optional
        Optional title to print above the matrix
    """
    if not isinstance(m, np.ndarray):
        m = np.array(m)
    
    if m.ndim != 2:
        raise ValueError("Input must be a 2D matrix")
    
    # Convert to float and round
    rounded = np.round(m.astype(float), decimals=decimals)
    
    # Find the maximum width needed for any number (including sign and decimal point)
    max_width = 0
    for row in rounded:
        for val in row:
            width = len(f"{val:{decimals}f}")
            if val < 0:  # account for the minus sign
                width = max(width, len(f"{val:{decimals}f}"))
            max_width = max(max_width, width)
    
    # Add a bit of padding for readability
    col_width = max_width + 2
    
    # Optional title
    if title:
        print(f"\n{title}")
        print("   " + "─" * (col_width * rounded.shape[1] + 4))
    
    # Print the matrix
    print("   " + "─" * (col_width * rounded.shape[1] + 4))
    
    for row in rounded:
        line = "   │ " + " ".join(f"{val:{col_width}.{decimals}f}" for val in row) + " │"
        print(line)
    
    print("   " + "─" * (col_width * rounded.shape[1] + 4))


def scale_to_closest_integers(matrix: np.ndarray, tol: float = 1e-10,
                              max_abs_value: int = 100) -> np.ndarray:
    """
    Scale each column AFTER the first k columns so that it becomes
    as close as possible to an integer vector, while ensuring that
    all resulting integer entries satisfy |value| <= max_abs_value.

    The function dynamically increases the search upper bound until
    the optimized scaling factors keep the final integers safely
    within [-max_abs_value, max_abs_value].

    Parameters
    ----------
    matrix : np.ndarray
        Shape (k, n) reduced matrix [I | B] (float).
    tol : float
        Tolerance for warning about large rounding residuals.
    max_abs_value : int
        Maximum allowed absolute value in the final integer matrix (default 100).

    Returns
    -------
    np.ndarray (dtype=int)
        Same shape with left k columns unchanged and right columns
        converted to closest integers under the size constraint.
    """
    if matrix.ndim != 2:
        raise ValueError("Input must be a 2D matrix")

    k, n = matrix.shape
    if k > n:
        raise ValueError("Matrix must have at least as many columns as rows")

    result = matrix.astype(float).copy()

    # Start with a modest upper bound and increase if needed
    upper_bound = 70.0
    max_attempts = 20

    for j in range(k, n):                     # only scale the non-identity columns
        v = result[:, j]
        abs_v_max = np.max(np.abs(v))

        if abs_v_max < 1e-12:
            # Degenerate column → leave as zero
            result[:, j] = 0
            continue

        best_s = None
        best_residual = np.inf
        final_upper = upper_bound

        for attempt in range(max_attempts):
            def objective(s: float) -> float:
                if s <= 0:
                    return 1e30
                scaled = s * v
                nearest = np.round(scaled)
                return np.sum((scaled - nearest) ** 2)

            # Try optimization with current upper bound
            res = minimize_scalar(
                objective,
                bounds=(0.01, final_upper),
                method='bounded',
                tol=1e-14
            )

            candidate_s = res.x
            scaled = candidate_s * v
            integer_col = np.round(scaled)
            max_entry = np.max(np.abs(integer_col))

            # Check if this scaling keeps values within limits
            if max_entry <= max_abs_value:
                # Good scaling found
                best_s = candidate_s
                best_residual = np.max(np.abs(scaled - integer_col))
                break
            else:
                # Need larger range — the optimal s is likely near the boundary
                final_upper *= 1.8   # increase by 80%
                if attempt == max_attempts - 1:
                    print(f"  Warning: Could not find scaling for column {j} "
                          f"within |x| <= {max_abs_value} after {max_attempts} attempts.")
                    # Fall back to the last tried scaling and clip
                    best_s = candidate_s
                    best_residual = np.max(np.abs(scaled - integer_col))

        # Apply best scaling found
        if best_s is None:
            best_s = 1.0

        scaled = best_s * v
        integer_col = np.round(scaled)

        # Final safety clip (should rarely trigger)
        integer_col = np.clip(integer_col, -max_abs_value/2, max_abs_value/2)

        # Quality check
        # residual = np.max(np.abs(scaled - integer_col))
        # if residual > tol:
        #     print(f"  Warning: column {j} has max residual {residual:.2e} "
        #           f"(scaling factor s = {best_s:.8f}, upper_bound used = {final_upper:.1f})")

        result[:, j] = integer_col

    return result.astype(int)


# def main():
#     n = 9
#     k = 4
#     m = 5

#     nonIvectors = approximate_etf(n, k, )

#     verify_etf(nonIvectors, f"Best Approximate ETF (n={n}, k={k})")
#     display(nonIvectors.T)
#     print('\n')

#     result = reduced_row_echelon_left_identity(nonIvectors.T)
#     for num in range(1, n-k+1):
#         print(f'{num}_height of result: {m_height(result, num):>10f}')
#         display(result)

#     integer_matrix = scale_to_closest_integers(result)
    
#     print(f'\n\nAfter finding closest integer approximation:\n')

#     for num in range(1, n-k+1):
#         print(f'{num}_height of result: {m_height(integer_matrix, num):>10f}')
#         display(integer_matrix, decimals=0)

#     P = integer_matrix[:, k:]

#     P, current_h = local_improve(P, n, k, m)

#     G = build_systematic_G(k, n, P)

#     print(f'Best {m}_height after local improvements: {current_h}')
#     display(G, 0)

# if __name__ == "__main__":
#     main()


# ====================== CONFIG ======================
GEN_PICKLE = "generatorMatrixTotalMerge"
MH_PICKLE = "mHeightTotalMerge"

PARAMS = [
    (9, 4, 2), (9, 4, 3), (9, 4, 4), (9, 4, 5),
    (9, 5, 2), (9, 5, 3), (9, 5, 4),
    (9, 6, 2), (9, 6, 3),
]


def load_state():
    best_generators = pickle.load(open(GEN_PICKLE, "rb")) if os.path.exists(GEN_PICKLE) else {}
    best_mheights = pickle.load(open(MH_PICKLE, "rb")) if os.path.exists(MH_PICKLE) else {}
    return best_generators, best_mheights


def save_state(best_generators, best_mheights):
    with open(GEN_PICKLE, "wb") as f:
        pickle.dump(best_generators, f)
    with open(MH_PICKLE, "wb") as f:
        pickle.dump(best_mheights, f)


def main():
    best_generators, best_mheights = load_state()

    print("Starting search for better generator matrices...\n")

    for n, k, m in PARAMS:
        print(f"Processing parameter (n={n}, k={k}, m={m}) ...")

        nonIvectors = approximate_etf(n, k)
        result = reduced_row_echelon_left_identity(nonIvectors.T)
        integer_matrix = scale_to_closest_integers(result)

        P = integer_matrix[:, k:]
        P, current_h = local_improve(P, n, k, m)
        G = build_systematic_G(k, n, P)

        print(f'Best {m}_height after local improvements: {current_h}')
        display(G, 0)

        param_tuple = (n, k, m)
        stored_h = best_mheights.get(param_tuple, float('inf'))

        print(f"   Computed m={m}_height: {current_h:.8f}")

        # === SAVE LOGIC ===
        if current_h < stored_h:
            print(f"   ✓ NEW BEST for {param_tuple}: {stored_h:.8f} → {current_h:.8f}")
            best_generators[param_tuple] = result.copy()   # save the reduced matrix [I | B]
            best_mheights[param_tuple] = float(current_h)
            save_state(best_generators, best_mheights)
        else:
            print(f"   No improvement (stored best = {stored_h:.8f})")

        print("-" * 60)

    # Final summary
    print("\n=== FINAL SUMMARY ===")
    for param in sorted(best_mheights.keys()):
        print(f"  {param}: best m_height = {best_mheights[param]:.8f}")


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    main()