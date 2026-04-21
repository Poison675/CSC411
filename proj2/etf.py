import os
import pickle
import random
from itertools import combinations
from typing import Dict, Tuple

import numpy as np
from scipy.optimize import linprog, minimize
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


def main():
    n = 9
    k = 4

    nonIvectors = approximate_etf(n, k, )

    verify_etf(nonIvectors, f"Best Approximate ETF (n={n}, k={k})")
    display(nonIvectors.T)
    print('\n')

    result = reduced_row_echelon_left_identity(nonIvectors.T)
    for num in range(1, n-k+1):
        # if num == n-k:
        #     imitation_n = 3
        #     result = approximate_etf(imitation_n, k, random_state=123)
        #     result = reduced_row_echelon_left_identity(result.T)

        #     for i in range((n-imitation_n)//(imitation_n-k)):
        #         result = duplicate_last_m_columns(result, (imitation_n-k))
            
        #     result = duplicate_last_m_columns(result, ((n-imitation_n)%(imitation_n-k)))

        print(f'{num}_height of result: {m_height(result, num):>10f}')
        display(result)

if __name__ == "__main__":
    main()