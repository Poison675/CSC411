import numpy as np
from itertools import combinations
from scipy.optimize import linprog
from typing import List, Tuple
from concurrent.futures import ProcessPoolExecutor


def _solve_lp(args):
    G, j, barS = args
    c = -G[:, j]
    barS_matrix = G[:, barS].T
    A_ub = np.vstack([barS_matrix, -barS_matrix])
    b_ub = np.ones(2 * len(barS))
    res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                  bounds=(None, None), method='highs',
                  options={'presolve': True, 'disp': False})
    return -res.fun if res.success else np.inf

def m_height(G: np.ndarray, m: int) -> float:
    n = G.shape[1]
    if m == 0:
        return 1.0
    tasks = [(G, j, [t for t in range(n) if t not in S]) 
             for S in combinations(range(n), m) for j in S]
    results = [_solve_lp(task) for task in tasks]
    return max(max(results), 1.0)


# =============================================================================
# Example / test (matches the document's Example 6 exactly)
# =============================================================================
if __name__ == "__main__":
    # Generator matrix
    # P = np.array([
    #     [0.4759809, 0.9938236, 0.819425],
    #     [-0.8960798, -0.7442706, 0.3345122],
    # ], dtype=float)

    P = np.array([
        [-1],
        [-1],
    ], dtype=float)

    I = np.eye(P.shape[0], dtype=float)
    P = np.concatenate([I, P.astype(float)], axis=1)

    for m in range(P.shape[1]):
        print(f"h_{m+1}(C) = {m_height(P, m)}")