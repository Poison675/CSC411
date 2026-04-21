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
        [1, 1, 1, 1,-1],
        [1, 1, 1,-1, 1],
        [1, 1,-1, 1, 1],
        [1,-1, 1, 1, 1],
    ], dtype=float)

    I = np.eye(P.shape[0], dtype=float)
    P = np.concatenate([I, P.astype(float)], axis=1)

    for m in range(P.shape[1]):
        print(f"h_{m+1}(C) = {m_height(P, m)}")

    # for m in [3]:
    #     print(f"h_{m}(C) = {m_height(P, m-1)}")







# if __name__ == "__main__":
#     original_P = np.array([[1.0], [1.0]])          # the column you want to scale

#     best_2_height = float('inf')
#     best_scaled_P = None
#     best_scale = None

#     print("Searching for best scaling factor...\n")

#     for i in np.linspace(0.01, 2, 1000):
#         scaled_P = original_P / i                     # scale the dynamic column

#         I = np.eye(2, dtype=float)
#         G = np.concatenate([I, scaled_P], axis=1)     # full G = [I | scaled_P]

#         h2 = m_height(G, m=1)

#         # Only consider finite (valid) heights
#         if np.isfinite(h2) and h2 < best_2_height:
#             best_2_height = h2
#             best_scaled_P = scaled_P.copy()
#             best_scale = i
#             print(f"New best → scale = {i:.4f} | 2-height = {h2:.8f}")

#     print("\n" + "="*70)
#     print("SEARCH FINISHED")
#     print("="*70)

#     if best_scale is None:
#         print("❌ No finite 2-height was found in the searched range.")
#         print("   (Most likely because very large scalings made the LP unbounded)")
#     else:
#         print(f"Best scale factor : {best_scale:.6f}")
#         print(f"Best dynamic P    :\n{best_scaled_P}")
#         print(f"Best 2-height     : {best_2_height:.8f}")
#         print(f"Full generator matrix G:\n{np.concatenate([np.eye(2), best_scaled_P], axis=1)}")



