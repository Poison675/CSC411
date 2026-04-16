import pickle
import random
import numpy as np
from itertools import combinations
from scipy.optimize import linprog
from typing import Tuple, Dict

# =============================================================================
# m-HEIGHT COMPUTATION (exact, for final evaluation)
# =============================================================================
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
# ANGLE-BASED COST (the criterion you asked for)
# =============================================================================
def angle_based_cost(G: np.ndarray, m: int) -> float:
    n = G.shape[1]
    norms = np.linalg.norm(G, axis=0) + 1e-12
    cos_matrix = np.abs(G.T @ G) / np.outer(norms, norms)
    ang_matrix = np.arccos(np.clip(cos_matrix, -1.0, 1.0))

    worst_angle = 0.0
    for i in range(n):
        other_mask = np.arange(n) != i
        ang_to_others = ang_matrix[i, other_mask]
        closest_rel = np.argsort(ang_to_others)[:m-1]
        other_indices = np.where(other_mask)[0]
        subset_idx = [i] + list(other_indices[closest_rel])

        sub_angles = ang_matrix[np.ix_(subset_idx, subset_idx)]
        max_in_subset = np.max(sub_angles)
        if max_in_subset > worst_angle:
            worst_angle = max_in_subset
    return worst_angle


# =============================================================================
# LOCAL IMPROVEMENT (greedy ±1 on angle cost)
# =============================================================================
def local_improve_angle(P: np.ndarray, m: int) -> np.ndarray:
    P = P.copy().astype(int)
    k, p_dim = P.shape
    n = k + p_dim

    G = np.hstack([np.eye(k), P.astype(float)])
    current_cost = angle_based_cost(G, m)

    improved = True
    while improved:
        improved = False
        for i in range(k):
            for j in range(p_dim):
                for delta in [-1, 1]:
                    P_new = P.copy()
                    P_new[i, j] += delta
                    if np.all(P_new[:, j] == 0):
                        continue

                    G_new = np.hstack([np.eye(k), P_new.astype(float)])
                    new_cost = angle_based_cost(G_new, m)

                    if new_cost < current_cost:
                        P = P_new
                        current_cost = new_cost
                        improved = True
                        print(f"    Angle-improve: P[{i},{j}] += {delta} → cost = {current_cost:.4f}")
                        break
                if improved:
                    break
            if improved:
                break
    return P


# =============================================================================
# PARAMETERS & MAIN LOOP
# =============================================================================
PARAMS = [
    (9, 4, 2), (9, 4, 3), (9, 4, 4), (9, 4, 5),
    (9, 5, 2), (9, 5, 3), (9, 5, 4),
    (9, 6, 2), (9, 6, 3),
]

TRIALS_PER_PARAM = 10000          # ← increase for better quality (slower)
LOW = -10
HIGH = 10


def main():
    best_generators: Dict[Tuple[int, int, int], np.ndarray] = {}
    best_mheights: Dict[Tuple[int, int, int], float] = {}

    print(f"Starting search for {len(PARAMS)} parameter sets "
          f"({TRIALS_PER_PARAM} random trials each + local improvement)\n")

    for n, k, m in PARAMS:
        p_dim = n - k
        print(f"\n{'='*70}")
        print(f"Processing (n={n}, k={k}, m={m})  →  P is {k}×{p_dim}")
        print(f"{'='*70}")

        best_cost = float('inf')
        best_P = None
        best_h = None

        # Random search phase
        for trial in range(TRIALS_PER_PARAM):
            # Random integer P with no all-zero columns
            while True:
                P = np.random.randint(LOW, HIGH + 1, size=(k, p_dim))
                if np.all(np.any(P != 0, axis=0)):
                    break

            G = np.hstack([np.eye(k), P.astype(float)])
            cost = angle_based_cost(G, m)

            if cost < best_cost:
                best_cost = cost
                best_P = P.copy()

                try:
                    h = m_height(G, m)
                    best_h = h
                except Exception:
                    best_h = None

                print(f"  Trial {trial+1:4d}/{TRIALS_PER_PARAM} | "
                      f"angle-cost = {cost:.4f} rad | m-height = {best_h:.6f}")

        # Local improvement phase
        if best_P is not None:
            print(f"\n  Running local ±1 improvement on angle cost...")
            best_P = local_improve_angle(best_P, m)
            G_final = np.hstack([np.eye(k), best_P.astype(float)])
            best_cost = angle_based_cost(G_final, m)
            best_h = m_height(G_final, m)
            print(f"  Final after local improve → angle-cost = {best_cost:.4f} rad | m-height = {best_h:.6f}")

        # Store results
        best_generators[(n, k, m)] = best_P
        best_mheights[(n, k, m)] = float(best_h)

    # Save final submission files (exactly as required by the project)
    with open("generatorMatrix", "wb") as f:
        pickle.dump(best_generators, f)

    with open("mHeight", "wb") as f:
        pickle.dump(best_mheights, f)

    print("\n" + "="*70)
    print("ALL DONE! Submission files created:")
    print("   • generatorMatrix")
    print("   • mHeight")
    print("="*70)
    print("Best m-heights found:")
    for key in sorted(best_mheights.keys()):
        print(f"   {key} → h_m = {best_mheights[key]:.6f}")


if __name__ == "__main__":
    main()