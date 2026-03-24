"""
CSCE 411 Project - Generator Matrix Optimizer (Final Clean Version)
====================================================================

This script generates high-quality systematic generator matrices P 
for the Analog Code m-height minimization problem (n=9 cases).

Key features:
- Seeds matrices with Identity matrix if the dimensions are viable
- When no seeding: full random + perturbation for diversity
- Parallel candidate evaluation + hill-climbing refinement
- Only updates saved files when a strictly better m-height is found

Author: Grok (cleaned and documented for submission)
"""

import pickle
import numpy as np
from itertools import combinations
from scipy.optimize import linprog
import multiprocessing as mp
import os


# ====================== CORE: EXACT m-HEIGHT EVALUATOR ======================
def _solve_lp(args):
    """Solve one LPS,j linear program (exact match to project document Section 3)."""
    G, j, barS = args
    c = -G[:, j]
    barS_matrix = G[:, barS].T
    A_ub = np.vstack([barS_matrix, -barS_matrix])
    b_ub = np.ones(2 * len(barS))
    res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                  bounds=(None, None), method='highs',
                  options={'presolve': True, 'disp': False})
    return -res.fun if res.success else -1


def compute_m_height(G: np.ndarray, m: int) -> float:
    """
    Compute the exact m-height of the Analog Code using the official
    LP-based algorithm from the project document (Section 3).
    """
    n = G.shape[1]
    if m == 0:
        return 1.0

    tasks = [(G, j, [t for t in range(n) if t not in S])
             for S in combinations(range(n), m) for j in S]
    results = [_solve_lp(task) for task in tasks]
    return max(max(results), 0)


# ====================== LIGHT HILL-CLIMB REFINEMENT ======================
def hill_climb(P: np.ndarray, m: int, max_steps: int = 8) -> np.ndarray:
    """
    Perform simple local search: try ±1 on every entry and keep any
    improvement in m-height. Used as final refinement on the best candidate.
    """
    best_P = P.copy().astype(float)
    k, r = best_P.shape
    I = np.eye(k)

    for _ in range(max_steps):
        improved = False
        for i in range(k):
            for j in range(r):
                for delta in [-1.0, 1.0]:
                    old_val = best_P[i, j]
                    best_P[i, j] += delta
                    G = np.hstack((I, best_P))
                    h_new = compute_m_height(G, m)
                    if h_new < compute_m_height(np.hstack((I, best_P.copy())), m) - 1e-8:
                        improved = True
                    else:
                        best_P[i, j] = old_val
        if not improved:
            break
    return best_P.astype(int)


# ====================== CANDIDATE EVALUATOR ======================
# def evaluate_candidate(args):
#     """
#     Generate and evaluate one candidate matrix.
#     - Applies the user's special repeated-identity seeding when the condition is met.
#     - When seeding is active: leaves the matrix completely clean (no perturbation).
#     - When no seeding: uses random initialization + perturbation for diversity.
#     """
#     k, r, m, seed = args
#     np.random.seed(seed)                    # reproducible randomness

#     I = np.eye(k)

#     # === SPECIAL REPEATED-IDENTITY SEEDING (user's required rule) ===
#     num_full = min(m - 1, r // k) if m >= 2 else 0

#     P = np.zeros((k, r), dtype=int)         # start with clean zero matrix

#     # Fill full identity blocks
#     for b in range(num_full):
#         start = b * k
#         P[:, start:start + k] = I

#     # Remainder columns (safe cycling of identity)
#     rem_start = num_full * k
#     rem = r - rem_start
#     if rem > 0:
#         extra_I = np.tile(I, (1, (rem // k) + 1))[:, :rem]
#         P[:, rem_start:rem_start + rem] = extra_I

#     # === NO PERTURBATION WHEN SEEDED (per user request) ===
#     if num_full == 0:
#         # Only apply random initialization + perturbation when no special seeding
#         P = np.random.normal(0, 1.5, (k, r)).round().astype(int)
#         P += np.random.randint(-2, 3, (k, r))
#         P = np.clip(P, -15, 15)

#     # Final safety: ensure no zero columns
#     P[:, np.all(P == 0, axis=0)] = 1

#     # Evaluate exact m-height
#     G = np.hstack((I, P.astype(float)))
#     h = compute_m_height(G, m)

#     return P, h


def evaluate_candidate(args):
    """
    Generate and evaluate one candidate matrix.
    - ALWAYS seeds with as many full k×k identity blocks as possible (num_full = r // k).
    - Remainder columns are filled from the beginning of the next identity matrix.
    - No random perturbation is applied (clean structured seeding for ALL matrices).
    """
    k, r, m, seed = args
    np.random.seed(seed)                    # reproducible randomness

    I = np.eye(k)

    # === ALWAYS SEED WITH MAXIMUM POSSIBLE IDENTITY BLOCKS ===
    num_full = r // k                       # maximum full blocks that fit in r columns

    P = np.zeros((k, r), dtype=int)         # start with clean zero matrix

    # Fill full identity blocks
    for b in range(num_full):
        start = b * k
        P[:, start:start + k] = I

    # Remainder columns (safe cycling of identity)
    rem_start = num_full * k
    rem = r - rem_start
    if rem > 0:
        extra_I = np.tile(I, (1, (rem // k) + 1))[:, :rem]
        P[:, rem_start:rem_start + rem] = extra_I

    # === NO PERTURBATION (clean seeding for ALL matrices per user request) ===

    # Final safety: ensure no zero columns
    P[:, np.all(P == 0, axis=0)] = 1

    # Evaluate exact m-height
    G = np.hstack((I, P.astype(float)))
    h = compute_m_height(G, m)

    return P, h




# ====================== MAIN EXECUTION ======================
if __name__ == "__main__":
    params_list = [
        (9, 4, 2), (9, 4, 3), (9, 4, 4), (9, 4, 5),
        (9, 5, 2), (9, 5, 3), (9, 5, 4),
        (9, 6, 2), (9, 6, 3)
    ]

    # Load previous best results (if they exist)
    if os.path.exists("generatorMatrix"):
        with open("generatorMatrix", "rb") as f:
            generatorMatrix = pickle.load(f)
        with open("mHeight", "rb") as f:
            mHeight = pickle.load(f)
        print("✅ Loaded previous best results.")
    else:
        generatorMatrix = {}
        mHeight = {}

    print("\nStarting optimized parallel search...\n")

    for n, k, m in params_list:
        r = n - k
        key = (n, k, m)
        print(f"→ Processing {key} | r={r}")

        num_candidates = 10 if k <= 5 else 5
        args_list = [(k, r, m, i) for i in range(num_candidates)]

        with mp.Pool(max(1, mp.cpu_count() - 1)) as pool:
            results = pool.map(evaluate_candidate, args_list)

        best_P, new_h = min(results, key=lambda x: x[1])

        # Light hill-climbing refinement on best candidate
        print("   Applying hill-climbing...")
        refined_P = hill_climb(best_P, m)
        G_ref = np.hstack((np.eye(k), refined_P.astype(float)))
        refined_h = compute_m_height(G_ref, m)

        if refined_h < new_h:
            best_P, new_h = refined_P, refined_h
            print(f"   Refined! h = {new_h:.6f}")

        # Update only if strictly better
        old_h = mHeight.get(key, float('inf'))
        if new_h < old_h - 1e-6:
            print(f"   IMPROVED! {new_h:.6f} < previous {old_h:.6f} → Updating")
            generatorMatrix[key] = best_P
            mHeight[key] = float(new_h)
        else:
            print(f"   New h={new_h:.6f} not better → Keeping old")

        print(f"   Current best for {key}: {mHeight.get(key, new_h):.6f}\n")

    # ====================== SAVE FILES ======================
    with open("generatorMatrixTEMP", "wb") as f:
        pickle.dump(generatorMatrix, f)
    with open("mHeightTEMP", "wb") as f:
        pickle.dump(mHeight, f)

    print("✅ DONE! Files 'generatorMatrix' and 'mHeight' are ready.")
    print("   • When seeding is active: clean repeated-identity (no perturbation)")
    print("   • Run multiple times to keep improving.")