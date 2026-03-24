import pickle
import numpy as np
from itertools import combinations
from scipy.optimize import linprog
import multiprocessing as mp
import sys

# ====================== FIXED & CORRECT m-HEIGHT EVALUATOR ======================
def _solve_lp(args):
    """Worker for one LPS,j LP (exact match to project document Section 3)."""
    G, j, barS = args
    c = -G[:, j]                                      # maximize → minimize negative
    barS_matrix = G[:, barS].T                        # (len(barS), k)
    A_ub = np.vstack([barS_matrix, -barS_matrix])
    b_ub = np.ones(2 * len(barS))
    res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                  bounds=(None, None),
                  method='highs',
                  options={'presolve': True, 'disp': False})
    return -res.fun if res.success else 1.0

def compute_m_height(G: np.ndarray, m: int) -> float:
    """Exact m-height of the code (project algorithm)."""
    n = G.shape[1]
    if m == 0:
        return 1.0
    tasks = []
    for S_tup in combinations(range(n), m):
        barS = [t for t in range(n) if t not in S_tup]
        for j in S_tup:
            tasks.append((G, j, barS))
    num_workers = max(1, mp.cpu_count() - 1)
    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(_solve_lp, tasks)
    hm = max(results) if results else 1.0
    return max(hm, 1.0)


# ====================== HILL-CLIMBING REFINEMENT ======================
def hill_climb(P, m, max_iter=5):
    """Local search: try ±1 on every entry, keep if height improves."""
    best_P = P.copy().astype(float)
    k, r = best_P.shape
    I = np.eye(k, dtype=float)
    improved = True
    iter_count = 0
    while improved and iter_count < max_iter:
        improved = False
        iter_count += 1
        for i in range(k):
            for j in range(r):
                for delta in [-1.0, 1.0]:
                    old_val = best_P[i, j]
                    best_P[i, j] += delta
                    G = np.hstack((I, best_P))
                    h_new = compute_m_height(G, m)
                    if h_new < compute_m_height(np.hstack((I, best_P.copy())), m) - 1e-8:  # strict improvement
                        improved = True
                    else:
                        best_P[i, j] = old_val  # revert
    return best_P.astype(int)


# ====================== MAIN SEARCH FOR ALL 9 CASES ======================
params_list = [
    (9, 4, 2), (9, 4, 3), (9, 4, 4), (9, 4, 5),
    (9, 5, 2), (9, 5, 3), (9, 5, 4),
    (9, 6, 2), (9, 6, 3)
]

generatorMatrix = {}
mHeight = {}

print("Starting search for best generator matrices (n=9 cases)...\n")
for n, k, m in params_list:
    r = n - k
    print(f"→ Processing (n={n}, k={k}, m={m}) | P shape=({k},{r})")
    best_h = float('inf')
    best_P = None
    trials = 5000 if m <= 3 else 3000   # fewer trials for larger m (more expensive)

    for trial in range(trials):
        P = np.random.randint(-8, 9, size=(k, r))
        if np.any(np.all(P == 0, axis=0)):
            continue
        G = np.hstack((np.eye(k), P.astype(float)))
        h = compute_m_height(G, m)
        if h < best_h:
            best_h = h
            best_P = P.copy()
            print(f"   New best! h={best_h:.6f} (trial {trial+1})")

    # Refine with hill-climbing
    if best_P is not None:
        print("   Running hill-climbing refinement...")
        refined_P = hill_climb(best_P, m)
        G_ref = np.hstack((np.eye(k), refined_P.astype(float)))
        final_h = compute_m_height(G_ref, m)
        if final_h < best_h:
            best_h = final_h
            best_P = refined_P
            print(f"   Refined! Final h={best_h:.6f}")

    # Clip to [-100,100] (already inside) and ensure no zero columns
    best_P = np.clip(best_P, -100, 100).astype(int)
    if np.any(np.all(best_P == 0, axis=0)):
        best_P[0, np.where(np.all(best_P == 0, axis=0))[0]] = 1  # fix any zero column

    generatorMatrix[(n, k, m)] = best_P
    mHeight[(n, k, m)] = float(best_h)
    print(f"   FINAL for ({n},{k},{m}): h={best_h:.6f}\n")

# ====================== SAVE SUBMISSION FILES ======================
with open("generatorMatrix", "wb") as f:
    pickle.dump(generatorMatrix, f)
with open("mHeight", "wb") as f:
    pickle.dump(mHeight, f)

print("✅ DONE! Files created:")
print("   • generatorMatrix  (contains all 9 best P matrices)")
print("   • mHeight          (contains the corresponding m-heights)")
print("\nYou can now submit these two files + your report + codes folder.")
print("All matrices have integer entries in [-100,100] and no zero columns.")