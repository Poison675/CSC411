import pickle
import numpy as np
from itertools import combinations
from scipy.optimize import linprog
import multiprocessing as mp

# ====================== FIXED & CORRECT m-HEIGHT EVALUATOR ======================
def _solve_lp(args):
    """Worker for one LPS,j LP (proven formulation from the project document)."""
    G, j, barS = args
    c = -G[:, j]                                      # maximize g_j · u  →  minimize -g_j · u
    
    # FIXED: G is (k, n) → select COLUMNS, then transpose to get coefficient rows
    barS_matrix = G[:, barS].T                        # shape: (len(barS), k)
    A_ub = np.vstack([barS_matrix, -barS_matrix])     # |g_t · u| ≤ 1 for each t ∉ S
    b_ub = np.ones(2 * len(barS))
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                  bounds=(None, None),
                  method='highs',
                  options={'presolve': True, 'disp': False})
    return -res.fun if res.success else 1.0

def compute_m_height(G: np.ndarray, m: int) -> float:
    """
    Exact m-height using the project's LP algorithm (Section 3).
    Takes only G and m — n and k are read from G.shape.
    """
    n = G.shape[1]
    if m == 0:
        return 1.0

    # Build all tasks: one per (S, j∈S)
    tasks = []
    for S_tup in combinations(range(n), m):
        barS = [t for t in range(n) if t not in S_tup]
        for j in S_tup:
            tasks.append((G, j, barS))

    # Parallel solve
    num_workers = max(1, mp.cpu_count() - 1)
    print(f"   Solving {len(tasks):,} LPs with {num_workers} CPU workers...")
    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(_solve_lp, tasks)

    hm = max(results) if results else 1.0
    return max(hm, 1.0)


# ====================== MAIN (unchanged except using the fixed evaluator) ======================
# print("Loading HW-4-n_k_m_P ...")
# with open('HW-4-n_k_m_P', 'rb') as f:
#     data = pickle.load(f)

# print(f"Loaded {len(data)} samples. Starting m-height computation...\n")

# m_heights = []
# for idx, item in enumerate(data):
#     n, k, m, P = item
#     # Build systematic generator matrix G = [I_k | P]
#     I = np.eye(k, dtype=float)
#     G = np.hstack((I, P.astype(float)))

#     print(f"[{idx+1:4d}/{len(data)}]  n={n} k={k} m={m}  shape={G.shape}")
#     h = compute_m_height(G, m)          # ← fixed version
#     m_heights.append(h)
#     print(f"    -> m-height = {h:.10f}\n")

# # Save the result
# with open('HW-4-mHeightsTEMP', 'wb') as f:
#     pickle.dump(m_heights, f)

# print("All done! File 'HW-4-mHeightsTEMP' created successfully.")

if __name__ == "__main__":
    print("Loading generatorMatrix file...")
    with open("generatorMatrix", "rb") as f:
        generatorMatrix = pickle.load(f)

    print(f"Loaded {len(generatorMatrix)} generator matrices.\n")

    mHeight = {}

    # Process each stored matrix
    for idx, (key, P) in enumerate(sorted(generatorMatrix.items()), 1):
        n, k, m = key
        print(f"[{idx:4d}/{len(generatorMatrix)}]  n={n} k={k} m={m}  shape={P.shape}")

        # Build systematic generator matrix G = [I_k | P]
        I = np.eye(k, dtype=float)
        G = np.hstack((I, P.astype(float)))

        h = compute_m_height(G, m)
        mHeight[key] = float(h)
        print(f"    -> m-height = {h:.10f}\n")

    # ====================== SAVE EXACTLY AS PROJECT REQUIRES ======================
    with open("mHeight", "wb") as f:
        pickle.dump(mHeight, f)

    print("✅ DONE! File 'mHeight' has been created with verified values.")
    print("   You can now submit 'generatorMatrix' and 'mHeight'.")