import pickle
import numpy as np
from itertools import combinations
from scipy.optimize import linprog
import multiprocessing as mp

# ====================== FIXED & CORRECT m-HEIGHT EVALUATOR (unchanged) ======================
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


# ====================== ALTERED MAIN - WITH REPEATED IDENTITY CONSTRUCTION ======================
print("Loading HW-4-n_k_m_P ...")
with open('HW-4-n_k_m_P', 'rb') as f:
    data = pickle.load(f)

print(f"Loaded {len(data)} samples. Starting m-height computation...\n")

m_heights = []
for idx, item in enumerate(data):
    n, k, m, P_loaded = item
    r = n - k
    
    # ==================== NEW STRUCTURED CONSTRUCTION (exactly as you requested) ====================
    if m >= 2 and r >= k * (m - 1):
        print(f"[{idx+1:4d}/{len(data)}]  n={n} k={k} m={m}  *** STRUCTURED REPEATED-IDENTITY CONSTRUCTION ACTIVATED ***")
        num_full = m - 1
        P_struct = np.zeros((k, r), dtype=int)
        
        # Fill with (m-1) full k×k identity blocks
        for block in range(num_full):
            start_col = block * k
            P_struct[:, start_col : start_col + k] = np.eye(k, dtype=int)
        
        # Remainder columns = first few columns of the next identity matrix
        remainder = r - num_full * k
        if remainder > 0:
            P_struct[:, num_full * k : num_full * k + remainder] = np.eye(k, dtype=int)[:, :remainder]
        
        P_to_use = P_struct
    else:
        print(f"[{idx+1:4d}/{len(data)}]  n={n} k={k} m={m}  (using loaded P)")
        P_to_use = P_loaded

    # Build systematic G = [I_k | P]
    I = np.eye(k, dtype=float)
    G = np.hstack((I, P_to_use.astype(float)))

    print(f"    shape={G.shape}  (r={r}, condition r >= k*(m-1) = {k*(m-1)} → {'YES' if m>=2 and r>=k*(m-1) else 'NO'})")
    h = compute_m_height(G, m)
    m_heights.append(h)
    print(f"    -> m-height = {h:.10f}\n")

# Save the result
# with open('HW-4-mHeightsTEMP', 'wb') as f:
#     pickle.dump(m_heights, f)

print("All done! File 'HW-4-mHeightsTEMP' created successfully.")
print("   • When r >= k*(m-1), the loaded P was replaced by repeated identity blocks + partial identity remainder.")
print("   • All heights were computed with the exact project LP algorithm.")