import pickle
import numpy as np
from itertools import combinations
from scipy.optimize import linprog
from typing import List, Tuple
import time as t
from concurrent.futures import ProcessPoolExecutor
import os


# ================================================================
# EXACT FUNCTIONS FROM THE PROJECT (unchanged)
# ================================================================
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


# ================================================================
# LOAD THE TWO SAMPLE FILES
# ================================================================
with open("CSCE-411-Project-sample-n_k_m_P", "rb") as f:
    samples = pickle.load(f)          # list of [n, k, m, P]

with open("CSCE-411-Project-sample-mHeights", "rb") as f:
    stored_heights = pickle.load(f)   # list of corresponding m-heights

print(f"Loaded {len(samples)} sample matrices and {len(stored_heights)} stored heights.\n")

# ================================================================
# VERIFY EACH SAMPLE
# ================================================================
s_time = t.perf_counter()
i = 0
all_match = True
for idx, (sample, stored_h) in enumerate(zip(samples, stored_heights)):
    n, k, m, P = sample
    
    # Build the full systematic generator matrix G = [I_k | P]
    I = np.eye(k, dtype=float)
    G = np.concatenate([I, P.astype(float)], axis=1)
    
    # Compute m-height using the official functions
    computed_h = m_height(G, m)
    
    # Check if they are essentially equal (floating-point tolerance)
    match = np.isclose(computed_h, stored_h, rtol=1e-8, atol=1e-10)
    
    status = "✅ MATCH" if match else "❌ MISMATCH"
    if not match:
        all_match = False
    
    print(f"Sample {idx:2d}  (n={n}, k={k}, m={m}) "
          f"→ computed = {computed_h:.10f}  |  stored = {stored_h:.10f}  → {status}")
    if i == 100:
        e_time = t.perf_counter()
        print(f'\n\nTIME: {e_time - s_time: 0.3f}')
        break
    i += 1

print("\n" + "="*80)
if all_match:
    print("✅ ALL SAMPLES VERIFIED SUCCESSFULLY")
    print("   The provided m-heights exactly match the official computation.")
else:
    print("❌ SOME MISMATCHES FOUND")
    print("   Check the printed lines above for details.")
print("="*80)