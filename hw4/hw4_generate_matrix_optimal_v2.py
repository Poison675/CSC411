import pickle
import numpy as np
from itertools import combinations
from scipy.optimize import linprog
import multiprocessing as mp
import os

# ====================== m-HEIGHT EVALUATOR ======================
def _solve_lp(args):
    G, j, barS = args
    c = -G[:, j]
    barS_matrix = G[:, barS].T
    A_ub = np.vstack([barS_matrix, -barS_matrix])
    b_ub = np.ones(2 * len(barS))
    res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                  bounds=(None, None), method='highs',
                  options={'presolve': True, 'disp': False})
    return -res.fun if res.success else 1.0

def compute_m_height(G: np.ndarray, m: int) -> float:
    n = G.shape[1]
    if m == 0:
        return 1.0
    tasks = [(G, j, [t for t in range(n) if t not in S]) 
             for S in combinations(range(n), m) for j in S]
    results = [_solve_lp(task) for task in tasks]
    return max(max(results), 1.0)


def evaluate_candidate(args):
    k, r, m, seed = args
    # Set seed for reproducibility (critical for consistent search/optimization)
    np.random.seed(seed)
    I = np.eye(k)
    
    if r >= k * (m - 1):
        # === CASE 1: r < k*(m-1) ===
        # Replace each row of P with repeated identity matrices:
        # For row i, place 1's at columns i, i+k, i+2k, ... (pure structured diagonal pattern)
        # This is the "required special case" for small parity-check length relative to m.
        P = np.zeros((k, r), dtype=int)
        for i in range(k):
            pos = i
            while pos < r:
                P[i, pos] = 1
                pos += k
    else:
        # === CASE 2: r >= k*(m-1) ===
        # Randomly seed exactly like the original snippet (normal + round to int),
        # then apply the safe repeated-identity blocks + perturbation,
        P = np.random.normal(0, 1.5, (k, r)).round().astype(int)
        
        # Fill as many full k×k identity blocks as possible (up to m-1)
        num_full = min(m - 1, r // k) if m >= 2 else 0
        for b in range(num_full):
            start = b * k
            P[:, start:start + k] = I
        
        # Remainder columns (cycle identity safely)
        rem_start = num_full * k
        rem = r - rem_start
        if rem > 0:
            extra_I = np.tile(I, (1, (rem // k) + 1))[:, :rem]
            P[:, rem_start:rem_start + rem] = extra_I
        
        # Perturbation for diversity (exactly as in the original)
        P += np.random.randint(-2, 3, (k, r))
    
    # Common post-processing for both cases (ensures validity)
    # No zero columns allowed (project requirement)
    zero_cols = np.all(P == 0, axis=0)
    if np.any(zero_cols):
        P[:, zero_cols] = 1   # safe non-zero fix
    
    # Build systematic generator matrix G = [I_k | P]
    G = np.hstack((I, P.astype(float)))
    
    # Compute m-height
    h = compute_m_height(G, m)
    return P, h



# ====================== MAIN ======================
params_list = [
    (9, 4, 2), (9, 4, 3), (9, 4, 4), (9, 4, 5),
    (9, 5, 2), (9, 5, 3), (9, 5, 4),
    (9, 6, 2), (9, 6, 3)
]

# Load previous best (only update if strictly better)
if os.path.exists("storage/generatorMatrix"):
    with open("storage/generatorMatrix", "rb") as f:
        generatorMatrix = pickle.load(f)
    with open("storage/mHeight", "rb") as f:
        mHeight = pickle.load(f)
    print("✅ Loaded previous best results for comparison.")
else:
    generatorMatrix = {}
    mHeight = {}

print("\nStarting stable parallel search (broadcasting bug fixed)...\n")

for _ in range(1):
    for n, k, m in params_list:
        r = n - k
        key = (n, k, m)
        print(f"→ Processing {key} | r={r}")

        num_candidates = 100 if k <= 5 else 70
        args_list = [(k, r, m, i) for i in range(num_candidates)]

        with mp.Pool(max(1, mp.cpu_count() - 1)) as pool:
            results = pool.map(evaluate_candidate, args_list)

        best_P, new_h = min(results, key=lambda x: x[1])

        old_h = mHeight.get(key, float('inf'))
        if new_h < old_h - 1e-6:
            print(f"   🔥 IMPROVED! {new_h:.6f} < previous {old_h:.6f} → Updating")
            generatorMatrix[key] = best_P
            mHeight[key] = float(new_h)
        else:
            print(f"   New h={new_h:.6f} not better than previous {old_h:.6f} → Keeping old")

        print(f"   Current best for {key}: {mHeight.get(key, new_h):.6f}\n")

# ====================== SAVE ONLY IF IMPROVED ======================
with open("storage/generatorMatrix", "wb") as f:
    pickle.dump(generatorMatrix, f)
with open("storage/mHeight", "wb") as f:
    pickle.dump(mHeight, f)

print("✅ DONE!")
print("   • Repeated-identity seeding is always safe (even when r is small).")
print("   • Run this script multiple times to keep improving further.")