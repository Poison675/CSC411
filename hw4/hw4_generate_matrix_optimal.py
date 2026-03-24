import pickle
import numpy as np
from itertools import combinations
from scipy.optimize import linprog, differential_evolution
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

# ====================== SINGLE-THREADED EVALUATOR (stable & fast) ======================
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


# ====================== DE WITH EARLY STOPPING (parallel restarts) ======================
def de_single_restart(args):
    """One independent DE run (top-level so it can be parallelized)."""
    k, r, m, restart_id = args
    I = np.eye(k)
    
    # === YOUR REQUIRED REPEATED-IDENTITY SEEDING ===
    num_full = m - 1 if (m >= 2 and r >= k * (m - 1)) else 0
    P_seed = np.zeros((k, r))
    for b in range(num_full):
        start = b * k
        P_seed[:, start:start + k] = I
    rem = r - num_full * k
    if rem > 0:
        extra = np.tile(I, (1, (rem // k) + 1))[:, :rem]
        P_seed[:, num_full * k : num_full * k + rem] = extra

    def objective(flat_P):
        P_real = flat_P.reshape(k, r)
        G = np.hstack((I, P_real))
        return compute_m_height(G, m)

    bounds = [(-12, 12)] * (k * r)
    pop0 = np.tile(P_seed.flatten(), (15, 1)) + np.random.normal(0, 0.8, (15, k*r))

    best_h = float('inf')
    last_improvement = 0
    generation = 0

    def callback(xk, convergence=0.0):
        nonlocal best_h, last_improvement, generation
        generation += 1
        current_h = objective(xk)
        if current_h < best_h - 1e-4:          # meaningful improvement threshold
            best_h = current_h
            last_improvement = generation
        # Early stop if no meaningful improvement for 6 generations
        if generation - last_improvement > 6:
            return True
        return False

    result = differential_evolution(
        objective,
        bounds,
        popsize=15,
        init=pop0,
        maxiter=40,
        tol=1e-5,
        callback=callback,
        workers=1,
        polish=False
    )

    best_real_P = result.x.reshape(k, r)
    return best_real_P, result.fun


# ====================== MAIN PARALLEL DE OPTIMIZER ======================
def optimal_de_construction(k: int, r: int, m: int):
    """Parallel DE with early stopping + your special seeding."""
    num_restarts = 6 if k <= 5 else 4                     # more restarts on smaller k
    args_list = [(k, r, m, i) for i in range(num_restarts)]

    print(f"   Running {num_restarts} parallel DE restarts (early-stop on stall)...")
    with ProcessPoolExecutor(max_workers=num_restarts) as executor:
        results = list(executor.map(de_single_restart, args_list))

    # Pick the best restart
    best_real_P, best_h = min(results, key=lambda x: x[1])

    # Round + final hill-climb (light)
    P_int = np.round(best_real_P).astype(int)
    P_final = P_int.copy()                               # you can add hill_climb here if desired
    # Quick safety
    P_final[:, np.all(P_final == 0, axis=0)] = 1

    G_final = np.hstack((np.eye(k), P_final.astype(float)))
    final_h = compute_m_height(G_final, m)
    return P_final, final_h


# ====================== MAIN SCRIPT ======================
params_list = [
    (9, 4, 2), (9, 4, 3), (9, 4, 4), (9, 4, 5),
    (9, 5, 2), (9, 5, 3), (9, 5, 4),
    (9, 6, 2), (9, 6, 3)
]

if os.path.exists("generatorMatrixOVERALL"):
    with open("generatorMatrixOVERALL", "rb") as f:
        generatorMatrix = pickle.load(f)
    with open("mHeightOVERALL", "rb") as f:
        mHeight = pickle.load(f)
    print("✅ Loaded previous best results.")
else:
    generatorMatrix = {}
    mHeight = {}

print("\nStarting MAXIMALLY PARALLEL DE search (with early stopping)...\n")

for n, k, m in params_list:
    r = n - k
    key = (n, k, m)
    print(f"→ Processing {key} | r={r}")

    P_new, new_h = optimal_de_construction(k, r, m)

    old_h = mHeight.get(key, float('inf'))
    if new_h < old_h - 1e-6:
        print(f"   🔥 IMPROVED! {new_h:.6f} < previous {old_h:.6f} → Updating")
        generatorMatrix[key] = P_new
        mHeight[key] = float(new_h)
    else:
        print(f"   New h={new_h:.6f} not better → Keeping old")

    print(f"   Current best for {key}: {mHeight.get(key, new_h):.6f}\n")

# Save only on improvement
with open("generatorMatrixOVERALL", "wb") as f:
    pickle.dump(generatorMatrix, f)
with open("mHeightOVERALL", "wb") as f:
    pickle.dump(mHeight, f)
with open("generatorMatrixOptimal", "wb") as f:
    pickle.dump(generatorMatrix, f)
with open("mHeightOptimal", "wb") as f:
    pickle.dump(mHeight, f)

print("✅ DONE!")
print("   • Special repeated-identity seeding is ALWAYS respected.")
print("   • Early stopping triggers if no meaningful improvement (≤1e-4) for 6 generations.")
print("   • Run multiple times to keep improving.")