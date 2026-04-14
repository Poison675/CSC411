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


# ====================== PERTURBATION (core of hill-climbing) ======================
def perturb_P(P: np.ndarray, intensity: int = 2) -> np.ndarray:
    """Small random integer perturbation + zero-column safety fix."""
    new_P = P.copy().astype(int)
    k, r = new_P.shape

    # Perturb ~10% of entries (or at least 1)
    num_perturb = max(1, (k * r) // 10)
    for _ in range(num_perturb):
        i = np.random.randint(0, k)
        j = np.random.randint(0, r)
        delta = np.random.randint(-intensity, intensity + 1)
        new_P[i, j] += delta
        new_P[i, j] = np.clip(new_P[i, j], -100, 100)

    # Guarantee no all-zero column (project requirement)
    zero_cols = np.all(new_P == 0, axis=0)
    if np.any(zero_cols):
        for col in np.where(zero_cols)[0]:
            new_P[:, col] = np.random.randint(-3, 4, k)
            if np.all(new_P[:, col] == 0):
                new_P[0, col] = 1  # guaranteed non-zero
    return new_P


# ====================== HILL-CLIMB CANDIDATE (replaces old random evaluator) ======================
def hill_climb_candidate(args):
    k, r, m, seed, current_P = args
    np.random.seed(seed)  # reproducible yet diverse climbs
    I = np.eye(k)

    # === INITIAL MATRIX (exactly mirrors original structure for fairness) ===
    if current_P is None:
        # No previous best → generate fresh structured/random start
        if r >= k * (m - 1):
            # Structured repeated-identity blocks (pure diagonal pattern)
            P = np.zeros((k, r), dtype=int)
            for i in range(k):
                pos = i
                while pos < r:
                    P[i, pos] = 1
                    pos += k
        else:
            # Random + as many full identity blocks as possible + remainder
            P = np.random.normal(0, 1.5, (k, r)).round().astype(int)
            num_full = min(m - 1, r // k) if m >= 2 else 0
            for b in range(num_full):
                start = b * k
                P[:, start:start + k] = I
            rem_start = num_full * k
            rem = r - rem_start
            if rem > 0:
                extra_I = np.tile(I, (1, (rem // k) + 1))[:, :rem]
                P[:, rem_start:rem_start + rem] = extra_I
            P += np.random.randint(-2, 3, (k, r))
    else:
        # Start from previous best (or its copy)
        P = current_P.copy()

    # Common safety fix
    zero_cols = np.all(P == 0, axis=0)
    if np.any(zero_cols):
        P[:, zero_cols] = 1

    # ====================== HILL CLIMBING LOOP ======================
    best_P = P.copy()
    G_best = np.hstack((I, best_P.astype(float)))
    best_h = compute_m_height(G_best, m)

    no_improve = 0
    max_steps = 300 if k <= 5 else 200          # heavy iteration budget
    patience = 50                                   # stop early if stuck

    for step in range(max_steps):
        candidate_P = perturb_P(best_P, intensity=2)
        G_cand = np.hstack((I, candidate_P.astype(float)))
        new_h = compute_m_height(G_cand, m)

        if new_h < best_h - 1e-6:          # strictly better → accept
            best_P = candidate_P
            best_h = new_h
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    return best_P, best_h


# ====================== MAIN (structure preserved + hill-climbing) ======================
params_list = [
    (9, 4, 2), (9, 4, 3), (9, 4, 4), (9, 4, 5),
    (9, 5, 2), (9, 5, 3), (9, 5, 4),
    (9, 6, 2), (9, 6, 3)
]

# Load previous best results (only improve if strictly better)
if os.path.exists("generatorMatrix"):
    with open("generatorMatrix", "rb") as f:
        generatorMatrix = pickle.load(f)
    with open("mHeight", "rb") as f:
        mHeight = pickle.load(f)
    print("✅ Loaded previous best results for comparison.")
else:
    generatorMatrix = {}
    mHeight = {}

print("\n🚀 Starting hill-climbing search (10 full iterations)...\n")

for iteration in range(10):                     # heavy outer iteration
    print(f"=== GLOBAL ITERATION {iteration + 1}/10 ===")
    for n, k, m in params_list:
        r = n - k
        key = (n, k, m)
        print(f"→ Processing {key} | r={r} | climbing from current best...")

        # Start from previous best (if any) or fresh structured random
        current_P = generatorMatrix.get(key)   # None if first time

        num_candidates = 30 if k <= 5 else 15   # parallel climbs (each does heavy local search)
        args_list = [(k, r, m, i, current_P) for i in range(num_candidates)]

        with mp.Pool(max(1, mp.cpu_count() - 1)) as pool:
            results = pool.map(hill_climb_candidate, args_list)

        # Best climb from this batch
        best_P, new_h = min(results, key=lambda x: x[1])

        old_h = mHeight.get(key, float('inf'))
        if new_h < old_h - 1e-6:
            print(f"   🔥 IMPROVED! {new_h:.6f} < previous {old_h:.6f} → Updating")
            generatorMatrix[key] = best_P
            mHeight[key] = float(new_h)
        else:
            print(f"   Best climb h={new_h:.6f} (no improvement over {old_h:.6f}) → Keeping old")

        print(f"   Current best for {key}: {mHeight.get(key, new_h):.6f}\n")

# ====================== SAVE (always write latest bests) ======================
with open("generatorMatrix", "wb") as f:
    pickle.dump(generatorMatrix, f)
with open("mHeight", "wb") as f:
    pickle.dump(mHeight, f)

print("✅ DONE! Generator matrices & m-heights saved.")
print("   • Repeated-identity structure + heavy hill-climbing used.")
print("   • Run this script again (or increase range(10)) to keep improving further.")
print("   • Files 'generatorMatrix' and 'mHeight' are ready for submission.")