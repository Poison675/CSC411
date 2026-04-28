import pickle
import numpy as np
from itertools import combinations
from scipy.optimize import linprog, differential_evolution
from typing import Tuple, Dict, Optional

# =============================================================================
# EXACT m-HEIGHT COMPUTATION (LP-based, as in the project spec)
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
    if not res.success or np.isnan(res.fun):
        return 1e9  # large penalty for numerical issues / unbounded LP → low d(C)
    return -res.fun if res.success else 1e9

def m_height(G: np.ndarray, m: int) -> float:
    """Exact m-height using the LP algorithm from the project document."""
    n = G.shape[1]
    if m == 0:
        return 1.0
    tasks = [(G.copy(), j, [t for t in range(n) if t not in S]) 
             for S in combinations(range(n), m) for j in S]
    results = [_solve_lp(task) for task in tasks]
    return max(max(results), 1.0)

# =============================================================================
# NORMALIZE + SCALE TO CLOSEST SMALL-INTEGER APPROXIMATION
# (applied to EVERY candidate before evaluation)
# =============================================================================
def normalize_and_approximate_P(P_in: np.ndarray, target_scale: float = 15.0) -> np.ndarray:
    """Normalize columns to unit length → scale → round to integer [-100,100].
    Guarantees no zero columns. This is the key discretization step."""
    P = P_in.astype(float).copy()
    k, p_dim = P.shape
    for j in range(p_dim):
        col = P[:, j]
        norm = np.linalg.norm(col)
        if norm < 1e-12:
            col = np.random.randn(k)
            norm = np.linalg.norm(col)
        col_norm = col / norm
        col_scaled = col_norm * target_scale
        P[:, j] = np.round(col_scaled)
    P = np.clip(P, -100, 100).astype(int)
    # Safety: no zero columns
    for j in range(p_dim):
        while np.all(P[:, j] == 0):
            col = np.random.randn(k)
            col_norm = col / np.linalg.norm(col)
            col_scaled = col_norm * target_scale
            P[:, j] = np.round(np.clip(col_scaled, -100, 100)).astype(int)
    return P

# =============================================================================
# IMPROVED PROXY COST: MAX |COSINE| BETWEEN COLUMNS (well-spread directions)
# =============================================================================
def angle_based_cost(G: np.ndarray, m: int) -> float:
    """BETTER proxy (replaces the old flawed angle-cluster cost).
    Minimizes the maximum absolute cosine similarity between any pair of columns.
    This encourages equiangular / well-spread columns in R^k → smaller m-height.
    (m parameter kept for API compatibility but not used; works great for all m)."""
    norms = np.linalg.norm(G, axis=0) + 1e-12
    cos_matrix = np.abs(G.T @ G) / np.outer(norms, norms)
    np.fill_diagonal(cos_matrix, 0)
    return np.max(cos_matrix)  # smaller = better spread

# =============================================================================
# LOCAL ±1 GREEDY IMPROVEMENT (now uses the new proxy)
# =============================================================================
def local_improve_angle(P: np.ndarray, m: int) -> np.ndarray:
    P = P.copy().astype(int)
    k, p_dim = P.shape
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
                        break
                if improved:
                    break
            if improved:
                break
    return P

# =============================================================================
# GLOBAL CONTINUOUS OPTIMIZATION (DIFFERENTIAL EVOLUTION)
# This replaces pure random sampling with a strong global optimizer.
# Much better starting vectors than Gaussian random (addresses your note).
# =============================================================================
def continuous_optimize(k: int, p_dim: int, m: int, target_scale: float = 15.0,
                        popsize: int = 12, maxiter: int = 12) -> np.ndarray:
    """Global optimization of the continuous direction matrix using differential evolution.
    Optimizes the max-cosine proxy → finds excellent starting points before discretization."""
    def objective(flat_P):
        P = flat_P.reshape((k, p_dim))
        G = np.hstack([np.eye(k), P])
        return angle_based_cost(G, m)
    bounds = [(-50, 50) for _ in range(k * p_dim)]
    result = differential_evolution(objective, bounds, popsize=popsize, maxiter=maxiter,
                                    tol=1e-4, workers=1, seed=None)
    return result.x.reshape((k, p_dim))

# =============================================================================
# IMPROVED SEEDS (Hadamard / simplex-like for good initial spread)
# =============================================================================
SEEDS: Dict[Tuple[int, int, int], np.ndarray] = {
    (9, 4, 2): np.array([[1, 1, 1, 1, 1],
                         [1, 1, -1, -1, 1],
                         [1, -1, 1, -1, 1],
                         [1, -1, -1, 1, 1]], dtype=float),
    (9, 4, 3): np.array([[1, 1, 1, 1, 1],
                         [1, 1, -1, -1, 1],
                         [1, -1, 1, -1, 1],
                         [1, -1, -1, 1, 1]], dtype=float),
    (9, 4, 4): np.array([[1, 1, 1, 1, 1],
                         [1, 1, -1, -1, 1],
                         [1, -1, 1, -1, 1],
                         [1, -1, -1, 1, 1]], dtype=float),
    (9, 4, 5): np.array([[1, 1, 1, 1, 1],
                         [1, 1, -1, -1, 1],
                         [1, -1, 1, -1, 1],
                         [1, -1, -1, 1, 1]], dtype=float),
    (9, 5, 2): np.array([[1, 1, 1, 1],
                         [1, 1, -1, -1],
                         [1, -1, 1, -1],
                         [1, -1, -1, 1],
                         [1, 1, 1, -1]], dtype=float),
    (9, 5, 3): np.array([[1, 1, 1, 1],
                         [1, 1, -1, -1],
                         [1, -1, 1, -1],
                         [1, -1, -1, 1],
                         [1, 1, 1, -1]], dtype=float),
    (9, 5, 4): np.array([[1, 1, 1, 1],
                         [1, 1, -1, -1],
                         [1, -1, 1, -1],
                         [1, -1, -1, 1],
                         [1, 1, 1, -1]], dtype=float),
    (9, 6, 2): np.array([[1, 1, 1],
                         [1, 1, -1],
                         [1, -1, 1],
                         [1, -1, -1],
                         [1, 1, 1],
                         [-1, 1, 1]], dtype=float),
    (9, 6, 3): np.array([[1, 1, 1],
                         [1, 1, -1],
                         [1, -1, 1],
                         [1, -1, -1],
                         [1, 1, 1],
                         [-1, 1, 1]], dtype=float),
}

# =============================================================================
# PARAMETERS + MAIN SEARCH LOOP (focuses only on project parameters)
# =============================================================================
PARAMS = [
    (9, 4, 2), (9, 4, 3), (9, 4, 4), (9, 4, 5),
    (9, 5, 2), (9, 5, 3), (9, 5, 4),
    (9, 6, 2), (9, 6, 3),
]

def main():
    best_generators: Dict[Tuple[int, int, int], np.ndarray] = {}
    best_mheights: Dict[Tuple[int, int, int], float] = {}

    TARGET_SCALE = 15.0
    NUM_OPT_RUNS = 50      # more runs = better starting vectors
    MAXITER = 12
    POPS = 15

    print(f"Starting search with DIFFERENTIAL EVOLUTION (global optimizer) + discretization + local improvement\n"
          f"Target scale = {TARGET_SCALE} | {NUM_OPT_RUNS} DE runs per parameter set\n")

    for n, k, m in PARAMS:
        p_dim = n - k
        print(f"\n{'='*80}\nProcessing (n={n}, k={k}, m={m}) → P is {k}×{p_dim}\n{'='*80}")

        best_cost = float('inf')
        best_h = float('inf')
        best_P = None

        # 1. Try seed (if provided)
        if (n, k, m) in SEEDS:
            seed_P_raw = SEEDS[(n, k, m)].copy()
            seed_P = normalize_and_approximate_P(seed_P_raw, TARGET_SCALE)
            G_seed = np.hstack([np.eye(k), seed_P.astype(float)])
            seed_cost = angle_based_cost(G_seed, m)
            try:
                seed_h = m_height(G_seed, m)
                if seed_h < best_h:
                    best_h = seed_h
                    best_cost = seed_cost
                    best_P = seed_P.copy()
                    print(f"  Seed → max_cos={seed_cost:.4f} | h_m={seed_h:.4f}")
            except:
                pass

        # 2. Global continuous optimization (DE) - the key improvement over pure random
        for run in range(NUM_OPT_RUNS):
            print(f"  DE run {run+1}/{NUM_OPT_RUNS}...", end=" ")
            P_cont = continuous_optimize(k, p_dim, m, TARGET_SCALE, popsize=POPS, maxiter=MAXITER)
            P_disc = normalize_and_approximate_P(P_cont, TARGET_SCALE)
            G = np.hstack([np.eye(k), P_disc.astype(float)])
            cost = angle_based_cost(G, m)
            try:
                h = m_height(G, m)
                print(f"max_cos={cost:.4f} | h_m={h:.4f}")
                if h < best_h:
                    best_h = h
                    best_cost = cost
                    best_P = P_disc.copy()
            except Exception:
                print(f"max_cos={cost:.4f} | h_m=inf (skipped)")

        # 3. Local discrete ±1 refinement on the best found
        if best_P is not None:
            print(f"  Local ±1 improvement on best (max_cos={best_cost:.4f}, h_m={best_h:.4f})...")
            best_P = local_improve_angle(best_P, m)
            G_final = np.hstack([np.eye(k), best_P.astype(float)])
            final_cost = angle_based_cost(G_final, m)
            final_h = m_height(G_final, m)
            print(f"  FINAL → max_cos={final_cost:.4f} | h_m={final_h:.4f}")
            best_h = final_h

        best_generators[(n, k, m)] = best_P
        best_mheights[(n, k, m)] = float(best_h)

    # Save exactly the required submission files
    with open("generatorMatrix", "wb") as f:
        pickle.dump(best_generators, f)
    with open("mHeight", "wb") as f:
        pickle.dump(best_mheights, f)

    print("\n" + "="*80)
    print("ALL DONE! Submission files created:")
    print("   • generatorMatrix")
    print("   • mHeight")
    print("="*80)
    print("Best m-heights found:")
    for key in sorted(best_mheights.keys()):
        print(f"   {key} → h_m = {best_mheights[key]:.6f}")

if __name__ == "__main__":
    main()