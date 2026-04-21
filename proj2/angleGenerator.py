import pickle
import random
import numpy as np
from itertools import combinations
from scipy.optimize import linprog
from typing import Tuple, Dict, Optional

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
# NORMALIZE + SCALE TO CLOSEST SMALL INTEGER APPROXIMATION
# (applied to EVERY candidate P before any testing/evaluation)
# =============================================================================
def normalize_and_approximate_P(P_in: np.ndarray, target_scale: float = 10.0) -> np.ndarray:
    """
    BEFORE TESTING ANY P:
      - Normalize each column of P to unit length.
      - Scale by target_scale.
      - Round to nearest integer.
      - Clip to [-100, 100].
      - Guarantee no all-zero column (retry random direction if needed).
    This turns any P (random or user-seeded) into its "closest small-integer
    direction approximation".
    """
    P = P_in.astype(float).copy()
    k, p_dim = P.shape

    for j in range(p_dim):
        col = P[:, j]
        norm = np.linalg.norm(col)
        if norm < 1e-12:  # degenerate column
            # replace with random non-zero direction
            col = np.random.randn(k)
            norm = np.linalg.norm(col)
        col_norm = col / norm
        col_scaled = col_norm * target_scale
        P[:, j] = np.round(col_scaled)

    P = np.clip(P, -100, 100).astype(int)

    # Final safety: no zero columns
    for j in range(p_dim):
        while np.all(P[:, j] == 0):
            col = np.random.randn(k)
            col_norm = col / np.linalg.norm(col)
            col_scaled = col_norm * target_scale
            P[:, j] = np.round(np.clip(col_scaled, -100, 100)).astype(int)
    return P


# =============================================================================
# RANDOM P GENERATOR (now uses normalization + integer approximation)
# =============================================================================
def generate_random_P(k: int, p_dim: int, target_scale: float = 10.0) -> np.ndarray:
    """Generate a fresh random direction matrix, then immediately normalize+approximate."""
    while True:
        # Gaussian random directions (better isotropic sampling than uniform)
        P_float = np.random.randn(k, p_dim)
        P = normalize_and_approximate_P(P_float, target_scale)
        if np.all(np.any(P != 0, axis=0)):
            return P


# =============================================================================
# ANGLE-BASED COST (purely directional, invariant to column scaling)
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
        closest_rel = np.argsort(ang_to_others)[:m - 1] if m > 1 else []
        other_indices = np.where(other_mask)[0]
        subset_idx = [i] + list(other_indices[closest_rel])

        sub_angles = ang_matrix[np.ix_(subset_idx, subset_idx)]
        max_in_subset = np.max(sub_angles)
        if max_in_subset > worst_angle:
            worst_angle = max_in_subset
    return worst_angle


# =============================================================================
# LOCAL IMPROVEMENT (±1 greedy on angle cost)
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
                        print(f"    Angle-improve: P[{i},{j}] += {delta} → cost = {current_cost:.4f}")
                        break
                if improved:
                    break
            if improved:
                break
    return P


# =============================================================================
# SEED DICTIONARY (YOUR CUSTOM STARTING MATRICES GO HERE)
# =============================================================================
# Format: key = (n, k, m), value = k × (n-k) integer NumPy array
# These will be normalized+approximated BEFORE testing (as requested).
SEEDS: Dict[Tuple[int, int, int], np.ndarray] = {
    # === ADD YOUR OWN SEEDS HERE (examples only) ===
    # (9, 4, 2): np.array([[1, 2, 0, 3, 4],
    #                      [0, 1, -2, 5, -1],
    #                      [3, -1, 4, 0, 2],
    #                      [2, 0, 1, -3, 5]], dtype=int),
    #
    # (9, 5, 3): np.array([[...], ...]),
    # ...
    # If a key is missing, we skip seeding for that parameter set.
    (9, 4, 2): np.array([
                [1, 0, 0, 0, 1],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
            ], dtype=float),
    (9, 4, 3): np.array([
                [1, 1, 1, 1,-1],
                [1, 1, 1,-1, 1],
                [1, 1,-1, 1, 1],
                [1,-1, 1, 1, 1],
            ], dtype=float),
    (9, 4, 4): np.array([
                [1, 1, 1, 1,-1],
                [1, 1, 1,-1, 1],
                [1, 1,-1, 1, 1],
                [1,-1, 1, 1, 1],
            ], dtype=float),
}


# =============================================================================
# PARAMETERS & MAIN LOOP
# =============================================================================
PARAMS = [
    (9, 4, 2), (9, 4, 3), (9, 4, 4), (9, 4, 5),
    (9, 5, 2), (9, 5, 3), (9, 5, 4),
    (9, 6, 2), (9, 6, 3),
]

TRIALS_PER_PARAM = 10000          # increase for better quality
TARGET_SCALE = 10.0               # scale used for normalization+rounding


def main():
    best_generators: Dict[Tuple[int, int, int], np.ndarray] = {}
    best_mheights: Dict[Tuple[int, int, int], float] = {}

    print(f"Starting search for {len(PARAMS)} parameter sets "
          f"({TRIALS_PER_PARAM} random trials each + local improvement)\n"
          f"Normalization + small-integer approximation (scale={TARGET_SCALE}) "
          f"applied to EVERY P before testing.\n")

    for n, k, m in PARAMS:
        p_dim = n - k
        print(f"\n{'='*80}")
        print(f"Processing (n={n}, k={k}, m={m})  →  P is {k}×{p_dim}")
        print(f"{'='*80}")

        best_cost = float('inf')
        best_P: Optional[np.ndarray] = None
        best_h: Optional[float] = None

        # 1. Try user-seeded matrix (if provided)
        if (n, k, m) in SEEDS:
            seed_P_raw = SEEDS[(n, k, m)].copy()
            # === APPLY NORMALIZE + APPROXIMATE BEFORE TESTING ===
            seed_P = normalize_and_approximate_P(seed_P_raw, TARGET_SCALE)
            G_seed = np.hstack([np.eye(k), seed_P.astype(float)])
            seed_cost = angle_based_cost(G_seed, m)

            print(f"  Using SEEDED P (after normalization/approx) → angle-cost = {seed_cost:.4f}")
            best_P = seed_P
            best_cost = seed_cost
            try:
                best_h = m_height(G_seed, m)
                print(f"    Exact m-height of seed = {best_h:.6f}")
            except Exception:
                best_h = None

        # 2. Random search phase (every random P also normalized+approximated)
        for trial in range(TRIALS_PER_PARAM):
            P_raw = generate_random_P(k, p_dim, TARGET_SCALE)   # already normalized
            G = np.hstack([np.eye(k), P_raw.astype(float)])
            cost = angle_based_cost(G, m)

            if cost < best_cost:
                best_cost = cost
                best_P = P_raw.copy()

                try:
                    h = m_height(G, m)
                    best_h = h
                except Exception:
                    best_h = None

                print(f"  Trial {trial+1:4d}/{TRIALS_PER_PARAM} | "
                      f"angle-cost = {cost:.4f} rad | m-height = {best_h:.6f}")

        # 3. Local improvement on the overall best so far
        if best_P is not None:
            print(f"\n  Running local ±1 improvement on best P...")
            best_P = local_improve_angle(best_P, m)
            G_final = np.hstack([np.eye(k), best_P.astype(float)])
            best_cost = angle_based_cost(G_final, m)
            best_h = m_height(G_final, m)
            print(f"  Final after local improve → angle-cost = {best_cost:.4f} rad | "
                  f"m-height = {best_h:.6f}")

        # Store results (P is already integer, normalized, no zero columns)
        best_generators[(n, k, m)] = best_P
        best_mheights[(n, k, m)] = float(best_h)

    # Save exactly the files required by the project spec
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