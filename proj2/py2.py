import os
import pickle
import random
from itertools import combinations
from typing import Dict, Tuple

import numpy as np
from scipy.optimize import linprog
from concurrent.futures import ProcessPoolExecutor, as_completed

# ============================================================
# === m-HEIGHT CORE (PARALLEL LP VERSION) ====================
# ============================================================

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


# ============================================================
# === IDENTITY-ROWS CANDIDATE ================================
# ============================================================

def create_identity_rows_P(k: int, n_minus_k: int) -> np.ndarray:
    P = np.zeros((k, n_minus_k), dtype=int)
    for col in range(n_minus_k):
        row = col % k
        P[row, col] = 1
    return P


# ============================================================
# === PARALLEL NEIGHBOR EVALUATOR ============================
# ============================================================

def evaluate_neighbor(args: tuple) -> float:
    P_new, n, k, m = args
    G_new = build_systematic_G(k, n, P_new)
    return m_height(G_new, m)


# ============================================================
# === LOCAL IMPROVEMENT ======================================
# ============================================================

def local_improve(P: np.ndarray, n: int, k: int, m: int) -> Tuple[np.ndarray, float]:
    """
    1. First check the "matrix of all identity rows" candidate.
    2. Then run greedy ±1 local search (parallel neighbors).
    """
    P = P.copy().astype(int)
    current_h = m_height(build_systematic_G(k, n, P), m)

    # Quick identity-rows candidate
    P_id = create_identity_rows_P(k, n - k)
    h_id = m_height(build_systematic_G(k, n, P_id), m)
    if h_id < current_h:
        P = P_id
        current_h = h_id
        print(f"  Identity-rows candidate beats current: h_m = {h_id:.6g}")

    # Parallel greedy ±1 local search
    iteration = 0
    while True:
        iteration += 1
        neighbors = []
        for i in range(k):
            for j in range(n - k):
                for delta in [-3, -2, -1, 1, 2, 3]:
                    P_new = P.copy()
                    P_new[i, j] += delta
                    if np.all(P_new[:, j] == 0):
                        continue
                    neighbors.append((P_new, n, k, m))

        if not neighbors:
            break

        num_workers = min(len(neighbors), os.cpu_count() or 4)
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            h_list = list(executor.map(evaluate_neighbor, neighbors))

        best_h_new = current_h
        best_P_new = None
        for idx, h_new in enumerate(h_list):
            if h_new < best_h_new:
                best_h_new = h_new
                best_P_new = neighbors[idx][0]

        if best_P_new is None:
            break

        P = best_P_new
        current_h = best_h_new
        print(f"  Local improve iter {iteration} → h_m = {current_h:.6g}")

    return P, current_h


# ============================================================
# === HELPERS ================================================
# ============================================================

def random_P(k: int, n_minus_k: int, low: int = -5, high: int = 5) -> np.ndarray:
    while True:
        P = np.random.randint(low, high + 1, size=(k, n_minus_k))
        if np.all(np.any(P != 0, axis=0)):
            return P


def build_systematic_G(k: int, n: int, P: np.ndarray) -> np.ndarray:
    I = np.eye(k, dtype=float)
    return np.concatenate([I, P.astype(float)], axis=1)


# ============================================================
# === WORKER (ALWAYS RETURNS VALID P) ========================
# ============================================================

def worker_task(param: Tuple[int, int, int], num_trials: int, seed: int = None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    n, k, m = param
    best_h = float("inf")
    best_P = None
    fallback_P = None

    for i in range(num_trials):
        P = random_P(k, n - k)
        G = build_systematic_G(k, n, P)
        h = m_height(G, m)

        if h < best_h:
            best_h = h
            best_P = P

        # Safety: always keep at least one valid P
        if fallback_P is None:
            fallback_P = P

    # If no finite h was found, return the last valid P with inf
    if best_P is None:
        best_P = fallback_P
        best_h = float("inf")

    return param, best_P, best_h


# ============================================================
# === MAIN ===================================================
# ============================================================

GEN_PICKLE = "generatorMatrixe"
MH_PICKLE = "mHeighte"

PARAMS = [
    (9, 4, 2), (9, 4, 3), (9, 4, 4), (9, 4, 5),
    (9, 5, 2), (9, 5, 3), (9, 5, 4),
    (9, 6, 2), (9, 6, 3),
]


def load_state():
    best_generators = pickle.load(open(GEN_PICKLE, "rb")) if os.path.exists(GEN_PICKLE) else {}
    best_mheights = pickle.load(open(MH_PICKLE, "rb")) if os.path.exists(MH_PICKLE) else {}
    return best_generators, best_mheights


def save_state(best_generators, best_mheights):
    with open(GEN_PICKLE, "wb") as f:
        pickle.dump(best_generators, f)
    with open(MH_PICKLE, "wb") as f:
        pickle.dump(best_mheights, f)


def main(
    total_trials_per_param: int = 1000,
    workers: int = None,
    batch_size: int = 50,
):
    best_generators, best_mheights = load_state()
    if workers is None:
        workers = os.cpu_count() or 4

    print(f"Using {workers} worker processes (parallel m_height enabled).")

    # === TARGETED FIX PHASE (m_height > 10) ===
    bad_params = [p for p in PARAMS if best_mheights.get(p, float("inf")) > 10]

    if bad_params:
        print(f"\n=== TARGETED FIX PHASE ===\n"
              f"Found {len(bad_params)} matrices with m_height > 10.\n"
              f"After each random batch we run local_improve on the best matrix of that batch.\n")

        with ProcessPoolExecutor(max_workers=workers) as executor:
            for param in bad_params:
                n, k, m = param
                current_h = best_mheights[param]
                print(f"→ Targeting {param} (current h_m = {current_h:.6g}) ...")

                improved = False
                batch_num = 0
                while not improved:
                    batch_num += 1

                    fut = executor.submit(worker_task, param, num_trials=batch_size,
                                          seed=random.randint(0, 2**31 - 1))
                    _, P_candidate, h_raw = fut.result()

                    print(f"  Batch {batch_num} raw best h = {h_raw:.6g}")
                    if h_raw != np.inf:
                        print("       → running local_improve...")
                        P_refined, h_refined = local_improve(P_candidate, n, k, m)
                    else:
                        P_refined, h_refined = P_candidate, h_raw

                    if h_refined < current_h:
                        print(f"  ✓ IMPROVED {param} at batch {batch_num}: "
                              f"{current_h:.6g} → {h_refined:.6g} (after local minimum)")
                        best_mheights[param] = float(h_refined)
                        best_generators[param] = P_refined
                        save_state(best_generators, best_mheights)
                        improved = True
                    else:
                        print(f"  Batch {batch_num} refined h = {h_refined:.6g} (no improvement yet)")

    else:
        print("\nNo stored matrices have m_height > 10.")

    # === FINAL POLISH ON EVERYTHING ===
    print("\n=== Final local improvement phase on all parameters ===")
    for param in PARAMS:
        if param not in best_generators:
            continue
        n, k, m = param
        print(f"Local improvement for {param} (current h_m = {best_mheights[param]:.6g}) ...")
        P_improved, h_improved = local_improve(best_generators[param], n, k, m)
        if h_improved < best_mheights[param]:
            print(f"  Improved {param}: {best_mheights[param]:.6g} → {h_improved:.6g}")
            best_generators[param] = P_improved
            best_mheights[param] = h_improved
            save_state(best_generators, best_mheights)
        else:
            print(f"  No further improvement for {param}")

    print("\n=== DONE ===")
    print("Final best m-heights:")
    for param in sorted(best_mheights.keys()):
        print(f"  {param}: h_m = {best_mheights[param]:.6g}")


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    main(
        total_trials_per_param=1000,
        workers=None,
        batch_size=50,
    )