import os
import pickle
import random
from itertools import combinations
from typing import Dict, Tuple, List

import numpy as np
from scipy.optimize import linprog
from concurrent.futures import ProcessPoolExecutor, as_completed

# ============================================================
# === m-HEIGHT CORE ==========================================
# ============================================================

def compute_z_S_j(G: np.ndarray, S: Tuple[int, ...], j: int) -> float:
    """Solve one LP: maximize G[:, j] @ u  s.t. -1 <= G[:, t] @ u <= 1 for t in bar{S}."""
    k, n = G.shape
    bar_S = [t for t in range(n) if t not in S]

    c = -G[:, j].astype(np.float64)

    A_ub_list: List[np.ndarray] = []
    b_ub_list: List[float] = []
    for t in bar_S:
        gt = G[:, t].astype(np.float64)
        A_ub_list.append(gt)
        b_ub_list.append(1.0)
        A_ub_list.append(-gt)
        b_ub_list.append(1.0)

    res = linprog(
        c=c,
        A_ub=np.array(A_ub_list),
        b_ub=np.array(b_ub_list),
        bounds=[(-1e6, 1e6)] * k,
        method='highs',
        options={'presolve': True, 'disp': False}
    )

    if res.success:
        return float(-res.fun)
    elif res.message and "unbounded" in res.message.lower():
        return float('inf')
    else:
        print(f"  LP warning for S={S}, j={j}: {res.message} → returning 1.0")
        return 1.0


def m_height(G: np.ndarray, m: int) -> float:
    """Compute m-height h_m(C)."""
    k, n = G.shape
    if not (0 <= m < n):
        raise ValueError(f"m must be in [0, {n-1}], got {m}")

    if m == 0:
        return 1.0

    subsets = combinations(range(n), m)
    max_h = 1.0

    for S in subsets:
        S = tuple(S)
        for j in S:
            z = compute_z_S_j(G, S, j)
            if z > max_h:
                max_h = z

    return max_h


# ============================================================
# === NEW HELPER: IDENTITY-ROWS CANDIDATE ====================
# ============================================================

def create_identity_rows_P(k: int, n_minus_k: int) -> np.ndarray:
    """
    Create the "matrix of all identity rows" candidate:
    Each column j gets a single 1 in row (j % k). Guarantees no all-zero columns.
    """
    P = np.zeros((k, n_minus_k), dtype=int)
    for col in range(n_minus_k):
        row = col % k
        P[row, col] = 1
    return P


# ============================================================
# === NEW HELPER: PARALLEL NEIGHBOR EVALUATOR ================
# ============================================================

def evaluate_neighbor(args: tuple) -> float:
    """Top-level worker for parallel map (must be picklable)."""
    P_new, n, k, m = args
    G_new = build_systematic_G(k, n, P_new)
    return m_height(G_new, m)


# ============================================================
# === LOCAL IMPROVEMENT (with identity check + parallel) =====
# ============================================================

def local_improve(P: np.ndarray, n: int, k: int, m: int) -> Tuple[np.ndarray, float]:
    """
    1. First check the "matrix of all identity rows" candidate.
       If it beats the current m-height, replace immediately.
    2. Then run greedy ±1 local search, with ALL possible neighbors
       evaluated in parallel each iteration (maximally parallel).
    Continues until no improving single flip exists.
    """
    P = P.copy().astype(int)
    current_h = m_height(build_systematic_G(k, n, P), m)

    # === Step 1: Quick identity-rows candidate check ===
    P_id = create_identity_rows_P(k, n - k)
    h_id = m_height(build_systematic_G(k, n, P_id), m)
    if h_id < current_h:
        P = P_id
        current_h = h_id
        print(f"  Identity-rows candidate beats current: h_m = {h_id:.6g}")

    # === Step 2: Parallel greedy ±1 local search ===
    iteration = 0
    while True:
        iteration += 1
        # Generate all valid single-flip neighbors
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

        # Parallel evaluation of ALL neighbors
        num_workers = min(len(neighbors), os.cpu_count() or 4)
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            h_list = list(executor.map(evaluate_neighbor, neighbors))

        # Find the SINGLE best improving neighbor (greedy)
        best_h_new = current_h
        best_P_new = None
        for idx, h_new in enumerate(h_list):
            if h_new < best_h_new:
                best_h_new = h_new
                best_P_new = neighbors[idx][0]

        if best_P_new is None:
            break  # no improvement possible

        # Accept the best flip and continue
        P = best_P_new
        current_h = best_h_new
        print(f"  Local improve iter {iteration} → h_m = {current_h:.6g} (best neighbor)")

    return P, current_h


# ============================================================
# === SEARCH SPACE & PARALLEL DRIVER =========================
# ============================================================

GEN_PICKLE = "generatorMatrixMerge"
MH_PICKLE = "mHeightMerge"

PARAMS = [
    (9, 4, 2), (9, 4, 3), (9, 4, 4), (9, 4, 5),
    (9, 5, 2), (9, 5, 3), (9, 5, 4),
    (9, 6, 2), (9, 6, 3),
]


def load_state():
    if os.path.exists(GEN_PICKLE):
        with open(GEN_PICKLE, "rb") as f:
            best_generators = pickle.load(f)
    else:
        best_generators = {}

    if os.path.exists(MH_PICKLE):
        with open(MH_PICKLE, "rb") as f:
            best_mheights = pickle.load(f)
    else:
        best_mheights = {}

    return best_generators, best_mheights


def save_state(best_generators, best_mheights):
    with open(GEN_PICKLE, "wb") as f:
        pickle.dump(best_generators, f)
    with open(MH_PICKLE, "wb") as f:
        pickle.dump(best_mheights, f)


def random_P(k: int, n_minus_k: int, low: int = -5, high: int = 5) -> np.ndarray:
    while True:
        P = np.random.randint(low, high + 1, size=(k, n_minus_k))
        if np.all(np.any(P != 0, axis=0)):
            return P


def build_systematic_G(k: int, n: int, P: np.ndarray) -> np.ndarray:
    I = np.eye(k, dtype=float)
    return np.concatenate([I, P.astype(float)], axis=1)


# ============================================================
# === WORKER FOR RANDOM SEARCH ===============================
# ============================================================

def worker_task(param: Tuple[int, int, int], num_trials: int, seed: int = None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    n, k, m = param
    best_h = float("inf")
    best_P = None

    for _ in range(num_trials):
        P = random_P(k, n - k)
        G = build_systematic_G(k, n, P)
        h = m_height(G, m)
        if h < best_h:
            best_h = h
            best_P = P

    return param, best_P, best_h


# ============================================================
# === MAIN ===================================================
# ============================================================

def main(
    total_trials_per_param: int = 200,
    workers: int = None,
    batch_size: int = 10,
):
    best_generators, best_mheights = load_state()

    if workers is None:
        workers = os.cpu_count() or 4

    print(f"Using {workers} worker processes for random search.")

    # === RANDOM SEARCH PHASE (already parallel) ===
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = []
        for param in PARAMS:
            n, k, m = param
            remaining = total_trials_per_param
            while remaining > 0:
                this_batch = min(batch_size, remaining)
                remaining -= this_batch
                seed = random.randint(0, 2**31 - 1)
                fut = executor.submit(worker_task, param, this_batch, seed)
                futures.append(fut)

        for fut in as_completed(futures):
            param, P_candidate, h_candidate = fut.result()
            if P_candidate is None:
                continue
            current_best = best_mheights.get(param, float("inf"))
            if h_candidate < current_best:
                print(f"Improved (random) for {param}: {current_best:.6g} → {h_candidate:.6g}")
                best_mheights[param] = float(h_candidate)
                best_generators[param] = P_candidate
                save_state(best_generators, best_mheights)

    # === LOCAL IMPROVEMENT PHASE (parallel neighbors inside each param) ===
    print("\n=== Starting local improvement phase (identity check + parallel ±1) ===")
    for param in PARAMS:
        if param not in best_generators:
            continue
        n, k, m = param
        print(f"Local improvement for {param} (current h_m = {best_mheights[param]:.6g}) ...")
        P_improved, h_improved = local_improve(
            best_generators[param], n, k, m
        )
        if h_improved < best_mheights[param]:
            print(f"  Improved {param}: {best_mheights[param]:.6g} → {h_improved:.6g}")
            best_generators[param] = P_improved
            best_mheights[param] = h_improved
            save_state(best_generators, best_mheights)
        else:
            print(f"  No further improvement for {param}")

    print("Search + local improvement finished.")
    print("Best m-heights found:")
    for param in sorted(best_mheights.keys()):
        print(f"  {param}: h_m = {best_mheights[param]:.6g}")


if __name__ == "__main__":
    main(
        total_trials_per_param=10000,
        workers=None,
        batch_size=10,
    )