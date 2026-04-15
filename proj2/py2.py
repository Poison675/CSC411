import os
import pickle
import random
from itertools import combinations
from typing import Dict, Tuple, List

import numpy as np
from scipy.optimize import linprog
from concurrent.futures import ProcessPoolExecutor, as_completed

# ============================================================
# === m-HEIGHT CORE (your given code, lightly cleaned) =======
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
        # Very rare fallback
        print(f"  LP warning for S={S}, j={j}: {res.message} → returning 1.0")
        return 1.0


def m_height(G: np.ndarray, m: int) -> float:
    """Compute m-height h_m(C) – single-process, fully optimized."""
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
# === NEW LOCAL IMPROVEMENT FUNCTION (added as requested) ====
# ============================================================

def local_improve(P: np.ndarray, n: int, k: int, m: int) -> Tuple[np.ndarray, float]:
    """
    Greedy local search: repeatedly try ±1 on any entry of P.
    If the change decreases m-height (and keeps no all-zero columns),
    accept it and restart the search until no further improvement.
    Called after the random-search phase for each (n,k,m).
    """
    P = P.copy().astype(int)  # ensure integer matrix
    current_G = build_systematic_G(k, n, P)
    current_h = m_height(current_G, m)

    improved = True
    iteration = 0
    while improved:
        improved = False
        iteration += 1
        # Try every entry in P
        for i in range(k):           # row
            for j in range(n - k):   # column in P
                for delta in [-1, 1]:
                    P_new = P.copy()
                    P_new[i, j] += delta

                    # Reject if this creates an all-zero column
                    if np.all(P_new[:, j] == 0):
                        continue

                    G_new = build_systematic_G(k, n, P_new)
                    h_new = m_height(G_new, m)

                    if h_new < current_h:
                        P = P_new
                        current_h = h_new
                        improved = True
                        print(f"  Local improve [{i},{j}] += {delta} | iter {iteration} → h_m = {h_new:.6g}")
                        break  # found an improvement → restart full sweep
                if improved:
                    break
            if improved:
                break

    return P, current_h


# ============================================================
# === SEARCH SPACE & PARALLEL DRIVER =========================
# ============================================================

GEN_PICKLE = "generatorMatrixTempCp"
MH_PICKLE = "mHeightTempCp"

# The parameter set you care about
PARAMS = [
    (9, 4, 2),
    (9, 4, 3),
    (9, 4, 4),
    (9, 4, 5),
    (9, 5, 2),
    (9, 5, 3),
    (9, 5, 4),
    (9, 6, 2),
    (9, 6, 3),
]


def load_state():
    """Load existing best matrices and m-heights if present."""
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
    """Persist current best results."""
    with open(GEN_PICKLE, "wb") as f:
        pickle.dump(best_generators, f)
    with open(MH_PICKLE, "wb") as f:
        pickle.dump(best_mheights, f)


def random_P(k: int, n_minus_k: int, low: int = -1, high: int = 1) -> np.ndarray:
    """Generate a random integer P with no all-zero columns."""
    while True:
        P = np.random.randint(low, high + 1, size=(k, n_minus_k))
        # Ensure no all-zero column
        if np.all(np.any(P != 0, axis=0)):
            return P


def build_systematic_G(k: int, n: int, P: np.ndarray) -> np.ndarray:
    """G = [I_k | P]."""
    I = np.eye(k, dtype=float)
    return np.concatenate([I, P.astype(float)], axis=1)


# ============================================================
# === WORKER FUNCTION FOR PARALLEL SEARCH ====================
# ============================================================

def worker_task(param: Tuple[int, int, int], num_trials: int, seed: int = None):
    """
    For a given (n, k, m), run num_trials random matrices and
    return the best (P, h_m) found in this worker.
    """
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
# === MAIN PARALLEL LOOP =====================================
# ============================================================

def main(
    total_trials_per_param: int = 200,
    workers: int = None,
    batch_size: int = 10,
):
    """
    Parallel search for good generator matrices.
    """
    best_generators, best_mheights = load_state()

    if workers is None:
        workers = os.cpu_count() or 4

    print(f"Using {workers} worker processes.")

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = []

        for param in PARAMS:
            n, k, m = param
            remaining = total_trials_per_param

            while remaining > 0:
                this_batch = min(batch_size, remaining)
                remaining -= this_batch

                # Seed to make runs reproducible-ish but independent
                seed = random.randint(0, 2**31 - 1)
                fut = executor.submit(worker_task, param, this_batch, seed)
                futures.append(fut)

        for fut in as_completed(futures):
            param, P_candidate, h_candidate = fut.result()
            n, k, m = param

            # Skip if worker didn't find anything (shouldn't happen)
            if P_candidate is None:
                continue

            # Compare with existing best
            current_best = best_mheights.get(param, float("inf"))

            if h_candidate < current_best:
                print(f"Improved for {param}: {current_best:.6g} → {h_candidate:.6g}")
                best_mheights[param] = float(h_candidate)
                best_generators[param] = P_candidate
                save_state(best_generators, best_mheights)

    # === LOCAL IMPROVEMENT PHASE (called after each parameter's random search) ===
    print("\n=== Starting local improvement phase for all parameters ===")
    for param in PARAMS:
        if param not in best_generators:
            continue
        print(f"Local improvement for {param} (current h_m = {best_mheights[param]:.6g}) ...")
        P_improved, h_improved = local_improve(
            best_generators[param], param[0], param[1], param[2]
        )
        if h_improved < best_mheights[param]:
            print(f"  Improved {param}: {best_mheights[param]:.6g} → {h_improved:.6g}")
            best_generators[param] = P_improved
            best_mheights[param] = h_improved
            save_state(best_generators, best_mheights)
        else:
            print(f"  No further improvement for {param}")
    print("Local improvement finished.")

    print("Search + local improvement finished.")
    print("Best m-heights found:")
    for param in sorted(best_mheights.keys()):
        print(f"  {param}: h_m = {best_mheights[param]:.6g}")


if __name__ == "__main__":
    # Tune these as you like
    main(
        total_trials_per_param=300,  # total random matrices per (n,k,m)
        workers=None,                # default: all cores
        batch_size=10,               # trials per worker call
    )