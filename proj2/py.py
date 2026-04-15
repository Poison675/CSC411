# =============================================================================
# CSCE-411 Project: Continuous Parallel Random + Hill-Climbing Search
# =============================================================================
# CHANGES FROM PREVIOUS VERSION:
#   • Continuous outer loop: cycles through ALL 9 (n,k,m) forever.
#   • Every batch now mixes RANDOM generation + HILL-CLIMBING (perturbations
#     from the current best P for that parameter set).
#   • Ctrl+C now safely interrupts the ENTIRE program at any time and saves
#     the latest best generator matrices + m-heights before exiting.
#   • Immediate pickle updates only on strict improvement (as requested).
#   • Pool is created once and reused for maximum efficiency.
#
# Usage: python search_optimal_generator.py
# (It will keep improving forever. Press Ctrl+C to stop gracefully.)
# =============================================================================

import os
import pickle
import numpy as np
from scipy.optimize import linprog
from itertools import combinations
from typing import Tuple, List, Optional
from multiprocessing import Pool, cpu_count
import sys

# =============================================================================
# === FAST SINGLE-PROCESS m-HEIGHT (with numerical fix) ===
# =============================================================================
def compute_z_S_j(G: np.ndarray, S: Tuple[int, ...], j: int) -> float:
    """Solve one LP: maximize G[:, j] @ u  s.t. -1 <= G[:, t] @ u <= 1 for t in bar{S}"""
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
        print(f"⚠️  LP warning for S={S}, j={j}: {res.message} → returning 1.0")
        return 1.0


def m_height(G: np.ndarray, m: int) -> float:
    """Compute m-height h_m(C) – single-process, fully optimized."""
    k, n = G.shape
    if not (0 <= m < n):
        raise ValueError(f"m must be in [0, {n-1}], got {m}")

    if m == 0:
        return 1.0

    subsets = list(combinations(range(n), m))
    max_h = 1.0

    for S_tup in subsets:
        S = tuple(S_tup)
        for j in S:
            z = compute_z_S_j(G, S, j)
            if z > max_h:
                max_h = z

    return max_h


# =============================================================================
# Worker: random OR hill-climb from current best
# =============================================================================
def worker(args):
    """Generate/perturb one candidate and compute its m-height."""
    n, k, m, seed, base_P = args
    np.random.seed(seed)

    if base_P is None:
        # Pure random candidate
        P = np.random.randint(-100, 101, size=(k, n - k))
    else:
        # Hill-climbing: perturb current best
        P = base_P.copy()
        num_perturb = np.random.randint(1, 5)          # 1–4 random entry changes
        for _ in range(num_perturb):
            r = np.random.randint(0, k)
            c = np.random.randint(0, n - k)
            delta = np.random.randint(-20, 21)         # reasonably large steps
            P[r, c] = np.clip(P[r, c] + delta, -100, 100)

    # Guarantee no all-zero column
    zero_cols = np.all(P == 0, axis=0)
    while np.any(zero_cols):
        P[:, zero_cols] = np.random.randint(-100, 101, size=(k, np.sum(zero_cols)))
        zero_cols = np.all(P == 0, axis=0)

    # Build G = [I_k | P]
    G = np.hstack((np.eye(k), P.astype(np.float64)))

    h = m_height(G, m)
    return P, h


# =============================================================================
# Main continuous search with safe Ctrl+C exit
# =============================================================================
if __name__ == "__main__":
    gen_file = "generatorMatrixTemp"
    mh_file = "mHeightTemp"

    # Load previous best results (or start fresh)
    if os.path.exists(gen_file):
        with open(gen_file, "rb") as f:
            generator_dict = pickle.load(f)
        print(f"✅ Loaded existing generatorMatrix ({len(generator_dict)} entries)")
    else:
        generator_dict = {}
        print("ℹ️  No generatorMatrix found → starting fresh")

    if os.path.exists(mh_file):
        with open(mh_file, "rb") as f:
            mHeight_dict = pickle.load(f)
        print(f"✅ Loaded existing mHeight ({len(mHeight_dict)} entries)")
    else:
        mHeight_dict = {}
        print("ℹ️  No mHeight found → starting fresh")

    # All required parameter sets
    params = [
        (9, 4, 2), (9, 4, 3), (9, 4, 4), (9, 4, 5),
        (9, 5, 2), (9, 5, 3), (9, 5, 4),
        (9, 6, 2), (9, 6, 3),
    ]

    num_workers = cpu_count()
    batch_size = num_workers * 8          # high throughput
    print(f"🚀 Starting continuous parallel search with {num_workers} cores")
    print(f"   Batch size = {batch_size} candidates per parameter set (random + hill-climbing)")

    trial_counter = 0

    with Pool(processes=num_workers) as pool:
        try:
            while True:                                      # ← continuous forever
                for n, k, m in params:
                    key = (n, k, m)
                    current_best_h = mHeight_dict.get(key, float("inf"))
                    current_best_P = generator_dict.get(key)

                    print(f"\n🔄 Batch {trial_counter:,} | {n},{k},{m} | best so far = {current_best_h:.8f}")

                    # Decide how many hill-climb vs random (bias toward hill-climb once we have a good matrix)
                    hill_prob = 0.65 if current_best_P is not None else 0.0

                    args_list = []
                    for i in range(batch_size):
                        seed = trial_counter * batch_size + i
                        if np.random.rand() < hill_prob:
                            base_P = current_best_P.copy()
                        else:
                            base_P = None
                        args_list.append((n, k, m, seed, base_P))

                    # PARALLEL evaluation
                    results = pool.map(worker, args_list)

                    improved = False
                    for P_cand, h_cand in results:
                        if 1.0 <= h_cand < current_best_h:   # strictly better
                            current_best_h = h_cand
                            current_best_P = P_cand.copy()
                            generator_dict[key] = current_best_P
                            mHeight_dict[key] = current_best_h

                            # Immediate save
                            with open(gen_file, "wb") as f:
                                pickle.dump(generator_dict, f)
                            with open(mh_file, "wb") as f:
                                pickle.dump(mHeight_dict, f)

                            print(f"🎉 NEW BEST for {key} → h_m = {current_best_h:.8f}")
                            improved = True

                    if improved:
                        print(f"   → Pickles updated with improved matrix for {key}")

                    trial_counter += 1

        except KeyboardInterrupt:
            print("\n\n🛑 Ctrl+C received – saving current best results before exit...")
            # Final save (in case the very last batch had no improvement)
            with open(gen_file, "wb") as f:
                pickle.dump(generator_dict, f)
            with open(mh_file, "wb") as f:
                pickle.dump(mHeight_dict, f)
            print("✅ All best generator matrices and m-heights saved.")
            print("   You can restart the script anytime – it will resume from these bests.")
            sys.exit(0)

    # (This line is unreachable because of the infinite loop + Ctrl+C handler)