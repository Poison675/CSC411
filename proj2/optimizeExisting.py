import numpy as np
import pickle
import os
from concurrent.futures import ProcessPoolExecutor
from scipy.optimize import linprog
from itertools import combinations
from typing import Tuple

# ============================================================
# === ALL PROVIDED FUNCTIONS (EXACTLY AS YOU GAVE THEM) =====
# ============================================================

def _solve_lp(args):
    """Solve one LPS,j linear program."""
    G, j, barS = args
    c = -G[:, j]
    barS_matrix = G[:, barS].T
    A_ub = np.vstack([barS_matrix, -barS_matrix])
    b_ub = np.ones(2 * len(barS))
    res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                  bounds=(None, None),
                  method='highs',
                  options={'presolve': True, 'disp': False})
    return -res.fun if res.success else np.inf


def m_height(G: np.ndarray, m: int) -> float:
    """Exact m-height using the LP formulation (parallelized)."""
    n = G.shape[1]
    if m == 0:
        return 1.0
    tasks = [(G, j, [t for t in range(n) if t not in S])
             for S in combinations(range(n), m) for j in S]
    with ProcessPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
        results = list(executor.map(_solve_lp, tasks))
    return max(max(results), 1.0)


def create_identity_rows_P(k: int, n_minus_k: int) -> np.ndarray:
    P = np.zeros((k, n_minus_k), dtype=int)
    for col in range(n_minus_k):
        row = col % k
        P[row, col] = 1
    return P


def evaluate_neighbor(args: tuple) -> float:
    P_new, n, k, m = args
    G_new = build_systematic_G(k, n, P_new)
    return m_height(G_new, m)


def local_improve(P: np.ndarray, n: int, k: int, m: int) -> Tuple[np.ndarray, float]:
    P = P.copy().astype(int)
    current_h = m_height(build_systematic_G(k, n, P), m)

    # Identity-rows baseline
    P_id = create_identity_rows_P(k, n - k)
    h_id = m_height(build_systematic_G(k, n, P_id), m)
    if h_id < current_h:
        P = P_id
        current_h = h_id
        print(f"  Identity-rows candidate beats current: h_m = {h_id:.6g}")

    # Greedy ±1/±2/±3 local search
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


def build_systematic_G(k: int, n: int, P: np.ndarray) -> np.ndarray:
    I = np.eye(k, dtype=float)
    return np.concatenate([I, P.astype(float)], axis=1)


# ============================================================
# === MAIN LOGIC: DOUBLE P + LOCAL OPTIMIZE + CONDITIONAL UPDATE
# ============================================================

def main():
    # Load current submissions
    with open('generatorMatrix', 'rb') as f:
        generatorMatrix = pickle.load(f)
    with open('mHeight', 'rb') as f:
        mHeight = pickle.load(f)

    print(f"Loaded {len(generatorMatrix)} matrices.\n")

    updated = False
    for key in list(generatorMatrix.keys()):
        n, k, m = key
        current_P = generatorMatrix[key]
        current_h = mHeight[key]

        print(f"🔄 Processing {key} | current h_m = {current_h:.10f}")

        # Double every non-identity column (the parity matrix P)
        doubled_P = 2 * current_P.astype(int)

        # Run full local improvement on the doubled matrix
        improved_P, improved_h = local_improve(doubled_P, n, k, m)

        # Update ONLY if strictly better
        if improved_h < current_h:
            print(f"✅ IMPROVED {key}: {current_h:.10f} → {improved_h:.10f}\n")
            generatorMatrix[key] = improved_P
            mHeight[key] = improved_h
            updated = True
        else:
            print(f"   No improvement for {key} ({improved_h:.10f} >= {current_h:.10f})\n")

    # Save only if at least one improvement was made
    if updated:
        with open('generatorMatrix', 'wb') as f:
            pickle.dump(generatorMatrix, f)
        with open('mHeight', 'wb') as f:
            pickle.dump(mHeight, f)
        print("🎉 Files updated with better matrices/heights!")
    else:
        print("No improvements found – files left unchanged.")

if __name__ == "__main__":
    main()