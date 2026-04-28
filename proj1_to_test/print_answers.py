import pickle
import os
import numpy as np
from itertools import combinations
from scipy.optimize import linprog
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple


def _solve_lp(args):
    """Solve one LP_{S,j} linear program."""
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
    """Exact m-height using the LP formulation (parallelized)."""
    n = G.shape[1]
    if m == 0:
        return 1.0
    
    # Generate all tasks: (G, j, barS) for every S of size m and every j in S
    tasks = [(G, j, [t for t in range(n) if t not in S])
             for S in combinations(range(n), m) for j in S]
    
    # Parallel execution
    with ProcessPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
        results = list(executor.map(_solve_lp, tasks))
    
    return max(max(results), 1.0)


def print_generator_matrices():
    """
    Reads the saved 'generatorMatrixe' pickle file
    and prints EVERY stored generator matrix with freshly computed m-height.
    
    - Loads generator matrices only.
    - For each (n, k, m):
        • Computes m-height on the fly using the exact LP method
        • Prints the parity matrix P (integers)
        • Prints the full systematic generator matrix G = [I_k | P] (floats)
    - Sorted by (n, k, m).
    """
    filename = "generatorMatrixSC"
    
    if not os.path.exists(filename):
        print(f"❌ No '{filename}' file found in the current directory.")
        print("   Run the main hill-climbing search first.")
        return

    # Load the saved generator matrices
    with open(filename, "rb") as f:
        generatorMatrix = pickle.load(f)

    if not generatorMatrix:
        print("⚠️  generatorMatrix file is empty.")
        return

    print("🚀 LOADED GENERATOR MATRICES + COMPUTING m-height")
    print("=" * 100)

    # Sort keys for consistent output
    for key in sorted(generatorMatrix.keys()):
        n, k, m = key
        P = generatorMatrix[key]          # k × (n-k) integer matrix
        r = n - k
        I = np.eye(k, dtype=float)
        G = np.hstack((I, P.astype(float)))
        
        # Compute m-height on the fly
        print(f"   Computing m-height for (n={n}, k={k}, m={m}) ...")
        h = m_height(G, m)

        # ====================== VISUALLY PLEASING PRINT ======================
        print(f"\n📌  (n={n}, k={k}, m={m})")
        print(f"   m-height h_m(C) = {h:.10f}")
        print(f"   P  (parity matrix, {k}×{r})")
        
        # Pretty P (integers)
        print("   " + "─" * (r * 8))
        for row in P:
            print("   " + " ".join(f"{int(x):6d}" for x in row))
        print("   " + "─" * (r * 8))

        # Full systematic G
        print(f"   Full systematic G = [I_k | P]   ({k}×{n})")
        np.set_printoptions(precision=0, suppress=True, linewidth=120, floatmode='fixed')
        print(G)
        np.set_printoptions()   # reset

        print("-" * 100)

    print(f"\n✅ Printed {len(generatorMatrix)} generator matrices successfully!")
    print("   m-heights were computed exactly using linear programming.")


# ====================== EXAMPLE USAGE ======================
if __name__ == "__main__":
    print_generator_matrices()