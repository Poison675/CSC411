import pickle
import os
import numpy as np
from itertools import combinations
from scipy.optimize import linprog
from typing import List, Tuple


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


def print_generator_matrices():
    """
    Reads the saved 'generatorMatrix' and 'mHeight' pickle files
    and prints EVERY stored generator matrix in a clean, visually pleasing format.
    
    - Loads both files automatically.
    - For each (n, k, m):
        • Shows the m-height
        • Prints the parity matrix P (integers, nicely aligned)
        • Prints the full systematic generator matrix G = [I_k | P] (floats)
    - Sorted by (n, k, m) for easy reading.
    - Uses fixed-width formatting, clear borders, and rounded values.
    """
    if not os.path.exists("generatorMatrixTempCp"):
        print("❌ No 'generatorMatrix' file found in the current directory.")
        print("   Run the main hill-climbing search first to generate matrices.")
        return

    # Load the saved results
    with open("generatorMatrixTempCp", "rb") as f:
        generatorMatrix = pickle.load(f)
    with open("mHeightTempCp", "rb") as f:
        mHeight = pickle.load(f)

    if not generatorMatrix:
        print("⚠️  generatorMatrix file is empty.")
        return

    print("🚀 LOADED GENERATOR MATRICES")
    print("=" * 90)

    # Sort keys for consistent output order
    for key in sorted(generatorMatrix.keys()):
        n, k, m = key
        P = generatorMatrix[key]          # k × (n-k) integer matrix
        r = n - k
        I = np.eye(k)
        G = np.hstack((I, P.astype(float)))
        h = m_height(G, m)

        # Build full systematic G = [I_k | P]

        # ====================== VISUALLY PLEASING PRINT ======================
        print(f"\n📌  (n={n}, k={k}, m={m})")
        print(f"   m-height h_m(C) = {h:.10f}")
        print(f"   P  (parity matrix, {k}×{r})")
        
        # Pretty P (integers, right-aligned, fixed width)
        print("   " + "─" * (r * 8))
        for row in P:
            print("   " + " ".join(f"{int(x):6d}" for x in row))
        print("   " + "─" * (r * 8))

        # Full G (floats, 6 decimal places, clean)
        print(f"   Full systematic G = [I_k | P]   ({k}×{n})")
        np.set_printoptions(precision=0, suppress=True, linewidth=120, floatmode='fixed')
        print(G)
        np.set_printoptions()   # reset global settings

        print("-" * 90)

    print(f"\n✅ Printed {len(generatorMatrix)} generator matrices successfully!")
    print("   (Files: generatorMatrix + mHeight)")


# ====================== EXAMPLE USAGE ======================
if __name__ == "__main__":
    # Just call this function whenever you want a beautiful overview
    print_generator_matrices()

    # Optional: you can also call it after the hill-climbing search
    # (just put print_generator_matrices() at the very end of your main script)