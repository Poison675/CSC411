import pickle
import numpy as np
from itertools import combinations
from scipy.optimize import linprog
import os
from concurrent.futures import ProcessPoolExecutor

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


def main():
    """
    Program that reads 'generatorMatrix' (pickled dict of (n,k,m) -> P matrix)
    and produces 'mHeight' (pickled dict of (n,k,m) -> m-height).
    
    - Constructs systematic G = [I_k | P] for each entry.
    - Computes exact m-height using the provided LP-based functions.
    - Saves the results to 'mHeight' file (ready for submission).
    """
    # Load the generatorMatrix file (as specified in the project)
    with open('generatorMatrixSC', 'rb') as f:
        gen_dict = pickle.load(f)
    
    mheight_dict = {}
    
    print(f"Processing {len(gen_dict)} generator matrices...")
    for idx, (key, P) in enumerate(gen_dict.items(), 1):
        n, k, m = key
        print(f"  [{idx}/{len(gen_dict)}] Computing m-height for (n={n}, k={k}, m={m}) ...")
        
        # Construct full systematic generator matrix G = [I_k | P]
        I = np.eye(k, dtype=float)
        G = np.hstack((I, P.astype(float)))
        
        # Verify dimensions
        assert G.shape == (k, n), f"Invalid shape for key {key}: {G.shape}"
        
        # Compute m-height using the provided exact LP method
        h = m_height(G, m)
        mheight_dict[key] = float(h)
        print(f'M_height: {h}')
    
    # Save the mHeight file (as required by the project)
    with open('mHeightSC', 'wb') as f:
        pickle.dump(mheight_dict, f)
    
    print("\n✅ mHeight file created successfully!")
    print(f"   Keys processed: {list(mheight_dict.keys())}")
    print("   Ready for submission (together with generatorMatrix and report.pdf).")


if __name__ == "__main__":
    main()