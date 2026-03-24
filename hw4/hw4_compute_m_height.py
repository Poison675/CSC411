import pickle
import numpy as np
from itertools import combinations, product
from scipy.optimize import linprog
import multiprocessing as mp
from functools import partial
import time

def _solve_one_lp(params):
    """Worker function for one LP (picklable for multiprocessing)."""
    c_obj, A_ub, b_ub, A_eq, b_eq = params
    res = linprog(c_obj,
                  A_ub=A_ub, b_ub=b_ub,
                  A_eq=A_eq, b_eq=b_eq,
                  bounds=(None, None),
                  method='highs',
                  options={'presolve': True, 'disp': False})
    if res.success:
        return -res.fun
    elif res.status == 3:   # unbounded
        return 1.0
    else:                   # infeasible
        return 0.0


def compute_m_height(G: np.ndarray, m: int, n: int, k: int) -> float:
    """
    Computes the exact m-height of the analog code with generator matrix G
    using the LP-based algorithm described in the project document (Theorem 1).
    """
    if m == 0:
        return 1.0

    # Pre-build ALL LP instances (tiny memory footprint)
    lp_instances = []
    cols = range(n)
    psi_values = list(product([-1.0, 1.0], repeat=m))

    for a in cols:
        for b in cols:
            if b == a:
                continue
            remaining = [x for x in cols if x != a and x != b]
            for X_tup in combinations(remaining, m - 1 if m > 1 else 0):
                X = list(X_tup)
                X_sorted = sorted(X)
                Y_sorted = sorted(set(remaining) - set(X))
                x_list = [a] + X_sorted + [b] + Y_sorted
                tau_inv = {x_list[j]: j for j in range(n)}

                for psi in psi_values:
                    s0 = psi[0]
                    c_obj = -s0 * G[:, a]

                    A_ub_list, b_ub_list = [], []

                    # X constraints
                    for jj, j_col in enumerate(X_sorted):
                        pos = tau_inv[j_col]
                        s_j = psi[pos]
                        row1 = s_j * G[:, j_col] - s0 * G[:, a]
                        A_ub_list.append(row1)
                        b_ub_list.append(0.0)
                        row2 = -s_j * G[:, j_col]
                        A_ub_list.append(row2)
                        b_ub_list.append(-1.0)

                    # Y constraints (|c_j| <= 1)
                    for j_col in Y_sorted:
                        A_ub_list.append(G[:, j_col])
                        b_ub_list.append(1.0)
                        A_ub_list.append(-G[:, j_col])
                        b_ub_list.append(1.0)

                    A_ub = np.array(A_ub_list, dtype=np.float64)
                    b_ub = np.array(b_ub_list, dtype=np.float64)
                    A_eq = np.array([G[:, b]], dtype=np.float64)
                    b_eq = np.array([1.0], dtype=np.float64)

                    lp_instances.append((c_obj, A_ub, b_ub, A_eq, b_eq))

    # Parallel solve on all CPU cores
    num_workers = max(1, mp.cpu_count())
    print(f"   Solving {len(lp_instances):,} LPs with {num_workers} CPU workers...")

    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(_solve_one_lp, lp_instances)

    hm = max(results) if results else 1.0
    return max(hm, 1.0)


# ====================== MAIN ======================
print("Loading HW-4-n_k_m_P ...")
with open('HW-4-n_k_m_P', 'rb') as f:
    data = pickle.load(f)

print(f"Loaded {len(data)} samples. Starting m-height computation...\n")

m_heights = []
for idx, item in enumerate(data):
    n, k, m, P = item
    # Build systematic generator matrix G = [I_k | P]
    I = np.eye(k, dtype=float)
    G = np.hstack((I, P.astype(float)))

    print(f"[{idx+1:4d}/{len(data)}]  n={n} k={k} m={m}  shape={G.shape}")
    h = compute_m_height(G, m, n, k)
    m_heights.append(h)
    print(f"    -> m-height = {h:.10f}\n")

# Save the result
with open('HW-4-mHeightsTEMP', 'wb') as f:
    pickle.dump(m_heights, f)

print("All done! File 'HW-4-mHeightsTEMP' created successfully.")



# print("=== CSCE-411 Project m-Height Algorithm Verification ===\n")

# # Load sample input
# with open('CSCE-411-Project-sample-n_k_m_P', 'rb') as f:
#     sample_data = pickle.load(f)
# print(f"Loaded {len(sample_data)} test cases from CSCE-411-Project-sample-n_k_m_P")

# # Load expected m-heights
# with open('CSCE-411-Project-sample-mHeights', 'rb') as f:
#     expected_heights = pickle.load(f)
# print(f"Loaded {len(expected_heights)} expected m-heights from CSCE-411-Project-sample-mHeights\n")

# if len(sample_data) != len(expected_heights):
#     print("ERROR: Number of samples and expected heights do not match!")
#     sys.exit(1)

# tolerance_abs = 1e-6
# tolerance_rel = 1e-4   # stricter than the 1% grading threshold
# max_rel_error = 0.0
# failures = 0

# start_total = time.time()

# for idx, (item, expected) in enumerate(zip(sample_data, expected_heights)):
#     n, k, m, P = item
#     I = np.eye(k, dtype=float)
#     G = np.hstack((I, P.astype(float)))

#     print(f"Test {idx+1:2d}/{len(sample_data)} | n={n} k={k} m={m:2d}  ", end="")
#     t0 = time.time()
#     computed = compute_m_height(G, m, n, k)
#     t1 = time.time()

#     abs_err = abs(computed - expected)
#     rel_err = abs_err / max(abs(expected), 1e-9) if expected > 0 else abs_err

#     if rel_err > tolerance_rel or abs_err > tolerance_abs:
#         status = "FAIL"
#         failures += 1
#     else:
#         status = "PASS"

#     print(f"expected={expected:.8f}  computed={computed:.8f}  "
#           f"rel_err={rel_err*100:6.4f}%  time={t1-t0:5.2f}s  → {status}")

#     if rel_err > max_rel_error:
#         max_rel_error = rel_err

# print("\n" + "="*70)
# print(f"VERIFICATION SUMMARY")
# print(f"Total tests      : {len(sample_data)}")
# print(f"Passed           : {len(sample_data) - failures}")
# print(f"Failed           : {failures}")
# print(f"Max relative error : {max_rel_error*100:.6f}%")
# print(f"Total time       : {time.time() - start_total:.1f} seconds")

# if failures == 0:
#     print("\nVERIFICATION PASSED - Your algorithm matches the reference implementation exactly!")
#     print("   You can now confidently run it on HW-4-n_k_m_P.")
# else:
#     print("\nSome tests failed. Check the LP implementation or floating-point precision.")

