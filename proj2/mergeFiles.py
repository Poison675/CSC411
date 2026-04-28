import pickle
from typing import Dict, Tuple, List
import numpy as np

# =============================================================================
# <<< EDIT THIS LIST WITH YOUR FILES >>>
# Each entry is a tuple: (generatorMatrix_pickle_path, mHeight_pickle_path)
# You can add as many pairs as you want.
# =============================================================================
SUBMISSIONS: List[Tuple[str, str]] = [
    # === ADD YOUR SUBMISSION FILES HERE ===
    ("generatorMatrix_best", "mHeight_best"),   # example: your first run
    ("generatorMatrixMerge", "mHeightMerge"),   # example: your second run
    ("generatorMatrixTemp", "mHeightTemp"),   # example: another run / different seed
    ("generatorMatrixTempCp", "mHeightTempCp"),
    ("generatorMatrix", "mHeight"),
    ("generatorMatrixTotalMerge", "mHeightTotalMerge"),
    # ("another_gen.pickle", "another_mh.pickle"),
    # ... add more pairs as needed
]

# Output filenames (change if you want a different name)
OUTPUT_GEN = "generatorMatrixTotalMerge"
OUTPUT_MH  = "mHeightTotalMerge"


def main():
    best_gen: Dict[Tuple[int, int, int], np.ndarray] = {}
    best_mh:  Dict[Tuple[int, int, int], float] = {}

    print("Comparing m-heights across multiple submission files...\n")
    print(f"Loaded {len(SUBMISSIONS)} submission pair(s):\n")
    for i, (g, h) in enumerate(SUBMISSIONS, 1):
        print(f"   {i:2d}. generator = {g}    mHeight = {h}")

    # Union of all (n, k, m) keys across all files
    all_keys = set()

    # Collect ALL candidates for each key
    candidates: Dict[Tuple[int, int, int], List[Tuple[float, np.ndarray, str]]] = {}

    for gen_path, mh_path in SUBMISSIONS:
        # Load this pair
        with open(gen_path, "rb") as f:
            gen_dict: Dict[Tuple[int, int, int], np.ndarray] = pickle.load(f)
        with open(mh_path, "rb") as f:
            mh_dict: Dict[Tuple[int, int, int], float] = pickle.load(f)

        # Add every key from this pair
        for key in gen_dict:
            if key in mh_dict:          # only consider complete entries
                all_keys.add(key)
                if key not in candidates:
                    candidates[key] = []
                candidates[key].append((
                    mh_dict[key],           # m-height
                    gen_dict[key],          # P matrix
                    f"{gen_path} + {mh_path}"  # source label for printing
                ))

    # For each key, pick the candidate with the SMALLEST m-height
    sorted_keys = sorted(all_keys)
    print("\nBest results:\n")
    for key in sorted_keys:
        if not candidates.get(key):
            continue

        # Sort by m-height (smallest first)
        candidates[key].sort(key=lambda x: x[0])
        best_h, best_P, source = candidates[key][0]

        best_gen[key] = best_P
        best_mh[key]  = best_h

        print(f"  {key}: h_m = {best_h:.6f}   ← best from {source}")

    if not best_gen:
        print("No valid entries found in any file.")
        return

    # Save the combined best files (exactly the format required by the project)
    with open(OUTPUT_GEN, "wb") as f:
        pickle.dump(best_gen, f)

    with open(OUTPUT_MH, "wb") as f:
        pickle.dump(best_mh, f)

    print(f"\n✅ DONE!")
    print(f"   Best combined files saved as:")
    print(f"      • {OUTPUT_GEN}")
    print(f"      • {OUTPUT_MH}")
    print(f"   Total parameters optimized: {len(best_gen)}")


if __name__ == "__main__":
    main()