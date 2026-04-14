import pickle
import numpy as np
import os


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
    if not os.path.exists("generatorMatrix"):
        print("❌ No 'generatorMatrix' file found in the current directory.")
        print("   Run the main hill-climbing search first to generate matrices.")
        return

    # Load the saved results
    with open("generatorMatrix", "rb") as f:
        generatorMatrix = pickle.load(f)
    with open("mHeight", "rb") as f:
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
        h = mHeight.get(key, float('nan'))

        # Build full systematic G = [I_k | P]
        I = np.eye(k)
        G = np.hstack((I, P.astype(float)))

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
        np.set_printoptions(precision=6, suppress=True, linewidth=120, floatmode='fixed')
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