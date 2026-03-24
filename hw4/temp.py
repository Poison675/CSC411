import pickle
import numpy as np

def print_all_generator_matrices(filename: str = "generatorMatrix"):
    """
    Load the generatorMatrix pickle file and print every P matrix
    in a clean, professional format with clear headers.
    """
    print("🔍 Loading generatorMatrix file...\n")
    
    with open(filename, "rb") as f:
        data = pickle.load(f)
    
    print(f"✅ Loaded {len(data)} generator matrices.\n")
    print("=" * 80)
    
    # Sort keys for consistent output order
    for key in sorted(data.keys()):
        n, k, m = key
        P = data[key]
        
        print(f"\n📋 MATRIX FOR (n={n}, k={k}, m={m})")
        print(f"   Shape: {P.shape} × {n-k}   |   Type: int")
        print("-" * 80)
        
        # Nice matrix printing with aligned columns
        np.set_printoptions(precision=0, suppress=True, linewidth=100)
        print(P)
        
        print("-" * 80)
        print(f"   This P forms the parity part of G = [I_{k} | P]\n")
    
    print("=" * 80)
    print("✅ All matrices printed successfully!")
    print("   (You can copy-paste any matrix directly into your report or LaTeX)")


# ====================== HOW TO USE ======================
if __name__ == "__main__":
    print_all_generator_matrices("generatorMatrixOVERALL")
    # Or specify a different filename:
    # print_all_generator_matrices("generatorMatrixOVERALL")