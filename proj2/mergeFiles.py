import pickle
import argparse
from typing import Dict, Tuple

def main():
    parser = argparse.ArgumentParser(
        description="Compare two generatorMatrix + mHeight pairs and create the best combined submission files."
    )
    parser.add_argument("gen1", help="Path to first generatorMatrix file (pickled dict)")
    parser.add_argument("mh1",  help="Path to first mHeight file (pickled dict)")
    parser.add_argument("gen2", help="Path to second generatorMatrix file (pickled dict)")
    parser.add_argument("mh2",  help="Path to second mHeight file (pickled dict)")
    parser.add_argument("--output_gen", default="generatorMatrix", help="Output filename for best generatorMatrix (default: generatorMatrix)")
    parser.add_argument("--output_mh",  default="mHeight",        help="Output filename for best mHeight (default: mHeight)")

    args = parser.parse_args()

    # Load both pairs
    with open(args.gen1, "rb") as f:
        gen1: Dict[Tuple[int, int, int], "np.ndarray"] = pickle.load(f)
    with open(args.mh1, "rb") as f:
        mh1: Dict[Tuple[int, int, int], float] = pickle.load(f)

    with open(args.gen2, "rb") as f:
        gen2: Dict[Tuple[int, int, int], "np.ndarray"] = pickle.load(f)
    with open(args.mh2, "rb") as f:
        mh2: Dict[Tuple[int, int, int], float] = pickle.load(f)

    best_gen = {}
    best_mh = {}

    # Union of all (n, k, m) keys
    all_keys = sorted(set(gen1.keys()) | set(gen2.keys()))

    print("Comparing m-heights...\n")
    for key in all_keys:
        candidates = []

        # Collect valid entries from first pair
        if key in gen1 and key in mh1:
            candidates.append((mh1[key], gen1[key], 1))

        # Collect valid entries from second pair
        if key in gen2 and key in mh2:
            candidates.append((mh2[key], gen2[key], 2))

        if not candidates:
            continue

        # Pick the one with the smallest m-height
        candidates.sort(key=lambda x: x[0])          # sort by height
        best_h, best_P, source = candidates[0]

        best_gen[key] = best_P
        best_mh[key] = best_h

        print(f"  {key}: best h_m = {best_h:.6f} (from file {source})")

    # Save the best combined files
    with open(args.output_gen, "wb") as f:
        pickle.dump(best_gen, f)

    with open(args.output_mh, "wb") as f:
        pickle.dump(best_mh, f)

    print(f"\n✅ Done! Best files saved as:")
    print(f"   • {args.output_gen}")
    print(f"   • {args.output_mh}")
    print(f"   Total parameters: {len(best_gen)}")


if __name__ == "__main__":
    main()