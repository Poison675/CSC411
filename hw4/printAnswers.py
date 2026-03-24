# print_mheights.py
import pickle
import sys
from pathlib import Path

def main():
    filename = "HW-4-mHeights"
    
    if not Path(filename).is_file():
        print(f"Error: File '{filename}' not found in the current directory.")
        print("Make sure you're running this script in the same folder as HW-4-mHeights.")
        sys.exit(1)
    
    try:
        with open(filename, 'rb') as f:
            m_heights = pickle.load(f)
    except Exception as e:
        print(f"Error reading the file: {e}")
        sys.exit(1)
    
    if not isinstance(m_heights, list):
        print("Error: The file does not contain a list of m-heights.")
        sys.exit(1)
    
    count = len(m_heights)
    print(f"Loaded {count} m-heights from {filename}\n")
    
    if count == 0:
        print("The list is empty.")
        return
    
    # Print header
    print("Index    m-height          Status")
    print("-----    ----------------  ----------------")
    
    for i, value in enumerate(m_heights):
        if value == float('inf'):
            status = "∞ (unbounded / d(C) ≤ m)"
            val_str = "inf"
        elif value == 1.0:
            status = "1.0 (likely minimum distance reached)"
            val_str = "1.00000000"
        else:
            status = "finite"
            val_str = f"{value:.8f}"
        
        print(f"{i:4d}     {val_str:>12}    {status}")
    
    print("\n" + "-" * 60)
    print(f"Summary:")
    print(f"  • Total entries: {count}")
    print(f"  • Number of inf values: {sum(1 for x in m_heights if x == float('inf'))}")
    print(f"  • Number of exactly 1.0: {sum(1 for x in m_heights if x == 1.0)}")
    
    finite_values = [x for x in m_heights if x != float('inf') and x > 0]
    if finite_values:
        print(f"  • Min finite m-height: {min(finite_values):.8f}")
        print(f"  • Max finite m-height: {max(finite_values):.8f}")
    else:
        print("  • No finite m-heights found (all inf or 1.0)")

if __name__ == "__main__":
    main()