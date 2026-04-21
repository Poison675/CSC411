import numpy as np

def regular_simplex_etf(k: int) -> np.ndarray:
    """
    Generate an Equiangular Tight Frame (ETF) consisting of 
    n = k + 1 unit vectors in k-dimensional real space (ℝ^k).
    
    This is the classic "regular simplex" ETF:
    - All vectors have unit norm.
    - Every pair of distinct vectors has the same inner product c = -1/k
      (equiangular with optimal angular separation).
    - They form a tight frame: for any x ∈ ℝ^k,
        ∑_{i=1}^n |⟨x, v_i⟩|² = (n / k) ‖x‖² = ((k + 1) / k) ‖x‖².
    
    This construction satisfies all defining principles of an ETF and is
    deterministic and exact (no approximation or randomness).
    
    Parameters
    ----------
    k : int
        Dimension of the space (k ≥ 1).
    
    Returns
    -------
    np.ndarray
        Array of shape (n, k) where each row is one unit vector of the ETF.
        n = k + 1.
    
    Example
    -------
    >>> vectors = regular_simplex_etf(2)   # 3 vectors in 2D (equilateral triangle)
    >>> print(vectors)
    >>> gram = vectors @ vectors.T
    >>> print(gram)   # diagonal = 1, off-diagonals = -0.5
    """
    n = k + 1
    # Place the simplex vertices in ambient ℝ^n using centered standard basis
    mean = np.ones(n) / n
    points = np.eye(n) - mean[:, None]          # shape (n, n); columns are centered e_j
    
    # Embed the (n-1)-dimensional subspace into ℝ^k via SVD
    U, S, Vt = np.linalg.svd(points, full_matrices=False)
    basis = U[:, :k]                            # n × k orthonormal basis for the column space
    
    # Coordinates of the n points in ℝ^k
    coords = basis.T @ points                   # k × n
    
    # Normalize each column to unit length (all norms are identical)
    norms = np.linalg.norm(coords, axis=0)
    coords /= norms
    
    # Return as (n, k) matrix (each row = one vector)
    return coords.T