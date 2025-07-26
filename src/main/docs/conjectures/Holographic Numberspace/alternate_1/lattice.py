#!/usr/bin/env python3
"""
lattice.py

Constructs and queries the 6D holographic lattice of Z‐metric embeddings
for integers up to N. Exposes functions to build, save, load, and query
the lattice via KD-Tree nearest‐neighbor lookup.
"""

import numpy as np
import argparse
import os
from math import log, exp, fmod, sqrt, atan2, e
from sklearn.neighbors import KDTree


def compute_ghost_mass(n: int) -> float:
    """
    Compute the 'ghost mass' of n:
      m(n) = ln(ln(n)) + 2.582
    """
    return log(log(n)) + 2.582


def compute_z_vector(n: int) -> np.ndarray:
    """
    Map integer n to its 6D Z-vector:
      [metric, curvature, resonance, magnitude, angle, Z_value]
    """
    # 1) spacetime metric
    metric = log(n)

    # 2) ghost mass
    m = compute_ghost_mass(n)

    # 3) approximated curvature
    curvature = (m * metric) / (e ** 2)

    # 4) resonance term
    resonance = fmod(n, metric) * (m / e)

    # 5) magnitude & angle in the (curvature, resonance) plane
    magnitude = sqrt(curvature**2 + resonance**2)
    angle = atan2(resonance, curvature)

    # 6) Z_value approximation
    Z_value = n / exp(curvature)

    return np.array([metric, curvature, resonance,
                     magnitude, angle, Z_value],
                    dtype=np.float64)


def build_holograph_db(N: int, out_path: str):
    """
    Build and save the holographic lattice DB for n = 2..N inclusive.
    Saves:
      - coords:     (N-1 x 6) array of Z-vectors
      - indices:    array of ints 2..N
      - tree:       KDTree built on coords
    """
    # Preallocate buffer
    coords = np.zeros((N - 1, 6), dtype=np.float64)
    indices = np.arange(2, N + 1, dtype=np.int32)

    print(f"Building Z-metric lattice embeddings for n=2..{N}...")
    for idx, n in enumerate(indices):
        coords[idx, :] = compute_z_vector(int(n))
        if idx and idx % 10000 == 0:
            print(f"  • computed {idx} embeddings...")

    # Build KD-tree for nearest-neighbor queries
    print("Constructing KDTree on Z-vectors...")
    tree = KDTree(coords, leaf_size=40, metric='euclidean')

    # Save to disk
    print(f"Saving database to '{out_path}'...")
    np.savez_compressed(out_path,
                        coords=coords,
                        indices=indices,
                        tree_data=tree.data,
                        tree_ind=tree.indices,
                        tree_ptr=tree.indptr,
                        tree_leaf_size=tree.leaf_size)
    print("Done.")


def load_holograph_db(db_path: str):
    """
    Load the holographic lattice DB and reconstruct KDTree.
    Returns (indices, coords, tree).
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"DB file not found: {db_path}")

    npz = np.load(db_path)
    coords = npz['coords']
    indices = npz['indices']

    # Rebuild KDTree from saved internals
    tree = KDTree.__new__(KDTree)
    tree.data = npz['tree_data']
    tree.indices = npz['tree_ind']
    tree.indptr = npz['tree_ptr']
    tree.leaf_size = int(npz['tree_leaf_size'])
    tree.metric = 'euclidean'
    return indices, coords, tree


def query_neighbors(tree: KDTree,
                    coords: np.ndarray,
                    indices: np.ndarray,
                    query_vec: np.ndarray,
                    radius: float = 4.0):
    """
    Return list of (n, distance) for all points within `radius`
    of `query_vec` in the Z-vector space.
    """
    idxs = tree.query_radius(query_vec.reshape(1, -1), r=radius)[0]
    dists = np.linalg.norm(coords[idxs] - query_vec, axis=1)
    results = list(zip(indices[idxs].tolist(), dists.tolist()))
    # sort by distance ascending
    return sorted(results, key=lambda x: x[1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build 6D holographic lattice of Z-metric embeddings.")
    parser.add_argument("--max", "-N", type=int, default=100000,
                        help="Maximum integer N to embed (default: 100000)")
    parser.add_argument("--out", "-o", type=str,
                        default="holograph_db.npz",
                        help="Output .npz path (default: holograph_db.npz)")
    args = parser.parse_args()

    build_holograph_db(N=args.max, out_path=args.out)
