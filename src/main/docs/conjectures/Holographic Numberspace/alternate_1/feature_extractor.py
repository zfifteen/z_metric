#!/usr/bin/env python3
"""
feature_extractor.py

Compute Z-metric and dynamic features for integers,
including velocity, curvature change, and resonance drift.
"""

import numpy as np
from lattice import compute_z_vector


def compute_dynamic_features(buffer: np.ndarray) -> np.ndarray:
    """
    Given a buffer of Z-vectors of shape (T, 6), compute dynamic features
    for each time step t >= 1:
      - metric_velocity       = metric[t]    - metric[t-1]
      - curvature_velocity    = curvature[t] - curvature[t-1]
      - resonance_velocity    = resonance[t] - resonance[t-1]
      - magnitude_velocity    = magnitude[t] - magnitude[t-1]
      - angle_change          = minimal signed difference of angles
      - Z_value_velocity      = Z_value[t]   - Z_value[t-1]

    Returns an array of shape (T-1, 6).
    """
    # raw differences (velocity)
    diffs = np.diff(buffer, axis=0)

    # compute minimal angular difference in [-π, +π]
    angles = buffer[:, 4]
    raw = angles[1:] - angles[:-1]
    angle_diff = (raw + np.pi) % (2 * np.pi) - np.pi

    # assemble dynamic features
    dyn_feats = np.hstack([
        diffs[:, [0]],            # metric_velocity
        diffs[:, [1]],            # curvature_velocity
        diffs[:, [2]],            # resonance_velocity
        diffs[:, [3]],            # magnitude_velocity
        angle_diff.reshape(-1, 1),# angle_change
        diffs[:, [5]]             # Z_value_velocity
    ])
    return dyn_feats


def make_feature_matrix(ns: np.ndarray) -> np.ndarray:
    """
    Given an array of integers ns (shape (N,)), compute a raw feature matrix
    where each row corresponds to:
      [metric, curvature, resonance, magnitude, angle, Z_value,
       metric_vel, curvature_vel, resonance_vel,
       magnitude_vel, angle_change, Z_vel]

    Returns an (N x 12) array.
    """
    N = ns.shape[0]
    # 1) compute Z-vectors for each n
    z_vectors = np.zeros((N, 6), dtype=np.float64)
    for i, n in enumerate(ns):
        z_vectors[i] = compute_z_vector(int(n))

    # 2) compute dynamic features (N-1 x 6)
    dyn = compute_dynamic_features(z_vectors)

    # 3) pad first row of dynamics with zeros
    pad = np.zeros((1, dyn.shape[1]), dtype=np.float64)
    dyn_full = np.vstack([pad, dyn])

    # 4) concatenate static and dynamic features
    features = np.hstack([z_vectors, dyn_full])
    return features


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build raw feature matrix for integers n=2..N."
    )
    parser.add_argument("--max", "-N", type=int, default=100000,
                        help="Maximum integer N to process (default: 100000)")
    parser.add_argument("--out", "-o", type=str, default="features.npz",
                        help="Output .npz path (default: features.npz)")
    args = parser.parse_args()

    # prepare integer range
    ns = np.arange(2, args.max + 1, dtype=np.int64)
    print(f"Building raw feature matrix for n=2..{args.max}...")

    feat = make_feature_matrix(ns)
    np.savez_compressed(args.out, ns=ns, features=feat)

    print(f"Saved feature matrix to '{args.out}'.")
