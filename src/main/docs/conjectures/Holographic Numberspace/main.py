#!/usr/bin/env python3
"""
Main script: loads the prebuilt holographic DB,
classifies primality without any factorization,
maintains a rolling 360-frame window, and predicts the next prime.
"""

import numpy as np
import sympy
from sklearn.neighbors import KDTree
from collections import deque
from sklearn.linear_model import LinearRegression


# Reuse mass-independent feature functions from batch
def compute_minkowski_norm(n):
    return float(np.sqrt(n))


def compute_vortex_index(n):
    return float((n & -n).bit_length() - 1)


def compute_prime_density(n, prime_list):
    import bisect
    idx = bisect.bisect_left(prime_list, n)
    lower = prime_list[idx - 1] if idx > 0 else prime_list[0]
    upper = prime_list[idx] if idx < len(prime_list) else prime_list[-1]
    gap = upper - lower if upper != lower else 1
    return 1.0 / gap


def compute_z_angle(n):
    return float(np.arctan2(n % 180, n % 90))


class RollingHologram:
    """Keeps the last `window_size` Z-vectors in memory."""
    def __init__(self, window_size=360):
        self.buffer = deque(maxlen=window_size)
        self._kdtree = None

    def add(self, n, zvec):
        self.buffer.append((n, zvec))
        self._kdtree = None

    def _build_tree(self):
        pts = np.stack([z for _, z in self.buffer])
        self._kdtree = KDTree(pts)

    def query(self, zvec, k=3):
        if self._kdtree is None:
            self._build_tree()
        dist, idx = self._kdtree.query([zvec], k=k)
        return [(self.buffer[i][0], dist[0][j]) for j, i in enumerate(idx[0])]

    def as_array(self):
        return np.stack([z for _, z in self.buffer])


def load_database(path="holograph_db.npz"):
    data = np.load(path, allow_pickle=True)
    return (
        data["ns"],
        data["feats"],
        data["stats_min"],
        data["stats_max"],
        data["prime_list"].tolist(),
    )


def normalize_raw(raw, stats_min, stats_max):
    span = np.where(stats_max > stats_min, stats_max - stats_min, 1.0)
    return (raw - stats_min) / span


def classify(n, stats_min, stats_max, global_tree, prime_flags, primes):
    """
    Classify n as prime/composite:
      - If preloaded (n ≤ max_n): read mass-dependent features (gm, κ)
      - Else: compute only mass-independent features, normalize, and query
    """
    max_n = primes[-1]
    if n <= max_n:
        idx = n - 2
        gm, curv = feats[idx, :2]
        return (gm < 1e-6 and curv < 1e-6)

    # compute raw mass-independent features
    raw = np.zeros(6, float)
    raw[0] = 0.0  # placeholder for gm
    raw[1] = 0.0  # placeholder for curvature
    raw[2] = compute_minkowski_norm(n)
    raw[3] = compute_vortex_index(n)
    raw[4] = compute_prime_density(n, primes)
    raw[5] = compute_z_angle(n)

    normed = normalize_raw(raw, stats_min, stats_max)
    # nearest neighbor in mass-independent subspace (dims 2:6)
    dist, idx = global_tree.query([normed[2:]], k=1)
    return prime_flags[idx[0][0]]  # True if neighbor is a prime


def fit_AR(window):
    """Fit independent AR(1) models to each Z-dimension."""
    X = window[:-1]
    Y = window[1:]
    coefs = np.zeros(window.shape[1])
    inter = np.zeros(window.shape[1])
    for d in range(window.shape[1]):
        model = LinearRegression().fit(
            X[:, d].reshape(-1, 1), Y[:, d]
        )
        coefs[d] = model.coef_[0]
        inter[d] = model.intercept_
    return coefs, inter


def predict_next(z_last, coefs, inter):
    return coefs * z_last + inter


if __name__ == "__main__":
    # Load database
    ns, feats, stats_min, stats_max, prime_list = load_database()
    max_n = ns[-1]
    prime_set = set(prime_list)
    prime_flags = np.isin(ns, prime_list)

    # Build global KD-Tree on mass-independent dims (2,3,4,5)
    global_tree = KDTree(feats[:, 2:])

    # Rolling buffer
    holo = RollingHologram(window_size=360)

    # Process primes in ascending order
    for n in prime_list:
        # classification (should always be True for n in prime_list)
        is_p = classify(n, stats_min, stats_max, global_tree, prime_flags, prime_list)
        print(f"{n} → {'prime' if is_p else 'composite'}")

        # add normalized mass-independent Z-vector to hologram
        zvec = feats[n - 2, 2:]
        holo.add(n, zvec)

        # Once we have 360 frames, forecast the next prime and exit
        if len(holo.buffer) == holo.buffer.maxlen:
            window = holo.as_array()
            coefs, inter = fit_AR(window)
            z_next = predict_next(window[-1], coefs, inter)

            # find candidates within Δ = 0.4 in mass-independent subspace
            idxs = global_tree.query_radius([z_next], r=0.4)[0]
            cands = [ns[i] for i in idxs]
            print("\nPredicted next prime candidates:", sorted(cands))
            break
