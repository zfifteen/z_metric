#!/usr/bin/env python3
"""
Batch script to build a 6D “holographic” NumberSpace database.
Factorizes each n ≤ N, computes Z-vector components, normalizes them,
and writes everything into an .npz file for fast runtime loading.
"""

import sympy
import numpy as np


def factorize(n):
    """Return prime factorization of n as {p: e, …}."""
    return sympy.factorint(n)


# Mass-Dependent Features (requires factorization)
def compute_ghost_mass(n, factors):
    """Placeholder for Z-minima energy. Zero for primes."""
    # Example proxy: total exponent count
    return float(sum(factors.values()))


def compute_curvature(n, factors):
    """Placeholder for discrete curvature."""
    # Example proxy: number of distinct prime factors minus 1
    return float(max(len(factors) - 1, 0))


# Mass-Independent Features
def compute_minkowski_norm(n):
    """Discrete spacetime norm proxy."""
    # Example proxy: sqrt(n) to simulate growth
    return float(np.sqrt(n))


import numpy as np

def compute_vortex_index(n):
    # n is numpy.int64
    lowbit = n & -n
    return float(np.log2(lowbit))


def compute_prime_density(n, prime_list):
    """Inverse average gap to nearest primes."""
    import bisect
    idx = bisect.bisect_left(prime_list, n)
    # find neighbors
    lower = prime_list[idx - 1] if idx > 0 else prime_list[0]
    upper = prime_list[idx] if idx < len(prime_list) else prime_list[-1]
    gap = upper - lower if upper != lower else 1
    return 1.0 / gap


def compute_z_angle(n):
    """Angle in Z-space proxy."""
    # Example proxy: arctan2(curvature, ghost_mass) placeholder
    return float(np.arctan2(n % 180, n % 90))


def build_holograph_db(N, out_path):
    """Main builder: compute raw features, normalize, and save."""
    ns = np.arange(2, N + 1)
    raw = np.zeros((len(ns), 6), dtype=float)

    # Precompute primes for density
    prime_list = list(sympy.primerange(2, N + 1))

    print(f"Building raw feature matrix for n=2…{N}")
    for i, n in enumerate(ns):
        fac = factorize(n)
        raw[i, 0] = compute_ghost_mass(n, fac)
        raw[i, 1] = compute_curvature(n, fac)
        raw[i, 2] = compute_minkowski_norm(n)
        raw[i, 3] = compute_vortex_index(n)
        raw[i, 4] = compute_prime_density(n, prime_list)
        raw[i, 5] = compute_z_angle(n)

    # Normalize each column to [0, 1]
    mins = raw.min(axis=0)
    maxs = raw.max(axis=0)
    span = np.where(maxs > mins, maxs - mins, 1.0)
    feats = (raw - mins) / span

    print(f"Saving database to {out_path}")
    np.savez_compressed(
        out_path,
        ns=ns,
        feats=feats,
        stats_min=mins,
        stats_max=maxs,
        prime_list=np.array(prime_list, dtype=int),
    )
    print("Done.")


if __name__ == "__main__":
    # Adjust N as needed for your target bound
    build_holograph_db(N=100_000, out_path="holograph_db.npz")
