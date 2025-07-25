#!/usr/bin/env python3
"""
prime_comparison_optimized.py

Benchmark pure Miller–Rabin vs. hybrid geometric-filter + Miller–Rabin.
"""

import math
import random
import time

# ------------------ Miller–Rabin Primality Test ------------------ #
def is_probable_prime(n, k=5):
    if n < 2:
        return False
    # quick check small primes
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23):
        if n % p == 0:
            return n == p
    # write n-1 = 2^r * d
    r, d = 0, n - 1
    while d % 2 == 0:
        d //= 2
        r += 1
    # witness loop
    for _ in range(k):
        a = random.randrange(2, n - 1)
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


# ------------------ Geometric Filter ------------------ #
class GeometricFilter:
    def __init__(self, z1_stats, z4_stats, phase_threshold=5000):
        # z1, z4 gaussian parameters (mu, sigma)
        self.mu1, self.sigma1 = z1_stats
        self.mu4, self.sigma4 = z4_stats
        self.phase_threshold = phase_threshold

    def _prime_factors(self, n):
        factors = []
        x = n
        # trial division up to √n
        for p in [2] + list(range(3, int(math.isqrt(n)) + 1, 2)):
            while x % p == 0:
                factors.append(p)
                x //= p
        if x > 1:
            factors.append(x)
        return factors

    def compute_z1(self, n):
        # sum of normalized logs of prime factors
        return sum(math.log(p, n) for p in self._prime_factors(n))

    def compute_z4(self, n):
        # count of prime factors
        return len(self._prime_factors(n))

    def curvature(self, z1, z4):
        # simple |z4 - z1| + ε
        return abs(z4 - z1) + 1e-9

    def passes(self, n):
        # always allow small n
        if n < self.phase_threshold:
            return True

        z1 = self.compute_z1(n)
        z4 = self.compute_z4(n)
        curv = self.curvature(z1, z4)

        # updated σ to be more aggressive
        sigma = 0.6 + (math.e / curv)

        cond1 = abs(z1 - self.mu1) < self.sigma1 * sigma
        cond4 = abs(z4 - self.mu4) < self.sigma4 * sigma
        return cond1 and cond4


# ------------------ Hybrid Check & Benchmark ------------------ #
def hybrid_is_prime(n, gf):
    if not gf.passes(n):
        return False
    return is_probable_prime(n)


def benchmark(limit_n, mode='pure', gf=None):
    start = time.time()
    mr_calls = 0
    skipped = 0

    for i in range(2, limit_n + 1):
        if mode == 'hybrid':
            # geometric filter first
            if not gf.passes(i):
                skipped += 1
                continue
            mr_calls += 1
            is_probable_prime(i)
        else:
            mr_calls += 1
            is_probable_prime(i)

    elapsed = time.time() - start
    return elapsed, mr_calls, skipped


# ------------------ Main Entry Point ------------------ #
if __name__ == '__main__':
    limits = [100_000, 200_000, 500_000, 1_000_000]
    gf = GeometricFilter(
        z1_stats=(0.49, 0.56),
        z4_stats=(2.22, 2.09),
        phase_threshold=5000
    )

    # Header
    print(f"{'n':>11} | {'Pure MR (s)':>12} | {'Hybrid (s)':>11} | {'MR Calls':>8} | {'Skipped':>7} | {'Ratio':>6}")
    print('-' * 68)

    # Run benchmarks
    for n in limits:
        t_pure, calls_pure, _ = benchmark(n, mode='pure')
        t_hyb, calls_hyb, skipped = benchmark(n, mode='hybrid', gf=gf)
        ratio = t_pure / t_hyb if t_hyb > 0 else float('inf')
        print(f"{n:>11,} | {t_pure:>12.3f} | {t_hyb:>11.3f} | {calls_hyb:>8,} | {skipped:>7,} | {ratio:>6.2f}")
