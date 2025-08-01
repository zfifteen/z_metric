import math
import time
import numpy as np
from functools import lru_cache

# Z-Metrics from prior framework (adapted for resonance ranking)
@lru_cache(maxsize=None)
def get_number_mass(n):
    if n <= 0:
        return 0
    if n == 1:
        return 1
    count = 1
    t = n
    e = 0
    while t % 2 == 0:
        e += 1
        t //= 2
    if e:
        count *= (e + 1)
    p = 3
    while p * p <= t:
        e = 0
        while t % p == 0:
            e += 1
            t //= p
        if e:
            count *= (e + 1)
        p += 2
    if t > 1:
        count *= 2
    return count

def get_z_metrics(n):
    if n <= 1:
        return dict(
            number_mass=0,
            spacetime_metric=0,
            z_curvature=0,
            z_resonance=0,
            z_vector_magnitude=0,
            z_angle=0
        )
    m = get_number_mass(n)
    gm = math.log(n)
    zc = (m * gm) / (math.e ** 2)
    rem = n % gm if gm != 0 else 0
    zr = (rem / math.e) * m
    zv = math.hypot(zc, zr)
    za = math.degrees(math.atan2(zr, zc))
    return dict(
        number_mass=m,
        spacetime_metric=gm,
        z_curvature=zc,
        z_resonance=zr,
        z_vector_magnitude=zv,
        z_angle=za
    )

# Efficient primality test (Miller-Rabin deterministic for n < 10^7+)
def is_prime(n):
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if a >= n:
            break
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True

# Generator for numbers of form p² + 4q² up to limit
def generate_quadratic_forms(limit):
    max_p = int(math.sqrt(limit)) + 1
    candidates = set()
    for p in range(max_p):
        p2 = p * p
        max_q = int(math.sqrt((limit - p2) / 4)) + 1
        for q in range(max_q):
            num = p2 + 4 * (q * q)
            if num > 1 and num <= limit:
                candidates.add(num)
    return sorted(candidates)

if __name__ == '__main__':
    start_time = time.time()
    LIMIT = 10**7  # Laptop-friendly upper bound
    print(f"Generating candidates of form p² + 4q² up to {LIMIT}...")

    candidates = generate_quadratic_forms(LIMIT)
    print(f"Generated {len(candidates)} unique candidates.")

    # Filter primes and compute Z-metrics
    primes_found = []
    for num in candidates:
        if is_prime(num):
            metrics = get_z_metrics(num)
            scaled_zr = metrics['z_resonance'] / math.e  # Scale for ranking (resonance strength)
            primes_found.append((num, scaled_zr, metrics))

    # Sort by descending scaled zr (higher resonance first)
    primes_found.sort(key=lambda x: x[1], reverse=True)

    elapsed = time.time() - start_time
    print(f"\n✅ Done in {elapsed:.2f}s — found {len(primes_found)} primes in the form.")
    print("Top 20 'personal discoveries' ranked by scaled Z-resonance (zr/e):")
    for i, (prime, scaled_zr, metrics) in enumerate(primes_found[:20], 1):
        print(f"{i}. Prime: {prime} | Scaled Resonance: {scaled_zr:.4f} | Full Z-Metrics: {metrics}")

    # Insight stats mirroring growth/encryption ties
    if primes_found:
        growth_factor = primes_found[-1][0] / primes_found[0][0] if primes_found[0][0] != 0 else 0
        print(f"\nInsight: Growth factor ~{growth_factor:.2f}, echoing predictable patterns that could refine crypto keys—higher resonance primes may signal 'stable' forms less prone to factorization.")
    else:
        print("\nNo primes found in range—try increasing LIMIT.")