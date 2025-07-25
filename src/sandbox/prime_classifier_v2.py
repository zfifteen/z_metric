import math
import sys
from functools import lru_cache

# --- Miller–Rabin Primality Test ---
def miller_rabin(n):
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17):
        if n == p:
            return True
        if n % p == 0:
            return False

    # write n-1 = d * 2^s
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1

    def try_composite(a):
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            return False
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                return False
        return True  # definitely composite

    for a in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        if a % n == 0:
            return True
        if try_composite(a):
            return False
    return True

# --- Approximate Mass via Small Primes ---
class SmallPrimeMass:
    def __init__(self, primes):
        self.primes = primes
        self.modulus = primes[-1]

    @lru_cache(maxsize=None)
    def mass(self, n):
        if n <= 1:
            return 1
        # Each small-prime factor adds 0.5 “mass units”
        return 1 + 0.5 * sum(1 for p in self.primes if n % p == 0)

# --- Z-Point in Discrete Spacetime ---
class ZPoint:
    def __init__(self, n, small_mass):
        self.n         = n
        self.modulus   = small_mass.modulus
        self.phase     = (n % self.modulus) / self.modulus * 2 * math.pi
        self.magnitude = math.log(n) if n > 1 else 0
        self.mass      = small_mass.mass(n)
        self.curvature = (self.mass * self.magnitude) / (math.e ** 2)
        self.resonance = ((n % self.magnitude) / math.e) * self.mass if self.magnitude else 0
        self.vector    = math.hypot(self.curvature, self.resonance)
        self.angle     = math.degrees(math.atan2(self.resonance, self.curvature))

# --- Geometric Filter with z1/z4 Transition Test ---
class GeometricFilter:
    def __init__(self, z1_stats, z4_stats, threshold=20000):
        self.z1_mean, self.z1_std = z1_stats
        self.z4_mean, self.z4_std = z4_stats
        self.phase_boundary       = threshold
        self.last_zp              = None
        self.distance_since_prime = 0
        self.filtered_count       = 0

    def passes(self, zp):
        # If we’ve wandered too far, clear the filter
        if self.distance_since_prime > self.phase_boundary or self.last_zp is None:
            return None  # signal “use oracle”
        gap = zp.n - self.last_zp.n
        if gap == 0:
            return None

        # compute transition scores
        z1 = (self.last_zp.vector / gap) * abs(self.last_zp.angle / 90.0)
        z4 = self.last_zp.curvature * (self.last_zp.vector / gap)

        # adaptive sigma
        σ = 1.3 + (math.e ** 2 / zp.mass)
        in_z1 = abs(z1 - self.z1_mean) <= σ * self.z1_std
        in_z4 = abs(z4 - self.z4_mean) <= σ * self.z4_std

        if in_z1 and in_z4:
            return True
        else:
            self.filtered_count += 1
            return False

    def record_prime(self, zp):
        self.last_zp = zp
        self.distance_since_prime = 0

    def record_composite(self):
        self.distance_since_prime += 1

# --- Main Prime Finder ---
def find_primes(limit=60000):
    small_mass  = SmallPrimeMass(primes=[2, 3, 5, 7, 11, 13, 17, 19, 23, 29])
    geo_filter  = GeometricFilter(z1_stats=(0.49, 0.56), z4_stats=(2.22, 2.09))
    primes      = []

    for n in range(2, limit + 1):
        zp = ZPoint(n, small_mass)

        decision = geo_filter.passes(zp)
        if decision is False:
            geo_filter.record_composite()
            continue

        # either decision is None (fallback) or True
        if miller_rabin(n):
            primes.append(n)
            geo_filter.record_prime(zp)
        else:
            geo_filter.record_composite()

    return primes, geo_filter.filtered_count

# --- Runner + Sanity Check + Prints ---
def main():
    target   = 6000
    limit    = 60000
    primes, fcount = find_primes(limit)

    print(f"✅ Total primes found up to {limit}: {len(primes)}")
    print(f"🔍  Filtered out (skipped) composites: {fcount}")
    if len(primes) >= target:
        p6000 = primes[target - 1]
        if p6000 == 59359:
            print(f"✔️  Sanity check passed: 6000th prime is {p6000}")
        else:
            print(f"❌  Sanity check failed: 6000th prime {p6000}, expected 59359")
    else:
        print(f"❌  Only {len(primes)} primes found; expected ≥ {target}")

if __name__ == '__main__':
    main()
