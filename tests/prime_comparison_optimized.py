import math
import time
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

    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1

    def is_composite(a):
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            return False
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                return False
        return True

    for a in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        if a % n == 0:
            return True
        if is_composite(a):
            return False
    return True

# --- Wheel Generator (mod 2310 = 2·3·5·7·11) ---
WHEEL_MOD = 2310
WHEEL = [r for r in range(1, WHEEL_MOD) if math.gcd(r, WHEEL_MOD) == 1]

def wheel_candidates(limit):
    yield from (2, 3, 5, 7, 11)
    for base in range(0, limit + 1, WHEEL_MOD):
        for r in WHEEL:
            n = base + r
            if n > limit:
                return
            if n >= 2:
                yield n

# --- Approximate Mass via Small Primes ---
class SmallPrimeMass:
    def __init__(self, primes):
        self.primes = primes
        self.modulus = primes[-1]

    @lru_cache(maxsize=None)
    def mass(self, n):
        if n <= 1:
            return 1.0
        return 1.0 + 0.5 * sum(1 for p in self.primes if n % p == 0)

# --- Geometric Filter with Z-metric ---
class GeometricFilter:
    def __init__(self, z1_stats, z4_stats, phase_threshold=20000):
        self.z1_mean, self.z1_std = z1_stats
        self.z4_mean, self.z4_std = z4_stats
        self.phase_boundary       = phase_threshold
        self.last_n               = None
        self.last_vector          = 0.0
        self.last_angle_factor    = 0.0
        self.last_curvature       = 0.0
        self.distance_since_prime = 0
        self.filtered_count       = 0

    def passes(self, n, curvature, resonance, vector):
        if self.distance_since_prime > self.phase_boundary or self.last_n is None:
            return None

        gap = n - self.last_n
        if gap == 0:
            return None

        z1 = (self.last_vector / gap) * self.last_angle_factor
        z4 = self.last_curvature * (self.last_vector / gap)

        sigma = 0.8 + (math.e**2 / curvature)  # was 1.3 + ...
        if (abs(z1 - self.z1_mean) <= sigma * self.z1_std and
            abs(z4 - self.z4_mean) <= sigma * self.z4_std):
            return True

        self.filtered_count += 1
        return False

    def record_prime(self, n, curvature, resonance, vector):
        self.last_n = n
        self.last_vector = vector
        self.last_angle_factor = min(1.0, abs(resonance / curvature)) if curvature else 0.0
        self.last_curvature = curvature
        self.distance_since_prime = 0

    def record_composite(self):
        self.distance_since_prime += 1

# --- Hybrid Prime Finder ---
def hybrid_find(limit):
    small_mass = SmallPrimeMass(primes=[2, 3, 5, 7, 11])
    geo_filter = GeometricFilter(z1_stats=(0.49, 0.56), z4_stats=(2.22, 2.09), phase_threshold=5000)

    primes = []
    mr_calls = 0

    for n in wheel_candidates(limit):
        logn = math.log(n)
        mass = small_mass.mass(n)
        curvature = (mass * logn) / (math.e**2)
        resonance = ((n % logn) / math.e) * mass
        vector = math.hypot(curvature, resonance)

        decision = geo_filter.passes(n, curvature, resonance, vector)
        if decision is False:
            geo_filter.record_composite()
            continue

        mr_calls += 1
        if miller_rabin(n):
            primes.append(n)
            geo_filter.record_prime(n, curvature, resonance, vector)
        else:
            geo_filter.record_composite()

    return primes, mr_calls, geo_filter.filtered_count

# --- Pure Miller–Rabin Prime Finder ---
def pure_mr_find(limit):
    primes = []
    calls = 0
    for n in range(2, limit + 1):
        calls += 1
        if miller_rabin(n):
            primes.append(n)
    return primes, calls

# --- Benchmark & Compare ---
if __name__ == "__main__":
    LIMITS = [100_000, 200_000, 500_000, 1_000_000]
    print(f"{'n':>12} | {'Pure MR (s)':>10} | {'Hybrid (s)':>10} | {'MR Calls':>9} | {'Skipped':>7} | {'Ratio':>6}")
    print("-" * 68)

    for LIM in LIMITS:
        t0 = time.time()
        _, mr_calls = pure_mr_find(LIM)
        t_mr = time.time() - t0

        t0 = time.time()
        _, hy_calls, hy_skipped = hybrid_find(LIM)
        t_hy = time.time() - t0

        ratio = t_mr / t_hy if t_hy > 0 else float("inf")
        print(f"{LIM:12,} | {t_mr:10.3f} | {t_hy:10.3f} | {mr_calls:9,} | {hy_skipped:7,} | {ratio:6.2f}")
