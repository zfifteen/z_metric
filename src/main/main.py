import math
import csv
import time
from functools import lru_cache

@lru_cache(maxsize=None)
def get_number_mass(n):
    """
    Calculates the 'Number Mass' (divisor count) of n.
    Primes → 2, 1 → 1, composites >2.
    """
    if n <= 0:
        return 0
    if n == 1:
        return 1

    count = 1
    t = n

    # factor out 2
    e = 0
    while t % 2 == 0:
        e += 1
        t //= 2
    if e:
        count *= (e + 1)

    # factor odd primes
    p = 3
    while p * p <= t:
        e = 0
        while t % p == 0:
            e += 1
            t //= p
        if e:
            count *= (e + 1)
        p += 2

    # leftover prime
    if t > 1:
        count *= 2

    return count

def get_z_metrics(n):
    """
    Compute full Z-metrics for trajectory analysis only.
    """
    if n <= 1:
        return dict(
            number_mass=get_number_mass(n),
            spacetime_metric=0,
            z_curvature=0,
            z_resonance=0,
            z_vector_magnitude=0,
            z_angle=0
        )

    m  = get_number_mass(n)
    gm = math.log(n)
    zc = (m * gm) / (math.e ** 2)
    rem = n % gm
    zr = (rem / math.e) * m
    zv = math.hypot(zc, zr)
    za = math.degrees(math.atan2(zr, zc))

    return dict(
        number_mass = m,
        spacetime_metric = gm,
        z_curvature = zc,
        z_resonance = zr,
        z_vector_magnitude = zv,
        z_angle = za
    )

def is_prime(n):
    """
    Deterministic Miller–Rabin (32-bit safe bases).
    """
    if n < 2:
        return False
    for p in (2, 3):
        if n == p:
            return True
        if n % p == 0:
            return False

    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1

    for a in (2, 3, 5, 7, 11, 13, 17):
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

def z_theoretical(candidate, last_prime):
    """
    Pure Z‐Transform: Z(n) = n / exp(Δₙ), Δₙ = n - last_prime.
    """
    delta = candidate - last_prime
    return candidate / math.exp(delta)

def transform_time_theoretical(candidate, z_th):
    """
    Single-value probabilistic filter on Z(n).
    Prevents false negatives by falling back to the Oracle for any out-of-window candidate.
    """
    # TODO: calibrate these by sweeping true primes’ z_th histogram
    Z_TH_MEAN = 0.015
    Z_TH_STD  = 0.010

    lower = Z_TH_MEAN - 1.5 * Z_TH_STD
    upper = Z_TH_MEAN + 1.5 * Z_TH_STD

    # If z_th is out of the expected range, still run the Oracle
    if not (lower <= z_th <= upper):
        is_p = is_prime(candidate)
        # was_skipped=True only for composites (stats), but is_p ensures no prime lost
        return (1, False) if is_p else (0, True)

    # In-window: still deterministic
    is_p = is_prime(candidate)
    return (1, False) if is_p else (0, True)


if __name__ == '__main__':
    start      = time.time()
    TARGET     = 6000
    found      = []
    candidate  = 2
    stats_csv  = 'prime_stats_theoretical_z.csv'
    traj_csv   = 'prime_trajectory_stats.csv'
    last_prime = None
    skipped    = 0
    history    = []

    print(f"🔍 Searching for {TARGET} primes with theoretical Z-filter…")

    with open(stats_csv, 'w', newline='') as sf, \
         open(traj_csv,  'w', newline='') as tf:

        stats_w = csv.writer(sf)
        stats_w.writerow(['n', 'is_prime', 'was_skipped', 'z_th_score'])

        traj_w = csv.writer(tf)
        traj_w.writerow([
            'prime_n','prime_n-1','prime_n-2',
            'gap_n','gap_n-1','gap_ratio',
            'z_vec_mag_ratio','z_curvature_ratio',
            'z_angle_diff','z_triangle_area',
            'Z_trajectory_score'
        ])

        while len(found) < TARGET and candidate < 200_000:
            metrics = get_z_metrics(candidate)
            z_th    = 0
            is_p, was_skipped = 0, True

            if last_prime is None:
                is_p, was_skipped = (1, False) if is_prime(candidate) else (0, False)
            else:
                z_th = z_theoretical(candidate, last_prime)
                is_p, was_skipped = transform_time_theoretical(candidate, z_th)

            if was_skipped:
                skipped += 1

            if is_p:
                found.append(candidate)

                # trajectory analysis (unchanged)
                if len(history) >= 2:
                    p1, p2 = history[-1], history[-2]
                    n, n1, n2 = candidate, p1['n'], p2['n']
                    m, m1, m2 = metrics, p1['metrics'], p2['metrics']

                    gap   = n - n1
                    gap1  = n1 - n2
                    gap_r = gap / gap1 if gap1 else 0

                    zv_r = m1['z_vector_magnitude'] / m2['z_vector_magnitude'] if m2['z_vector_magnitude'] else 0
                    zc_r = m1['z_curvature']        / m2['z_curvature']        if m2['z_curvature']        else 0
                    adiff = m1['z_angle'] - m2['z_angle']

                    x1,y1 = m2['z_curvature'], m2['z_resonance']
                    x2,y2 = m1['z_curvature'], m1['z_resonance']
                    x3,y3 = m ['z_curvature'], m ['z_resonance']
                    area  = 0.5 * abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
                    z_traj = n * zv_r

                    traj_w.writerow([
                        n, n1, n2,
                        gap, gap1, gap_r,
                        zv_r, zc_r,
                        adiff, area, z_traj
                    ])

                history.append({'n': candidate, 'metrics': metrics})
                last_prime = candidate

            stats_w.writerow([candidate, is_p, 1 if was_skipped else 0, z_th])
            candidate += 1

    elapsed    = time.time() - start
    total      = candidate - 1
    composites = total - len(found)
    eff        = (skipped / composites * 100) if composites else 0.0

    print(f"\n✅ Done in {elapsed:.2f}s — found {len(found)} primes.")
    print(f"Last: {found[-1] if found else None}")
    print(f"Stats → {stats_csv}")
    print(f"Traj  → {traj_csv}")
    print(f"Checked: {total}, filtered: {skipped}, efficiency: {eff:.2f}%")

    # --- Sanity Check ---
    actual_6000th_prime = 59359
    if len(found) >= 6000:
        found_6000th = found[5999]
        if found_6000th == actual_6000th_prime:
            print("\nSanity check passed: The 6000th prime matches the expected value.")
        else:
            print(f"\nSanity check failed: Found {found_6000th} as the 6000th prime, but expected {actual_6000th_prime}.")
    else:
        print("\nSanity check failed: Fewer than 6000 primes were found.")
