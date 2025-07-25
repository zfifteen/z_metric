# main.py

import math
import csv
import time
from functools import lru_cache

@lru_cache(maxsize=None)
def get_number_mass(n):
    if n <= 0:
        return 0
    if n == 1:
        return 1

    num_divisors = 1
    temp = n

    # factor out 2's
    exp = 0
    while temp % 2 == 0:
        exp += 1
        temp //= 2
    if exp:
        num_divisors *= (exp + 1)

    # factor out odd primes
    p = 3
    while p * p <= temp:
        exp = 0
        while temp % p == 0:
            exp += 1
            temp //= p
        if exp:
            num_divisors *= (exp + 1)
        p += 2

    if temp > 1:
        num_divisors *= 2

    return num_divisors


def get_z_metrics(n):
    if n <= 1:
        return {
            'number_mass': get_number_mass(n),
            'spacetime_metric': 0,
            'z_curvature': 0,
            'z_resonance': 0,
            'z_vector_magnitude': 0,
            'z_angle': 0
        }

    m = get_number_mass(n)
    sm = math.log(n)
    zc = (m * sm) / (math.e ** 2)
    r = n % sm
    zr = (r / math.e) * m
    zv = math.hypot(zc, zr)
    za = math.degrees(math.atan2(zr, zc))

    return {
        'number_mass': m,
        'spacetime_metric': sm,
        'z_curvature': zc,
        'z_resonance': zr,
        'z_vector_magnitude': zv,
        'z_angle': za
    }


def is_prime(n):
    if n <  2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0 or n % 3 == 0:
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
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def classify_with_z_score(candidate, z1, z4, candidate_metrics, prime_history):
    """
    Derive Z1_MEAN, Z1_STD_DEV, Z4_MEAN, Z4_STD_DEV from
    the current prime_history via the analytic model.
    """
    # require at least two primes for a stable estimate
    if len(prime_history) < 2:
        # fall back to deterministic check if not enough history
        return (1, False) if is_prime(candidate) else (0, False)

    # build list of ln(p) for all found primes
    ln_primes = [math.log(p['n']) for p in prime_history]
    mean_ln = sum(ln_primes) / len(ln_primes)
    std_ln  = math.sqrt(sum((x - mean_ln) ** 2 for x in ln_primes) / len(ln_primes))

    # analytic constants from derivation
    Z1_MEAN     = 0.27
    Z1_STD_DEV  = 0.27 / math.sqrt(mean_ln)
    Z4_MEAN     = 0.124 * mean_ln
    Z4_STD_DEV  = 0.124 * std_ln

    # adaptive sigma multiplier (stronger shrink for large d_n)
    d_n = candidate_metrics['number_mass']
    if d_n <= 1:
        d_n = 2
    sigma_multiplier = 1.0 + (math.e ** 2) / (d_n ** 0.8)

    # observational window
    z1_lo = Z1_MEAN - sigma_multiplier * Z1_STD_DEV
    z1_hi = Z1_MEAN + sigma_multiplier * Z1_STD_DEV
    z4_lo = Z4_MEAN - sigma_multiplier * Z4_STD_DEV
    z4_hi = Z4_MEAN + sigma_multiplier * Z4_STD_DEV

    if not (z1_lo <= z1 <= z1_hi and z4_lo <= z4 <= z4_hi):
        return 0, True

    # if inside the window, collapse with deterministic prime check
    return (1, False) if is_prime(candidate) else (0, False)


if __name__ == '__main__':
    start = time.time()

    primes_to_find = 500
    found_primes   = []
    prime_history  = []
    skipped_tests  = 0
    candidate      = 1
    distance_from_last = 0
    PHASE_BOUNDARY = 20_000

    csv_name       = 'prime_stats_hybrid_filter.csv'
    traj_name      = 'prime_trajectory_stats.csv'

    with open(csv_name, 'w', newline='') as f, \
         open(traj_name, 'w', newline='') as tf:

        writer = csv.writer(f)
        writer.writerow(['n','is_prime','was_skipped','z1_score','z4_score'])

        traj_writer = csv.writer(tf)
        traj_writer.writerow([
            'prime_n','prime_n-1','prime_n-2',
            'gap_n','gap_n-1','gap_ratio',
            'z_vec_mag_ratio','z_curvature_ratio','z_angle_diff',
            'z_triangle_area','Z_trajectory_score'
        ])

        print(f"Searching for {primes_to_find} primes...")

        while len(found_primes) < primes_to_find and candidate < 150_000:
            m   = get_z_metrics(candidate)
            z1  = z4 = 0
            status, skipped = 0, True

            last = prime_history[-1] if prime_history else None

            # hybrid filter logic
            if distance_from_last > PHASE_BOUNDARY:
                status, skipped = (1, False) if is_prime(candidate) else (0, False)

            elif last:
                gap = candidate - last['n']
                if gap > 0:
                    zv_p   = last['metrics']['z_vector_magnitude']
                    za_p   = last['metrics']['z_angle']
                    zc_p   = last['metrics']['z_curvature']
                    z1     = (zv_p / gap) * abs(za_p / 90.0) if za_p else 0
                    z4     = zc_p * (zv_p / gap)
                    status, skipped = classify_with_z_score(candidate, z1, z4, m, prime_history)

            else:
                status, skipped = (1, False) if is_prime(candidate) else (0, False)

            if skipped:
                skipped_tests += 1

            # if prime, record and maybe write trajectory row
            if status == 1:
                found_primes.append(candidate)

                if len(prime_history) >= 2:
                    p_n2 = prime_history[-2]
                    p_n1 = prime_history[-1]
                    p_n  = {'n': candidate, 'metrics': m}

                    # gaps & ratios
                    gap_n  = p_n['n']  - p_n1['n']
                    gap_n1 = p_n1['n'] - p_n2['n']
                    gap_ratio = gap_n / gap_n1 if gap_n1 else 0

                    # metric ratios & angle diff
                    zm_n1 = p_n1['metrics']['z_vector_magnitude']
                    zm_n2 = p_n2['metrics']['z_vector_magnitude']
                    zc_n1 = p_n1['metrics']['z_curvature']
                    zc_n2 = p_n2['metrics']['z_curvature']
                    z_vec_mag_ratio   = zm_n1 / zm_n2 if zm_n2 else 0
                    z_curvature_ratio = zc_n1 / zc_n2 if zc_n2 else 0
                    z_angle_diff      = p_n1['metrics']['z_angle'] - p_n2['metrics']['z_angle']

                    # Z-triangle area
                    x1,y1 = p_n2['metrics']['z_curvature'], p_n2['metrics']['z_resonance']
                    x2,y2 = p_n1['metrics']['z_curvature'], p_n1['metrics']['z_resonance']
                    x3,y3 = m['z_curvature'],          m['z_resonance']
                    area = 0.5 * abs(
                        x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)
                    )

                    Z_traj_score = p_n['n'] * z_vec_mag_ratio

                    traj_writer.writerow([
                        p_n['n'], p_n1['n'], p_n2['n'],
                        gap_n, gap_n1, gap_ratio,
                        z_vec_mag_ratio, z_curvature_ratio, z_angle_diff,
                        area, Z_traj_score
                    ])

                prime_history.append({'n': candidate, 'metrics': m})
                distance_from_last = 0

            else:
                distance_from_last += 1

            writer.writerow([candidate, status, int(skipped), z1, z4])
            candidate += 1

    total_nums  = candidate - 1
    total_comp  = total_nums - len(found_primes)
    elapsed     = time.time() - start

    print(f"\n✅ Found {len(found_primes)} primes up to {total_nums}.")
    print(f"Composites filtered out: {skipped_tests}/{total_comp} ({100*skipped_tests/total_comp:.2f}% efficiency)")
    print(f"Elapsed time: {elapsed:.2f}s")
