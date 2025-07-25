# main.py

import math
import csv
import time
from functools import lru_cache

@lru_cache(maxsize=None)
def get_number_mass(n):
    if n <= 0: return 0
    if n == 1: return 1
    cnt, t = 1, n
    # factor 2
    e = 0
    while t % 2 == 0:
        e += 1; t //= 2
    if e: cnt *= (e + 1)
    # odd factors
    p = 3
    while p * p <= t:
        e = 0
        while t % p == 0:
            e += 1; t //= p
        if e: cnt *= (e + 1)
        p += 2
    if t > 1:
        cnt *= 2
    return cnt

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
    m  = get_number_mass(n)
    sm = math.log(n)
    zc = (m * sm) / math.e**2
    zr = (n % sm) * m / math.e
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
    if n < 2: return False
    if n in (2, 3): return True
    if n % 2 == 0 or n % 3 == 0: return False
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2; r += 1
    for a in (2, 3, 5, 7, 11, 13, 17):
        if a >= n: break
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

def classify_with_z_score(cand, z1, z4, mets,
                          mean_ln, std_ln, count_ln):
    if count_ln < 2:
        return (1, False) if is_prime(cand) else (0, False)

    # analytic constants
    Z1_MEAN, Z1_STD = 0.27, 0.27 / math.sqrt(mean_ln)
    Z4_MEAN, Z4_STD = 0.124 * mean_ln, 0.124 * std_ln

    d = max(mets['number_mass'], 2)
    sigma = 1.0 + (math.e ** 2) / (d ** 0.8)

    if not (Z1_MEAN - sigma*Z1_STD <= z1 <= Z1_MEAN + sigma*Z1_STD
            and Z4_MEAN - sigma*Z4_STD <= z4 <= Z4_MEAN + sigma*Z4_STD):
        return 0, True

    return (1, False) if is_prime(cand) else (0, False)

def main():
    start = time.time()

    target_primes = 6000
    found = []
    history = []
    skipped_tests = 0
    distance_from_last = 0
    PHASE_BOUNDARY = 20000
    n = 1

    # Running Welford for ln(prime)
    count_ln = 0
    mean_ln  = 0.0
    m2_ln    = 0.0

    with open('prime_stats_hybrid_filter.csv', 'w', newline='') as f, \
         open('prime_trajectory_stats.csv', 'w', newline='') as tf:

        w  = csv.writer(f)
        tw = csv.writer(tf)

        w.writerow(['n','is_prime','was_skipped','z1_score','z4_score'])
        tw.writerow([
            'prime_n','prime_n-1','prime_n-2',
            'gap_n','gap_n-1','gap_ratio',
            'z_vec_mag_ratio','z_curvature_ratio','z_angle_diff',
            'z_triangle_area','Z_trajectory_score'
        ])

        print(f"🔍 Starting search for {target_primes} primes...")

        while len(found) < target_primes and n < 150_000:
            mets = get_z_metrics(n)
            z1 = z4 = 0.0
            status, skipped = 0, True
            last = history[-1] if history else None

            if distance_from_last > PHASE_BOUNDARY:
                status, skipped = (1, False) if is_prime(n) else (0, False)
            elif last:
                gap = n - last['n']
                if gap > 0:
                    zv_p = last['metrics']['z_vector_magnitude']
                    za_p = last['metrics']['z_angle']
                    zc_p = last['metrics']['z_curvature']
                    z1   = (zv_p / gap) * abs(za_p / 90.0) if za_p else 0
                    z4   = zc_p * (zv_p / gap)
                    status, skipped = classify_with_z_score(
                        n, z1, z4, mets,
                        mean_ln,
                        math.sqrt(m2_ln/count_ln) if count_ln>0 else 0.0,
                        count_ln
                    )
            else:
                status, skipped = (1, False) if is_prime(n) else (0, False)

            if skipped:
                skipped_tests += 1

            if status == 1:
                found.append(n)

                # Welford update for ln(n)
                ln_n = math.log(n)
                count_ln += 1
                delta = ln_n - mean_ln
                mean_ln += delta / count_ln
                m2_ln   += delta * (ln_n - mean_ln)

                # trajectory stats
                if len(history) >= 2:
                    p2 = history[-2]
                    p1 = history[-1]
                    gap_n  = n - p1['n']
                    gap_n1 = p1['n'] - p2['n']
                    gap_ratio = gap_n / gap_n1 if gap_n1 else 0

                    zv1 = p1['metrics']['z_vector_magnitude']
                    zv2 = p2['metrics']['z_vector_magnitude']
                    zc1 = p1['metrics']['z_curvature']
                    zc2 = p2['metrics']['z_curvature']
                    z_vec_mag_ratio   = zv1 / zv2 if zv2 else 0
                    z_curvature_ratio = zc1 / zc2 if zc2 else 0
                    z_angle_diff      = p1['metrics']['z_angle'] - p2['metrics']['z_angle']

                    x1, y1 = p2['metrics']['z_curvature'], p2['metrics']['z_resonance']
                    x2, y2 = p1['metrics']['z_curvature'], p1['metrics']['z_resonance']
                    x3, y3 = mets['z_curvature'],          mets['z_resonance']
                    area = 0.5 * abs(
                        x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
                    )
                    Z_traj_score = n * z_vec_mag_ratio

                    tw.writerow([
                        n, p1['n'], p2['n'],
                        gap_n, gap_n1, gap_ratio,
                        z_vec_mag_ratio, z_curvature_ratio, z_angle_diff,
                        area, Z_traj_score
                    ])

                history.append({'n': n, 'metrics': mets})
                distance_from_last = 0
            else:
                distance_from_last += 1

            w.writerow([n, status, int(skipped), z1, z4])
            n += 1

            if n % 10000 == 0:
                print(f"  … reached n={n:,}, found {len(found)} primes so far")

    elapsed = time.time() - start
    total_checked = n - 1
    total_composite = total_checked - len(found)
    efficiency = 100 * skipped_tests / total_composite if total_composite else 0

    print(f"\n✅ Search finished:")
    print(f"    • Primes found:         {len(found)} / {target_primes}")
    print(f"    • Last n checked:       {total_checked}")
    print(f"    • Composites filtered:  {skipped_tests}/{total_composite} ({efficiency:.2f}% eff.)")
    print(f"    • Elapsed time:         {elapsed:.2f}s")

    # Sanity check for the 6000th prime
    if len(found) >= 6000:
        if found[5999] == 59359:
            print("✔️  Sanity check passed: 6000th prime is 59359.")
        else:
            print(f"❌  Sanity check failed: 6000th prime = {found[5999]}, expected 59359.")

if __name__ == '__main__':
    main()
