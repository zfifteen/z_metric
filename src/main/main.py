import math
import csv
import time
from functools import lru_cache
import numpy as np
from sklearn.mixture import GaussianMixture
from sympy import isprime as sympy_isprime  # For Mersenne checks, leveraging symbolic precision

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
            number_mass=get_number_mass(n),
            spacetime_metric=0,
            z_curvature=0,
            z_resonance=0,
            z_vector_magnitude=0,
            z_angle=0
        )
    m = get_number_mass(n)
    gm = math.log(n)
    zc = (m * gm) / (math.e ** 2)
    rem = n % gm
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

def is_prime(n):
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

def apply_vortex_filter(n):
    if n > 3 and (n % 2 == 0 or n % 3 == 0):
        return (0, True)
    is_p = is_prime(n)
    return (1, False) if is_p else (0, False)

# Integrated techniques: Frame shift residues for density clustering
phi = (1 + np.sqrt(5)) / 2

def frame_shift_residues(n_vals, k):
    mod_phi = np.mod(n_vals, phi) / phi
    return phi * np.power(mod_phi, k)

def bin_densities(theta_all, theta_pr, nbins=20):
    bins = np.linspace(0, phi, nbins + 1)
    all_counts, _ = np.histogram(theta_all, bins=bins)
    pr_counts, _ = np.histogram(theta_pr, bins=bins)
    all_d = all_counts / len(theta_all)
    pr_d = pr_counts / len(theta_pr)
    with np.errstate(divide='ignore', invalid='ignore'):
        enh = (pr_d - all_d) / all_d * 100
    enh = np.where(all_d > 0, enh, -np.inf)
    return all_d, pr_d, enh

def fourier_fit(theta_pr, M=5, nbins=100):
    x = (theta_pr % phi) / phi
    y, edges = np.histogram(theta_pr, bins=nbins, density=True)
    centers = (edges[:-1] + edges[1:]) / 2 / phi
    def design(x):
        cols = [np.ones_like(x)]
        for k in range(1, M + 1):
            cols.append(np.cos(2 * np.pi * k * x))
            cols.append(np.sin(2 * np.pi * k * x))
        return np.vstack(cols).T
    A = design(centers)
    coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
    a = coeffs[0::2]
    b = coeffs[1::2]
    return a, b

def gmm_fit(theta_pr, n_components=5):
    X = ((theta_pr % phi) / phi).reshape(-1, 1)
    gm = GaussianMixture(n_components=n_components,
                         covariance_type='full',
                         random_state=0).fit(X)
    sigmas = np.sqrt([gm.covariances_[i].flatten()[0]
                      for i in range(n_components)])
    return gm, np.mean(sigmas)

def compute_mersenne_primes(primes):
    return [p for p in primes if sympy_isprime(2 ** p - 1)]

def statistical_summary(primes, mersenne_primes):
    total_primes = len(primes)
    total_mersenne = len(mersenne_primes)
    hit_rate = (total_mersenne / total_primes) * 100 if total_primes > 0 else 0
    miss_rate = 100 - hit_rate

    print("\n=== Statistical Summary ===")
    print(f"Total Primes Checked: {total_primes}")
    print(f"Total Mersenne Primes Found: {total_mersenne}")
    print(f"Hit Rate: {hit_rate:.2f}%")
    print(f"Miss Rate: {miss_rate:.2f}%")

    prime_array = np.array(primes)
    print("\nPrime Distribution Statistics:")
    print(f"Mean of Primes: {np.mean(prime_array):.2f}")
    print(f"Median of Primes: {np.median(prime_array):.2f}")
    print(f"Standard Deviation of Primes: {np.std(prime_array):.2f}")

    if mersenne_primes:
        mersenne_values = [(1 << p) - 1 for p in mersenne_primes]
        print("\nMersenne Prime Growth:")
        print(f"Smallest Mersenne Prime: {min(mersenne_values)}")
        print(f"Largest Mersenne Prime: {max(mersenne_values)}")
        print(f"Mersenne Growth Factor: {max(mersenne_values) / min(mersenne_values):.2f}")

if __name__ == '__main__':
    # --- Configuration ---
    TARGET = 100

    SANITY_CHECKS = {
        6000: 59359,
        10000: 104729,
        100000: 1299709,
        600000: 7980000
    }

    # --- Initialization ---
    start      = time.time()
    found      = []
    candidate  = 2
    stats_csv  = 'prime_stats_vortex_filter.csv'
    traj_csv   = 'prime_trajectory_stats.csv'
    last_prime = None
    skipped    = 0
    history    = []

    print(f"🔍 Searching for {TARGET} primes with the Vortex Method filter…")

    with open(stats_csv, 'w', newline='') as sf, \
            open(traj_csv,  'w', newline='') as tf:

        stats_w = csv.writer(sf)
        stats_w.writerow(['n', 'is_prime', 'was_skipped'])

        traj_w = csv.writer(tf)
        traj_w.writerow([
            'prime_n','prime_n-1','prime_n-2',
            'gap_n','gap_n-1','gap_ratio',
            'z_vec_mag_ratio','z_curvature_ratio',
            'z_angle_diff','z_triangle_area',
            'Z_trajectory_score'
        ])

        while len(found) < TARGET:
            is_p, was_skipped = apply_vortex_filter(candidate)

            if was_skipped:
                skipped += 1

            if is_p:
                found.append(candidate)
                metrics = get_z_metrics(candidate)

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

            stats_w.writerow([candidate, is_p, 1 if was_skipped else 0])
            candidate += 1

    # --- Results ---
    elapsed    = time.time() - start
    total      = candidate - 1
    composites = total - len(found)
    eff        = (skipped / composites * 100) if composites else 0.0

    print(f"\n✅ Done in {elapsed:.2f}s — found {len(found)} primes.")
    print(f"Last: {found[-1] if found else None}")
    print(f"Stats → {stats_csv}")
    print(f"Traj  → {traj_csv}")
    print(f"Checked: {total}, filtered: {skipped}, efficiency: {eff:.2f}%")

    if TARGET in SANITY_CHECKS:
        actual_prime = SANITY_CHECKS[TARGET]
        if len(found) >= TARGET:
            found_prime = found[TARGET - 1]
            if found_prime == actual_prime:
                print(f"\n✅ Sanity check passed: The {TARGET}th prime found ({found_prime}) matches the expected value.")
            else:
                print(f"\n❌ Sanity check failed: Found {found_prime} as the {TARGET}th prime, but expected {actual_prime}.")
        else:
            print(f"\n❌ Sanity check failed: Fewer than {TARGET} primes were found.")
    else:
        print(f"\nℹ️ No sanity check value available for a target of {TARGET}.")

    # --- Integrated Curvature Analysis ---
    primes_list = found
    k_values = np.arange(0.2, 0.4001, 0.002)
    results = []

    for k in k_values:
        theta_all = frame_shift_residues(np.arange(1, last_prime + 1), k)
        theta_pr = frame_shift_residues(np.array(primes_list), k)
        all_d, pr_d, enh = bin_densities(theta_all, theta_pr, nbins=20)
        max_enh = np.max(enh)
        _, sigma_prime = gmm_fit(theta_pr, n_components=5)
        _, b_coeffs = fourier_fit(theta_pr, M=5)
        sum_b = np.sum(np.abs(b_coeffs))
        results.append({
            'k': k,
            'max_enhancement': max_enh,
            'sigma_prime': sigma_prime,
            'fourier_b_sum': sum_b
        })

    valid_results = [r for r in results if np.isfinite(r['max_enhancement'])]
    best = max(valid_results, key=lambda r: r['max_enhancement'])
    k_star, enh_star = best['k'], best['max_enhancement']

    print("\n=== Refined Prime Curvature Analysis Results ===")
    print(f"Optimal curvature exponent k* = {k_star:.3f}")
    print(f"Max mid-bin enhancement = {enh_star:.1f}%")
    print(f"GMM σ' at k* = {best['sigma_prime']:.3f}")
    print(f"Σ|b_k| at k* = {best['fourier_b_sum']:.3f}\n")

    print("Sample of k-sweep metrics (every 10th k):")
    for entry in valid_results[::10]:
        print(f" k={entry['k']:.3f} | enh={entry['max_enhancement']:.1f}%"
              f" | σ'={entry['sigma_prime']:.3f}"
              f" | Σ|b|={entry['fourier_b_sum']:.3f}")

    # --- Mersenne Prime Validation ---
    mersenne_primes = compute_mersenne_primes(primes_list)
    print("\nValidated Mersenne Prime Exponents:")
    print(", ".join(map(str, mersenne_primes)))

    statistical_summary(primes_list, mersenne_primes)