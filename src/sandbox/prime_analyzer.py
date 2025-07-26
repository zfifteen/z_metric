# prime_analyzer.py
#
# A high-performance prime number classifier and trajectory analyzer.
# This script integrates key learnings from previous versions:
#   1.  A centralized, easy-to-use configuration section.
#   2.  An efficient 'Circle Method' pre-filter tightly integrated with a
#       robust primality test for maximum performance.
#   3.  Modular functions for clarity and separation of concerns (classification vs. analysis).
#   4.  Clean, descriptive output and logging for better data interpretation.

import math
import csv
import time
from functools import lru_cache

# --- Configuration ---
# Set the desired number of primes to find. This is the main control variable.
TARGET_PRIME_COUNT = 10000

# Known prime values for dynamic sanity checks. Add more as needed.
SANITY_CHECKS = {
    100: 541,
    1000: 7919,
    6000: 59359,
    10000: 104729,
    100000: 1299709,
}

# --- Core Engine: Classification and Analysis ---

def is_prime_optimized(n):
    """
    Determines if a number is prime using a highly efficient, two-stage process.

    Stage 1: The Circle Filter
        - Handles edge cases (n <= 3).
        - Immediately rejects any number divisible by 2 or 3, which accounts
          for ~67% of all integers and over 70% of composites. This is the
          fastest way to discard the majority of non-prime candidates.

    Stage 2: Optimized Primality Test
        - If a number passes the circle filter, it must be of the form 6k ± 1.
        - This test only checks for divisors of the same form, drastically
          reducing the number of checks required compared to standard trial division.

    Returns:
        (bool): True if the number is prime, False otherwise.
    """
    # Stage 1: Circle Filter (and edge cases)
    if n <= 1: return False
    if n <= 3: return True
    if n % 2 == 0 or n % 3 == 0: return False

    # Stage 2: Optimized Primality Test (for numbers of the form 6k ± 1)
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

@lru_cache(maxsize=None)
def get_divisor_count(n):
    """
    Calculates the number of divisors for n (its 'Number Mass').
    Uses caching to avoid re-calculating for the same number.
    """
    if n <= 0: return 0
    if n == 1: return 1
    count, t = 1, n
    p = 2
    while p * p <= t:
        e = 0
        while t % p == 0:
            e += 1
            t //= p
        if e > 0:
            count *= (e + 1)
        p = p + 1 if p == 2 else p + 2
    if t > 1:
        count *= 2
    return count

def get_trajectory_analytics(p1, p2, p3):
    """
    Calculates the Z-metrics and other trajectory data between three primes.
    This function isolates the complex analysis from the main search loop.
    """
    # Calculate Z-metrics for each prime
    metrics = []
    for n in [p1, p2, p3]:
        m = get_divisor_count(n)  # Always 2 for primes
        gm = math.log(n)
        zc = (m * gm) / (math.e ** 2)
        rem = n % gm
        zr = (rem / math.e) * m
        metrics.append({
            'n': n,
            'z_curvature': zc,
            'z_resonance': zr,
            'z_vector_magnitude': math.hypot(zc, zr),
            'z_angle': math.degrees(math.atan2(zr, zc))
        })

    m1, m2, m3 = metrics[0], metrics[1], metrics[2]

    # Calculate relational metrics
    gap2 = m3['n'] - m2['n']
    gap1 = m2['n'] - m1['n']
    gap_ratio = gap2 / gap1 if gap1 else 0

    zv_ratio = m2['z_vector_magnitude'] / m1['z_vector_magnitude'] if m1['z_vector_magnitude'] else 0
    zc_ratio = m2['z_curvature'] / m1['z_curvature'] if m1['z_curvature'] else 0
    angle_diff = m2['z_angle'] - m1['z_angle']

    # Geometric analysis (Triangle Area in Z-space)
    x1, y1 = m1['z_curvature'], m1['z_resonance']
    x2, y2 = m2['z_curvature'], m2['z_resonance']
    x3, y3 = m3['z_curvature'], m3['z_resonance']
    triangle_area = 0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

    return {
        'prime_n': m3['n'], 'prime_n-1': m2['n'], 'prime_n-2': m1['n'],
        'gap_n': gap2, 'gap_n-1': gap1, 'gap_ratio': gap_ratio,
        'z_vec_mag_ratio': zv_ratio, 'z_curvature_ratio': zc_ratio,
        'z_angle_diff': angle_diff, 'z_triangle_area': triangle_area
    }


# --- Main Execution ---

if __name__ == '__main__':
    # --- Setup ---
    start_time = time.time()
    found_primes = []
    candidate = 2
    total_composites_found = 0
    composites_skipped_by_filter = 0

    # Output files
    classification_log_file = 'classification_log.csv'
    trajectory_log_file = 'prime_trajectory_log.csv'

    print(f"🚀 Initializing Prime Analyzer...")
    print(f"🎯 Target: Find {TARGET_PRIME_COUNT} prime numbers.")
    print("-" * 40)

    with open(classification_log_file, 'w', newline='') as cf, \
         open(trajectory_log_file, 'w', newline='') as tf:

        # Setup CSV writers and headers
        class_writer = csv.writer(cf)
        class_writer.writerow(['number', 'is_prime', 'was_filtered'])

        traj_writer = csv.DictWriter(tf, fieldnames=[
            'prime_n', 'prime_n-1', 'prime_n-2', 'gap_n', 'gap_n-1', 'gap_ratio',
            'z_vec_mag_ratio', 'z_curvature_ratio', 'z_angle_diff', 'z_triangle_area'
        ])
        traj_writer.writeheader()

        # --- Main Loop ---
        while len(found_primes) < TARGET_PRIME_COUNT:
            is_p = is_prime_optimized(candidate)
            was_filtered = not is_p and (candidate > 3 and (candidate % 2 == 0 or candidate % 3 == 0))

            class_writer.writerow([candidate, 1 if is_p else 0, 1 if was_filtered else 0])

            if is_p:
                found_primes.append(candidate)
                # Perform trajectory analysis once we have three consecutive primes
                if len(found_primes) >= 3:
                    p1, p2, p3 = found_primes[-3], found_primes[-2], found_primes[-1]
                    trajectory_data = get_trajectory_analytics(p1, p2, p3)
                    traj_writer.writerow(trajectory_data)
            else:
                total_composites_found += 1
                if was_filtered:
                    composites_skipped_by_filter += 1

            candidate += 1

    # --- Reporting ---
    end_time = time.time()
    elapsed_time = end_time - start_time
    total_numbers_checked = candidate - 1
    filter_efficiency = (composites_skipped_by_filter / total_composites_found * 100) if total_composites_found else 0

    print("\n" + "=" * 40)
    print("✅ Analysis Complete")
    print("=" * 40)
    print(f"⏱️  Time Elapsed: {elapsed_time:.2f} seconds")
    print(f"📈  Primes Found: {len(found_primes):,}")
    print(f"🔢  Last Prime Found: {found_primes[-1]:,}")
    print(f"🔍  Total Numbers Checked: {total_numbers_checked:,}")
    print(f"🗑️  Composites Filtered by Circle Method: {composites_skipped_by_filter:,} / {total_composites_found:,}")
    print(f"⚡  Filter Efficiency: {filter_efficiency:.2f}%")
    print("-" * 40)
    print(f"📊  Classification Log: '{classification_log_file}'")
    print(f"📉  Trajectory Log: '{trajectory_log_file}'")

    # --- Dynamic Sanity Check ---
    if TARGET_PRIME_COUNT in SANITY_CHECKS:
        expected_prime = SANITY_CHECKS[TARGET_PRIME_COUNT]
        actual_prime = found_primes[-1]
        if actual_prime == expected_prime:
            print(f"\n✅ Sanity Check Passed: The {TARGET_PRIME_COUNT:,}th prime found ({actual_prime:,}) matches the expected value.")
        else:
            print(f"\n❌ Sanity Check Failed: Found {actual_prime:,} as the {TARGET_PRIME_COUNT:,}th prime, but expected {expected_prime:,}.")
    else:
        print(f"\nℹ️ No sanity check value available for a target of {TARGET_PRIME_COUNT:,}.")

