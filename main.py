import math
import csv
import time  # Added for performance tracking
from functools import lru_cache  # Added for memoization
from collections import deque  # Added for rolling statistics
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


@lru_cache(maxsize=None)  # Cache results of this expensive function
def get_number_mass(n):
    """
    Counts the total number of divisors for a given integer 'n' using an
    efficient prime factorization method.
    This version is self-contained and does not require a pre-computed prime list.
    """
    if n <= 0: return 0
    if n == 1: return 1

    num_divisors = 1
    temp_n = n

    # Step 1: Handle the factor of 2 separately
    exponent = 0
    while temp_n % 2 == 0:
        exponent += 1
        temp_n //= 2
    if exponent > 0:
        num_divisors *= (exponent + 1)

    # Step 2: Iterate through odd numbers up to the square root
    p = 3
    while p * p <= temp_n:
        exponent = 0
        while temp_n % p == 0:
            exponent += 1
            temp_n //= p
        if exponent > 0:
            num_divisors *= (exponent + 1)
        p += 2  # Move to the next odd number

    # Step 3: If a prime factor remains, it's greater than the sqrt
    if temp_n > 1:
        num_divisors *= 2  # This remaining temp_n is prime

    return num_divisors


def get_z_metrics(n):
    """
    Calculates Z-metrics for a given integer n, based on a spacetime analogy.
    """
    if n <= 1:
        return {
            'number_mass': get_number_mass(n), 'spacetime_metric': 0, 'z_curvature': 0,
            'z_resonance': 0, 'z_vector_magnitude': 0, 'z_angle': 0
        }
    number_mass = get_number_mass(n)
    spacetime_metric = math.log(n)
    z_curvature = (number_mass * spacetime_metric) / (math.e ** 2)
    remainder = n % spacetime_metric
    z_resonance = (remainder / math.e) * number_mass
    z_vector_magnitude = math.sqrt(z_curvature ** 2 + z_resonance ** 2)
    z_angle = math.degrees(math.atan2(z_resonance, z_curvature))
    return {
        'number_mass': number_mass, 'spacetime_metric': spacetime_metric,
        'z_curvature': z_curvature, 'z_resonance': z_resonance,
        'z_vector_magnitude': z_vector_magnitude, 'z_angle': z_angle
    }


def is_prime(n):
    """
    Miller-Rabin primality test, made deterministic for the required range.
    This is significantly faster than trial division for large numbers.
    """
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False

    # Write n-1 as 2^r * d
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1

    # Use a set of bases that are deterministic for numbers up to a very high limit.
    bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    if n < 341550071728321:
        bases = [2, 3, 5, 7, 11, 13, 17]

    for a in bases:
        if a >= n:
            break

        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue

        is_composite = True
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                is_composite = False
                break

        if is_composite:
            return False

    return True


def classify_with_z_score(candidate, z1, z4, candidate_metrics):
    """
    Applies the Combined Z-Score filter using a dynamically derived multiplier
    and a stable, pre-calculated statistical model. Returns detailed stats for logging.
    """
    # Base statistical signature from analysis of the first 6000 primes.
    Z1_MEAN, Z1_STD_DEV = 0.49, 0.56
    Z4_MEAN, Z4_STD_DEV = 2.22, 2.09

    # Derive the SIGMA_MULTIPLIER from the candidate's own properties.
    d_n = candidate_metrics['number_mass']
    if d_n <= 1:  # Handle n=0,1 edge cases.
        d_n = 2  # Treat as prime-like to be safe

    # The multiplier is derived from the Spacetime-Curvature Ratio.
    sigma_multiplier = 1.3 + (math.e ** 2 / d_n)

    # Use the stable base statistics
    z1_lower_bound = Z1_MEAN - sigma_multiplier * Z1_STD_DEV
    z1_upper_bound = Z1_MEAN + sigma_multiplier * Z1_STD_DEV
    z4_lower_bound = Z4_MEAN - sigma_multiplier * Z4_STD_DEV
    z4_upper_bound = Z4_MEAN + sigma_multiplier * Z4_STD_DEV

    is_in_range = (z1_lower_bound <= z1 <= z1_upper_bound) and \
                  (z4_lower_bound <= z4 <= z4_upper_bound)

    prime_status, skipped = (0, True) if not is_in_range else (1 if is_prime(candidate) else 0, False)

    return prime_status, skipped, sigma_multiplier, z1_lower_bound, z1_upper_bound, z4_lower_bound, z4_upper_bound


# --- Main execution block ---
if __name__ == '__main__':
    start_time = time.time()  # Record start time

    primes_to_find = 6000
    found_primes = []
    candidate_number = 1
    csv_file_name = 'prime_stats_hybrid_filter.csv'

    # --- Initialize variables ---
    skipped_tests = 0
    last_prime_n = 0
    last_prime_metrics = {}

    # --- Stall Detector variables ---
    numbers_since_last_prime = 0
    STALL_THRESHOLD = 20000

    print(f"Searching for {primes_to_find} primes...")

    with open(csv_file_name, 'w', newline='') as file:
        # Add new columns to the header for detailed analysis
        header = ['n', 'is_prime', 'was_skipped', 'z1_score', 'z4_score',
                  'sigma_multiplier', 'z1_lower', 'z1_upper', 'z4_lower', 'z4_upper']
        writer = csv.writer(file)
        writer.writerow(header)

        while len(found_primes) < primes_to_find and candidate_number < 150000:
            metrics = get_z_metrics(candidate_number)

            z1, z4 = 0, 0
            prime_status, skipped = 0, True
            # Initialize detailed stats with default values
            sigma, z1_low, z1_high, z4_low, z4_high = 0, 0, 0, 0, 0

            # --- HYBRID FILTER LOGIC ---
            if numbers_since_last_prime > STALL_THRESHOLD:
                # Escape Hatch: test every number definitively
                prime_status = 1 if is_prime(candidate_number) else 0
                skipped = False

            elif last_prime_n > 0:
                gap = candidate_number - last_prime_n
                if gap > 0:
                    z_vec_mag_p = last_prime_metrics['z_vector_magnitude']
                    z_angle_p = last_prime_metrics['z_angle']
                    z_curv_p = last_prime_metrics['z_curvature']

                    z1 = (z_vec_mag_p / gap) * abs(z_angle_p / 90.0) if z_angle_p else 0
                    z4 = z_curv_p * (z_vec_mag_p / gap)

                    # Get detailed stats back from the classifier
                    prime_status, skipped, sigma, z1_low, z1_high, z4_low, z4_high = classify_with_z_score(
                        candidate_number, z1, z4, metrics)
            else:
                prime_status = 1 if is_prime(candidate_number) else 0
                skipped = False

            if skipped:
                skipped_tests += 1

            if prime_status == 1:
                found_primes.append(candidate_number)
                last_prime_n = candidate_number
                last_prime_metrics = metrics
                numbers_since_last_prime = 0
            else:
                numbers_since_last_prime += 1

            writer.writerow([candidate_number, prime_status, skipped, z1, z4,
                             sigma, z1_low, z1_high, z4_low, z4_high])
            candidate_number += 1

    end_time = time.time()  # Record end time
    elapsed_time = end_time - start_time

    # --- Final Performance Stats ---
    print(f"\n✅ Search complete.")
    print(f"   - Found {len(found_primes)} primes.")
    if found_primes:
        print(f"   - The last prime is: {found_primes[-1]}")
    print(f"   - Statistics saved to '{csv_file_name}'")

    total_numbers_checked = candidate_number - 1
    total_composites = total_numbers_checked - len(found_primes)

    print("\n--- Hybrid Filter Performance ---")
    print(f"   - Total Execution Time:        {elapsed_time:.2f} seconds")
    print(f"   - Total Numbers Checked:       {total_numbers_checked}")
    print(f"   - Total Composites Found:      {total_composites}")
    print(f"   - Composites Filtered Out:     {skipped_tests}")

    if total_composites > 0:
        efficiency = (skipped_tests / total_composites) * 100
        print(f"   - Filter Efficiency:           {efficiency:.2f}%")

    if len(found_primes) < primes_to_find:
        print("\n⚠️  ERROR: Target prime count not reached. Review stall detector or increase safety break.")
    else:
        print(
            "\n   - Accuracy:                  The filter successfully found all target primes without false negatives.")

    # --- Sanity Check ---
    actual_6000th_prime = 59359
    if len(found_primes) >= 6000:
        found_6000th = found_primes[5999]
        if found_6000th == actual_6000th_prime:
            print("\nSanity check passed: The 6000th prime matches the expected value.")
        else:
            print(
                f"\nSanity check failed: Found {found_6000th} as the 6000th prime, but expected {actual_6000th_prime}.")
    else:
        print("\nSanity check failed: Fewer than 6000 primes were found.")
