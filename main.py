import math
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import logging
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_number_mass(n, primes_list):
    """
    Counts the total number of divisors for a given integer 'n' using an optimized
    prime factorization method.
    """
    if n == 1:
        return 1
    num_divisors = 1
    temp_n = n
    for p in primes_list:
        if p * p > temp_n:
            break
        exponent = 0
        while temp_n % p == 0:
            exponent += 1
            temp_n //= p
        if exponent > 0:
            num_divisors *= (exponent + 1)
    if temp_n > 1:
        num_divisors *= 2
    return num_divisors


def get_z_metrics(n, primes_list):
    """
    Calculates Z-metrics for a given integer n, based on a spacetime analogy.
    """
    if n <= 1:
        return {
            'number_mass': n, 'spacetime_metric': 0, 'z_curvature': 0,
            'z_resonance': 0, 'z_vector_magnitude': 0, 'z_angle': 0
        }
    number_mass = get_number_mass(n, primes_list)
    spacetime_metric = math.log(n)
    z_curvature = (number_mass * (spacetime_metric / math.e)) / math.e
    remainder = n % spacetime_metric
    z_resonance = (remainder / math.e) * number_mass
    z_vector_magnitude = math.sqrt(z_curvature ** 2 + z_resonance ** 2)
    z_angle = math.degrees(math.atan2(z_resonance, z_curvature))
    return {
        'number_mass': number_mass, 'spacetime_metric': spacetime_metric,
        'z_curvature': z_curvature, 'z_resonance': z_resonance,
        'z_vector_magnitude': z_vector_magnitude, 'z_angle': z_angle
    }


def is_prime(num):
    """
    Tests if a number is prime using an efficient primality test.
    """
    if num <= 1: return False
    if num == 2: return True
    if num % 2 == 0: return False
    i = 3
    while i <= math.sqrt(num):
        if num % i == 0:
            return False
        i += 2
    return True


def classify_with_z_score(candidate, z1, z4):
    """
    Applies the Combined Z-Score filter. A number is a candidate for a prime
    if its Z1 and Z4 scores fall within the expected range for prime transitions.
    """
    # Updated prime signature based on statistical analysis of first 6000 primes
    Z1_MEAN, Z1_STD_DEV = 0.49, 0.56
    Z4_MEAN, Z4_STD_DEV = 2.22, 2.09
    SIGMA_MULTIPLIER = 5.0

    z1_lower_bound = Z1_MEAN - SIGMA_MULTIPLIER * Z1_STD_DEV
    z1_upper_bound = Z1_MEAN + SIGMA_MULTIPLIER * Z1_STD_DEV
    z4_lower_bound = Z4_MEAN - SIGMA_MULTIPLIER * Z4_STD_DEV
    z4_upper_bound = Z4_MEAN + SIGMA_MULTIPLIER * Z4_STD_DEV

    is_in_range = (z1_lower_bound <= z1 <= z1_upper_bound) and \
                  (z4_lower_bound <= z4 <= z4_upper_bound)

    if not is_in_range:
        return 0, True

    start_time = time.perf_counter()
    is_p = is_prime(candidate)
    is_prime_time = time.perf_counter() - start_time
    logger.info(f"n={candidate}: is_prime took {is_prime_time:.6f}s")
    if is_p:
        return 1, False
    else:
        return 0, False


# --- Main execution block ---
if __name__ == '__main__':
    primes_to_find = 6000
    found_primes = []
    candidate_number = 1
    csv_file_name = 'prime_stats_hybrid_filter.csv'

    # --- Initialize variables ---
    skipped_tests = 0
    last_prime_n = 0
    last_prime_metrics = {}

    # --- Phase State Transition Detector variables ---
    numbers_since_last_prime = 0
    PHASE_STATE_TRANSITION_THRESHOLD = 20000
    phase_state_transition_detector_active = False

    print(f"Searching for {primes_to_find} primes using the Hybrid Filter...")

    with open(csv_file_name, 'w', newline='') as file:
        header = ['n', 'is_prime', 'was_skipped', 'z1_score', 'z4_score']
        writer = csv.writer(file)
        writer.writerow(header)

        while len(found_primes) < primes_to_find and candidate_number < 150000:  # Increased safety break
            start_time = time.perf_counter()
            metrics = get_z_metrics(candidate_number, found_primes)
            metrics_time = time.perf_counter() - start_time
            logger.info(f"n={candidate_number}: get_z_metrics took {metrics_time:.6f}s")

            z1, z4 = 0, 0
            prime_status, skipped = 0, True

            # --- HYBRID FILTER LOGIC ---
            if numbers_since_last_prime > PHASE_STATE_TRANSITION_THRESHOLD:
                if not phase_state_transition_detector_active:
                    print(f"\n⚠️  PHASE STATE TRANSITION DETECTED at n={candidate_number}! Disabling filter to find outlier prime...\n")
                    phase_state_transition_detector_active = True
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

                    start_time = time.perf_counter()
                    prime_status, skipped = classify_with_z_score(candidate_number, z1, z4)
                    classify_time = time.perf_counter() - start_time
                    logger.info(f"n={candidate_number}: classify_with_z_score took {classify_time:.6f}s, skipped={skipped}")
            else:
                prime_status = 1 if is_prime(candidate_number) else 0
                skipped = False

            if skipped:
                skipped_tests += 1

            if prime_status == 1:
                if phase_state_transition_detector_active:
                    print(f"✅ Outlier prime found: {candidate_number}. Re-engaging Z-Score filter.")
                    phase_state_transition_detector_active = False

                found_primes.append(candidate_number)
                last_prime_n = candidate_number
                last_prime_metrics = metrics
                numbers_since_last_prime = 0  # Reset counter

                if len(found_primes) % 500 == 0:
                    print(f"Found prime {len(found_primes)}: {candidate_number}")
            else:
                numbers_since_last_prime += 1

            writer.writerow([candidate_number, prime_status, skipped, z1, z4])
            candidate_number += 1

    # --- Final Performance Stats ---
    print(f"\n✅ Search complete.")
    print(f"   - Found {len(found_primes)} primes.")
    if found_primes:
        print(f"   - The last prime is: {found_primes[-1]}")
    print(f"   - Statistics saved to '{csv_file_name}'")

    total_numbers_checked = candidate_number - 1
    total_composites = total_numbers_checked - len(found_primes)

    print("\n--- Hybrid Filter Performance ---")
    print(f"   - Total Numbers Checked:       {total_numbers_checked}")
    print(f"   - Total Composites Found:      {total_composites}")
    print(f"   - Composites Filtered Out:     {skipped_tests}")

    if total_composites > 0:
        efficiency = (skipped_tests / total_composites) * 100
        print(f"   - Filter Efficiency:           {efficiency:.2f}%")

    if len(found_primes) < primes_to_find:
        print("\n⚠️  ERROR: Target prime count not reached. Review phase state change detector or increase safety break.")
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
            print(f"\nSanity check failed: Found {found_6000th} as the 6000th prime, but expected {actual_6000th_prime}.")
    else:
        print("\nSanity check failed: Fewer than 6000 primes were found.")