import math
import csv
from collections import deque
import numpy as np


# ==============================================================================
# Z-FIELD METRIC CALCULATIONS
# ==============================================================================

def get_number_mass(n, primes_list):
    """Calculates the 'number_mass' of an integer 'n'."""
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
    """Calculates the full set of Z-Field metrics for an integer 'n'."""
    if n <= 1:
        return {
            'number_mass': n, 'spacetime_metric': 0, 'z_curvature': 0,
            'z_resonance': 0, 'z_vector_magnitude': 0, 'z_angle': 0
        }
    spacetime_metric = math.log(n)
    number_mass = get_number_mass(n, primes_list)
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


# ==============================================================================
# FIELD CLASSIFICATION AND PRIMALITY TESTING
# ==============================================================================

def is_prime(num):
    """A deterministic primality test based on trial division."""
    if num <= 1: return False
    if num == 2: return True
    if num % 2 == 0: return False
    i = 3
    while i <= math.sqrt(num):
        if num % i == 0:
            return False
        i += 2
    return True


def classify_with_quad_filter(gap, log_p, z1, z4, z_resonance_n, z_angle_n, z1_stats, z4_stats):
    """
    Classifies a number using a quad-filter system to identify "Field",
    "Cluster", "Isolation", and "Inertial" primes.

    Returns:
        tuple: (is_field, is_cluster, is_isolation, is_inertial)
    """
    # --- Filter 1: The "Field Prime" Signature ---
    SIGMA_MULTIPLIER = 4.0
    z1_mean, z1_std = z1_stats['mean'], z1_stats['std']
    z4_mean, z4_std = z4_stats['mean'], z4_stats['std']
    z1_std = max(z1_std, 0.2)
    z4_std = max(z4_std, 0.2)
    z1_lower = z1_mean - SIGMA_MULTIPLIER * z1_std
    z1_upper = z1_mean + SIGMA_MULTIPLIER * z1_std
    z4_lower = z4_mean - SIGMA_MULTIPLIER * z4_std
    z4_upper = z4_mean + SIGMA_MULTIPLIER * z4_std
    is_field_candidate = (z1_lower <= z1 <= z1_upper) and \
                         (z4_lower <= z4 <= z4_upper)

    # --- Filter 2: The "Cluster Prime" Signature ---
    CLUSTER_RESONANCE_MEAN = 1.06
    CLUSTER_RESONANCE_STD = 0.86
    res_lower = CLUSTER_RESONANCE_MEAN - 3 * CLUSTER_RESONANCE_STD
    res_upper = CLUSTER_RESONANCE_MEAN + 3 * CLUSTER_RESONANCE_STD
    is_cluster_candidate = (res_lower <= z_resonance_n <= res_upper)

    # --- Filter 3: The "Isolation Prime" Signature ---
    is_isolation_candidate = False
    if log_p > 0 and (gap / log_p) > 3.0:
        is_isolation_candidate = True

    # --- Filter 4: The "Inertial Prime" Signature ---
    is_inertial_candidate = (z_angle_n < 20.0)

    return is_field_candidate, is_cluster_candidate, is_isolation_candidate, is_inertial_candidate


# ==============================================================================
# MAIN EXECUTION BLOCK: THE QUAD-FILTER ALGORITHM
# ==============================================================================

# --- Configuration ---
primes_to_find = 6000
search_limit = 70000
WINDOW_SIZE = 500
log_file_name = 'quad_filter_analysis.csv'

# --- State Variables ---
found_primes = []
candidate_number = 1
last_prime_n = 0
last_prime_metrics = {}
z1_history = deque(maxlen=WINDOW_SIZE)
z4_history = deque(maxlen=WINDOW_SIZE)
z1_stats = {'mean': 1.10, 'std': 0.65}
z4_stats = {'mean': 0.50, 'std': 0.35}

print(f"Initializing quad-filter Z-Field search for {primes_to_find} primes...")
print(f"Logging detailed analysis to '{log_file_name}'")

with open(log_file_name, 'w', newline='') as file:
    writer = csv.writer(file)
    header = [
        'n', 'is_prime', 'gap', 'z1', 'z4', 'z_resonance_n', 'z_angle_n',
        'is_field', 'is_cluster', 'is_isolation', 'is_inertial'
    ]
    writer.writerow(header)

    # Main loop
    while len(found_primes) < primes_to_find and candidate_number < search_limit:
        metrics = get_z_metrics(candidate_number, found_primes)
        prime_status = 0
        candidates = (False,) * 4
        gap, z1, z4, z_resonance_n, z_angle_n = 0, 0, 0, 0, 0

        if last_prime_n > 0:
            gap = candidate_number - last_prime_n
            if gap > 0:
                z_vec_mag_p = last_prime_metrics['z_vector_magnitude']
                z_angle_p = last_prime_metrics['z_angle']
                z_curv_p = last_prime_metrics['z_curvature']
                log_p = last_prime_metrics['spacetime_metric']

                z1 = (z_vec_mag_p / gap) * abs(z_angle_p / 90.0) if z_angle_p else 0
                z4 = z_curv_p * (z_vec_mag_p / gap)
                z_resonance_n = metrics['z_resonance']
                z_angle_n = metrics['z_angle']

                candidates = classify_with_quad_filter(
                    gap, log_p, z1, z4, z_resonance_n, z_angle_n, z1_stats, z4_stats
                )

                if any(candidates):
                    if is_prime(candidate_number):
                        prime_status = 1
        else:  # Handle first prime
            if is_prime(candidate_number):
                prime_status = 1

        # --- LOGGING ---
        if last_prime_n > 0:
            writer.writerow([
                candidate_number, prime_status, gap, z1, z4, z_resonance_n, z_angle_n,
                *candidates
            ])

        # --- STATE UPDATE ---
        if prime_status == 1:
            found_primes.append(candidate_number)

            if last_prime_n > 0:
                z1_history.append(z1)
                z4_history.append(z4)

                if len(z1_history) > 10:
                    z1_stats['mean'] = np.mean(z1_history)
                    z1_stats['std'] = np.std(z1_history)
                    z4_stats['mean'] = np.mean(z4_history)
                    z4_stats['std'] = np.std(z4_history)

            last_prime_n = candidate_number
            last_prime_metrics = metrics

            if len(found_primes) % 500 == 0:
                print(f"Found prime {len(found_primes)}/{primes_to_find}: {candidate_number}")

        candidate_number += 1

# --- Final Summary ---
print(f"\n✅ Z-Field search complete.")
print(f"   - Found {len(found_primes)} primes.")
if found_primes:
    print(f"   - The last prime found is: {found_primes[-1]}")

if len(found_primes) < primes_to_find:
    print(f"\n⚠️  WARNING: Search limit of {search_limit} reached. "
          f"Found {len(found_primes)} out of {primes_to_find} primes.")
else:
    print("\n   - SUCCESS: Target prime count reached within the search limit.")
    # --- SANITY CHECK ---
    actual_6000th_prime = 57671
    found_6000th_prime = found_primes[5999]
    print(f"   - Ground Truth 6000th prime: {actual_6000th_prime}")
    print(f"   - Algorithm's 6000th prime:  {found_6000th_prime}")
    if actual_6000th_prime == found_6000th_prime:
        print("   - ACCURACY CONFIRMED: The filter correctly identified all primes.")
    else:
        print(f"   - ACCURACY FAILED: The filter missed {actual_6000th_prime - found_6000th_prime} primes.")
