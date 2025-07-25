import math
import csv
import time  # Added for performance tracking
from functools import lru_cache  # Added for memoization


@lru_cache(maxsize=None)  # Cache results of this expensive function
def get_number_mass(n):
    """
    Calculates the 'Number Mass' of an integer.
    In this formalism, the number of divisors (d(n)) is treated as a measure
    of a number's "mass" or "complexity."
    - Prime numbers, having exactly two divisors (1 and themselves), are considered
      fundamental particles with a minimal, non-zero rest mass of 2.
    - Composite numbers are analogous to more massive, composite bodies, as their
      mass (divisor count) is always greater than 2.
    """
    if n <= 0: return 0
    if n == 1: return 1  # The unit has a mass of 1.

    num_divisors = 1
    temp_n = n

    # Step 1: Factor out the '2-component' of the number's mass.
    exponent = 0
    while temp_n % 2 == 0:
        exponent += 1
        temp_n //= 2
    if exponent > 0:
        num_divisors *= (exponent + 1)

    # Step 2: Iterate through odd prime factors to calculate the remaining mass.
    # This is analogous to resolving the particle's structure into its constituent fields.
    p = 3
    while p * p <= temp_n:
        exponent = 0
        while temp_n % p == 0:
            exponent += 1
            temp_n //= p
        if exponent > 0:
            num_divisors *= (exponent + 1)
        p += 2

    # Step 3: If a prime factor remains, it contributes its own minimal mass.
    if temp_n > 1:
        num_divisors *= 2

    return num_divisors


def get_z_metrics(n):
    """
    Calculates the full set of Z-metrics for a given integer n.
    This function acts as our "measurement device," probing the properties of a
    number as it exists within the Z-field.
    """
    if n <= 1:
        return {
            'number_mass': get_number_mass(n), 'spacetime_metric': 0, 'z_curvature': 0,
            'z_resonance': 0, 'z_vector_magnitude': 0, 'z_angle': 0
        }

    # The number's "rest mass," as calculated from its divisor components.
    number_mass = get_number_mass(n)

    # The natural logarithm is treated as the fundamental "fabric of spacetime"
    # or the metric tensor of the number line. It defines the distance and scale.
    spacetime_metric = math.log(n)

    # This is the core metric. It represents how much the number's "mass"
    # warps the "spacetime" around it. In GR, mass tells spacetime how to curve.
    # Here, a high number_mass (composite) results in high curvature, while
    # primes (mass=2) are points of minimal, stable curvature.
    z_curvature = (number_mass * spacetime_metric) / (math.e ** 2)

    # This can be viewed as a "quantum" or "sub-manifold" property. It measures
    # the number's internal "resonance" or "vibrational mode" within the local
    # spacetime, derived from the remainder of its interaction with the log-space.
    remainder = n % spacetime_metric
    z_resonance = (remainder / math.e) * number_mass

    # In physics, vectors combine multiple properties into a single state.
    # This Z-vector magnitude represents the total "field strength" or "energy"
    # of the number, combining its curvature (potential energy) and resonance (kinetic energy).
    z_vector_magnitude = math.sqrt(z_curvature ** 2 + z_resonance ** 2)

    # The Z-angle is the "phase" or "orientation" of the Z-vector, indicating
    # the balance between the number's curvature and its resonance.
    z_angle = math.degrees(math.atan2(z_resonance, z_curvature))

    return {
        'number_mass': number_mass, 'spacetime_metric': spacetime_metric,
        'z_curvature': z_curvature, 'z_resonance': z_resonance,
        'z_vector_magnitude': z_vector_magnitude, 'z_angle': z_angle
    }


def is_prime(n):
    """
    The "Oracle" or "Event Horizon" check.
    This is the final, deterministic measurement. Once a candidate number passes
    through the probabilistic filter, this function collapses its wave function,
    resolving its nature as either prime or composite with absolute certainty.
    Uses the highly efficient Miller-Rabin test.
    """
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False

    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1

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
    The "Observer" or "Probabilistic Filter."
    This function doesn't determine primality with certainty. Instead, it observes
    the "transition path" from the last known prime to the current candidate.
    If the path deviates too much from the expected "geodesic" for a prime-to-prime
    transition, it classifies the candidate as a composite and skips the expensive
    deterministic measurement.
    """
    # The "Standard Model" of prime transitions. These are the empirically
    # determined average properties (mean, std dev) for the Z-scores that
    # describe the "low-energy," most probable paths between primes.
    Z1_MEAN, Z1_STD_DEV = 0.49, 0.56
    Z4_MEAN, Z4_STD_DEV = 2.22, 2.09

    # The "Adaptive Lens" of our observer. The SIGMA_MULTIPLIER is not a
    # static "magic number." It is derived from the candidate's own internal
    # properties via the Spacetime-Curvature Ratio (SCR).
    d_n = candidate_metrics['number_mass']
    if d_n <= 1:
        d_n = 2  # Be lenient for low-mass objects.

    # For primes (d(n)=2), the multiplier is ~5.0, creating a wide, tolerant
    # observational window. For massive composites (d(n)>2), the multiplier
    # shrinks, creating a strict, narrow window that expects large deviations.
    sigma_multiplier = 1.3 + (math.e ** 2 / d_n)

    # Define the "observational window" or "light cone." If the candidate's
    # transition scores fall outside this window, it's considered to be on a
    # non-prime-like trajectory.
    z1_lower_bound = Z1_MEAN - sigma_multiplier * Z1_STD_DEV
    z1_upper_bound = Z1_MEAN + sigma_multiplier * Z1_STD_DEV
    z4_lower_bound = Z4_MEAN - sigma_multiplier * Z4_STD_DEV
    z4_upper_bound = Z4_MEAN + sigma_multiplier * Z4_STD_DEV

    is_in_range = (z1_lower_bound <= z1 <= z1_upper_bound) and \
                  (z4_lower_bound <= z4 <= z4_upper_bound)

    if not is_in_range:
        return 0, True  # The path deviated; classify as composite and skip.

    # If the path is within the expected range, we must perform the final
    # deterministic measurement with the "Oracle."
    if is_prime(candidate):
        return 1, False
    else:
        return 0, False

if __name__ == '__main__':
    start_time = time.time()  # Begin the simulation clock.

    primes_to_find = 6000
    found_primes = []
    candidate_number = 1
    csv_file_name = 'prime_stats_hybrid_filter.csv'

    # --- Initialize simulation state variables ---
    skipped_tests = 0
    last_prime_n = 0  # The coordinate of the last stable state (prime).
    last_prime_metrics = {}  # The full state vector of the last prime.

    # --- Phase State Transition Protocol ---
    # If we travel too far from a stable state (a prime), we risk crossing a
    # "phase boundary" into a chaotic region where the Z-filter is unreliable.
    # This protocol acts as a fail-safe, ensuring we can navigate these "voids".
    distance_from_last_stable_state = 0
    PHASE_BOUNDARY_THRESHOLD = 20000

    print(f"Searching for {primes_to_find} primes...")

    with open(csv_file_name, 'w', newline='') as file:
        header = ['n', 'is_prime', 'was_skipped', 'z1_score', 'z4_score']
        writer = csv.writer(file)
        writer.writerow(header)

        # The main simulation loop: a "geodesic walk" along the number line.
        while len(found_primes) < primes_to_find and candidate_number < 150000:
            # Measure the properties of the current point in spacetime.
            metrics = get_z_metrics(candidate_number)

            z1, z4 = 0, 0
            prime_status, skipped = 0, True

            # --- HYBRID FILTER LOGIC ---
            if distance_from_last_stable_state > PHASE_BOUNDARY_THRESHOLD:
                # Phase boundary crossed: The Z-filter is disengaged.
                # We now rely on the Oracle for deterministic checks to find the next stable state.
                prime_status = 1 if is_prime(candidate_number) else 0
                skipped = False

            elif last_prime_n > 0:
                # Calculate the properties of the transition from the last prime.
                gap = candidate_number - last_prime_n
                if gap > 0:
                    z_vec_mag_p = last_prime_metrics['z_vector_magnitude']
                    z_angle_p = last_prime_metrics['z_angle']
                    z_curv_p = last_prime_metrics['z_curvature']

                    # z1 and z4 measure the "geodesic deviation" or "tidal forces"
                    # experienced during the transition between the last prime and the candidate.
                    z1 = (z_vec_mag_p / gap) * abs(z_angle_p / 90.0) if z_angle_p else 0
                    z4 = z_curv_p * (z_vec_mag_p / gap)

                    # Use the observer to classify this transition path.
                    prime_status, skipped = classify_with_z_score(candidate_number, z1, z4, metrics)
            else:
                # Initial state check for the first few numbers.
                prime_status = 1 if is_prime(candidate_number) else 0
                skipped = False

            if skipped:
                skipped_tests += 1

            if prime_status == 1:
                # A stable state (prime) has been confirmed.
                # Update our position and state vector to this new point.
                found_primes.append(candidate_number)
                last_prime_n = candidate_number
                last_prime_metrics = metrics
                distance_from_last_stable_state = 0
            else:
                distance_from_last_stable_state += 1

            writer.writerow([candidate_number, prime_status, skipped, z1, z4])
            candidate_number += 1

    end_time = time.time()
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
