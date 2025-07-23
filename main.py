import math
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


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


def classify_number(candidate, metrics, current_ceiling):
    """
    Applies the low-pass filter and performs the final primality test if necessary.
    """
    if metrics['z_curvature'] > current_ceiling and candidate > 10:
        return 0, True, 0
    if is_prime(candidate):
        return 1, False, metrics['z_curvature']
    else:
        return 0, False, 0


# --- Main execution block ---
if __name__ == '__main__':
    primes_to_find = 500
    found_primes = []
    candidate_number = 1
    csv_file_name = 'prime_stats_gr_triangles.csv'

    # --- Initialize variables ---
    max_prime_curvature = 0.0
    skipped_tests = 0
    last_prime_n = 0
    last_prime_metrics = {}

    print(f"Searching for {primes_to_find} primes and logging GR-inspired triangle data...")

    with open(csv_file_name, 'w', newline='') as file:
        header = [
            'n', 'is_prime', 'number_mass', 'spacetime_metric', 'z_curvature', 'z_resonance',
            'z_vector_magnitude', 'z_angle', 'curvature_ceiling',
            # GR Triangle 1: Gravitational Lensing
            'gr1_mass_A', 'gr1_path_B', 'gr1_bending_C',
            # GR Triangle 2: Metric Tensor
            'gr2_gap', 'gr2_vec_mag_C',
            # GR Triangle 3: Frame-Dragging
            'gr3_delta_curv_A', 'gr3_delta_res_B', 'gr3_norm_gap_C'
        ]
        writer = csv.writer(file)
        writer.writerow(header)

        while len(found_primes) < primes_to_find:
            metrics = get_z_metrics(candidate_number, found_primes)
            curvature_ceiling = max(3.5, max_prime_curvature)
            prime_status, skipped, new_max_curvature = classify_number(candidate_number, metrics, curvature_ceiling)

            if skipped:
                skipped_tests += 1

            # --- Initialize new GR triangle metrics ---
            gr1_mass_A, gr1_path_B, gr1_bending_C = 0, 0, 0
            gr2_gap, gr2_vec_mag_C = 0, 0
            gr3_delta_curv_A, gr3_delta_res_B, gr3_norm_gap_C = 0, 0, 0

            # --- New GR Triangle Calculations ---
            if last_prime_n > 0:
                log_p = last_prime_metrics['spacetime_metric']
                if log_p > 0: # Avoid division by zero for the theoretical case of p=1
                    gap = candidate_number - last_prime_n
                    normalized_gap = gap / log_p

                    # Triangle 1: Gravitational Lensing
                    gr1_mass_A = last_prime_metrics['z_curvature']
                    gr1_path_B = normalized_gap
                    gr1_bending_C = last_prime_metrics['z_angle'] / 90.0

                    # Triangle 2: Metric Tensor
                    gr2_gap = gap
                    gr2_vec_mag_C = last_prime_metrics['z_vector_magnitude']

                    # Triangle 3: Frame-Dragging
                    gr3_delta_curv_A = abs(metrics['z_curvature'] - last_prime_metrics['z_curvature'])
                    gr3_delta_res_B = abs(metrics['z_resonance'] - last_prime_metrics['z_resonance'])
                    gr3_norm_gap_C = normalized_gap

            # Update state AFTER a prime is confirmed
            if prime_status == 1:
                found_primes.append(candidate_number)
                max_prime_curvature = max(max_prime_curvature, new_max_curvature)
                last_prime_n = candidate_number
                last_prime_metrics = metrics

            writer.writerow([
                candidate_number, prime_status, metrics['number_mass'], metrics['spacetime_metric'],
                metrics['z_curvature'], metrics['z_resonance'], metrics['z_vector_magnitude'],
                metrics['z_angle'], curvature_ceiling,
                gr1_mass_A, gr1_path_B, gr1_bending_C,
                gr2_gap, gr2_vec_mag_C,
                gr3_delta_curv_A, gr3_delta_res_B, gr3_norm_gap_C
            ])
            candidate_number += 1

    total_composites_checked = (candidate_number - 1) - primes_to_find

    print(f"\n✅ Success! Found {len(found_primes)} primes.")
    print(f"   - The last prime is: {found_primes[-1]}")
    print(f"   - Statistics saved to '{csv_file_name}'")
    print("\n--- Classifier Performance & Accuracy ---")
    print(f"   - Total Composites Found:      {total_composites_checked}")
    print(f"   - Composites Filtered Out:     {skipped_tests}")
    if total_composites_checked > 0:
        efficiency = (skipped_tests / total_composites_checked) * 100
        print(f"   - Filter Efficiency:           {efficiency:.2f}% of composites were correctly filtered.")

    # --- Generate the Plots from the CSV data ---
    print(f"\n📈 Now generating graphs from the statistics...")
    df = pd.read_csv(csv_file_name)
    primes_df = df[df['is_prime'] == 1]
    composites_df = df[df['is_prime'] == 0]

    # --- Graph 1: Cumulative Primes and Z Curvature ---
    plt.figure(figsize=(12, 7))
    plt.plot(df['n'], df['is_prime'].cumsum(), color='tab:blue', label='Cumulative Primes')
    plt.xlabel('Number (n)')
    plt.ylabel('Cumulative Prime Count', color='tab:blue')
    plt.twinx()
    plt.plot(df['n'], df['z_curvature'], color='tab:red', alpha=0.75, label='Z Curvature')
    plt.ylabel('Z Curvature', color='tab:red')
    plt.title("Cumulative Primes and Z Curvature vs. Number (n)")
    plt.tight_layout()
    plt.savefig('prime_distribution_and_curvature_graph.png')
    print("   - Graph 1 saved as 'prime_distribution_and_curvature_graph.png'")

    # --- Graph 2: Z Resonance ---
    plt.figure(figsize=(12, 7))
    plt.scatter(composites_df['n'], composites_df['z_resonance'], color='gray', alpha=0.5, s=10)
    plt.scatter(primes_df['n'], primes_df['z_resonance'], color='red', s=20)
    plt.title('Z Resonance vs. Number (n)')
    plt.savefig('resonance_factor_graph.png')
    print("   - Graph 2 saved as 'resonance_factor_graph.png'")

    # --- Graph 3: Z Curvature and Ceiling ---
    plt.figure(figsize=(12, 7))
    plt.scatter(composites_df['n'], composites_df['z_curvature'], color='gray', alpha=0.5, s=10)
    plt.scatter(primes_df['n'], primes_df['z_curvature'], color='red', s=20)
    plt.plot(df['n'], df['curvature_ceiling'], color='blue', linestyle='--')
    plt.title('Z Curvature vs. Dynamic Ceiling')
    plt.savefig('curvature_and_ceiling_graph.png')
    print("   - Graph 3 saved as 'curvature_and_ceiling_graph.png'")

    # --- Graph 4: Z Vector Magnitude ---
    plt.figure(figsize=(12, 7))
    plt.scatter(composites_df['n'], composites_df['z_vector_magnitude'], color='gray', alpha=0.5, s=10)
    plt.scatter(primes_df['n'], primes_df['z_vector_magnitude'], color='red', s=20)
    plt.title('Z Vector Magnitude vs. Number (n)')
    plt.savefig('z_vector_magnitude_graph.png')
    print("   - Graph 4 saved as 'z_vector_magnitude_graph.png'")

    # --- Graph 5: Z Angle ---
    plt.figure(figsize=(12, 7))
    plt.scatter(composites_df['n'], composites_df['z_angle'], color='gray', alpha=0.5, s=10)
    plt.scatter(primes_df['n'], primes_df['z_angle'], color='red', s=20)
    plt.title('Z Angle vs. Number (n)')
    plt.savefig('z_angle_graph.png')
    print("   - Graph 5 saved as 'z_angle_graph.png'")