import math
from collections import deque
import csv
import time

# --- CONFIGURATION ---
MAX_N = 6000
Z_HISTORY_SIZE = 3
CONFIDENCE_THRESHOLD = 0.85

# Target number of primes
TARGET = 6000  # Set your desired number of primes here

# Known prime values for sanity checks. Add more as needed.
SANITY_CHECKS = {
    6000: 59359,
    10000: 104729,
    100000: 1299709,
    200000: 2750159,
    500000: 7368787,
    1000000: 15485863
}

# --- MASS-FREE Z-POINT ---
def z_point_mass_free(n):
    log_n = math.log(n)
    zc = log_n / (math.e ** 2)
    zr = (n % log_n) / math.e
    return zc, zr

# --- TRAJECTORY METRICS ---
def z_triangle_area(z_points):
    (_, x1, y1), (_, x2, y2), (_, x3, y3) = z_points
    return 0.5 * abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))

def z_angle_diff(z_points):
    def angle(x, y):
        return math.atan2(y, x)
    (_, x1, y1), (_, x2, y2), (_, x3, y3) = z_points
    a1 = angle(x2 - x1, y2 - y1)
    a2 = angle(x3 - x2, y3 - y2)
    return math.degrees(abs(a2 - a1))

# --- MASS ESTIMATION ---
def estimate_mass(triangle_area, angle_diff):
    if triangle_area < 0.001 and abs(angle_diff) < 1:
        return 2  # Prime-like
    return 4  # Composite-like

# --- Z METRICS ---
def get_mass_free_z_metrics(n, proxy_mass):
    zc, zr = z_point_mass_free(n)
    z_curvature = proxy_mass * zc
    z_resonance = proxy_mass * zr
    zv_mag = math.hypot(z_curvature, z_resonance)
    angle = math.degrees(math.atan2(z_resonance, z_curvature))
    return z_curvature, z_resonance, zv_mag, angle

# --- CONFIDENCE SCORE ---
def confidence_score(triangle_area, angle_diff):
    return math.exp(-10 * triangle_area) * math.exp(-abs(angle_diff))

# --- MAIN CLASSIFIER ---
def classify_mass_free_z(max_n=MAX_N):
    Z_buffer = deque(maxlen=Z_HISTORY_SIZE)
    prime_candidates = []
    all_metrics = []

    total_processed = 0

    for n in range(3, max_n + 1):
        total_processed += 1
        zc, zr = z_point_mass_free(n)
        Z_buffer.append((n, zc, zr))

        if len(Z_buffer) == 3:
            # Unpack for clarity
            (n1, zc1, zr1), (n2, zc2, zr2), (n3, zc3, zr3) = Z_buffer

            tri_area = z_triangle_area(Z_buffer)
            angle_d = z_angle_diff(Z_buffer)
            confidence = confidence_score(tri_area, angle_d)

            est_mass = estimate_mass(tri_area, angle_d)
            z_curv, z_res, zv_mag, ang = get_mass_free_z_metrics(n3, est_mass)

            is_prime_like = est_mass == 2 and confidence >= CONFIDENCE_THRESHOLD
            if is_prime_like:
                prime_candidates.append(n3)

            all_metrics.append({
                'n_minus_2': n1,
                'n_minus_1': n2,
                'n': n3,
                'z_curvature_1': zc1,
                'z_resonance_1': zr1,
                'z_curvature_2': zc2,
                'z_resonance_2': zr2,
                'z_curvature_3': zc3,
                'z_resonance_3': zr3,
                'delta_zc_21': zc2 - zc1,
                'delta_zc_32': zc3 - zc2,
                'delta_zr_21': zr2 - zr1,
                'delta_zr_32': zr3 - zr2,
                'gap_21': n2 - n1,
                'gap_32': n3 - n2,
                'z_curvature': z_curv,
                'z_resonance': z_res,
                'z_vector_magnitude': zv_mag,
                'angle': ang,
                'triangle_area': tri_area,
                'angle_diff': angle_d,
                'estimated_mass': est_mass,
                'confidence': confidence,
                'prime_like': is_prime_like
            })

    # Sanity check
    assert len(prime_candidates) > 0, "Sanity check failed: no prime-like candidates found."

    return prime_candidates, all_metrics, total_processed

# --- SAVE RESULTS ---
def save_metrics_csv(metrics, filename="mass_free_z_stats.csv"):
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = metrics[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics:
            writer.writerow(row)

# --- ENTRY POINT ---
if __name__ == "__main__":
    print("🚀 Running mass-free Z-space classifier...")
    start_time = time.time()

    primes, metrics, total_processed = classify_mass_free_z()
    save_metrics_csv(metrics)

    end_time = time.time()
    elapsed = end_time - start_time

    total_composites = total_processed - len(primes)
    filter_efficiency = (total_composites / total_processed) * 100 if total_processed else 0

    print(f"✅ Search complete. Found {len(primes)} prime-like candidates.")
    print(f"📊 Results saved to 'mass_free_z_stats.csv'")
    print(f"🔍 Last candidate: {primes[-1] if primes else 'None'}")
    print("-" * 40)
    print(f"⏱️  Time Elapsed: {elapsed:.2f} seconds")
    print(f"📈  Total Numbers Processed: {total_processed}")
    print(f"🗑️  Composite-like Numbers (filtered): {total_composites}")
    print(f"⚡  Filtering Efficiency: {filter_efficiency:.2f}%")

    # --- Sanity Check Validation ---
    if TARGET in SANITY_CHECKS:
        expected_prime = SANITY_CHECKS[TARGET]
        actual_prime = primes[-1] if primes else None
        if actual_prime == expected_prime:
            print(f"\n✅ Sanity Check Passed: The {TARGET:,}th prime-like candidate ({actual_prime:,}) matches the expected prime.")
        else:
            print(f"\n❌ Sanity Check Failed: Found {actual_prime} as the {TARGET:,}th prime-like candidate, but expected {expected_prime}.")
    else:
        print(f"\nℹ️ No sanity check value available for target {TARGET:,}.")
