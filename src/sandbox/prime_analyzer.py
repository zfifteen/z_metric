import math
from collections import deque
import csv
import time

# --- CONFIGURATION ---
MAX_N = 6000
Z_HISTORY_SIZE = 3
CONFIDENCE_THRESHOLD = 0.75  # Lowered to capture more candidates
ALPHA = 0.05  # Tuning parameter for curvature shift ratio

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
    return 0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))


def z_angle_diff(z_points):
    def angle(x, y):
        return math.atan2(y, x)

    (_, x1, y1), (_, x2, y2), (_, x3, y3) = z_points
    a1 = angle(x2 - x1, y2 - y1)
    a2 = angle(x3 - x2, y3 - y2)
    return math.degrees(abs(a2 - a1))


# --- MASS ESTIMATION ---
def estimate_mass(triangle_area, angle_diff, delta_zc_21, delta_zc_32):
    # Avoid division by zero in curvature shift ratio
    if abs(delta_zc_21) > 1e-10:
        curvature_shift_ratio = delta_zc_32 / delta_zc_21
    else:
        curvature_shift_ratio = 0.0

    # Refined heuristic: Incorporate curvature shift ratio
    z_kappa_refined = 2.0 * triangle_area + ALPHA * abs(curvature_shift_ratio)

    # Relaxed thresholds based on empirical data
    if triangle_area < 3.5e-7 and abs(angle_diff) < 0.0005 and z_kappa_refined < 0.003:
        return 2, z_kappa_refined  # Prime-like
    return 4, z_kappa_refined  # Composite-like


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
    # Clamp angle_diff for numerical stability
    clamped_angle_diff = max(min(abs(angle_diff), 1.0), 1e-10)
    # Base confidence with softer adjustment
    confidence = math.exp(-10 * triangle_area) * math.exp(-0.5 * clamped_angle_diff)  # Scaled down angle_diff impact
    return max(min(confidence, 1.0), 0.0)  # Clamp to [0, 1]


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

            # Pass curvature deltas to mass estimation
            delta_zc_21 = zc2 - zc1
            delta_zc_32 = zc3 - zc2
            est_mass, z_kappa_refined = estimate_mass(tri_area, angle_d, delta_zc_21, delta_zc_32)
            z_curv, z_res, zv_mag, ang = get_mass_free_z_metrics(n3, est_mass)

            is_prime_like = est_mass == 2 and confidence >= CONFIDENCE_THRESHOLD
            if is_prime_like:
                prime_candidates.append(n3)
            else:
            # Debug: Log why candidate failed (uncomment for detailed logging)
            # reason = f"est_mass={est_mass}" if est_mass != 2 else f"confidence={confidence:.4f} < {CONFIDENCE_THRESHOLD}"
            # print(f"n={n3}: Failed prime-like check: {reason}")

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
                'delta_zc_21': delta_zc_21,
                'delta_zc_32': delta_zc_32,
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
                'z_kappa_refined': z_kappa_refined,
                'prime_like': is_prime_like
            })

    print(f"Found {len(prime_candidates)} prime-like candidates.")
    if len(prime_candidates) == 0:
        First, the
        user
        provided
        the
        code
        with bugs, and the output shows NameError for 'delta_zc3'.


I
need
to
fix
the
code.

The
tool
call
was
a
print, but
that
's not necessary.

My
response
should
be
the
fixed
code.

The
user
message is the
code
with bugs, and the output.

So, I
need
to
provide
the
integrated, fixed
code.

From
my
previous
thinking, the
fixed
version is:

Modify
estimate_mass
to
return est_mass, z_kappa_refined

In
the
loop, est_mass, z_kappa_refined = estimate_mass(...)

Fix
the
dict:

'delta_zr_32': zr3 - zr2,

'z_resonance': z_res,

'z_vector_magnitude': zv_mag,

Remove
'g': triangle_area, 'Delta_zc': delta_zc,

Remove
duplicate
'triangle_area': tri_area,

'confidence': confidence,

Add
'z_kappa_refined': z_kappa_refined,

In
entry
point, fix
elapsed = end_time - start_time

total_composites = total_processed - len(primes)

Fix
the
if len(prime_candidates) == 0: print("Warning...")

Remove
the
"Found"
inside if 0.

In
the
code, it
's:

if len(prime_candidates) == 0:
    print("⚠️ Warning: No prime-like candidates found. Check thresholds.")

The
"Found {len} " is outside.

In
the
provided
code, it
's:

if len(prime_candidates) == 0:
    print("⚠️ Warning: No prime-like candidates found. Check thresholds.")

return prime_candidates, all_metrics, total_processed

Then in main, print(f"✅ Search complete. Found {len(primes)} prime-like candidates.")

It
's correct, but in the output, it printed 0, and the warning.

But in the
last
user
message, it
's "⚠️ Warning: No prime-like candidates found. Check thresholds."

And
Found
0.

To
fix
the
NameError, and other
typos.

Also, in the
code, there is 'delta_zr_32': delta_zc3 - zr3, fix
to
zr3 - zr2

'z_resonance': delta_z_res, fix
to
z_res

'z_vector_magnitude': delta_zv_mag, fix
to
zv_mag

'angle': ang,

'triangle_area': tri_area,

'angle_diff': angle_d,

'estimated_mass': est_mass,

'confidence': confidence,

'prime_like': is_prime_like

Remove
extra
'g', 'Delta_zc', 'z_confidence'(since
it
's the same as confidence), '
z_curvature_refined
': z_kappa_refined (add this)

In
the
code, it
's '
z_confidence
': confidence, '
z_curvature_refined
': z_kappa_refined, '
confidence
': confidence

But
z_kappa_refined is not defined, so
need
to
return it
from estimate_mass.

Let
's write the fixed code.```python
import math
from collections import deque
import csv
import time

# --- CONFIGURATION ---
MAX_N = 6000
Z_HISTORY_SIZE = 3
CONFIDENCE_THRESHOLD = 0.75  # Lowered to capture more candidates
ALPHA = 0.05  # Tuning parameter for curvature shift ratio

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
    return 0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))


def z_angle_diff(z_points):
    def angle(x, y):
        return math.atan2(y, x)

    (_, x1, y1), (_, x2, y2), (_, x3, y3) = z_points
    a1 = angle(x2 - x1, y2 - y1)
    a2 = angle(x3 - x2, y3 - y2)
    return math.degrees(abs(a2 - a1))


# --- MASS ESTIMATION ---
def estimate_mass(triangle_area, angle_diff, delta_zc_21, delta_zc_32):
    # Avoid division by zero in curvature shift ratio
    if abs(delta_zc_21) > 1e-10:
        curvature_shift_ratio = delta_zc_32 / delta_zc_21
    else:
        curvature_shift_ratio = 0.0

    # Refined heuristic: Incorporate curvature shift ratio
    z_kappa_refined = 2.0 * triangle_area + ALPHA * abs(curvature_shift_ratio)

    # Relaxed thresholds based on empirical data
    if triangle_area < 3.5e-7 and abs(angle_diff) < 0.0005 and z_kappa_refined < 0.003:
        return 2, z_kappa_refined  # Prime-like
    return 4, z_kappa_refined  # Composite-like


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
    # Clamp angle_diff for numerical stability
    clamped_angle_diff = max(min(abs(angle_diff), 1.0), 1e-10)
    # Base confidence with softer adjustment
    confidence = math.exp(-10 * triangle_area) * math.exp(-0.5 * clamped_angle_diff)  # Scaled down angle_diff impact
    return max(min(confidence, 1.0), 0.0)  # Clamp to [0, 1]


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

            # Pass curvature deltas to mass estimation
            delta_zc_21 = zc2 - zc1
            delta_zc_32 = zc3 - zc2
            est_mass, z_kappa_refined = estimate_mass(tri_area, angle_d, delta_zc_21, delta_zc_32)
            z_curv, z_res, zv_mag, ang = get_mass_free_z_metrics(n3, est_mass)

            is_prime_like = est_mass == 2 and confidence >= CONFIDENCE_THRESHOLD
            if is_prime_like:
                prime_candidates.append(n3)
            # Optional debug logging
            # else:
            #     reason = f"est_mass={est_mass}" if est_mass != 2 else f"confidence={confidence:.4f} < {CONFIDENCE_THRESHOLD}"
            #     print(f"n={n3}: Failed - {reason}")

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
                'delta_zc_21': delta_zc_21,
                'delta_zc_32': delta_zc_32,
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
                'z_kappa_refined': z_kappa_refined,
                'prime_like': is_prime_like
            })

    if len(prime_candidates) == 0:
        print("⚠️ Warning: No prime-like candidates found. Check thresholds.")

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
            print(
                f"\n✅ Sanity Check Passed: The {TARGET:,}th prime-like candidate ({actual_prime:,}) matches the expected prime.")
        else:
            print(
                f"\n❌ Sanity Check Failed: Found {actual_prime} as the {TARGET:,}th prime-like candidate, but expected {expected_prime}.")
    else:
        print(f"\nℹ️ No sanity check value available for target {TARGET:,}.")