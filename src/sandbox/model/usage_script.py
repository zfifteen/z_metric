# usage_script.py
# This script demonstrates the usage of the Z Universal Form API
# with examples from physical and discrete domains.

from z_universal_api import compute_z_band

# Example 1: Physical Domain
# Z = T * (v / c)
# Assume T = 10 (time in seconds), v = 3e8 / 2 (half speed of light), c = 3e8 (m/s)
physical_A = 10  # Reference frame-dependent time
physical_B = 1.5e8  # Velocity rate
physical_C = 3e8  # Speed of light invariant
physical_Z = compute_z_band(physical_A, physical_B, physical_C)
print(f"Physical Domain Z: {physical_Z}")
# Output interpretation: In a reference frame where time dilation effects are considered,
# this Z represents the transformed quantity under velocity constraints approaching the universal limit.

# Example 2: Discrete Domain
# Z = n * (Δ_n / Δ_max)
# Assume n = 5 (observation count), Δ_n = 3 (frame shift at position n), Δ_max = 10 (max possible shift)
discrete_A = 5  # Reference frame-dependent integer observation
discrete_B = 3  # Measured frame shift
discrete_C = 10  # Maximum possible frame shift# extended_band_usage_script.py

from z_universal_api import compute_z_band, UNIVERSAL_C

# Physical Regime Band: Vary B (velocities)
A_phys = 10
B_phys_range = [0, 1e8, 2e8, 3e8]  # Rates approaching C
phys_band = compute_z_band(A_phys, B_phys_range, 'physical')
print(f"Physical Band: Range {phys_band['band_range']}, Avg dZ/dB: {phys_band['avg_derivative']} (invariance info band)")

# Discrete Regime Band: Vary B (shifts)
A_disc = 5
B_disc_range = [0, 3, 6, 9]  # Shifts toward max
disc_band = compute_z_band(A_disc, B_disc_range, 'discrete')
print(f"Discrete Band: Range {disc_band['band_range']}, Avg dZ/dB: {disc_band['avg_derivative']} (historical derivative band)")

# Demonstrates band as additional info layer from C's invariance
print(f"Universal C: {UNIVERSAL_C} generates this informational band.")
discrete_Z = compute_z_band(discrete_A, discrete_B, discrete_C)
print(f"Discrete Domain Z: {discrete_Z}")
# Output interpretation: In a discrete sequence of observations, this Z captures the proportional
# shift within the domain's curvature, bounded by the invariant maximum.

# Example 3: General Usage
# Arbitrary values to show flexibility
general_A = 100  # Measured quantity in some frame
general_B = 50  # Rate
general_C = 200  # Universal limit
general_Z = compute_z_band(general_A, general_B, general_C)
print(f"General Z: {general_Z}")
# This demonstrates a frame shift transformation where the rate approaches half the limit,
# scaling the measured quantity accordingly.