import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Constants and Parameters
c = 3e8  # Speed of light in m/s (invariant universal limit)
v_seismic = 1500  # Average speed of seismic waves in seawater (m/s)
v_tsunami = 200  # Average speed of tsunami waves (m/s)
earthquake_magnitude = 8.0  # Magnitude of the Kamchatka earthquake
earthquake_depth = 30  # Approximate depth of the earthquake in km
ocean_depth = 50  # Average ocean depth in meters (from supporting data)
sampling_rate = 100  # Sampling rate for simulation (Hz)

# Convert units to consistent system (meters, seconds)
earthquake_depth_m = earthquake_depth * 1000
ocean_depth_m = ocean_depth

# Z Transformation Function (Universal Form: Z = T(v/c))
def z_transform(time, velocity):
    return time * (velocity / c)

# Inverse Z Transformation for universal frame time
def inverse_z_transform(z_time, velocity):
    return z_time / (velocity / c)

# Seismic Wave Propagation Model
def seismic_wave_propagation(distance, time, velocity):
    return distance / velocity <= time

# Tsunami Wave Propagation Model
def tsunami_wave_propagation(distance, time, velocity):
    return distance / velocity <= time

# Tsunami Wave Height Estimation (based on earthquake magnitude)
def tsunami_wave_height(magnitude, distance):
    # Empirical relationship with power-law decay
    base_height = 10 ** (magnitude / 2 - 4.5)  # in meters
    decay_factor = distance ** (-1.5)  # Power-law decay (approximate)
    return base_height * decay_factor

# Simulation Parameters
total_time = 3600  # Total simulation time in seconds (1 hour)
distance_range = np.linspace(0, 1e7, 1000)  # Distance from epicenter in meters (up to 10,000 km)

# Initialize arrays for results
seismic_arrival = np.zeros(len(distance_range))
tsunami_arrival = np.zeros(len(distance_range))
tsunami_height = np.zeros(len(distance_range))

# Calculate seismic and tsunami arrival times and tsunami heights
for i, distance in enumerate(distance_range):
    # Seismic wave arrival time
    seismic_arrival[i] = distance / v_seismic
    # Tsunami wave arrival time
    tsunami_arrival[i] = distance / v_tsunami
    # Tsunami wave height at distance
    tsunami_height[i] = tsunami_wave_height(earthquake_magnitude, distance)

# Apply Z Transformation to correct for frame bias
z_corrected_seismic_time = z_transform(seismic_arrival, v_seismic)
z_corrected_tsunami_time = z_transform(tsunami_arrival, v_tsunami)

# Inverse Z Transformation for universal frame time
universal_seismic_time = inverse_z_transform(z_corrected_seismic_time, v_seismic)
universal_tsunami_time = inverse_z_transform(z_corrected_tsunami_time, v_tsunami)

# Interpolation for smoother visualization
interp_seismic = interp1d(distance_range, z_corrected_seismic_time, kind='linear')
interp_tsunami = interp1d(distance_range, tsunami_height, kind='linear')
interp_universal_seismic = interp1d(distance_range, universal_seismic_time, kind='linear')

# Plot results
plt.figure(figsize=(12, 12))

# Seismic arrival times
plt.subplot(4, 1, 1)
plt.plot(distance_range / 1000, seismic_arrival, 'g--', label='Original Seismic Arrival Time')
plt.plot(distance_range / 1000, z_corrected_seismic_time, 'b-', label='Z-Transformed Seismic Time')
plt.plot(distance_range / 1000, universal_seismic_time, 'r-', label='Universal Frame Seismic Time')
plt.xlabel('Distance from Epicenter (km)')
plt.ylabel('Time (s)')
plt.title('Seismic Wave Propagation (Original vs Z-Transformed vs Universal)')
plt.legend()

# Tsunami arrival times
plt.subplot(4, 1, 2)
plt.plot(distance_range / 1000, tsunami_arrival, 'm--', label='Original Tsunami Arrival Time')
plt.plot(distance_range / 1000, z_corrected_tsunami_time, 'c-', label='Z-Transformed Tsunami Time')
plt.plot(distance_range / 1000, universal_tsunami_time, 'y-', label='Universal Frame Tsunami Time')
plt.xlabel('Distance from Epicenter (km)')
plt.ylabel('Time (s)')
plt.title('Tsunami Wave Propagation (Original vs Z-Transformed vs Universal)')
plt.legend()

# Tsunami wave heights
plt.subplot(4, 1, 3)
plt.plot(distance_range / 1000, tsunami_height, 'r-', label='Tsunami Wave Height')
plt.xlabel('Distance from Epicenter (km)')
plt.ylabel('Wave Height (m)')
plt.title('Tsunami Wave Height Estimation')
plt.legend()

# Warning time differential between seismic and tsunami arrival
plt.subplot(4, 1, 4)
plt.plot(distance_range / 1000, tsunami_arrival - seismic_arrival, 'g--', label='Original ΔTime')
plt.plot(distance_range / 1000, z_corrected_tsunami_time - z_corrected_seismic_time, 'b-', label='Z-Frame ΔTime')
plt.plot(distance_range / 1000, universal_tsunami_time - universal_seismic_time, 'r-', label='Universal ΔTime')
plt.xlabel('Distance from Epicenter (km)')
plt.ylabel('ΔTime (s)')
plt.title('Warning Time Differential (Tsunami - Seismic)')
plt.legend()

plt.tight_layout()
plt.show()

# Calculate potential warning time improvement
def warning_time_improvement(original_time, z_transformed_time, universal_time):
    return {
        'seismic': original_time[0] - z_transformed_time[0],
        'tsunami': original_time[1] - z_transformed_time[1],
        'universal_seismic': original_time[0] - universal_time[0],
        'universal_tsunami': original_time[1] - universal_time[1]
    }

# Example calculation for a specific distance (e.g., 5000 km)
distance_example = 5e6  # 5000 km in meters
index = np.abs(distance_range - distance_example).argmin()
original_seismic_time = seismic_arrival[index]
original_tsunami_time = tsunami_arrival[index]
z_transformed_seismic_time = z_corrected_seismic_time[index]
z_transformed_tsunami_time = z_corrected_tsunami_time[index]
universal_seismic_time = universal_seismic_time[index]
universal_tsunami_time = universal_tsunami_time[index]

improvement = warning_time_improvement(
    (original_seismic_time, original_tsunami_time),
    (z_transformed_seismic_time, z_transformed_tsunami_time),
    (universal_seismic_time, universal_tsunami_time)
)

print(f"Original seismic arrival time at 5000 km: {original_seismic_time:.2f} seconds")
print(f"Z-transformed seismic arrival time at 5000 km: {z_transformed_seismic_time:.2f} seconds")
print(f"Universal frame seismic time at 5000 km: {universal_seismic_time:.2f} seconds")
print(f"Original tsunami arrival time at 5000 km: {original_tsunami_time:.2f} seconds")
print(f"Z-transformed tsunami arrival time at 5000 km: {z_transformed_tsunami_time:.2f} seconds")
print(f"Universal frame tsunami time at 5000 km: {universal_tsunami_time:.2f} seconds")
print(f"Potential warning time improvements:")
print(f"  Seismic (Z-transform): {improvement['seismic']:.2f} seconds")
print(f"  Tsunami (Z-transform): {improvement['tsunami']:.2f} seconds")
print(f"  Seismic (Universal): {improvement['universal_seismic']:.2f} seconds")
print(f"  Tsunami (Universal): {improvement['universal_tsunami']:.2f} seconds")