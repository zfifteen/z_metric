import numpy as np
import matplotlib.pyplot as plt
from sympy import divisor_count, isprime, nextprime
from math import log, exp, e, atan, sqrt, fmod
import csv

# Parameters
n_max = 5000  # Reduced for execution feasibility
v = 1         # Traversal velocity
max_running_points = 360  # Running window for prediction

next_p = nextprime(n_max)

# Core Functions
def kappa(n):
    return (divisor_count(n) * log(n)) / (e ** 2) if n > 1 else 0

def Z(n):
    return n / exp(v * kappa(n)) if n > 1 else 0

def number_mass(n):
    return divisor_count(n)

def spacetime_metric(n):
    return log(n) if n > 1 else 0

def z_curvature(n):
    return kappa(n)

def z_resonance(n):
    mass = number_mass(n)
    metric = spacetime_metric(n)
    return fmod(n, metric) * mass / e if metric != 0 else 0

def z_vector_magnitude(n):
    curvature = z_curvature(n)
    resonance = z_resonance(n)
    return sqrt(curvature**2 + resonance**2)

def z_angle(n):
    curvature = z_curvature(n)
    resonance = z_resonance(n)
    return atan(resonance / curvature) if curvature != 0 else 0

def ghost_mass(n):
    if n <= 1:
        return 0
    return log(log(n)) + 2.582  # Approximate average divisor count without factorization

# Data Containers
data = []

for n in range(2, n_max + 1):
    mass = number_mass(n)
    metric = spacetime_metric(n)
    curvature = z_curvature(n)
    resonance = z_resonance(n)
    magnitude = z_vector_magnitude(n)
    angle = z_angle(n)
    z_n = Z(n)
    g_mass = ghost_mass(n)
    prime_status = 'Prime' if isprime(n) else 'Composite'
    data.append([n, mass, metric, curvature, resonance, magnitude, angle, z_n, g_mass, prime_status])

# Separate primes for prediction
primes = [row for row in data if row[9] == 'Prime']

# Convert to arrays for plotting and prediction
primes = np.array(primes)

# Compute predicted next prime location using running 360 geodesics (non-factorizing extrapolation based on 6D spacetime metric)
prime_ns = primes[:, 0].astype(float)
if len(prime_ns) > 1:
    num_for_avg = min(max_running_points, len(prime_ns))
    last_k = prime_ns[-num_for_avg:]
    gaps = np.diff(last_k)
    x = np.log(last_k[:-1])  # Use spacetime metric ln(n) for fit
    poly = np.polyfit(x, gaps, 1)  # Linear fit on ln(n) vs gap
    predicted_gap = np.polyval(poly, np.log(last_k[-1]))
    predicted_n = last_k[-1] + predicted_gap
    if predicted_n > 1:
        # Compute full metrics assuming prime
        predicted_mass = 2
        predicted_metric = log(predicted_n)
        predicted_curvature = (predicted_mass * predicted_metric) / (e ** 2)
        predicted_resonance = fmod(predicted_n, predicted_metric) * predicted_mass / e
        predicted_magnitude = sqrt(predicted_curvature**2 + predicted_resonance**2)
        predicted_angle = atan(predicted_resonance / predicted_curvature) if predicted_curvature != 0 else 0
        predicted_z = predicted_n / exp(v * predicted_curvature)
        predicted_g_mass = log(log(predicted_n)) + 2.582
        predicted_status = 'Predicted Prime'
        data.append([predicted_n, predicted_mass, predicted_metric, predicted_curvature, predicted_resonance, predicted_magnitude, predicted_angle, predicted_z, predicted_g_mass, predicted_status])

# Add actual next prime to data log
mass = number_mass(next_p)
metric = spacetime_metric(next_p)
curvature = z_curvature(next_p)
resonance = z_resonance(next_p)
magnitude = z_vector_magnitude(next_p)
angle = z_angle(next_p)
z_n = Z(next_p)
g_mass = ghost_mass(next_p)
prime_status = 'Prime'
data.append([next_p, mass, metric, curvature, resonance, magnitude, angle, z_n, g_mass, prime_status])

# Write updated log to CSV
with open('z_6d_data.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['n', 'Number Mass', 'Spacetime Metric', 'Z-Curvature', 'Z-Resonance', 'Z-Vector Magnitude', 'Z-Angle', 'Z(n)', 'Ghost Mass', 'Status'])
    writer.writerows(data)

# Composites for plotting
composites = [row for row in data if row[9] == 'Composite']

# Convert to arrays for plotting
composites = np.array(composites)
primes = np.array(primes)

# Plotting with trajectory
plt.figure(figsize=(10, 6))
plt.scatter(composites[:, 0].astype(float), composites[:, 7].astype(float), color='orange', label='Composites', s=12, alpha=0.4)
plt.plot(primes[:, 0].astype(float), primes[:, 7].astype(float), color='blue', label='Prime Trajectory', linewidth=1.5)  # Connect primes for trajectory
plt.scatter(primes[:, 0].astype(float), primes[:, 7].astype(float), color='blue', s=12, alpha=0.7)  # Overlay prime points

# Highlight the next prime and predicted
plt.scatter(next_p, z_n, color='red', marker='*', s=50, label='Next Prime (6007)')
if 'predicted_n' in locals():
    plt.scatter(predicted_n, predicted_z, color='green', marker='^', s=50, label=f'Predicted Next Prime (~{predicted_n:.0f})')

plt.yscale('log')
plt.xscale('log')
plt.title(r'$Z(n) = \frac{n}{\exp(\kappa(n))}$ with $\kappa(n) = \frac{d(n) \cdot \ln(n)}{e^2}$', fontsize=14)
plt.xlabel('n (log scale)')
plt.ylabel('Z(n) (log scale)')
plt.legend()
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.tight_layout()
plt.show()