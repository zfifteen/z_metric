import numpy as np
import matplotlib.pyplot as plt
from sympy import divisor_count, isprime
from math import log, exp, e, atan, sqrt, fmod
import csv

# Parameters
n_max = 6000  # Reduced for execution feasibility
v = 1         # Traversal velocity

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

# Write to CSV
with open('z_6d_data.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['n', 'Number Mass', 'Spacetime Metric', 'Z-Curvature', 'Z-Resonance', 'Z-Vector Magnitude', 'Z-Angle', 'Z(n)', 'Ghost Mass', 'Status'])
    writer.writerows(data)

# Separate primes and composites for plotting
primes = [row for row in data if row[9] == 'Prime']
composites = [row for row in data if row[9] == 'Composite']

# Convert to arrays for plotting
primes = np.array(primes)
composites = np.array(composites)

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(primes[:, 0].astype(float), primes[:, 7].astype(float), color='blue', label='Primes', s=12, alpha=0.7)
plt.scatter(composites[:, 0].astype(float), composites[:, 7].astype(float), color='orange', label='Composites', s=12, alpha=0.4)
plt.yscale('log')
plt.xscale('log')
plt.title(r'$Z(n) = \frac{n}{\exp(\kappa(n))}$ with $\kappa(n) = \frac{d(n) \cdot \ln(n)}{e^2}$', fontsize=14)
plt.xlabel('n (log scale)')
plt.ylabel('Z(n) (log scale)')
plt.legend()
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.tight_layout()
plt.show()