import numpy as np
import matplotlib.pyplot as plt
from sympy import divisor_count, isprime
from math import log, exp, e

# Parameters
n_max = 1000000  # You can increase this up to ~10,000 or more for broader visualization
v = 1         # Traversal velocity

# Core Functions
def kappa(n):
    return (divisor_count(n) * log(n)) / (e ** 2)

def Z(n):
    return n / exp(v * kappa(n))

# Data Containers
primes = []
composites = []

for n in range(2, n_max + 1):
    z_n = Z(n)
    if isprime(n):
        primes.append((n, z_n))
    else:
        composites.append((n, z_n))

# Convert to arrays for plotting
primes = np.array(primes)
composites = np.array(composites)

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(primes[:, 0], primes[:, 1], color='blue', label='Primes', s=12, alpha=0.7)
plt.scatter(composites[:, 0], composites[:, 1], color='orange', label='Composites', s=12, alpha=0.4)
plt.yscale('log')
plt.xscale('log')
plt.title(r'$Z(n) = \frac{n}{\exp(\kappa(n))}$ with $\kappa(n) = \frac{d(n) \cdot \ln(n)}{e^2}$', fontsize=14)
plt.xlabel('n (log scale)')
plt.ylabel('Z(n) (log scale)')
plt.legend()
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.tight_layout()
plt.show()
