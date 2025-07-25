"""
Z(p) Spectral Analysis Toolkit
Author: Dionisio Alberto Lopez III
Date: July 2025

Overview:
This script explores the spectral behavior of the function:
    Z(p) = p * (φ(p - 1) / (p - 1))
where φ is Euler's totient function and p is prime.

Conceptual Journey:
- The Z-metric was originally conceived as a discrete analog of Lorentzian geometry.
- Z(p) = A * (B / C), where:
    A = p
    B = φ(p - 1)
    C = p - 1
- This formulation captures the "internal symmetry" of the prime via its predecessor.
- Empirical analysis revealed that Z(p)/p clusters into discrete bands — akin to energy levels.
- Modular residue analysis (mod 20) exposed forbidden states and band structure.
- These findings support the hypothesis of a prime energy landscape.

Modules:
1. Core Z(p) computation
2. Statistical analysis
3. Modular residue analysis
4. Visualization suite
"""

# === Imports ===
from sympy import primerange, totient
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

# === 1. Core Z(p) Computation ===
def compute_Z(p: int) -> float:
    """
    Computes Z(p) = p * (φ(p - 1) / (p - 1))
    This reflects the 'symmetry density' of the prime's predecessor.
    """
    return p * (totient(p - 1) / (p - 1))

# === 2. Statistical Analysis ===
def analyze_Z_function(max_prime=10 ** 5):
    primes = list(primerange(3, max_prime))

    # Convert Z(p) values to native floats for compatibility with NumPy
    Z_values = [float(compute_Z(p)) for p in primes]
    Z_ratios = [Z / p for Z, p in zip(Z_values, primes)]

    # Now safe to use NumPy
    mean_Z = np.mean(Z_values)
    std_Z = np.std(Z_values)
    mean_ratio = np.mean(Z_ratios)
    min_ratio, max_ratio = min(Z_ratios), max(Z_ratios)
    Z_gaps = [Z_values[i + 1] - Z_values[i] for i in range(len(Z_values) - 1)]
    mean_gap = np.mean(Z_gaps)

    print(f"=== Z(p) Statistical Summary ===")
    print(f"Sample size: {len(primes)} primes")
    print(f"Prime range: {primes[0]} to {primes[-1]}")
    print(f"\nZ(p)/p Mean: {mean_ratio:.6f}")
    print(f"Z(p)/p Std Dev: {std_Z:.6f}")
    print(f"Z(p)/p Min: {min_ratio:.6f}")
    print(f"Z(p)/p Max: {max_ratio:.6f}")
    print(f"Mean Z-gap: {mean_gap:.6f}")

    return primes, Z_values, Z_ratios


# === 3. Modular Residue Analysis ===
def modular_residue_analysis(Z_values, modulus=20):
    """
    Analyzes Z(p) mod modulus to detect band structure and forbidden states.
    """
    residues = [int(Z) % modulus for Z in Z_values]
    residue_counts = Counter(residues)

    print(f"\n=== Modular Residue Analysis (mod {modulus}) ===")
    for r in range(modulus):
        count = residue_counts.get(r, 0)
        percentage = (count / len(Z_values)) * 100
        print(f"  Z ≡ {r} (mod {modulus}): {count} primes ({percentage:.2f}%)")

    return residue_counts

# === 4. Visualization Suite ===
def visualize_Z(primes, Z_values, Z_ratios):
    """
    Plots Z(p) and normalized Z(p)/p against prime index.
    Highlights clustering and spectral bands.
    """
    plt.figure(figsize=(12, 6))

    # Z(p) vs p
    plt.subplot(1, 2, 1)
    plt.plot(primes, Z_values, color='blue', label='Z(p)')
    plt.xlabel("Prime p")
    plt.ylabel("Z(p)")
    plt.title("Z(p) vs Prime p")
    plt.grid(True)
    plt.legend()

    # Z(p)/p vs p
    plt.subplot(1, 2, 2)
    plt.plot(primes, Z_ratios, color='green', label='Z(p)/p')
    plt.axhline(y=6/np.pi**2, color='red', linestyle='--', label='6/π² ≈ 0.6079')
    plt.xlabel("Prime p")
    plt.ylabel("Z(p)/p")
    plt.title("Normalized Z(p)/p vs Prime p")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

def visualize_Z_gaps(Z_values):
    """
    Plots histogram of Z(p) gaps to reveal quantized transitions.
    """
    Z_gaps = [Z_values[i+1] - Z_values[i] for i in range(len(Z_values)-1)]
    plt.figure(figsize=(10, 5))
    plt.hist(Z_gaps, bins=50, color='purple', edgecolor='black')
    plt.xlabel("Z(p_{n+1}) - Z(p_n)")
    plt.ylabel("Frequency")
    plt.title("Histogram of Z(p) Gaps")
    plt.grid(True)
    plt.show()

# === 5. Execution ===
if __name__ == "__main__":
    # Step 1: Analyze Z(p) up to a chosen bound
    primes, Z_values, Z_ratios = analyze_Z_function(max_prime=10**4)

    # Step 2: Modular residue analysis to detect spectral bands
    modular_residue_analysis(Z_values, modulus=20)

    # Step 3: Visualize Z(p) and its normalized form
    visualize_Z(primes, Z_values, Z_ratios)

    # Step 4: Visualize Z(p) gaps to explore quantization
    visualize_Z_gaps(Z_values)
