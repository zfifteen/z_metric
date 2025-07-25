import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sympy import primerange, totient

# -------------------------------
# 🧱 Step 1: Define Z-Metric Point
# -------------------------------

class PrimeZPoint:
    def __init__(self, p, modulus=20):
        self.p = p
        self.phi = float(totient(p - 1))  # Cast to float
        self.Z = self.compute_Z()
        self.angle = self.compute_angle()
        self.magnitude = self.compute_magnitude()
        self.mod_residue = self.Z % modulus

    def compute_Z(self):
        return self.p * self.phi / (self.p - 1)

    def compute_angle(self):
        return math.atan(self.phi / (self.p - 1))

    def compute_magnitude(self):
        return math.sqrt(self.p**2 + self.phi**2)

# -----------------------------------
# 🔍 Step 2: Generate Z-Space Points
# -----------------------------------

def generate_z_points(start=5, end=500, modulus=20):
    primes = list(primerange(start, end))
    return [PrimeZPoint(p, modulus) for p in primes]

# -----------------------------------
# 📊 Step 3: Visualize Z-Space
# -----------------------------------

def plot_z_vs_p(points):
    plt.figure(figsize=(10, 5))
    plt.plot([pt.p for pt in points], [pt.Z for pt in points], marker='o', linestyle='-', color='blue')
    plt.title("Z(p) vs Prime p")
    plt.xlabel("Prime p")
    plt.ylabel("Z(p)")
    plt.grid(True)
    plt.show()

def plot_angle_vs_magnitude(points):
    plt.figure(figsize=(10, 5))
    plt.scatter([pt.angle for pt in points], [pt.magnitude for pt in points],
                c=[pt.mod_residue for pt in points], cmap='viridis', s=50)
    plt.title("Z-Angle vs Z-Magnitude (Colored by Z mod modulus)")
    plt.xlabel("Z-Angle (radians)")
    plt.ylabel("Z-Magnitude")
    plt.colorbar(label="Z mod modulus")
    plt.grid(True)
    plt.show()

# -----------------------------------
# 🔁 Step 4: Gradient Analysis
# -----------------------------------

def compute_gradients(points):
    gradients = []
    for i in range(1, len(points)):
        dz = points[i].Z - points[i-1].Z
        dp = points[i].p - points[i-1].p
        gradients.append(dz / dp)
    return gradients

def plot_gradients(points):
    gradients = compute_gradients(points)
    plt.figure(figsize=(10, 5))
    plt.plot([pt.p for pt in points[1:]], gradients, marker='x', linestyle='-', color='red')
    plt.title("Gradient of Z(p) Across Primes")
    plt.xlabel("Prime p")
    plt.ylabel("ΔZ / Δp")
    plt.grid(True)
    plt.show()

# -----------------------------------
# 📤 Step 5: Remainder Statistics
# -----------------------------------

def compute_remainder_statistics(points):
    from collections import defaultdict
    stats = defaultdict(list)
    for pt in points:
        stats[int(pt.mod_residue)].append(pt)

    summary = []
    for mod_class, pts in stats.items():
        z_vals = [p.Z for p in pts]
        angles = [p.angle for p in pts]
        magnitudes = [p.magnitude for p in pts]
        summary.append({
            "Modular Class": mod_class,
            "Count": len(pts),
            "Mean Z": np.mean(z_vals),
            "Mean Angle": np.mean(angles),
            "Mean Magnitude": np.mean(magnitudes),
            "Std Z": np.std(z_vals),
            "Std Angle": np.std(angles)
        })
    return pd.DataFrame(summary).sort_values("Modular Class")

def export_remainder_stats(points, filename="remainder_stats.csv"):
    df = compute_remainder_statistics(points)
    df.to_csv(filename, index=False)
    print(f"📊 Remainder statistics exported to {filename}")

# -----------------------------------
# 🧭 Step 6: Predictive Filtering
# -----------------------------------

def filter_prime_candidates(points, allowed_residues={3, 7, 11, 13, 17}, angle_threshold=0.6):
    return [pt.p for pt in points if int(pt.mod_residue) in allowed_residues and pt.angle < angle_threshold]

# -----------------------------------
# 🚀 Run the Prototype
# -----------------------------------

if __name__ == "__main__":
    # Generate exactly the first 6000 primes
    primes = list(primerange(2, 600000))  # Overshoot to ensure coverage
    target_primes = primes[:6000]
    z_points = [PrimeZPoint(p, modulus=20) for p in target_primes]

    # ✅ Sanity check: 6000th prime should be 59359
    expected = 59359
    actual = target_primes[-1]
    print(f"\n✅ Search complete.")
    print(f"   - Found {len(target_primes)} primes.")
    print(f"   - The last prime is: {actual}")
    if actual == expected:
        print(f"   - Sanity check passed: The 6000th prime matches the expected value.")
    else:
        print(f"   ❌ Sanity check failed: Expected {expected}, but got {actual}")

    # Continue with analysis
    plot_z_vs_p(z_points)
    plot_angle_vs_magnitude(z_points)
    plot_gradients(z_points)

    export_remainder_stats(z_points)

    candidates = filter_prime_candidates(z_points)
    print("\n🔮 Predicted Prime Candidates (Filtered by Z-space):")
    print(candidates)
