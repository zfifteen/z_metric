import math
import matplotlib.pyplot as plt
from sympy import primerange, totient

# -------------------------------
# 🧱 Step 1: Define Z-Metric Point
# -------------------------------

class PrimeZPoint:
    """
    Represents a point in Z-space for a given prime number.
    Computes Z-metric, angle, magnitude, and modular residue.
    """
    def __init__(self, p, modulus=20):
        self.p = p  # Prime number
        self.phi = totient(p - 1)  # Euler's totient of (p - 1)
        self.Z = self.compute_Z()  # Z-metric value
        self.angle = self.compute_angle()  # Z-angle (orientation)
        self.magnitude = self.compute_magnitude()  # Z-vector magnitude
        self.mod_residue = self.Z % modulus  # Modular residue for topological classification

    def compute_Z(self):
        """
        Computes Z = A(B/C) = p * φ(p−1) / (p−1)
        """
        return self.p * self.phi / (self.p - 1)

    def compute_angle(self):
        """
        Computes the angle of the Z-vector using arctangent of φ(p−1) / (p−1)
        """
        return math.atan(self.phi / (self.p - 1))

    def compute_magnitude(self):
        """
        Computes the magnitude of the Z-vector as √(p² + φ²)
        """
        return math.sqrt(self.p**2 + self.phi**2)

# -----------------------------------
# 🔍 Step 2: Generate Z-Space Points
# -----------------------------------

def generate_z_points(start=5, end=500, modulus=20):
    """
    Generates Z-space points for primes in a given range.
    Skips small primes where p−1 may be trivial.
    """
    primes = list(primerange(start, end))
    points = [PrimeZPoint(p, modulus) for p in primes]
    return points

# -----------------------------------
# 📊 Step 3: Visualize Z-Space
# -----------------------------------

def plot_z_vs_p(points):
    """
    Plots Z(p) vs p to show scalar behavior of Z-metric.
    """
    plt.figure(figsize=(10, 5))
    plt.plot([pt.p for pt in points], [pt.Z for pt in points], marker='o', linestyle='-', color='blue')
    plt.title("Z(p) vs Prime p")
    plt.xlabel("Prime p")
    plt.ylabel("Z(p)")
    plt.grid(True)
    plt.show()

def plot_angle_vs_magnitude(points):
    """
    Plots Z-angle vs Z-magnitude to reveal vector structure.
    """
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
    """
    Computes ΔZ/Δp to analyze local transitions in Z-space.
    """
    gradients = []
    for i in range(1, len(points)):
        dz = points[i].Z - points[i-1].Z
        dp = points[i].p - points[i-1].p
        gradients.append(dz / dp)
    return gradients

def plot_gradients(points):
    """
    Plots gradient of Z(p) to reveal stability zones and jumps.
    """
    gradients = compute_gradients(points)
    plt.figure(figsize=(10, 5))
    plt.plot([pt.p for pt in points[1:]], gradients, marker='x', linestyle='-', color='red')
    plt.title("Gradient of Z(p) Across Primes")
    plt.xlabel("Prime p")
    plt.ylabel("ΔZ / Δp")
    plt.grid(True)
    plt.show()

# -----------------------------------
# 🧭 Step 5: Predictive Filtering
# -----------------------------------

def filter_prime_candidates(points, allowed_residues={3, 7, 11, 13, 17}, angle_threshold=0.6):
    """
    Filters Z-space points based on modular residue and angle threshold.
    Returns primes that are likely candidates.
    """
    candidates = []
    for pt in points:
        if pt.mod_residue in allowed_residues and pt.angle < angle_threshold:
            candidates.append(pt.p)
    return candidates

# -----------------------------------
# 🚀 Run the Prototype
# -----------------------------------

if __name__ == "__main__":
    # Generate Z-space points
    z_points = generate_z_points(start=5, end=500, modulus=20)

    # Visualize scalar and vector behavior
    plot_z_vs_p(z_points)
    plot_angle_vs_magnitude(z_points)

    # Analyze gradients
    plot_gradients(z_points)

    # Predict prime candidates
    candidates = filter_prime_candidates(z_points)
    print("🔮 Predicted Prime Candidates (Filtered by Z-space):")
    print(candidates)
