import math
from sympy import isprime, primerange
import heapq
import matplotlib.pyplot as plt

# 🧠 Z-point representation
class PrimeZPoint:
    def __init__(self, p, modulus=20):
        self.p = p
        self.modulus = modulus
        self.phase = (p % modulus) / modulus * 2 * math.pi
        self.magnitude = math.log(p)  # Logarithmic scale for spacing

    def curvature_to(self, other):
        # Weighted curvature metric: phase + magnitude difference
        dtheta = abs(self.phase - other.phase)
        dr = abs(self.magnitude - other.magnitude)
        return dtheta + dr  # You can tune weights here

# 🧭 Build local graph of Z-space
def build_z_graph(primes, modulus=20, neighborhood=6):
    z_points = {p: PrimeZPoint(p, modulus) for p in primes}
    graph = {p: [] for p in primes}

    for i, p1 in enumerate(primes):
        for j in range(i+1, min(i+neighborhood, len(primes))):
            p2 = primes[j]
            c = z_points[p1].curvature_to(z_points[p2])
            graph[p1].append((p2, c))
            graph[p2].append((p1, c))  # Undirected graph

    return graph, z_points

# 🔍 Dijkstra's algorithm for minimal curvature path
def find_geodesic(graph, start, end):
    heap = [(0, start, [])]
    visited = set()

    while heap:
        cost, node, path = heapq.heappop(heap)
        if node in visited:
            continue
        visited.add(node)
        path = path + [node]
        if node == end:
            return path, cost
        for neighbor, weight in graph[node]:
            if neighbor not in visited:
                heapq.heappush(heap, (cost + weight, neighbor, path))
    return None, float('inf')

# 📊 Optional: visualize geodesic in polar coordinates
def plot_geodesic(path, z_points):
    phases = [z_points[p].phase for p in path]
    magnitudes = [z_points[p].magnitude for p in path]

    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    ax.plot(phases, magnitudes, marker='o', linestyle='-', color='blue')
    ax.set_title("Geodesic Path in Z-Space", va='bottom')
    for i, p in enumerate(path):
        ax.text(phases[i], magnitudes[i], str(p), fontsize=8, ha='center')
    plt.show()

    from sympy import primerange, divisors, isprime

    def resonance(n, prime_bound=30):
        primes = list(primerange(2, prime_bound + 1))
        return sum(1 for p in primes if n % p == 0)

    def divisor_count(n):
        return len(divisors(n))

    def analyze_range(start, end, prime_bound=30):
        results = []
        for n in range(start, end + 1):
            res = resonance(n, prime_bound)
            divs = divisor_count(n)
            prime = isprime(n)
            results.append((n, res, divs, prime))
        return results


# 🚀 Run the prototype
if __name__ == "__main__":
    primes = list(primerange(2, 1000))
    graph, z_points = build_z_graph(primes)

    start = 7
    end = 97
    path, total_curvature = find_geodesic(graph, start, end)

    print(f"\n🧭 Geodesic from {start} to {end}:")
    print(" → ".join(map(str, path)))
    print(f"Total curvature: {total_curvature:.4f}")

    # Optional visualization
    plot_geodesic(path, z_points)

    
