import csv
import matplotlib.pyplot as plt
from sympy import primerange, divisors, isprime

def resonance(n, prime_bound=30):
    primes = list(primerange(2, prime_bound + 1))
    return sum(1 for p in primes if n % p == 0)

def divisor_count(n):
    return len(divisors(n))

def analyze_until_nth_prime(nth_prime, prime_bound=30):
    results = []
    prime_count = 0
    n = 1

    while prime_count < nth_prime:
        res = resonance(n, prime_bound)
        divs = divisor_count(n)
        prime_status = isprime(n)
        if prime_status:
            prime_count += 1
        results.append((n, res, divs, prime_status))
        n += 1

    return results

def save_to_csv(results, filename="modular_resonance_data.csv"):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["n", "Resonance", "DivisorCount", "IsPrime"])
        for row in results:
            writer.writerow(row)

def plot_resonance_vs_divisors(results):
    x = [r[1] for r in results]  # Resonance
    y = [r[2] for r in results]  # Divisor count
    colors = ['red' if r[3] else 'blue' for r in results]  # Prime = red, Composite = blue

    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, c=colors, alpha=0.6, edgecolors='k')
    plt.xlabel("Modular Resonance (primes ≤ 30)")
    plt.ylabel("Divisor Count")
    plt.title("Resonance vs Divisor Count with Prime Highlighting")
    plt.grid(True)
    plt.legend(handles=[
        plt.Line2D([0], [0], marker='o', color='w', label='Prime', markerfacecolor='red', markersize=8),
        plt.Line2D([0], [0], marker='o', color='w', label='Composite', markerfacecolor='blue', markersize=8)
    ])
    plt.tight_layout()
    plt.show()

# Run the analysis
if __name__ == "__main__":
    nth_prime = 6000
    prime_bound = 30
    results = analyze_until_nth_prime(nth_prime, prime_bound)
    save_to_csv(results)
    plot_resonance_vs_divisors(results)
