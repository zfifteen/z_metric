import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Cognitive parameters (empirically estimated constants)
COGNITIVE_VELOCITY = 1.0  # v_c (baseline neural processing speed)
COGNITIVE_COUPLING = 1.0  # k (subliminal-conscious coupling)
LOAD_FACTOR = 0.5  # Cognitive load amplification


def compute_cognitive_curvature(n):
    """Compute Zκ(n) = d(n)·ln(n)/e²"""
    if n < 2:
        return 0.0
    # Count divisors (mass-energy density)
    divisors = 0
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            divisors += 1 if i * i == n else 2
    # Compute curvature
    return divisors * math.log(n) / (math.e ** 2)


def subliminal_frame_shift(n, cognitive_load=0.0):
    """Compute Δs = k·v_c(1 + load)·Zκ(n)"""
    v_c = COGNITIVE_VELOCITY * (1 + cognitive_load * LOAD_FACTOR)
    return COGNITIVE_COUPLING * v_c * compute_cognitive_curvature(n)


def conscious_perception(n, cognitive_load=0.0):
    """Simulate conscious perception: n_conscious = 𝒵(n)·exp(Δs)"""
    delta_s = subliminal_frame_shift(n, cognitive_load)
    bias_factor = math.exp(delta_s)
    return n * bias_factor  # Inflated conscious perception


def z_transform(perceived_n, cognitive_load=0.0):
    """Recover true invariant: 𝒵(n) = n_conscious / exp(Δs)"""
    delta_s = subliminal_frame_shift(perceived_n, cognitive_load)
    return perceived_n / math.exp(delta_s)


def simulate_cognitive_experiment(n_values, cognitive_load=0.0):
    """Run full perception simulation for number range"""
    results = []
    for n in n_values:
        true_invariant = z_transform(n, cognitive_load)
        perceived = conscious_perception(n, cognitive_load)
        distortion = perceived - n
        results.append((n, true_invariant, perceived, distortion))
    return results


def neural_processing_simulation(n_values):
    """Model neural response to primes vs composites"""
    prime_responses = []
    comp_responses = []

    for n in n_values:
        # Neural activity proportional to curvature (simulated fMRI)
        neural_activity = compute_cognitive_curvature(n)

        if number_of_divisors(n) == 2:  # Prime
            prime_responses.append((n, neural_activity))
        else:  # Composite
            comp_responses.append((n, neural_activity))

    return prime_responses, comp_responses


def ai_validation_experiment(max_n=1000):
    """Test if Z-transformation improves prime prediction"""
    # Generate labeled data
    X = np.array([[n, compute_cognitive_curvature(n)] for n in range(2, max_n)])
    y_raw = np.array([1 if number_of_divisors(n) == 2 else 0 for n in range(2, max_n)])

    # Apply Z-transformation
    X_transformed = np.array([[z_transform(n), compute_cognitive_curvature(n)]
                              for n in range(2, max_n)])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y_raw, test_size=0.2)
    Xt_train, Xt_test, _, _ = train_test_split(X_transformed, y_raw, test_size=0.2)

    # Train models
    model_raw = RandomForestClassifier().fit(X_train, y_train)
    model_transformed = RandomForestClassifier().fit(Xt_train, y_train)

    # Evaluate
    raw_score = model_raw.score(X_test, y_test)
    trans_score = model_transformed.score(Xt_test, y_test)

    return raw_score, trans_score


# Helper functions
def number_of_divisors(n):
    """Count divisors of n"""
    if n < 2:
        return 0
    count = 0
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            count += 1 if i * i == n else 2
    return count


# Visualization
def plot_perceptual_distortion(results):
    """Plot conscious perception vs true invariants"""
    n_vals = [r[0] for r in results]
    true_vals = [r[1] for r in results]
    perceived_vals = [r[2] for r in results]

    plt.figure(figsize=(12, 8))
    plt.plot(n_vals, n_vals, 'g--', label="Ideal Perception")
    plt.plot(n_vals, true_vals, 'b-', linewidth=2, label="True Invariant 𝒵(n)")
    plt.plot(n_vals, perceived_vals, 'r-', label="Conscious Perception")

    # Highlight primes
    primes = [n for n in n_vals if number_of_divisors(n) == 2]
    plt.scatter(primes, [r[2] for r in results if r[0] in primes],
                c='purple', s=50, label="Primes")

    plt.xlabel("Actual Number (n)")
    plt.ylabel("Perceived Value")
    plt.title("Cognitive Distortion in Numerical Perception")
    plt.legend()
    plt.grid(True)
    plt.show()


# Main simulation
if __name__ == "__main__":
    # Experimental parameters
    numbers = range(2, 51)
    cognitive_load = 0.7  # 70% cognitive load

    print("=== Cognitive Frame Shift Simulation ===")
    print("n | True 𝒵(n) | Conscious Perception | Distortion")
    print("-" * 50)

    results = simulate_cognitive_experiment(numbers, cognitive_load)
    for n, true, perceived, distortion in results:
        print(f"{n:2} | {true:8.4f} | {perceived:18.4f} | {distortion:+.4f}")

    # Neural processing simulation
    prime_resp, comp_resp = neural_processing_simulation(numbers)

    # AI validation
    raw_acc, trans_acc = ai_validation_experiment(1000)
    print(f"\nAI Validation Results:")
    print(f"Raw Data Accuracy: {raw_acc:.4f}")
    print(f"Z-Transformed Accuracy: {trans_acc:.4f}")
    print(f"Improvement: {(trans_acc - raw_acc) * 100:.2f}%")

    # Visualization
    plot_perceptual_distortion(results)