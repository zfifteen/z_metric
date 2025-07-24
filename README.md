# Z-Prime: A Hybrid Probabilistic Prime Number Filter

This script implements a novel, experimental algorithm for identifying prime numbers. It uses a hybrid approach that combines a probabilistic filter based on a custom "Z-metric" with a deterministic primality test (Miller-Rabin). The goal is to efficiently filter out composite numbers, thereby reducing the number of expensive primality tests required.

The core of this project is a theoretical framework called the **"Ontology of Z,"** which treats numbers as entities in a "number-spacetime." Their properties are measured by a set of "Z-metrics" that describe how a number's "mass" (divisor count) "warps" the spacetime around it.

## Core Concepts 💡

### The Ontology of Z

The Z-formalism is built on a few core ideas:

* **Number Mass (`get_number_mass`)**: This is the foundational concept, where the "mass" of a number is its total count of divisors.
    * **Prime numbers**, with exactly two divisors, are seen as fundamental particles with a "rest mass" of 2.
    * **Composite numbers** are more massive bodies, as their divisor count is always greater than 2.
* **Spacetime Metric (`math.log(n)`)**: The natural logarithm of a number is treated as the fabric of "number-spacetime," defining scale and distance.
* **Z-Curvature**: This metric measures how much a number's "mass" warps the "spacetime" around it. It's the primary indicator of a number's complexity.
* **Z-Resonance**: This can be viewed as a number's internal "vibrational mode" within the local spacetime.
* **Z-Vector**: This combines curvature (potential energy) and resonance (kinetic energy) into a single "field strength" or "energy" measurement for the number.

### Primality as a Right Angle

A key insight of this model is the idea that a **prime number corresponds to a right angle (90°)** in the Z-triangle, which is formed by the `z_curvature` and `z_resonance` components. When a number is prime, one of these components tends to vanish, causing the "Z-angle" to approach 90 degrees.

## How the Algorithm Works ⚙️

The script walks the number line, testing each integer to see if it's prime. It uses a "geodesic" approach, analyzing the "transition path" from the last known prime to the current candidate number.

1.  **Calculate Z-Metrics**: For each number, the full set of Z-metrics is calculated using `get_z_metrics()`.
2.  **Probabilistic Filtering (`classify_with_z_score`)**:
    * Instead of immediately testing for primality, the algorithm first calculates "geodesic deviation" scores (`z1`, `z4`) that describe the path from the last prime found.
    * These scores are compared against an "observational window" derived from empirically determined means and standard deviations for prime-to-prime transitions.
    * If a candidate's transition path deviates too far from the expected path for a prime, it is classified as **composite** and the expensive primality test is **skipped**.
3.  **Deterministic Check (`is_prime`)**:
    * If the candidate's path is within the expected range for a prime, it is then sent to the "Oracle"—the highly efficient Miller-Rabin primality test—for a definitive answer.
4.  **State Updates**: When a prime is confirmed, its number and Z-metrics are stored as the new "stable state" for the next set of comparisons.

This hybrid approach allows the algorithm to quickly discard a large percentage of composite numbers without performing a full primality test, leading to significant performance gains.

## How to Run the Code 🚀

1.  **Prerequisites**: You need Python 3.x installed. No external libraries are required.
2.  **Execution**: Simply run the script from your terminal:
    ```bash
    python main.py
    ```
3.  **Configuration**: You can adjust the following parameters in the `if __name__ == '__main__':` block:
    * `primes_to_find`: The total number of primes the script should find before stopping.
    * `candidate_number`: The starting integer for the search.
    * `csv_file_name`: The name of the output file where statistics will be saved.

## Output and Performance 📊

The script will print its progress to the console and generate a CSV file (`prime_stats_hybrid_filter.csv`) with the following columns for each number checked:

* `n`: The integer being tested.
* `is_prime`: `1` if the number is prime, `0` otherwise.
* `was_skipped`: `True` if the Miller-Rabin test was skipped for this number, `False` otherwise.
* `z1_score`: The calculated z1 geodesic deviation score.
* `z4_score`: The calculated z4 geodesic deviation score.

At the end of the run, the script prints a performance summary, including:

* Total execution time.
* The number of primes found and the value of the last prime.
* The total number of composites filtered out by the Z-score method.
* **Filter Efficiency**: The percentage of composite numbers that were correctly identified and skipped by the probabilistic filter.

The script also includes a sanity check to verify that the 6000th prime found matches the known correct value.