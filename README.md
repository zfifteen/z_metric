# 🧭 Z-Metric: A Relativistic Temporal Classifier on the Integer Manifold

This repository presents a novel framework for prime classification by reinterpreting the number line as a discrete spacetime lattice. The core construct, termed the **Z-metric**, defines a relativistic distance function over integers based on their prime factorizations. This metric is not merely a mathematical abstraction—it functions as a **temporal transformer**, correcting for arithmetic curvature in a manner analogous to time dilation in Minkowski spacetime.

The classifier implemented here navigates this curved integer manifold deterministically, identifying prime numbers by tracing geodesic-like paths through a structured metric space. It avoids explicit factorization by leveraging the intrinsic geometry of the number line, treating it as a four-dimensional spacetime construct with embedded curvature.

---

## 📐 Conceptual Foundations

In classical number theory, the number line is treated as a flat, one-dimensional structure. This project reimagines it as a **discrete manifold** endowed with a non-Euclidean metric derived from prime decomposition. Each integer is a point in a curved space, where the curvature reflects its multiplicative complexity.

The **Z-metric** is defined as:

\[
Z = A\left(\frac{B}{C}\right)
\]

This formulation maps directly onto the relativistic time transformation:

\[
Z = T\left(\frac{v}{c}\right)
\]

Where:
- \( A \) is a scaling factor
- \( B/C \) represents structural distortion due to factor density
- \( T(v/c) \) is the Lorentzian time dilation factor from special relativity

Thus, the Z-metric serves as a **temporal correction operator**, normalizing integer behavior relative to a fixed computational frame—analogous to how relativistic observers reconcile proper time and coordinate time.

---

## 🧪 Computational Implementation

The classifier operates deterministically, applying the Z-metric to traverse the integer manifold and isolate prime numbers. It performs the following:

- Constructs a Z-metric space over integers up to a specified bound
- Applies a hybrid filtering algorithm that avoids explicit factorization
- Validates accuracy and performance against known prime distributions

### ✅ Sample Output

```
Searching for 6000 primes...

✅ Search complete.
   - Found 6000 primes.
   - The last prime is: 59359
   - Statistics saved to 'prime_stats_hybrid_filter.csv'

--- Hybrid Filter Performance ---
   - Total Execution Time:        0.35 seconds
   - Total Numbers Checked:       59359
   - Total Composites Found:      53359
   - Composites Filtered Out:     8525
   - Filter Efficiency:           15.98%

   - Accuracy:                    The filter successfully found all target primes without false negatives.

Sanity check passed: The 6000th prime matches the expected value.
```

---

## 🧭 Physical Interpretation

The classifier navigates a **spacetime-like lattice** of integers, where:

- **Primes** behave as inertial reference points—minimal curvature, geodesically isolated
- **Composites** exhibit gravitational distortion—metric deviation due to factor density
- **Filtering** corresponds to identifying geodesic paths that avoid curvature anomalies

This analogy is functional: the classifier succeeds precisely because it treats the number line as a relativistic manifold, where metric structure governs traversal and classification.

---

## 📊 Mathematical and Physical Significance

| Domain              | Implications                                                                 |
|---------------------|------------------------------------------------------------------------------|
| **Number Theory**   | Introduces a metric-based topology over integers; novel classification of primes |
| **Computational Physics** | Demonstrates metric filtering analogous to geodesic motion in curved discrete space |
| **Cryptography**    | Offers efficient prime detection for secure key generation                   |
| **Mathematical Physics** | Bridges discrete mathematics with relativistic geometry and metric theory |

---

## 🛠️ Usage

To execute the prime search and filtering algorithm:

```bash
python main.py
```

Results are saved to `prime_stats_hybrid_filter.csv` for further analysis.

---

## 📁 Repository Structure

- `main.py` — Primary execution script  
- `z_metric.py` — Definition of the Z-metric and filtering logic  
- `prime_stats_hybrid_filter.csv` — Output statistics from the filtering process  

---

## 🔭 Future Directions

This framework opens several avenues for exploration:

- Formalizing the Z-metric as a discrete analog of Lorentzian geometry
- Extending the manifold to include negative integers or algebraic structures
- Visualizing curvature and geodesics in the integer space
- Benchmarking against classical sieves and probabilistic primality tests
- Publishing a theoretical paper on metric spaces over discrete sets

---

## 👤 Author

Developed by Dionisio ([@zfifteen](https://github.com/zfifteen)), with interests in mathematical physics, computational number theory, and geometric abstraction.

---

## 📜 License

This project is released under the [MIT License](LICENSE).


