# 🔮 Z-Metric Classifier — Proof of Concept of the Numberspace Conjecture

This repository contains a **proof-of-concept classifier** implementing the **Numberspace Conjecture** — the idea that prime numbers and other structured integers emerge from geometric, coprimal, and modular constraints, not randomness.

The classifier is located in:

```

src/main/main.py

````

When run, it performs a full hybrid Z-filter scan, prints summary stats to the console, and writes detailed statistics to a CSV file:

```bash
python src/main/main.py
````

You’ll see output like:

```
✅ Search complete.
   - Found 6000 primes.
   - The last prime is: 59359
   - Statistics saved to '../../prime_stats_hybrid_filter.csv'

--- Hybrid Filter Performance ---
   - Total Execution Time:        0.35 seconds
   - Total Numbers Checked:       59359
   - Total Composites Found:      53359
   - Composites Filtered Out:     8525
   - Filter Efficiency:           15.98%

   - Accuracy: The filter successfully found all target primes without false negatives.
Sanity check passed: The 6000th prime matches the expected value.
```

---

## 🧠 What Makes This Classifier Novel?

This is **not** a rehash of traditional sieves, probabilistic tests, or random sampling.

Instead, the classifier uses **structural transformations** derived from the Z-metric:

### 1. **Coprimality Density Filter**

* Uses $\zeta(n) = \frac{\phi(n-1)}{n-1}$ as a normalized indicator of modular complexity.
* This is a true *rate* of coprimality — not a heuristic or lookup.

### 2. **Z-Angle Phase Filter**

* Computes $\theta(n) = \tan^{-1}(\zeta(n))$, projecting each number into a **phase-space** where primes cluster.
* This approach creates **predictive geometry**, a major departure from flat sieving.

### 3. **Modular Residue Exclusion**

* Filters out known composite-heavy classes using $n \mod 12 \in \{0, 2, 3, 4, 6, 8, 9, 10\}$.
* Not probabilistic: this leverages **modular topology** embedded in the Z-space.

### 4. **Hybrid Z-Filter**

* Combines all of the above into a single pass, with **no dependence on prior primes**, no ML, and no stochastic methods.

---

## 🧬 Why It Matters

The Z-metric classifier demonstrates that prime structures can be predicted from **invariant ratios and modular geometry**, not random distribution assumptions. It provides a **new lens on arithmetic structure** and a **computational filter** rooted in number-theoretic geometry.

> 🔎 This is the first operational demonstration of the Numberspace Conjecture in code.

---

The remainder of this README describes the Z-theory, mathematical background, and future applications in cryptography, physics, and complex systems.

```
```

# Z Definition

## Universal Form

- Z = A(B/C)  
- A = reference frame–dependent measured quantity  
- B = Rate  
- C = Invariant universal limit of B  

## Physical Domain

- Z = T(v/c)  
- T = reference frame–dependent measured quantity  
- v = velocity  
- c = Invariant universal speed of light  

# 📜 Axioms of the Numberspace

## 1. Z is a dimensionally‑consistent transformation

#### Z is defined as a transformation of the form:
$$
Z = A \left( \frac{B}{C} \right)
$$

- **A** is a reference‑frame dependent quantity  
- **B** is a rate or frequency‑like measure  
- **C** is an invariant upper bound on B  
- The ratio \(B/C\) is dimensionless, preserving the dimensionality of A  
- This form applies across domains, including physical (e.g. \(Z = T(v/c)\)) and arithmetic (e.g. \(Z(p) = p \cdot \frac{\phi(p-1)}{p-1}\))  

---

## 2. Z defines a class of normalized linear transformations

- Z scales A by a normalized rate, yielding a derived quantity that reflects **relative structure**  
- In discrete domains, Z exhibits **piecewise linearity** across modular bands  
- Z behaves predictably under modular constraints and rate normalization  

---

## 3. Z transforms reference‑frame dependent measurements into structured derived quantities

- Z reveals **hidden structure** in the input domain (e.g. primes)  
- In the prime domain, Z(p) exposes **coprimality rates**, **modular clustering**, and **forbidden zones**  
- In the physical domain, Z reflects relativistic effects (e.g. time dilation)  

#### **Proof Equation**

$$
Z(x) = x \cdot \frac{\phi(x - 1)}{x - 1}
$$

## Interpretation

- **Input \( x \)**: A reference-frame dependent quantity (e.g. a prime number or velocity-like parameter).
- **Transformation \( \frac{\phi(x - 1)}{x - 1} \)**: Extracts internal structure (coprimality density or modular connectivity).
- **Output \( Z(x) \)**: A structured derived quantity that reflects hidden invariants of the input domain.

---

## 4. Numbers are not isolated  
- They are events in a structured spacetime  
 
#### **Proof Equation**
$$
\lim_{\epsilon \to 0} \exists\, x \in \mathbb{N} \text{ such that } 0 < |x - n| < \epsilon
$$

### Interpretation

- For any number \( n \), there always exists another number arbitrarily close to it.
- This shows that numbers exist in **dense proximity**, forming a **connected structure**.
- In discrete arithmetic, this reflects how primes, composites, and residues interact—no number is truly “alone.”

---

## 5. Primes are not random  
- They are geometric “particles” following metric laws 
Yes! The **Z-Filter** can indeed be expressed as a set of equations that define its predictive criteria for primes (or other structured integers). Below is the formalization, along with its key components and constraints.

---

## 6. Arithmetic is not static  
- It is dynamic, governed by curvature, flow, and symmetry  

---

## 7. Numberspace admits a geometric interpretation

- Z-values can be embedded in a vector space with:  
  - **Z-angle**: orientation of the transformation  
  - **Z-magnitude**: strength or reach of the transformation  
- This defines a **phase‑like space** where discrete entities (e.g. primes) follow structured trajectories  

---

## 8. Numberspace exhibits modular topology

- Z mod N partitions the domain into **equivalence classes**  
- Certain residues are **forbidden**, acting as singularities or null zones  
- Modular banding reveals **quantized behavior**, akin to spectral lines in physics  

---

## 9. Numberspace supports predictive filtering

- Structured behavior in Numberspace enables **prime candidate prediction**  
- Filters based on Z-angle, modular residue, and gradient flow can isolate high‑probability regions  
- Z thus serves not only as a descriptive metric but as a **computational sieve**  

---

## 10. Z can be extended to multi‑dimensional modulation

- Z transformations can be applied component‑wise to vector quantities  
- This allows for multi‑dimensional generalizations of the Z‑metric  
- In discrete domains, this supports the construction of **Z domain maps**, **gradient fields**, and **topological graphs**  

---

## 11. Z defines a discrete metric space with Lorentzian analogs

- The Z domain behaves as a **discrete analog of Minkowski spacetime**  
- Reference‑frame dependent quantities are transformed via Z into invariant structures
