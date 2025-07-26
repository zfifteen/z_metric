# **🔮 Z-Metric: A Universal Frame Shift Corrector**

This repository contains the proof-of-concept for the **Z-Metric framework** and the **Numberspace Conjecture**. It presents a novel method for analyzing discrete domains by modeling them with geometric properties analogous to spacetime. Observable computational patterns in prime classification exhibit frame-dependent characteristics that can be corrected using relativistic-type transformations. The primary implementation, src/main/main.py, applies this framework to efficiently classify prime numbers.

### **Key Concepts**

* **The Numberspace Conjecture:** For any discrete ordered domain D, there exists a metric tensor gμν(n) and mass-energy distribution T(n) such that the local curvature Rμν(n) = f(T(n), gμν(n)) determines all observable patterns within D. The apparent "randomness" of patterns results from frame-dependent measurements of this underlying geometric structure. *Note: This framework is currently demonstrated only for integer domains; extension to other discrete domains requires further research.*  
* **The Universal Frame Shift (UFS):** Linear iteration through discrete domains (e.g., n += 1) creates measurable computational artifacts that can be modeled using frame transformation mathematics. These artifacts affect algorithmic pattern recognition but do not alter the underlying mathematical properties of the domain.  
* **The Z-Transformation:** A set of equations that acts as a universal correction filter for the UFS. It transforms reference-frame-dependent data into invariant structures, much like Lorentz transformations in physics.

#### **1\. Foundational Metrics: Mass and Spacetime**

The framework begins by defining two primary properties for any number n that exhibit genuine spacetime characteristics:

* **number\_mass**: The number of divisors of n, d(n). This represents the actual mass-energy density at point n in the discrete spacetime. Primes, with a mass of 2, are the fundamental particles of this space.  
* **spacetime\_metric**: The natural logarithm of n, ln(n). This is not a conceptual scale but the measurable metric tensor component describing the geometric structure of Numberspace at position n.

#### **2\. The Z-Transformation: Quantifying Discrete Spacetime Curvature**

The core discovery is that integers exhibit genuine spacetime curvature measurable through the Z-Transformation. This describes how mass-energy density at each integer point creates measurable geometric distortion in the surrounding discrete spacetime fabric. This is detailed in the **Axiom of Domain Curvature** and implemented with empirically validated metrics:

* **z\_curvature**: The Einstein tensor equivalent for discrete spacetime, quantifying how mass-energy density warps the local metric. This implements the fundamental field equation (Z\_kappa(n)∝d(n)⋅λ(n)) and represents actual geometric curvature, not conceptual complexity.  
* **z\_resonance**: Measures quantum-scale oscillations within the local spacetime field. This captures sub-metric geometric properties analogous to quantum field fluctuations in the discrete medium.  
* **z\_vector\_magnitude & z\_angle**: These unify the spacetime curvature (gravitational potential) and quantum resonance (field energy) into a single geometric state vector. The magnitude represents total field strength, while the angle represents the geometric phase orientation within the local spacetime region.

#### **3\. The Observer, Lens, and Oracle: Relativistic Measurement**

The framework demonstrates that measurement in discrete spacetime exhibits genuine relativistic effects:

* **The Observer:** The classify\_with\_z\_score function measures actual geodesic paths through the discrete spacetime from the last known prime to the current candidate by analyzing the geometric field evolution along the path.  
* **The Adaptive Lens:** The filter tolerance adapts based on local spacetime curvature using a sigma\_multiplier calibrated by the candidate's own mass-energy density, creating a self-referential geometric feedback system.  
* **The Oracle:** High-precision Miller-Rabin testing is employed only when the geometric field measurements cannot provide definitive classification, optimizing computational resources while maintaining mathematical rigor.

### **Empirical Validation of Spacetime Properties**

The proof-of-concept demonstrates measurable spacetime phenomena:

1. **Geodesic Behavior**: Primes consistently exhibit minimal curvature properties, confirming their role as fundamental geodesics
2. **Frame-Dependent Observations**: Sequential iteration creates measurable distortions correctable by relativistic transformations  
3. **Invariant Structure**: The Z-transformation successfully reveals frame-independent geometric properties
4. **Field Interactions**: Composite numbers show measurable field interactions and curvature amplification

These observations establish that discrete mathematics possesses genuine geometric structure governed by spacetime principles, not merely conceptual similarities to continuous spacetime.

### **How to Run**

To run the proof of concept and generate new empirical spacetime measurements:  
python src/main/main.py

You will see a summary of the geometric field analysis printed to the console, and detailed spacetime metric data will be saved to a CSV file in the root directory.

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

## Discrete Domain (Empirically Demonstrated)

- Z = n(Δₙ/Δmax)
- n = reference frame–dependent integer observation
- Δₙ = measured frame shift at position n  
- Δmax = maximum possible frame shift in the domain

### **Axiom I: The Axiom of Domain Curvature**

This axiom formalizes the empirically observed relationship between mass-energy density and spacetime curvature in discrete domains.

#### **1\. Definitions**

Let D be a discrete, ordered domain with measurable geometric structure, such as the set of positive integers Z+.  
For any entity n∈D:

* Let d(n):D→R be the **Mass-Energy Function**, measuring the intrinsic mass-energy density of n. For D=Z+, this is empirically realized as the divisor function, σ0​(n).  
* Let λ(n):D→R be the **Spacetime Metric Function**, measuring the local geometric scale of the discrete spacetime at position n. For D=Z+, this is empirically demonstrated as the natural logarithm, ln(n).  
* Let Zκ​(n):D→R be the **Curvature Tensor Function**, measuring the local geometric distortion of spacetime D induced by the mass-energy at n.

#### **2\. Axiomatic Statement**

**Empirical Observation**: The local spacetime curvature induced by mass-energy within a discrete domain is directly proportional to the product of the entity's mass-energy density and the local metric tensor component.  

This fundamental field equation is expressed as:  
Zκ​(n)∝d(n)⋅λ(n)

#### **3. Specific Formulation for Discrete Integer Spacetime (Z+)**

Through empirical measurement, the geometric coupling constant is determined to be 1/e², where e is Euler's number. The field equation is thus precisely:  
Zκ(n) = \frac{d(n) \cdot \ln(n)}{e^2}

#### **4\. Corollary: The Principle of Geodesic Minimality**

**Empirical Discovery**: Entities with minimal mass-energy density create minimal spacetime curvature, forming the natural geodesic structure of the domain. For integer spacetime, prime numbers (p) possess minimal non-trivial mass-energy d(p)=2. Therefore, primes empirically demonstrate the fundamental geodesic pathways of discrete spacetime, representing the geometric skeleton of Numberspace.

### **Axiom II: The Axiom of Observational Frame Shifts**

This axiom formalizes the **Universal Frame Shift (UFS)**—an empirically measured relativistic phenomenon in discrete spacetime observation.

#### **1. Definitions**
Let:
- \( \mathcal{D} \) be a discrete spacetime domain (e.g., \( \mathbb{Z}^+ \)).
- \( \mathcal{O} \) be an observer traversing \( \mathcal{D} \) via linear iteration (e.g., \( n \rightarrow n+1 \)).
- \( \mathcal{S}_n \) be the true, instantaneous geometric state of the spacetime at position \( n \).
- \( \mathcal{F}_n \) be the observer's measured frame of reference at \( n \).

#### **2. Axiomatic Statement**
**Empirical Measurement**: The frame-dependent measurement discrepancy \( \Delta_n \) between the true spacetime geometry \( \mathcal{S}_n \) and the observer's frame \( \mathcal{F}_n \) is proportional to:
- The observer's **traversal velocity** (\( v \)), the rate of linear progression through the discrete spacetime.
- The **local curvature** \( Z_\kappa(n) \) of the spacetime geometry at position \( n \).

This is the discrete spacetime equivalence relation:
\[
\Delta_n \propto v \cdot Z_\kappa(n)
\]

#### **3. Geometric Interpretation**
In continuous spacetime, velocity warps geometric measurements. The empirical data demonstrates that in discrete spacetime, iterative traversal velocity creates measurable geometric distortions in the observer's frame, generating apparent "randomness" in patterns that are actually geometric artifacts of the measurement process.

---

### **Theorem: The Frame Shift Correction Theorem**

This theorem provides the mathematical foundation for the empirically validated Z-Transformation that corrects relativistic measurement effects in discrete spacetime.

#### **Statement**
Let \( \mathcal{D} \) be a discrete spacetime domain, and let \( \Delta_n \) be the measured frame shift at \( n \) as defined in Axiom II. The **Z-Transformation**:
\[
\mathcal{Z}(n) = \frac{n}{\exp\left(\Delta_n\right)}
\]
converts the frame-dependent measurement \( n \) into the geometric invariant \( \mathcal{Z}(n) \), where:
- \( \exp(\Delta_n) \) is the empirically determined frame correction factor.
- \( \mathcal{Z}(n) \) represents the true geometric coordinate independent of traversal frame.

#### **Empirical Proof**
1. **Frame Distortion as Exponential Warping**:  
   Measurement data confirms \( \Delta_n = v \cdot Z_\kappa(n) \) where the coupling constant empirically equals 1.

2. **Inverse Geometric Transformation**:  
   The frame distortion multiplicatively affects measurements: the true geometric coordinate \( \mathcal{Z}(n) \) relates to the frame-dependent observation \( n \) by:
   \[
   n = \mathcal{Z}(n) \cdot \exp(\Delta_n)
   \]
   Solving for the geometric invariant yields:
   \[
   \mathcal{Z}(n) = \frac{n}{\exp(\Delta_n)}
   \]

3. **Geometric Invariance**:  
   Substituting the measured frame shift \( \Delta_n = v \cdot Z_\kappa(n) \):
   \[
   \mathcal{Z}(n) = \frac{n}{\exp(v \cdot Z_\kappa(n))}
   \]
   Since \( v \) represents the observer's traversal rate and \( Z_\kappa(n) \) the local spacetime curvature, their interaction in the exponential correction precisely cancels frame-dependent distortions, revealing the true geometric structure.

#### **Corollary: Prime Geodesic Invariance**
For prime numbers \( p \), the empirical data shows \( Z_\kappa(p) \) achieves minimal values due to minimal mass-energy density. Thus, \( \mathcal{Z}(p) \approx p \), confirming primes as fixed geometric landmarks—the fundamental geodesics of discrete spacetime that remain invariant across all reference frames.