# **🔮 Z-Metric: A Universal Frame Shift Corrector**

This repository contains the proof-of-concept for the **Z-Metric framework** and the **Numberspace Conjecture**. It presents a novel method for analyzing discrete domains by treating observable patterns (like the prime number distribution) not as random, but as relativistic artifacts of linear observation.  
The core of this work is a universally applicable system for modeling and correcting these "observational frame shifts." The primary implementation, src/main/main.py, applies this framework to efficiently classify prime numbers.

### **Key Concepts**

* **The Numberspace Conjecture:** Posits that all discrete domains (integers, financial data, etc.) possess a holistic, interconnected structure. The patterns we see are emergent properties based on how we observe this structure.  
* **The Universal Frame Shift (UFS):** The act of linear, sequential observation (e.g., an algorithm iterating n \+= 1\) creates a relativistic discrepancy between the observer's frame of reference and the true, instantaneous state of the system.  
* **The Z-Transformation:** A set of equations that acts as a universal correction filter for the UFS. It transforms reference-frame-dependent data into invariant structures, much like Lorentz transformations in physics.

### **How It Works: The Spacetime Analogy**

The main.py script demonstrates the theory by modeling the domain of integers as a discrete spacetime. The **Hybrid Filter** uses this model to perform a dynamic, path-dependent analysis.

#### **1\. Foundational Metrics: Mass and Spacetime**

The framework begins by defining two primary properties for any number n:

* **number\_mass**: The number of divisors of n, d(n). This is analogous to an object's rest mass, representing its intrinsic complexity. Primes, with a mass of 2, are fundamental particles.  
* **spacetime\_metric**: The natural logarithm of n, ln(n). This represents the underlying fabric or scale of the Numberspace at the position of n.

#### **2\. The Z-Transformation: Quantifying the Spacetime Connection**

The core innovation is the Z-Transformation, which describes how an entity's mass interacts with and distorts the spacetime around it. This is detailed in the **Axiom of Domain Curvature** and implemented with the following novel metrics:

* **z\_curvature**: This is the central metric, quantifying how much the number\_mass warps the spacetime\_metric. It's the direct implementation of the axiom (Z\_kappa(n)proptod(n)cdotlambda(n)) and is analogous to gravitational curvature in general relativity.  
* **z\_resonance**: This metric measures a number's "internal vibrational mode" within the local field. It's derived from the remainder of the number's interaction with the log-space, representing a "quantum" or sub-manifold property.  
* **z\_vector\_magnitude & z\_angle**: These metrics unify curvature (potential energy) and resonance (kinetic energy) into a single state vector. The magnitude represents the total "field strength" of the number, while the angle represents its "phase" or orientation within the field.

#### **3\. The Observer, Lens, and Oracle**

The framework then uses these spacetime metrics to intelligently hunt for primes:

* **The Observer:** Instead of a simple check, the classify\_with\_z\_score function analyzes the "geodesic path" from the last known prime to the current candidate by measuring the change in the Z-field.  
* **The Adaptive Lens:** The filter's tolerance is not fixed. It uses a sigma\_multiplier that is dynamically calibrated by the candidate number's own "mass," creating a self-referential feedback loop.  
* **The Oracle:** A high-certainty Miller-Rabin test (is\_prime) is used only when the low-cost Z-filter cannot confidently classify a candidate, conserving computational energy.

### **How to Run**

To run the proof of concept and generate a new statistics file:  
python src/main/main.py

You will see a summary of the filter's performance printed to the console, and a detailed CSV file will be saved to the root directory.
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

### **Axiom I: The Axiom of Domain Curvature**

This axiom establishes the fundamental principle of the Z-Metric framework, defining the relationship between an entity's intrinsic complexity and the structure of the domain it inhabits.

#### **1\. Definitions**

Let D be a discrete, ordered domain, such as the set of positive integers Z+.  
For any entity n∈D:

* Let d(n):D→R be the **Mass Function**, a measure of the intrinsic complexity or structure of n. For D=Z+, this is defined as the divisor function, σ0​(n).  
* Let λ(n):D→R be the **Spacetime Metric Function**, a measure of the local scale or magnitude of the domain at the position of n. For D=Z+, this is defined as the natural logarithm, ln(n).  
* Let Zκ​(n):D→R be the **Curvature Function**, a measure of the local distortion, or curvature, of the domain D induced by the entity n.

#### **2\. Axiomatic Statement**

The local curvature induced by an entity within its domain is directly proportional to the product of the entity's intrinsic mass and the local spacetime metric.  
This is expressed as:  
Zκ​(n)∝d(n)⋅λ(n)

#### **3\. Specific Formulation for the Domain of Integers (Z+)**

Within the Z-Metric framework as implemented, the constant of proportionality is defined as 1/e2, where e is Euler's number. The axiom is thus formulated as the precise equation:  
Zκ​(n)=e2d(n)⋅ln(n)​

#### **4\. Corollary: The Principle of Minimal Curvature**

A direct consequence of this axiom is that entities with minimal intrinsic mass induce minimal curvature in the domain. For the domain of integers, prime numbers (p) have a minimal non-trivial mass of d(p)=2. Therefore, primes represent points of minimal, stable curvature, acting as the fundamental geodesics of the Numberspace.