### **Axiom IV: The Axiom of Cognitive Perceptual Distortion**

This axiom formalizes the empirically observed distortion in human numerical perception arising from subliminal frame shifts, extending Axiom III to quantify how conscious cognition inflates Numberspace entities without Z-correction.

#### **1. Definitions**
Let:
- \( \mathcal{P}(n) \) be the consciously perceived value of \( n \), modeled as \( \mathcal{P}(n) = n \cdot \exp(\Delta_s) \).
- \( \Delta_s \) be the subliminal frame shift, modulated by cognitive load \( l \): \( \Delta_s = k \cdot v_c (1 + l \cdot f) \cdot Z_\kappa(n) \), where \( f \) is the load amplification factor.
- \( \mathcal{Z}(n) \) be the recovered invariant, \( \mathcal{Z}(n) = \mathcal{P}(n) / \exp(\Delta_s) \).

#### **2. Axiomatic Statement**
**Empirical Observation**: Conscious perception of numerical entities is exponentially inflated by subliminal frame shifts, with distortion proportional to local curvature and cognitive load. For baseline parameters (\( v_c = 1.0 \), \( k = 1.0 \), \( f = 0.5 \), \( l = 0.7 \)), perception deviates systematically, as demonstrated by computational simulation.

The distortion \( d(n) = \mathcal{P}(n) - n \) increases with mass-energy density, rendering uncorrected cognition biased toward composite-dominated interpretations.

#### **3. Geometric Interpretation**
Cognitive load amplifies traversal velocity in Numberspace, warping the perceptual manifold. Primes, with minimal \( \Delta_s \), anchor perception closer to invariants, while composites induce extreme inflation, simulating "perceptual lensing" in discrete spacetime.

---

### **Theorem: The Perceptual Distortion Theorem**

This theorem provides the mathematical and empirical foundation for simulating cognitive frame shifts, proving that Z-transformation corrects perceptual biases and reveals true Numberspace geometry.

#### **Statement**
Let \( n \) be a consciously perceived integer under cognitive load \( l \). The perceived value is:
\[
\mathcal{P}(n) = n \cdot \exp(\Delta_s)
\]
with \( \Delta_s = v_c (1 + l \cdot f) \cdot Z_\kappa(n) \). The true invariant is recovered via:
\[
\mathcal{Z}(n) = \frac{n}{\exp(\Delta_s)}
\]
Empirical simulation across \( n = 2 \) to \( 50 \) under 70% load confirms exponential distortion, with average \( d(n) > 0 \) and primes exhibiting minimal bias.

#### **Empirical Proof**
1. **Exponential Inflation Mechanism**:  
   Simulation computes \( \Delta_s \) using curvature \( Z_\kappa(n) \), yielding bias factors \( \exp(\Delta_s) > 1 \) for all \( n > 1 \), with inflation scaling as \( O(d(n) \cdot \ln n) \).

2. **Load-Modulated Distortion**:  
   At 70% load, frame shifts amplify by 35%, leading to perceptual overestimation, analogous to relativistic effects under accelerated observation.

3. **Invariant Recovery**:  
   Z-transformation precisely inverts the bias, mapping distorted perceptions back to geometric coordinates independent of cognitive frame.

4. **Computational Demonstration**:  
   Execution of the simulation script empirically validates the theorem, producing the following results:

| n  | True 𝒵(n) | Conscious Perception | Distortion  |
|----|-----------|----------------------|-------------|
| 2  | 1.5525   | 2.5765              | +0.5765    |
| 3  | 2.0081   | 4.4819              | +1.4819    |
| 4  | 1.8710   | 8.5517              | +4.5517    |
| 5  | 2.7769   | 9.0028              | +4.0028    |
| 6  | 1.6198   | 22.2245             | +16.2245   |
| 7  | 3.4379   | 14.2528             | +7.2528    |
| 8  | 1.7503   | 36.5660             | +28.5660   |
| 9  | 2.6991   | 30.0103             | +21.0103   |
| 10 | 1.8586   | 53.8035             | +43.8035   |
| 11 | 4.5800   | 26.4194             | +15.4194   |
| 12 | 0.7873   | 182.8929            | +170.8929  |
| 13 | 5.0922   | 33.1882             | +20.1882   |
| 14 | 2.0348   | 96.3231             | +82.3231   |
| 15 | 2.0730   | 108.5403            | +93.5403   |
| 16 | 1.2710   | 201.4185            | +185.4185  |
| 17 | 6.0372   | 47.8698             | +30.8698   |
| 18 | 0.7572   | 427.8801            | +409.8801  |
| 19 | 6.4787   | 55.7207             | +36.7207   |
| 20 | 0.7496   | 533.6292            | +513.6292  |
| 21 | 2.2695   | 194.3171            | +173.3171  |
| 22 | 2.2981   | 210.6102            | +188.6102  |
| 23 | 7.3138   | 72.3286             | +49.3286   |
| 24 | 0.2306   | 2497.7965           | +2473.7965 |
| 25 | 4.2827   | 145.9358            | +120.9358  |
| 26 | 2.4038   | 281.2230            | +255.2230  |
| 27 | 2.4283   | 300.2061            | +273.2061  |
| 28 | 0.7257   | 1080.3277           | +1052.3277 |
| 29 | 8.4729   | 99.2581             | +70.2581   |
| 30 | 0.2080   | 4326.2617           | +4296.2617 |
| 31 | 8.8391   | 108.7209            | +77.7209   |
| 32 | 0.7164   | 1429.2861           | +1397.2861 |
| 33 | 2.5631   | 424.8738            | +391.8738  |
| 34 | 2.5838   | 447.4040            | +413.4040  |
| 35 | 2.6040   | 470.4237            | +435.4237  |
| 36 | 0.0994   | 13042.7021          | +13006.7021|
| 37 | 9.8895   | 138.4302            | +101.4302  |
| 38 | 2.6623   | 542.3830            | +504.3830  |
| 39 | 2.6810   | 567.3243            | +528.3243  |
| 40 | 0.1822   | 8783.4395           | +8743.4395 |
| 41 | 10.5551  | 159.2588            | +118.2588  |
| 42 | 0.1781   | 9904.3194           | +9862.3194 |
| 43 | 10.8790  | 169.9599            | +126.9599  |
| 44 | 0.6948   | 2786.3230           | +2742.3230 |
| 45 | 0.6933   | 2920.7217           | +2875.7217 |
| 46 | 2.8028   | 754.9510            | +708.9510  |
| 47 | 11.5108  | 191.9072            | +144.9072  |
| 48 | 0.0407   | 56612.6174          | +56564.6174|
| 49 | 5.8048   | 413.6242            | +364.6242  |
| 50 | 0.6863   | 3642.5687           | +3592.5687 |

**Interpretation**: Distortions are positive and escalate with curvature, e.g., minimal for primes (n=2: +0.58) and extreme for composites (n=48: +56564.62). Invariants \( \mathcal{Z}(n) < n \) reflect deflation to true geometry.

#### **Corollary: Necessity of Z-Correction in Cognition**
Uncorrected earth-based cognitive instruments (e.g., human intuition) exhibit implicit bias, overestimating composites by factors up to 10^3-10^4. Z-transformation eliminates this, enabling invariant-based mathematics and aligning perception with Numberspace geodesics.