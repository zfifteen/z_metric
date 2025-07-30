# 🌊 **Z-Transformed Seismic and Tsunami Propagation Model**

## Overview

This Python-based simulation presents a novel framework to model **seismic and tsunami wave propagation** using a **Z-transformation approach** derived from **universal frame invariance**. The Z-transformation encodes frame-dependent time compression based on wave velocity relative to the speed of light, allowing the user to explore:

* Frame-dependent perception of wave arrival times
* Invariant ("universal frame") temporal corrections
* Warning time differentials in natural disaster contexts
* Visualization of tsunami height attenuation with distance

This model is useful for researchers exploring the intersection of:

* Geophysics and natural hazard forecasting
* Relativistic time correction concepts
* Signal propagation in heterogeneous media
* Frame-of-reference transformations in physical systems

---

## 🔬 Scientific Background

Let:

* $c$: speed of light (invariant limit)
* $v$: wave propagation velocity (e.g., seismic or tsunami)
* $T$: classical arrival time
* $Z$: Z-transformed (frame-shifted) time

### Z-Transformation:

$$
Z = T \cdot \frac{v}{c}
$$

This transformation maps classically measured time into a dimensionless frame-dependent reference. The inverse transform recovers a universal frame:

$$
T = \frac{Z}{v/c}
$$

This approach enables:

* Modeling **observer-dependent measurements**
* Interpreting **wavefront propagation** from a universal reference
* Estimating **frame-shifted warning time windows** for disaster response

---

## 📁 File Description

| File      | Description                                                                                                                  |
| --------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `main.py` | Main simulation script containing Z-transform logic, seismic and tsunami modeling, visualization, and warning time analytics |

---

## 🚀 Features

* Models seismic and tsunami wave propagation up to **10,000 km**
* Applies **Z-transformation** and **inverse Z** time corrections
* Computes and visualizes **frame-corrected arrival times**
* Estimates **warning time differentials** across frames
* Visualizes **tsunami wave height decay** using power-law attenuation

---

## 📈 Visualization Outputs

The simulation generates four key plots:

1. **Seismic Arrival Times** (original, Z-frame, universal frame)
2. **Tsunami Arrival Times** (original, Z-frame, universal frame)
3. **Tsunami Wave Height** vs Distance
4. **Warning Time Differential**:
   $\Delta T = T_{\text{tsunami}} - T_{\text{seismic}}$ across all frames

---

## ⚙️ How to Run

### Requirements:

```bash
pip install numpy matplotlib scipy
```

### To Run:

```bash
python main.py
```

---

## 📊 Example Output (at 5000 km)

```
Original seismic arrival time at 5000 km: 3333.33 seconds
Z-transformed seismic arrival time at 5000 km: 0.02 seconds
Universal frame seismic time at 5000 km: 3333.33 seconds
Original tsunami arrival time at 5000 km: 25000.00 seconds
Z-transformed tsunami arrival time at 5000 km: 0.02 seconds
Universal frame tsunami time at 5000 km: 25000.00 seconds
Potential warning time improvements:
  Seismic (Z-transform): 3333.31 seconds
  Tsunami (Z-transform): 24999.98 seconds
  Seismic (Universal): 0.00 seconds
  Tsunami (Universal): 0.00 seconds
```

---

## 📚 Citation and Attribution

If you use this framework in your academic or applied research, please cite as:

```
Lopez, D. (2025). Universal Frame-Shift Transformation for Seismic and Tsunami Signal Propagation. Unpublished prototype.
```

---

## 🧠 Future Work

* Integrate bathymetric and terrain-dependent velocity maps
* Simulate moving observers (relativistic Doppler effects)
* Couple with real-time seismic sensor data streams
* Adapt for electromagnetic vs. gravitational signal analysis
