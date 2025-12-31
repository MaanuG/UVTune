# UVTune

**UV-Tune: Modeling an Electrically Tunable Deep-UV AlGaN Metasurface Pixel for Phase-Modulated Optical Signaling**

This repository contains a physics-based simulation project that models an **electrically tunable deep-UV (270 nm) AlGaN metasurface pixel** and evaluates how its resonant and electro-optic properties impact **coherent optical communication performance**, using **bit error rate (BER)** as a system-level metric.

The project emphasizes **relative performance trends** rather than absolute device optimization, making it well suited for **simulation-driven undergraduate research**.

---

## Repository Contents

- **`code20aesthetic.py`**  
  Main simulation script combining electromagnetic modeling, electrostatics, and communication-theory analysis.

- **Deliverables (PDF)**  
  Written project deliverables explaining the motivation, physics background, modeling approach, results, and interpretation.

---

## Overview of `code20aesthetic.py`

The script links **metasurface resonance physics** to **communication-level performance** through the following stages:

---

### 1. Target Wavelength & Operating Regime
- Models operation at **270 nm (deep-UV)**.
- Defines optical frequency and angular frequency consistently for both electromagnetic simulation and material dispersion models.

---

### 2. Materials & Optical Physics
- **AlGaN resonator** on a **sapphire substrate**, using UV-appropriate refractive indices.
- Incorporates a **Drude-based plasma dispersion model** to convert electrically induced carrier density changes into a **permittivity shift**.

This allows electrical tuning to directly modify the optical response.

---

### 3. Metasurface Geometry & Boundary Conditions
- Periodic unit cell representing a **single metasurface pixel**.
- Cylindrical AlGaN resonator placed on a sapphire substrate.
- **Perfectly Matched Layers (PMLs)** applied along the propagation direction.
- Geometry and boundaries are chosen to correctly extract transmitted optical flux.

---

### 4. Reference Flux Normalization
- A free-space reference simulation measures the **incident optical flux**.
- All device transmission values are normalized against this reference, isolating metasurface effects from source power.

---

### 5. Device Transmission Simulation
- Runs the metasurface simulation with and without electrical tuning.
- Extracts transmitted flux and computes the **normalized transmission amplitude** of the pixel.

---

### 6. Electrostatics → Carrier Density → Permittivity
- Models a simplified **gate-capacitor structure**.
- Converts applied voltage into:
  - Gate capacitance  
  - Carrier density modulation  
  - Resulting change in AlGaN permittivity  

This bridges **electrical control** and **optical phase tuning**.

---

### 7. Resonator Q-Factor → Phase Efficiency
- Uses a perturbative relationship between:
  - Resonator **Q-factor**
  - Achievable **optical phase shift**
- Demonstrates how higher-Q resonators enable more efficient phase modulation.

---

### 8. Communication System Modeling (BPSK)
- Models a **coherent BPSK optical communication link**.
- Compares two scenarios:
  - **Untuned pixel** → random phase scrambling
  - **Tuned pixel** → stable phase states
- Adds complex Gaussian noise and computes **BER vs SNR**.

This shows how resonant electro-optic tuning enables coherent signaling.

---

### 9. BER vs Resonator Q-Factor
- Evaluates how limited phase swing at low Q degrades communication performance.
- Shows BER improving as resonator Q-factor increases.

---

### 10. Visualization & Intuition
The script generates figures including:
- Phase shift vs resonator Q-factor
- BER vs SNR (tuned vs untuned)
- BER vs Q-factor
- Histograms comparing received signal distributions

These visualizations build intuition for how **resonant EO tuning preserves BPSK signal integrity**.

---

## What the Deliverables Show

The deliverables PDF provides:

- Motivation for using **deep-UV metasurfaces** in optical communication
- Discussion of why visible/IR links struggle under daylight and turbulence
- Physical justification for **electrically tunable AlGaN metasurfaces**
- Explanation of modeling assumptions and simplifications
- Interpretation of simulation results
- Limitations of the current model and potential future extensions

Together, the deliverables contextualize the code and explain **why the observed trends matter** for free-space and satellite optical communication systems.

---

## Scope & Limitations

- Focuses on **relative performance trends**, not fabrication-ready designs
- Simplified electrostatics and material dispersion models
- Intended as a **conceptual bridge** between nanophotonics and communication theory

---

## Dependencies

- Python
- `meep`
- `numpy`
- `scipy`
- `matplotlib`

---

## License

This project is intended for educational and research use.
