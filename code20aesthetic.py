"""
UV-Tune: Modeling an Electrically Tunable Deep-UV AlGaN Metasurface Pixel for Phase-Modulated Optical Signaling
Linking resonance physics to BER-based performance metrics
Deep-UV (270 nm), AlGaN-on-sapphire metasurface unit cell
Flux-normalized transmission with abstracted phase-state system modeling
Includes:
• Physics-consistent metasurface electromagnetic boundary conditions
• Electrostatics → carrier density → permittivity shift (Drude-based)
• BER vs SNR evaluation under coherent BPSK assumptions
• Resonator Q-factor → phase-efficiency relationship
• BER vs Q-factor performance trends
This project focuses on relative performance trends rather than absolute device optimization, making it suitable for simulation-driven undergraduate research.
"""

import meep as mp
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import e, epsilon_0, m_e, c
from scipy.special import erfc

plt.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 15,
    "axes.titlesize": 16,
    "legend.fontsize": 13,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "lines.linewidth": 2.5,
    "axes.linewidth": 1.4,
    "grid.alpha": 0.3
})

# ============================================================
# 1. Target Wavelength & Frequency
# ============================================================

lambda0 = 0.270            # microns
freq0 = 1 / lambda0
omega0 = 2 * np.pi * c / (lambda0 * 1e-6)

# ============================================================
# 2. Materials (UV-appropriate)
# ============================================================

n_AlGaN = 2.2
eps_AlGaN = n_AlGaN**2

n_sapphire = 1.78
eps_sapphire = n_sapphire**2

m_eff = 0.22 * m_e

def delta_eps_from_deltaN(deltaN):
    """
    Real-valued Drude plasma dispersion (Meep-compatible)
    """
    return -(e**2 / (epsilon_0 * m_eff * omega0**2)) * deltaN

# ============================================================
# 3. Geometry & Boundary Conditions (Correct for Metasurface)
# ============================================================

period = 0.25
radius = 0.085
height = 0.45
substrate_thickness = 1.0

cell = mp.Vector3(period, period, 6.0)
resolution = 80
pml_thickness = 1.0

boundary_layers = [mp.PML(pml_thickness, direction=mp.Z)]

# ============================================================
# 4. Reference Flux (Normalization)
# ============================================================

def run_reference_flux():
    sim = mp.Simulation(
        cell_size=cell,
        resolution=resolution,
        boundary_layers=boundary_layers,
        sources=[mp.Source(
            mp.GaussianSource(freq0, fwidth=0.3*freq0),
            component=mp.Ez,
            center=mp.Vector3(z=-2.0)
        )]
    )

    fr = mp.FluxRegion(center=mp.Vector3(z=2.0),
                       size=mp.Vector3(period, period, 0))
    flux = sim.add_flux(freq0, 0, 1, fr)

    sim.run(until_after_sources=mp.stop_when_fields_decayed(
        50, mp.Ez, mp.Vector3(z=2.0), 1e-7))

    val = mp.get_fluxes(flux)[0]
    sim.reset_meep()
    return val

# ============================================================
# 5. Device Flux
# ============================================================

def run_device_flux(delta_eps=0.0):

    geometry = [
        mp.Block(
            size=mp.Vector3(period, period, substrate_thickness),
            center=mp.Vector3(z=-1.5),
            material=mp.Medium(epsilon=eps_sapphire)
        ),
        mp.Cylinder(
            radius=radius,
            height=height,
            center=mp.Vector3(z=0.0),
            material=mp.Medium(epsilon=eps_AlGaN + delta_eps)
        )
    ]

    sim = mp.Simulation(
        cell_size=cell,
        geometry=geometry,
        resolution=resolution,
        boundary_layers=boundary_layers,
        sources=[mp.Source(
            mp.GaussianSource(freq0, fwidth=0.3*freq0),
            component=mp.Ez,
            center=mp.Vector3(z=-2.0)
        )]
    )

    fr = mp.FluxRegion(center=mp.Vector3(z=2.0),
                       size=mp.Vector3(period, period, 0))
    flux = sim.add_flux(freq0, 0, 1, fr)

    sim.run(until_after_sources=mp.stop_when_fields_decayed(
        50, mp.Ez, mp.Vector3(z=2.0), 1e-7))

    val = mp.get_fluxes(flux)[0]
    sim.reset_meep()
    return val

# ============================================================
# 6. Electrostatics → Carrier Density
# ============================================================

V_drive = 5.0
d_gate = 100e-9
eps_r = 9.0

A_gate = np.pi * (radius * 1e-6)**2
C = epsilon_0 * eps_r * A_gate / d_gate

V_active = np.pi * (radius * 1e-6)**2 * (height * 1e-6)
deltaN = (C * V_drive) / (e * V_active)

delta_eps = delta_eps_from_deltaN(deltaN)

# ============================================================
# 7. Normalized Transmission Extraction
# ============================================================

incident = run_reference_flux()
T = run_device_flux(delta_eps) / incident
A0 = np.sqrt(np.abs(T))

print(f"Normalized transmission amplitude A ≈ {A0:.3f}")

# ============================================================
# 8. Resonator Q → Phase Efficiency Scaling
# ============================================================

Q_values = np.array([50, 100, 200, 400])

# Perturbative phase-shift estimate:
# Δφ ≈ (Q / 2) · |Δε| / ε
delta_phi = (Q_values / 2) * np.abs(delta_eps) / eps_AlGaN

plt.figure(figsize=(7, 5))

plt.plot(Q_values, delta_phi,
         marker="o",
         markersize=7)

plt.xlabel("Resonator Q-Factor")
plt.ylabel("Achievable Phase Shift Δφ (rad)")
plt.title("Resonator Q-Factor vs Phase Efficiency")

plt.grid(True, which="major")
plt.grid(True, which="minor", linestyle=":", alpha=0.25)

plt.tight_layout()
plt.show()

# ============================================================
# 9. BER vs SNR Sweep (Correct System-Level Model)
# ============================================================

N = 60000
bits = np.random.randint(0, 2, N)
symbols = 1 - 2*bits   # +1 / -1

noise_levels = np.linspace(0.05, 0.6, 25)

BER_untuned = []
BER_tuned = []

for sigma in noise_levels:

    noise = sigma * (np.random.randn(N) + 1j*np.random.randn(N))

    # Untuned: phase scrambled
    rand_phase = np.random.uniform(-np.pi, np.pi, N)
    rx_u = A0 * np.exp(1j * rand_phase) + noise
    I_u = np.real(rx_u)
    BER_untuned.append(np.mean((I_u < 0) != bits))

    # Tuned: coherent BPSK
    rx_t = A0 * symbols + noise
    I_t = np.real(rx_t)
    BER_tuned.append(np.mean((I_t < 0) != bits))

# ============================================================
# 10. BER vs SNR
# ============================================================

SNR = A0**2 / (2 * noise_levels**2)

plt.figure(figsize=(7.5, 5))

plt.semilogy(10*np.log10(SNR), BER_untuned,
             marker="s", linestyle="--",
             markersize=7,
             label="Untuned (phase-scrambled)")

plt.semilogy(10*np.log10(SNR), BER_tuned,
             marker="o", linestyle="-",
             markersize=7,
             label="Tuned (coherent BPSK)")

plt.xlabel("SNR (dB)")
plt.ylabel("Bit Error Rate (BER)")
plt.title("BER vs SNR for Resonantly Tuned EO Pixel")

plt.grid(True, which="major")
plt.grid(True, which="minor", linestyle=":", alpha=0.25)

plt.legend(frameon=True)
plt.tight_layout()
plt.show()

# ============================================================
# 11. BER vs Resonator Q-Factor (Phase-Efficiency Limited)
# ============================================================

noise_std = 0.25
BER_vs_Q = []

for phi in delta_phi:
    # Effective constellation collapse under insufficient phase swing
    rx = A0 * np.cos(phi) * symbols \
         + noise_std * np.random.randn(N)

    BER_vs_Q.append(np.mean((rx < 0) != bits))

plt.figure(figsize=(7, 5))

plt.semilogy(Q_values, BER_vs_Q,
             marker="o",
             linestyle="-",
             markersize=7)

plt.xlabel("Resonator Q-Factor")
plt.ylabel("Bit Error Rate (BER)")
plt.title("Communication Performance vs Resonator Q-Factor")

plt.grid(True, which="major")
plt.grid(True, which="minor", linestyle=":", alpha=0.25)

plt.tight_layout()
plt.show()

# ============================================================
# 12. Histogram Visualization (Intuition Figure)
# ============================================================

plt.figure(figsize=(7.5, 5))

plt.hist(
    np.real(A0 * np.exp(1j * np.random.uniform(-np.pi, np.pi, N))),
    bins=80,
    density=True,
    alpha=0.55,
    label="Untuned (random phase)"
)

plt.hist(
    A0 * symbols + noise_std * np.random.randn(N),
    bins=80,
    density=True,
    alpha=0.55,
    label="Tuned (BPSK phase states)"
)

# Decision boundary
plt.axvline(0, linestyle="--", linewidth=1.5)

plt.xlabel("In-Phase Component")
plt.ylabel("Probability Density")
plt.title("Resonant EO Tuning Preserves BPSK Signal Integrity")

plt.legend(frameon=True)
plt.tight_layout()
plt.show()
