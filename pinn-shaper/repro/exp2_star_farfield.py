"""
Experiment 2: Star-shaped flat-top (far field)
===============================================
Faithful reproduction of "Star-shaped flat-top (far field).ipynb":
  - Gaussian source (w_I=0.253mm) over 1mm square aperture, lambda=1um
  - Star-shaped far-field target in (alpha,beta) direction-cosine coordinates
  - Farfield_PINN: 3 hidden layers x 150 neurons, phi_scale=8, hard radial
    constraint phi = r^2 + f(x,y)
  - Training: Adam(1000x1000) -> Adam(2000x15000) -> LBFGS(22500, 1000 it) -> LBFGS(22500, 800 it)
  - Validation: scalar diffraction far field (diffractsim, 2048x2048)
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--no-train", action="store_true")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

import torch
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.set_default_device("cuda")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pinn_shaper import Farfield_PINN
from pinn_shaper import um, mm
from pinn_shaper import load_image, create_interpolator
from scipy.ndimage import gaussian_filter as scipy_gaussian_filter

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE)
os.makedirs("results", exist_ok=True)
os.makedirs("reproduced_models", exist_ok=True)

# ------------------------------------------------------------------
# Problem setup (notebook cell 4)
# ------------------------------------------------------------------
extent_source = 1000 * um
w_I = 50.6 * um / (100 * um) * extent_source / 2
wavelength = 1.0 * um


def I_fun(x, y):
    r = torch.sqrt(x**2 + y**2)
    return torch.exp(-(r / w_I) ** 2) ** 2


target_image = np.asarray(scipy_gaussian_filter(load_image("images/star.png"), 10, mode="nearest", cval=0), dtype=np.float32)
Irad_fun = create_interpolator(torch.tensor(target_image), mode="bilinear", region=[-1, 1, -1, 1], align_corners=True)


def constrain_fn(points):
    # Radial constraint used by the original star-shaped notebook.
    return points[:, 0:1] ** 2 + points[:, 1:2] ** 2


solver = Farfield_PINN(I_fun, Irad_fun, extent_source, wavelength,
                       integration_points=5000,
                       hidden_layers=3, num_features=150,
                       constrain=True, symmetry=None,
                       constrain_fn=constrain_fn)

# ------------------------------------------------------------------
# Training (notebook cell 8)
# ------------------------------------------------------------------
residual_power = 1.5
loss_weights = [100.0, 1, 1]
noise_amplitude = 0.0
add_noise = False
effective_noise = noise_amplitude if add_noise else 0.0
num_test_interior = 1000
num_test_boundary = 200

if not args.no_train:
    print("=" * 70)
    print("Experiment 2 (Star far-field): FULL TRAINING FROM SCRATCH")
    print("=" * 70)
    print("Run Phase 1: Adam")
    solver.run_phase(optimizer_type="adam", iterations=1000, num_domain=1000, num_boundary=100,
                     residual_power=residual_power, loss_weights=loss_weights, effective_noise=effective_noise,
                     num_test_interior=num_test_interior, num_test_boundary=num_test_boundary, lr=0.001)
    print("Run Phase 2: Adam")
    solver.run_phase(optimizer_type="adam", iterations=2000, num_domain=1500 * 10, num_boundary=100 * 10,
                     residual_power=residual_power, loss_weights=loss_weights, effective_noise=effective_noise,
                     num_test_interior=num_test_interior, num_test_boundary=num_test_boundary, lr=0.0001)
    print("Run Phase 3: L-BFGS")
    solver.run_phase(optimizer_type="lbfgs", num_domain=1500 * 15, num_boundary=100 * 15,
                     residual_power=residual_power, loss_weights=loss_weights, effective_noise=0.001,
                     num_test_interior=num_test_interior, num_test_boundary=num_test_boundary,
                     max_iterations_lbfgs=1000, lr_lbfgs=1)
    print("Run Phase 4: L-BFGS")
    solver.run_phase(optimizer_type="lbfgs", num_domain=1500 * 15, num_boundary=100 * 15,
                     residual_power=residual_power, loss_weights=loss_weights, effective_noise=0.0,
                     num_test_interior=num_test_interior, num_test_boundary=num_test_boundary,
                     max_iterations_lbfgs=800, lr_lbfgs=1)
    solver.print_final_test_loss(num_test_interior, num_test_boundary, residual_power, loss_weights, effective_noise)
    torch.save(solver.net.state_dict(), "reproduced_models/star_flat-top-farfield_model.pth")
    print("Saved model -> reproduced_models/star_flat-top-farfield_model.pth")
else:
    print("Loading reproduced model (no training)...")
    solver.net.load_state_dict(torch.load("reproduced_models/star_flat-top-farfield_model.pth"))

# ------------------------------------------------------------------
# Phase profile (notebook cell 12)
# ------------------------------------------------------------------
phi_spline = solver.get_phi_spline()
nx, ny = 2000, 2000
x = np.linspace(-solver.M, solver.M, nx)
y = np.linspace(-solver.M, solver.M, ny)

fig, axs = plt.subplots(1, 2, figsize=(12, 6))
ax1 = axs[0]
im = ax1.contourf(phi_spline(x, y), levels=30, extent=[x[0] / mm, x[-1] / mm, y[0] / mm, y[-1] / mm])
fig.colorbar(im, ax=ax1, fraction=0.045, label=r"$\Phi$ [rad]")
ax1.set_xlabel("$x$ [mm]"); ax1.set_ylabel("$y$ [mm]")
ax1.set_title(r"Phase profile $\Phi(x,y)$ (contourf)"); ax1.set_aspect(1)

ax2 = axs[1]
im = ax2.imshow((phi_spline(x, y)) % (np.pi * 2), origin="lower", cmap="hsv",
                extent=[x[0] / mm, x[-1] / mm, y[0] / mm, y[-1] / mm])
fig.colorbar(im, ax=ax2, fraction=0.045, label=r"$\Phi$ [rad]")
ax2.set_xlabel("$x$ [mm]"); ax2.set_ylabel("$y$ [mm]")
ax2.set_title(r"Phase profile $\Phi(x,y)$")
ax2.set_xlim([-0.2, 0.2]); ax2.set_ylim([-0.2, 0.2]); ax2.set_aspect(1)
fig.tight_layout()
fig.savefig("results/exp2_star_farfield_phase.png", dpi=150)
print("Saved results/exp2_star_farfield_phase.png")

# ------------------------------------------------------------------
# Far-field validation (notebook cell 14)
# ------------------------------------------------------------------
import diffractsim
diffractsim.set_backend("CPU")
from diffractsim import MonochromaticField

Nx, Ny = 2048, 2048
farfield_simulator = MonochromaticField(wavelength=wavelength, extent_x=extent_source, extent_y=extent_source, Nx=Nx, Ny=Ny)


def I_fun_numpy(x, y):
    r = np.sqrt(x**2 + y**2)
    return np.exp(-(r / w_I) ** 2) ** 2


farfield_simulator.E = np.sqrt(I_fun_numpy(farfield_simulator.xx, farfield_simulator.yy)) * np.exp(1j * phi_spline(farfield_simulator.x, farfield_simulator.y))
alpha, beta, radiant_intensity_percos = farfield_simulator.get_farfield()

# ------------------------------------------------------------------
# Comparison plot (notebook cell 16)
# ------------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
ax1 = axs[0]
im = ax1.imshow(radiant_intensity_percos, origin="lower", cmap="inferno",
                extent=[alpha[0], alpha[-1], beta[0], beta[-1]], aspect="auto")
cb = fig.colorbar(im, ax=ax1, orientation="vertical", fraction=0.045)
cb.set_label(r"$\frac{\partial P(\alpha,\beta)}{\partial\Omega\cos(\theta)}$ [a.u.]", size=14)
ax1.set_ylabel(r"$\beta$", size=12); ax1.set_xlabel(r"$\alpha$", size=12)
ax1.set_title("Simulated far-field profile"); ax1.set_aspect("equal")
ax1.set_xlim([-1.0, 1.0]); ax1.set_ylim([-1.0, 1.0])

ax2 = axs[1]
ax2.tick_params(axis="both", which="both", direction="in", right=True, top=True)
ax2.minorticks_on(); ax2.grid(which="major", linestyle="-", linewidth=1.0)
theta_degrees = np.arcsin(np.clip(alpha, -1, 1)) * 180 / np.pi
ax2.plot(theta_degrees, radiant_intensity_percos[Ny // 2, :], label="simulated profile")
network_device = next(solver.net.parameters()).device
alpha_torch = torch.from_numpy(np.asarray(alpha, dtype=np.float32)).to(network_device)
target_cross_section = solver.Irad_fun_norm(alpha_torch, torch.zeros_like(alpha_torch)).detach().cpu().numpy().ravel()
ax2.plot(theta_degrees, target_cross_section, "--", label="target profile")
ax2.set_xlim([-90, 90])
ax2.set_title(r"Cross section at $\beta=0$")
ax2.set_xlabel(r"$\theta$ [degrees]", size=12)
ax2.legend()
fig.tight_layout()
fig.savefig("results/exp2_star_farfield.png", dpi=150)
print("Saved results/exp2_star_farfield.png")

# ------------------------------------------------------------------
# Quantitative cross-section error (beta=0)
# ------------------------------------------------------------------
sim_cs = radiant_intensity_percos[Ny // 2, :]
tar_cs = target_cross_section
m = (alpha > -1) & (alpha < 1)
p = sim_cs[m].astype(np.float64); t = tar_cs[m].astype(np.float64)
scale = np.sum(p * t) / np.sum(p * p)
rms_line = np.sqrt(np.mean((scale * p - t) ** 2))
rel_rms_line = rms_line / np.sqrt(np.mean(t ** 2))
print(f"\n[Exp2 metrics, beta=0 cross-section, |alpha|<1]")
print(f"  LS rescale factor: {scale:.4f}")
print(f"  RMS error: {rms_line:.6e}")
print(f"  Relative RMS: {rel_rms_line:.3%}")