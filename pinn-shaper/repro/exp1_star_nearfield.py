"""
Experiment 1: Star-shaped flat-top (near field)
================================================
Faithful reproduction of "Star-shaped flat-top (near field).ipynb":
  - Gaussian source (w_I=1mm) over 6mm square aperture, lambda=1um
  - Star-shaped flat-top target in a 6mm target plane at z=10cm
  - Nearfield_PINN: 4 hidden layers x 150 neurons, tanh, phi_scale=30
  - Training: Adam(1000x1000) -> Adam(2000x15000) -> LBFGS(22500, 2000 it) -> LBFGS(22500, 2000 it)
  - Validation: scalar angular-spectrum propagation (diffractsim, 2048x2048)
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--no-train", action="store_true", help="skip training, load model from reproduced_models/")
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
import pinn_shaper
from pinn_shaper import Nearfield_PINN
from pinn_shaper import um, mm, cm
from pinn_shaper import load_image, create_interpolator
from scipy.ndimage import gaussian_filter as scipy_gaussian_filter

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE)
os.makedirs("results", exist_ok=True)
os.makedirs("reproduced_models", exist_ok=True)

# ------------------------------------------------------------------
# Problem setup (identical to notebook cell 4)
# ------------------------------------------------------------------
extent_source = 6 * mm
extent_target = 6 * mm
w_I = 1000 * um


def I_fun(x, y):
    # Source function. Gaussian beam with beam waist = w_I
    r = torch.sqrt(x**2 + y**2)
    return torch.exp(-(r / w_I) ** 2) ** 2


# Target function. Star-shaped Flat-Top beam
image = np.array(scipy_gaussian_filter(load_image("images/star.png"), 10, mode="nearest", cval=0), dtype="float32")
E_fun = create_interpolator(torch.tensor(image), mode="bilinear",
                            region=[-extent_source / 2, extent_source / 2, -extent_source / 2, extent_source / 2],
                            align_corners=True)

lam = 1.0 * um   # wavelength
z = 10 * cm      # source -> target distance

solver = Nearfield_PINN(I_fun, E_fun, extent_source, extent_target, z, lam,
                        integration_points=5000,
                        hidden_layers=4, num_features=150,
                        constrain=False, symmetry=None)

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
    print("Experiment 1 (Star near-field): FULL TRAINING FROM SCRATCH")
    print("=" * 70)
    print("Run Phase 1: Adam")
    solver.run_phase(optimizer_type="adam", iterations=1000, num_domain=1000, num_boundary=100,
                     residual_power=residual_power, loss_weights=loss_weights, effective_noise=effective_noise,
                     num_test_interior=num_test_interior, num_test_boundary=num_test_boundary, lr=0.001)
    print("Run Phase 2: Adam")
    solver.run_phase(optimizer_type="adam", iterations=2000, num_domain=1500 * 10, num_boundary=100 * 10,
                     residual_power=residual_power, loss_weights=loss_weights, effective_noise=effective_noise,
                     num_test_interior=num_test_interior, num_test_boundary=num_test_boundary, lr=0.0001)
    print("Run Phase 3: LBFGS")
    solver.run_phase(optimizer_type="lbfgs", num_domain=1500 * 15, num_boundary=100 * 15,
                     residual_power=residual_power, loss_weights=loss_weights, effective_noise=0.001,
                     num_test_interior=num_test_interior, num_test_boundary=num_test_boundary,
                     max_iterations_lbfgs=2000, lr_lbfgs=1)
    print("Run Phase 4: LBFGS")
    solver.run_phase(optimizer_type="lbfgs", num_domain=1500 * 15, num_boundary=100 * 15,
                     residual_power=residual_power, loss_weights=loss_weights, effective_noise=0.0,
                     num_test_interior=num_test_interior, num_test_boundary=num_test_boundary,
                     max_iterations_lbfgs=2000, lr_lbfgs=1)
    solver.print_final_test_loss(num_test_interior, num_test_boundary, residual_power, loss_weights, effective_noise)
    torch.save(solver.net.state_dict(), "reproduced_models/star_flat-top-nearfield_model.pth")
    print("Saved model -> reproduced_models/star_flat-top-nearfield_model.pth")
else:
    print("Loading reproduced model (no training)...")
    solver.net.load_state_dict(torch.load("reproduced_models/star_flat-top-nearfield_model.pth"))

# ------------------------------------------------------------------
# Phase profile (notebook cell 12)
# ------------------------------------------------------------------
phi_spline = solver.get_phi_spline()
nx, ny = 1000, 1000
x = np.linspace(-solver.M, solver.M, nx)
y = np.linspace(-solver.M, solver.M, ny)

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
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
ax2.set_title(r"Phase profile $\Phi(x,y)$"); ax2.set_aspect(1)
fig.tight_layout()
fig.savefig("results/exp1_star_nearfield_phase.png", dpi=150)
print("Saved results/exp1_star_nearfield_phase.png")

# ------------------------------------------------------------------
# Angular-spectrum validation (notebook cell 20)
# ------------------------------------------------------------------
import diffractsim
diffractsim.set_backend("CPU")
from diffractsim import MonochromaticField


def I_fun_numpy(x, y):
    r = np.sqrt(x**2 + y**2)
    return np.exp(-(r / w_I) ** 2) ** 2


F = MonochromaticField(wavelength=1 * um, extent_x=extent_source, extent_y=extent_source, Nx=2048, Ny=2048)
F.E = np.sqrt(I_fun_numpy(F.xx, F.yy)) * np.exp(1j * (phi_spline(F.x, F.y)))
F.propagate(z)

# ------------------------------------------------------------------
# Comparison plot (notebook cell 22)
# ------------------------------------------------------------------
device = next(solver.net.parameters()).device
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
ax1 = axs[0]
im = ax1.imshow(np.abs(F.E) ** 2, origin="lower", cmap="inferno",
                extent=[F.x[0] / mm, F.x[-1] / mm, F.y[0] / mm, F.y[-1] / mm], aspect="auto")
fig.colorbar(im, ax=ax1, fraction=0.045, label="Intensity [a.u.]")
ax1.set_xlabel("$x$ [mm]"); ax1.set_ylabel("$y$ [mm]")
ax1.set_title("Simulated profile at the target plane"); ax1.set_aspect(1)

ax2 = axs[1]
ax2.tick_params(axis="both", which="both", direction="in", right=True, top=True)
ax2.minorticks_on(); ax2.grid(which="major", linestyle="-", linewidth=1.0)
ax2.plot(F.x / mm, np.abs(F.E)[F.Ny // 2, :] ** 2, label="simulated profile")
x_torch = torch.from_numpy(np.array(F.x, dtype="float32")).to(device)
ax2.plot(F.x / mm, solver.E_fun_norm(x_torch, x_torch * 0).cpu().numpy().ravel(), "--", label="target profile")
ax2.set_title("Cross section at y = 0")
ax2.set_xlabel("$x$ [mm]")
ax2.legend()
fig.tight_layout()
fig.savefig("results/exp1_star_nearfield_propagation.png", dpi=150)
print("Saved results/exp1_star_nearfield_propagation.png")

# ------------------------------------------------------------------
# Quantitative cross-section error (y=0 line)
# ------------------------------------------------------------------
sim_cs = np.abs(F.E)[F.Ny // 2, :] ** 2
tar_cs = solver.E_fun_norm(x_torch, x_torch * 0).cpu().numpy().ravel()
# least-squares rescale, then relative RMS on the central line
mask = x_torch.cpu().numpy() / mm >= -3
p = sim_cs[mask].astype(np.float64); t = tar_cs[mask].astype(np.float64)
scale = np.sum(p * t) / np.sum(p * p)
rms_line = np.sqrt(np.mean((scale * p - t) ** 2))
rel_rms_line = rms_line / np.sqrt(np.mean(t ** 2))
print(f"\n[Exp1 metrics, y=0 cross-section]")
print(f"  LS rescale factor: {scale:.4f}")
print(f"  RMS error: {rms_line:.6e}")
print(f"  Relative RMS: {rel_rms_line:.3%}")