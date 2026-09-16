"""
Experiment 4: DT-letters far field + Gerchberg-Saxton comparison
=================================================================
Faithful reproduction of "DT-letters-shaped flat-top (far field) - with GS comparison.ipynb":
  - Gaussian source (w_I=0.253mm) over 1mm aperture, lambda=1um
  - DT-logo far-field target in (alpha,beta) coordinates
  - Farfield_PINN: 4 hidden layers x 150, phi_scale=8, hard r^2 constraint
  - Training: Adam(1000x1000) -> Adam(2000x15000) -> LBFGS(22500,4000) -> LBFGS(22500,2800)
  - GS baseline: 60 iterations, same grid
  - Metrics on [-1,1]^2: LS-rescaled MSE / relative RMS; energy efficiency eta
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
from pinn_shaper import um, mm, cm
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
lam = 1.0 * um


def I_fun(x, y):
    r = torch.sqrt(x**2 + y**2)
    return torch.exp(-(r / w_I) ** 2) ** 2


# DT-logo target.
image = np.array(scipy_gaussian_filter(load_image("images/DT.png"), 10, mode="nearest", cval=0), dtype="float32")
Irad_fun = create_interpolator(torch.tensor(image), mode="bilinear", region=[-1, 1, -1, 1], align_corners=True)


def constrain_fn(x):
    return x[:, 0:1] ** 2 + x[:, 1:2] ** 2


solver = Farfield_PINN(I_fun, Irad_fun, extent_source, lam,
                       integration_points=5000,
                       hidden_layers=4, num_features=150,
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
    print("Experiment 4 (DT far-field): FULL TRAINING FROM SCRATCH")
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
                     max_iterations_lbfgs=4000, lr_lbfgs=1)
    print("Run Phase 4: LBFGS")
    solver.run_phase(optimizer_type="lbfgs", num_domain=1500 * 15, num_boundary=100 * 15,
                     residual_power=residual_power, loss_weights=loss_weights, effective_noise=0.0,
                     num_test_interior=num_test_interior, num_test_boundary=num_test_boundary,
                     max_iterations_lbfgs=2800, lr_lbfgs=1)
    solver.print_final_test_loss(num_test_interior, num_test_boundary, residual_power, loss_weights, effective_noise)
    torch.save(solver.net.state_dict(), "reproduced_models/DT_flat-top-farfield_model.pth")
    print("Saved model -> reproduced_models/DT_flat-top-farfield_model.pth")
else:
    print("Loading reproduced model (no training)...")
    solver.net.load_state_dict(torch.load("reproduced_models/DT_flat-top-farfield_model.pth"))

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
fig.savefig("results/exp4_dt_farfield_phase.png", dpi=150)
print("Saved results/exp4_dt_farfield_phase.png")

# ------------------------------------------------------------------
# PINN scalar far-field validation (notebook cell 16)
# ------------------------------------------------------------------
import jax
jax.config.update("jax_enable_x64", True)

import diffractsim
diffractsim.set_backend("CPU")
from diffractsim import MonochromaticField

Nx, Ny = 2048, 2048
extent_x = extent_source
extent_y = extent_source

F = MonochromaticField(wavelength=lam, extent_x=extent_x, extent_y=extent_y, Nx=Nx, Ny=Ny)


def I_fun_numpy(x, y):
    r = np.sqrt(x**2 + y**2)
    return np.exp(-(r / w_I) ** 2) ** 2


F.E = np.sqrt(I_fun_numpy(F.xx, F.yy)) * np.exp(1j * phi_spline(F.x, F.y))
alpha, beta, radiant_intensity_percos = F.get_farfield()

# ------------------------------------------------------------------
# PINN far-field plot (notebook cell 18)
# ------------------------------------------------------------------
network_device = next(solver.net.parameters()).device
alpha_torch = torch.from_numpy(np.array(alpha, dtype="float32")).to(network_device)
target_cross = solver.Irad_fun_norm(alpha_torch, alpha_torch * 0).cpu().numpy().ravel()

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
ax1 = axs[0]
im = ax1.imshow(radiant_intensity_percos, origin="lower", cmap="inferno",
                extent=[alpha[0], alpha[-1], beta[0], beta[-1]], aspect="auto")
cb = fig.colorbar(im, ax=ax1, orientation="vertical", fraction=0.045)
cb.set_label(label=r"$\frac{\partial P(\alpha, \beta)}{\partial \Omega \cos(\theta)}$ [a.u.]", size=14)
ax1.set_ylabel(r"$\beta$", size=12); ax1.set_xlabel(r"$\alpha$", size=12)
ax1.set_title("Simulated profile at the far-field")
ax1.set_aspect(1); ax1.set_xlim([-1.0, 1.0]); ax1.set_ylim([-1.0, 1.0])

ax2 = axs[1]
ax2.tick_params(axis="both", which="both", direction="in", right=True, top=True)
ax2.minorticks_on(); ax2.grid(which="major", linestyle="-", linewidth=1.0)
ax2.plot(np.arcsin(np.clip(alpha, -1, 1)) * 180 / np.pi, radiant_intensity_percos[Ny // 2, :], label="simulated profile")
ax2.set_xlim([-90, 90])
ax2.plot(np.arcsin(np.clip(alpha, -1, 1)) * 180 / np.pi, target_cross, "--", label="target profile")
ax2.set_title("Cross section at beta = 0")
ax2.set_xlabel(r"$\theta$ [degrees]", size=12)
ax2.legend()
fig.tight_layout()
fig.savefig("results/exp4_dt_farfield_pinn.png", dpi=150)
print("Saved results/exp4_dt_farfield_pinn.png")

# ------------------------------------------------------------------
# Gerchberg-Saxton baseline (notebook cells 24-28)
# ------------------------------------------------------------------
import jax.numpy as jnp


def _to_numpy(a):
    try:
        return np.asarray(a)
    except Exception:
        return np.array(a)


def evaluate_target_on_alpha_beta_grid(alpha, beta, chunk_rows=128):
    """Evaluate normalized target radiant intensity solver.Irad_fun_norm on a grid."""
    device = next(solver.net.parameters()).device
    dtype = torch.float32
    alpha_np = np.asarray(alpha, dtype=np.float32)
    beta_np = np.asarray(beta, dtype=np.float32)
    alpha_t = torch.as_tensor(alpha_np, dtype=dtype, device=device)
    rows = []
    with torch.no_grad():
        for j0 in range(0, len(beta_np), chunk_rows):
            beta_chunk = torch.as_tensor(beta_np[j0:j0 + chunk_rows], dtype=dtype, device=device)
            AA, BB = torch.meshgrid(alpha_t, beta_chunk, indexing="xy")
            vals = solver.Irad_fun_norm(AA, BB).detach().cpu().numpy()
            rows.append(vals)
    return np.vstack(rows)


def gs_phase_retrieval_from_arrays(source_intensity, target_intensity, dx, dy, num_iter=60, random_seed=1):
    rng = np.random.default_rng(random_seed)
    source_amplitude = np.sqrt(np.maximum(source_intensity, 0.0))
    target_amplitude = np.sqrt(np.maximum(target_intensity, 0.0)) / (dx * dy)

    source_amplitude_j = jnp.asarray(np.fft.ifftshift(source_amplitude))
    target_amplitude_j = jnp.asarray(np.fft.ifftshift(target_amplitude))

    initial_phase = rng.uniform(-np.pi, np.pi, size=source_intensity.shape)
    U_p = source_amplitude_j * jnp.exp(1j * jnp.asarray(np.fft.ifftshift(initial_phase)))

    error_list = []
    for it in range(num_iter):
        U = source_amplitude_j * jnp.exp(1j * jnp.angle(U_p))
        Uf = jnp.fft.fft2(U)
        Uf_p = target_amplitude_j * jnp.exp(1j * jnp.angle(Uf))
        err = jnp.sqrt(jnp.mean((jnp.abs(Uf) - target_amplitude_j) ** 2))
        error_list.append(float(err))
        U_p = jnp.fft.ifft2(Uf_p)
    retrieved_phase = jnp.fft.fftshift(jnp.angle(U_p))
    return np.asarray(retrieved_phase), error_list


# Build the GS problem on the same grid (notebook cell 26)
num_iter_gs = 60
xx, yy = F.xx, F.yy
dx, dy = F.dx, F.dy
source_intensity_gs = I_fun_numpy(xx, yy)
target_intensity_gs = evaluate_target_on_alpha_beta_grid(alpha, beta, chunk_rows=128)

retrieved_phase_gs, gs_error_list = gs_phase_retrieval_from_arrays(
    source_intensity=source_intensity_gs,
    target_intensity=target_intensity_gs,
    dx=dx, dy=dy,
    num_iter=num_iter_gs,
    random_seed=1,
)

fig = plt.figure(figsize=(5, 3.5))
ax = fig.add_subplot(1, 1, 1)
ax.plot(np.arange(1, len(gs_error_list) + 1), gs_error_list)
ax.set_xlabel("GS iteration"); ax.set_ylabel("Fourier-amplitude RMS error")
ax.set_title("Gerchberg-Saxton convergence"); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("results/exp4_gs_convergence.png", dpi=150)
print("Saved results/exp4_gs_convergence.png")

# Propagate the GS phase (notebook cell 28)
E_gs = np.sqrt(np.maximum(source_intensity_gs, 0.0)) * np.exp(1j * retrieved_phase_gs)
F.E = np.asarray(E_gs, dtype=np.complex128)
alpha_gs, beta_gs, radiant_intensity_percos_gs = F.get_farfield()

alpha_np = np.asarray(alpha); beta_np = np.asarray(beta)
radiant_intensity_percos_pinn = np.asarray(radiant_intensity_percos)
alpha_gs_np = np.asarray(alpha_gs); beta_gs_np = np.asarray(beta_gs)
radiant_intensity_percos_gs = np.asarray(radiant_intensity_percos_gs)

print("Max |alpha_PINN - alpha_GS|:", np.max(np.abs(alpha_np - alpha_gs_np)))
print("Max |beta_PINN - beta_GS|:", np.max(np.abs(beta_np - beta_gs_np)))

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
im0 = axs[0].imshow(retrieved_phase_gs % (2 * np.pi), origin="lower", cmap="hsv",
                    extent=[x[0] / mm, x[-1] / mm, y[0] / mm, y[-1] / mm])
axs[0].set_title("GS phase profile")
axs[0].set_xlabel("x [mm]"); axs[0].set_ylabel("y [mm]")
axs[0].set_xlim([-0.2, 0.2]); axs[0].set_ylim([-0.2, 0.2]); axs[0].set_aspect(1)
fig.colorbar(im0, ax=axs[0], fraction=0.045, label=r"$\Phi$ [rad]")

im1 = axs[1].imshow(radiant_intensity_percos_gs, origin="lower", cmap="inferno",
                    extent=[alpha_gs_np[0], alpha_gs_np[-1], beta_gs_np[0], beta_gs_np[-1]], aspect="auto")
axs[1].set_title("GS far-field prediction")
axs[1].set_xlabel("alpha"); axs[1].set_ylabel("beta")
axs[1].set_xlim([-1, 1]); axs[1].set_ylim([-1, 1]); axs[1].set_aspect(1)
cb = fig.colorbar(im1, ax=axs[1], fraction=0.045)
cb.set_label(label=r"$\frac{\partial P(\alpha, \beta)}{\partial \Omega \cos(\theta)}$ [a.u.]", size=14)
fig.tight_layout()
fig.savefig("results/exp4_gs_validation.png", dpi=150)
print("Saved results/exp4_gs_validation.png")

# ------------------------------------------------------------------
# Shape-error comparison on [-1,1]^2 (notebook cell 37)
# ------------------------------------------------------------------
def least_squares_rescale(pred, target, mask):
    p = np.asarray(pred, dtype=np.float64)[mask]
    t = np.asarray(target, dtype=np.float64)[mask]
    denom = np.sum(p * p)
    scale = 0.0 if denom == 0 else np.sum(p * t) / denom
    return scale * np.asarray(pred, dtype=np.float64), scale


alpha_mask = (alpha_np >= -1.0) & (alpha_np <= 1.0)
beta_mask = (beta_np >= -1.0) & (beta_np <= 1.0)
region_mask = np.outer(beta_mask, alpha_mask)
target_compare = np.asarray(target_intensity_gs, dtype=np.float64)

pinn_scaled, pinn_scale = least_squares_rescale(radiant_intensity_percos_pinn, target_compare, region_mask)
gs_scaled, gs_scale = least_squares_rescale(radiant_intensity_percos_gs, target_compare, region_mask)

rms_pinn = np.sqrt(np.mean((pinn_scaled[region_mask] - target_compare[region_mask]) ** 2))
rms_gs = np.sqrt(np.mean((gs_scaled[region_mask] - target_compare[region_mask]) ** 2))

target_rms = np.sqrt(np.mean(target_compare[region_mask] ** 2))
rel_rms_pinn = rms_pinn / target_rms
rel_rms_gs = rms_gs / target_rms

mse_pinn = rms_pinn ** 2
mse_gs = rms_gs ** 2

print("\n" + "=" * 70)
print("SHAPE ERROR ON (alpha,beta) in [-1,1]^2  --  KEY PAPER METRICS")
print("=" * 70)
print(f"  PINN: MSE = {mse_pinn:.6e}   RMS = {rms_pinn:.6e}   relative RMS = {rel_rms_pinn:.6e}   scale = {pinn_scale:.6e}")
print(f"  GS:   MSE = {mse_gs:.6e}   RMS = {rms_gs:.6e}   relative RMS = {rel_rms_gs:.6e}   scale = {gs_scale:.6e}")
print(f"  Paper: PINN MSE ~ 3.59e6 (rel RMS 2.97%), GS MSE ~ 2.08e8 (rel RMS 22.6%)")

fig, axs = plt.subplots(1, 3, figsize=(14, 4))
vmax = np.percentile(target_compare[region_mask], 99.5)
for ax, img, title in [
    (axs[0], target_compare, "Target"),
    (axs[1], pinn_scaled, f"PINN prediction\nRMS = {rms_pinn:.3e}"),
    (axs[2], gs_scaled, f"GS prediction\nRMS = {rms_gs:.3e}"),
]:
    im = ax.imshow(img, origin="lower", cmap="inferno",
                   extent=[alpha_np[0], alpha_np[-1], beta_np[0], beta_np[-1]],
                   vmin=0, aspect="auto")
    ax.set_title(title); ax.set_xlabel("alpha"); ax.set_ylabel("beta")
    ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1]); ax.set_aspect(1)
    fig.colorbar(im, ax=ax, fraction=0.045)
fig.tight_layout()
fig.savefig("results/exp4_shape_error_comparison.png", dpi=150)
print("Saved results/exp4_shape_error_comparison.png")

# ------------------------------------------------------------------
# Cross-section comparison (notebook cell 39)
# ------------------------------------------------------------------
beta0_idx = int(np.argmin(np.abs(beta_np)))
fig = plt.figure(figsize=(4, 3))
ax = fig.add_subplot(1, 1, 1)
ax.plot(alpha_np, gs_scaled[beta0_idx, :], color="skyblue", label="GS design")
ax.plot(alpha_np, pinn_scaled[beta0_idx, :], color="green", label="PINN design")
ax.plot(alpha_np, target_compare[beta0_idx, :], "--", color="red", label="target")
ax.set_xlim([-1, 1])
ax.set_xlabel("alpha")
ax.set_ylabel(r"$\partial P/(\partial\Omega\cos\theta)$ [a.u.]")
ax.set_title("Simulated profiles at the far-field\nCross-section at beta = 0")
ax.grid(alpha=0.3); ax.legend()
fig.tight_layout()
fig.savefig("results/exp4_cross_section.png", dpi=150)
print("Saved results/exp4_cross_section.png")

# ------------------------------------------------------------------
# Energy-efficiency comparison (notebook cell 41)
# ------------------------------------------------------------------
def regular_grid_spacing(grid_1d):
    grid_1d = np.asarray(grid_1d, dtype=np.float64)
    if grid_1d.size < 2:
        return 1.0
    return float(np.abs(np.mean(np.diff(grid_1d))))


def integrate_on_grid(values, dx0, dx1, mask=None):
    vals = np.asarray(values, dtype=np.float64)
    if mask is not None:
        vals = np.where(mask, vals, 0.0)
    return float(np.sum(vals) * dx0 * dx1)


def energy_efficiency_farfield(prediction, target_support_mask, incident_flux, d_alpha, d_beta):
    target_flux = integrate_on_grid(prediction, d_alpha, d_beta, mask=target_support_mask)
    eta = target_flux / incident_flux if incident_flux != 0 else np.nan
    return eta, target_flux


incident_flux = integrate_on_grid(source_intensity_gs, dx, dy)
d_alpha = regular_grid_spacing(alpha_np)
d_beta = regular_grid_spacing(beta_np)
target_support_threshold = 1e-3 * np.nanmax(target_compare)
target_support_mask = region_mask & (target_compare > target_support_threshold)

eff_pinn, target_flux_pinn = energy_efficiency_farfield(
    radiant_intensity_percos_pinn, target_support_mask, incident_flux, d_alpha, d_beta)
eff_gs, target_flux_gs = energy_efficiency_farfield(
    radiant_intensity_percos_gs, target_support_mask, incident_flux, d_alpha, d_beta)

target_area = float(np.sum(target_support_mask) * d_alpha * d_beta)
print("\n" + "=" * 70)
print("ENERGY EFFICIENCY  eta = energy in target support / incident energy")
print("=" * 70)
print(f"  Target support threshold: {target_support_threshold:.6e}")
print(f"  Target support area in alpha-beta space: {target_area:.6e}")
print(f"  Incident energy flux: {incident_flux:.6e}")
print(f"  PINN target-area flux: {target_flux_pinn:.6e}")
print(f"  GS target-area flux:   {target_flux_gs:.6e}")
print(f"  PINN efficiency: {eff_pinn:.6%}")
print(f"  GS efficiency:   {eff_gs:.6%}")
print(f"  Paper: PINN ~ 99.98%, GS ~ 95.90%")