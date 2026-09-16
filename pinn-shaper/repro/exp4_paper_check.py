"""
Paper-number check: reproduce Section 4.3 metrics using the OFFICIAL
pretrained model (saved_models/DT_flat-top-farfield_model.pth).
Expected: PINN MSE ~3.59e6, rel-RMS ~2.97%, eta ~99.98%; GS MSE ~2.08e8, rel-RMS ~22.6%, eta ~95.90%.
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import torch
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

# --- setup identical to notebook cell 4 ---
extent_source = 1000 * um
w_I = 50.6 * um / (100 * um) * extent_source / 2
lam = 1.0 * um


def I_fun(x, y):
    r = torch.sqrt(x**2 + y**2)
    return torch.exp(-(r / w_I) ** 2) ** 2


image = np.array(scipy_gaussian_filter(load_image("images/DT.png"), 10, mode="nearest", cval=0), dtype="float32")
Irad_fun = create_interpolator(torch.tensor(image), mode="bilinear", region=[-1, 1, -1, 1], align_corners=True)


def constrain_fn(x):
    return x[:, 0:1] ** 2 + x[:, 1:2] ** 2


solver = Farfield_PINN(I_fun, Irad_fun, extent_source, lam,
                       integration_points=5000, hidden_layers=4, num_features=150,
                       constrain=True, symmetry=None, constrain_fn=constrain_fn)
solver.net.load_state_dict(torch.load("saved_models/DT_flat-top-farfield_model.pth"))
print("Loaded official pretrained model: saved_models/DT_flat-top-farfield_model.pth")

# --- far-field validation ---
import jax
jax.config.update("jax_enable_x64", True)
import diffractsim
diffractsim.set_backend("CPU")
from diffractsim import MonochromaticField

Nx, Ny = 2048, 2048
F = MonochromaticField(wavelength=lam, extent_x=extent_source, extent_y=extent_source, Nx=Nx, Ny=Ny)
phi_spline = solver.get_phi_spline()


def I_fun_numpy(x, y):
    r = np.sqrt(x**2 + y**2)
    return np.exp(-(r / w_I) ** 2) ** 2


F.E = np.sqrt(I_fun_numpy(F.xx, F.yy)) * np.exp(1j * phi_spline(F.x, F.y))
alpha, beta, radiant_intensity_percos = F.get_farfield()

# --- GS baseline ---
import jax.numpy as jnp

def evaluate_target_on_alpha_beta_grid(alpha, beta, chunk_rows=128):
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
    for _ in range(num_iter):
        U = source_amplitude_j * jnp.exp(1j * jnp.angle(U_p))
        Uf = jnp.fft.fft2(U)
        Uf_p = target_amplitude_j * jnp.exp(1j * jnp.angle(Uf))
        U_p = jnp.fft.ifft2(Uf_p)
    retrieved_phase = jnp.fft.fftshift(jnp.angle(U_p))
    return np.asarray(retrieved_phase)


xx, yy = F.xx, F.yy
dx, dy = F.dx, F.dy
source_intensity_gs = I_fun_numpy(xx, yy)
target_intensity_gs = evaluate_target_on_alpha_beta_grid(alpha, beta, chunk_rows=128)

retrieved_phase_gs = gs_phase_retrieval_from_arrays(source_intensity_gs, target_intensity_gs, dx, dy, num_iter=60)

E_gs = np.sqrt(np.maximum(source_intensity_gs, 0.0)) * np.exp(1j * retrieved_phase_gs)
F.E = np.asarray(E_gs, dtype=np.complex128)
alpha_gs, beta_gs, radiant_intensity_percos_gs = F.get_farfield()

alpha_np = np.asarray(alpha); beta_np = np.asarray(beta)
radiant_intensity_percos_pinn = np.asarray(radiant_intensity_percos)
radiant_intensity_percos_gs = np.asarray(radiant_intensity_percos_gs)

# --- metrics ---
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

print("\n================= PAPER NUMBER CHECK (official pretrained model) =================")
print(f"  PINN: MSE = {rms_pinn**2:.6e}   RMS = {rms_pinn:.6e}   relative RMS = {rel_rms_pinn:.6e}")
print(f"  GS:   MSE = {rms_gs**2:.6e}   RMS = {rms_gs:.6e}   relative RMS = {rel_rms_gs:.6e}")
print("  Paper: PINN MSE 3.59e6 (2.97%), GS MSE 2.08e8 (22.6%)")

# efficiency
def regular_grid_spacing(g):
    g = np.asarray(g, dtype=np.float64)
    return float(np.abs(np.mean(np.diff(g)))) if g.size >= 2 else 1.0


def integrate_on_grid(values, dx0, dx1, mask=None):
    vals = np.asarray(values, dtype=np.float64)
    if mask is not None:
        vals = np.where(mask, vals, 0.0)
    return float(np.sum(vals) * dx0 * dx1)


incident_flux = integrate_on_grid(source_intensity_gs, dx, dy)
d_alpha = regular_grid_spacing(alpha_np)
d_beta = regular_grid_spacing(beta_np)
target_support_threshold = 1e-3 * np.nanmax(target_compare)
target_support_mask = region_mask & (target_compare > target_support_threshold)

eff_pinn = integrate_on_grid(radiant_intensity_percos_pinn, d_alpha, d_beta, mask=target_support_mask) / incident_flux
eff_gs = integrate_on_grid(radiant_intensity_percos_gs, d_alpha, d_beta, mask=target_support_mask) / incident_flux

print(f"  PINN efficiency: {eff_pinn:.6%}")
print(f"  GS efficiency:   {eff_gs:.6%}")
print("  Paper: PINN ~99.98%, GS ~95.90%")

# save comparison figure
fig, axs = plt.subplots(1, 3, figsize=(14, 4))
for ax, img, title in [
    (axs[0], target_compare, "Target"),
    (axs[1], pinn_scaled, f"PINN\nRMS={rms_pinn:.3e}"),
    (axs[2], gs_scaled, f"GS\nRMS={rms_gs:.3e}"),
]:
    im = ax.imshow(img, origin="lower", cmap="inferno",
                   extent=[alpha_np[0], alpha_np[-1], beta_np[0], beta_np[-1]], vmin=0, aspect="auto")
    ax.set_title(title); ax.set_xlabel("alpha"); ax.set_ylabel("beta")
    ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1]); ax.set_aspect(1)
    fig.colorbar(im, ax=ax, fraction=0.045)
fig.tight_layout()
fig.savefig("results/exp4_paper_check.png", dpi=150)
print("Saved results/exp4_paper_check.png")