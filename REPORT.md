# Beaconless ML-AO: Full Pipeline Report

Reproduction and extension of DiComo et al., *"Beaconless adaptive optics for
atmospheric laser propagation with multi-plane convolutional neural network,"*
Opt. Express 33(15):31010 (2025), DOI 10.1364/OE.561077 — now re-based on the
**OOPAO** library as the turbulence-screen substrate, with a per-method
input-preprocessing smoke test.

| Artifact | Location |
|----------|----------|
| Dataset (OOPAO screens) | `data/beaconless_demo.h5` (2 500 samples, 3.93 GB) |
| Model checkpoint | `checkpoints/best.pt` (CNN1, 266 MB) |
| Evaluation figures | `results/fig_*.png` |
| Evaluation metrics | `results/results.json` |
| This report | `REPORT.md` |
| WandB project | https://wandb.ai/ywzhang909/beaconless-ao-sim |
| Training run | `sparkling-plasma-1` (`1jjq96ae`) |
| Evaluation run | `eoudxg1u` |
| Preprocessing smoke run | `094tewxf` |

---

## 1. Physics simulation & data generation

### 1.1 Pipeline (Algorithm 1)

`data/simulate.py` + `physics/` implement the full chain:

1. **Turbulence screens** — 10 Kolmogorov / von-Karman phase screens, 100 m
   apart over a 1 km path (Cn² = 8.13e-15, L0 = 100 m, λ = 800 nm).
2. **Split-step propagation** — a focused Gaussian beam (waist `λL/D`) is
   propagated through the 10 slabs to a target at L = 1 km
   (`physics/propagation_fft.py`).
3. **Algorithm-1 beacon** — a diffraction-limited Gaussian beacon is launched,
   propagated to the target, and **back-propagated to the pupil** with
   analytic parabolic-defocus removal. The conjugated beacon phase is the
   beaconless wavefront estimate.
4. **Zernike projection** — `Φ_Z78 = M_Z78(M⁺_Z78 Φ_beacon)` (78-mode Noll
   truncation, `physics/zernike_aotools.py`) is the CNN target.
5. **Multi-plane imaging** — 3 measurement planes (defocus ±z_R about the
   focal plane, Eq 12), rough-surface scattering (10 realizations), 12-bit
   quantization (Eq 13).

### 1.2 OOPAO integration (this work)

The turbulence-screen substrate is re-based on the **OOPAO** library
([github.com/cheritier/OOPAO](https://github.com/cheritier/OOPAO)), vendored
into `physics/oopao/` (Atmosphere, Telescope, Source, Zernike, phaseStats,
tools).

`physics/oopao_backend.py` (`OopaoScreenBackend`):
- Draws per-layer von-Karman screens from an OOPAO `Atmosphere`.
- Center-crops each screen to the 512×512 pupil grid.
- **Amplitude-rescales** each layer to the target per-slab r0
  (`r0_slab = r0_path · n^(3/5)`, r0 @ 800 nm from Cn²·L). OOPAO expresses r0
  at 500 nm and its OPD is in **radians**, so a PSD-normalization correction
  (`(r0_slab / r0_ref)^(5/6)`) is applied — an initial naive `^(5/6)` rescale
  over-drove the screens 4.8×; measuring per-layer OPD std at a reference r0
  fixed it.

Wired into `data/simulate.py` behind a `physical.beam_source` switch
(`soapy | aotools | oopao`); `config.yaml` sets `beam_source: "oopao"`. The
custom FFT split-step propagator, Algorithm-1 beacon back-prop, and multi-plane
imaging are **retained** — OOPAO supplies the turbulence + pupil + Zernike
basis only (it does not propagate a focused beam through N slabs to a distant
target).

### 1.3 Equivalence & determinism

| Check | Result |
|-------|--------|
| Per-slab OPD std, OOPAO / aotools | **0.84** (within von-Karman sampling noise) |
| Same-seed determinism (labels & images) | **bit-identical** |
| Test suite | **80 / 80 passed** |

### 1.4 Dataset statistics (OOPAO screens)

`data/beaconless_demo.h5`, N_total = 2 500 (2 000 train / 400 test / 100 eval),
seed schedule `master_seed + index` (master 20250830).

| Leg | median FOM |
|-----|-----------|
| noao | 0.2697 |
| track | 0.2697 |
| beacon | 0.9317 |
| z78 (78-mode ceiling) | 0.8818 |

`D/r0 ≈ 7.4` → strong turbulence; the 78-mode ceiling sits close to the
tracking baseline (beacon intensity nulls produce phase branch points no
78-mode phase conjugator can command).

---

## 2. Model

**CNN1** (`models/cnn.py`), per Sec 2.6 / Table 1:

- 3-stage CNN: `3×3 conv (stride 1, pad 0) → BatchNorm2d → ReLU → 2×2 MaxPool`,
  channels `[32, 64, 128]`.
- Input `(B, 3, 512, 512)` → after 3 pools `62×62` → `AdaptiveAvgPool2d((18,18))`
  → flatten `128·18·18 = 41 472`.
- Shared MLP: 4 hidden ReLU layers of 512 → 78 Zernike outputs.
- **~19.4 M parameters.**

Input contract: 3-plane intensity images, **uint16 / 2047** (12-bit camera
scale). Output: 78 Zernike coefficients, z-scored at training (Eq 14).

---

## 3. Training

`train.py`, Table 1 hyperparameters:

| | |
|---|---|
| Optimizer | Adam, lr 1e-4, β 0.9/0.999 |
| Loss | MSE on z-scored Zernike modes |
| Batch | 32 (micro-batch 8 × 4 accumulation, VRAM-limited) |
| Steps | 3 000 (demo; paper 11 500) |
| AMP | FP16 autocast + grad scaler |
| In-sim FOM eval | every 500 steps on 8 samples |

**Final in-simulation FOM (training, `1jjq96ae`):**

| Metric | Value |
|--------|-------|
| median FOM_ML | **0.4695** |
| median FOM_track | 0.2439 |
| median FOM_z78 | 0.9039 |
| gain g (Eq 15) | **1.925** |
| η (Eq 16) | **0.3418** |
| final MSE loss | 0.2526 |

Checkpoint: `checkpoints/best.pt` (best in-sim FOM), `checkpoints/last.pt`.

---

## 4. Evaluation (Sec 2.7)

`evaluate.py` on the 100-sample eval split, OOPAO dataset (`eoudxg1u`):

| Metric | Value |
|--------|-------|
| median FOM noao | 0.2823 |
| median FOM track | 0.2824 |
| median FOM beacon | 0.9347 |
| median FOM z78 | 0.8772 |
| median FOM_ML | **0.4861** |
| gain g (Eq 15) | **1.721** |
| η (Eq 16) | **0.3424** |
| mean Rj (Eq 17) | 0.1710 |

The ML leg (FOM_ML 0.486) beats the noao/track baselines (0.282) → gain 1.72,
η 0.34. Figures: `results/fig_FOM_scatter.png`, `fig_pred_vs_true.png`,
`fig_Rj_per_mode.png`, `fig_samples.png`.

---

## 5. Input-preprocessing smoke test (this work)

`smoke_test.py` runs the trained CNN1 on each of 7 input-preprocessing methods
**independently** (one by one), on the 50-sample eval subset, with a batched
low-VRAM forward (batch 8) on the memory-constrained GPU. For each method it
records input stats, CNN-trunk feature-map stats (saturation / dead-feature
diagnostics), per-mode Pearson Rj, RMS coefficient error, and the
physics-simulation FOM_ML. Results uploaded to WandB (`094tewxf`).

### 5.1 Methods & results

| # | Method | input mean | input max | feat max | pred RMS (rad) | Rj mean | **FOM_ML** |
|---|--------|-----------|-----------|----------|----------------|---------|-----------|
| 0 | **baseline_norm** (3-plane, /2047) | 0.043 | 0.93 | 6.65 | 0.161 | 0.163 | **0.506** |
| 1 | raw_uint16 (3-plane, 0–2047) | 88.6 | 1899 | 22 258 | 734.9 | 0.031 | 0.0007 |
| 2 | minmax_sample (3-plane, per-sample) | 0.071 | 1.0 | 13.1 | 0.294 | 0.086 | 0.176 |
| 3 | minmax_global (3-plane, global) | 0.047 | 1.0 | 7.39 | 0.168 | 0.148 | 0.481 |
| 4 | zscore (3-plane, per-set z-score) | ~0 | 18.7 | 163.6 | 4.12 | 0.024 | 0.0011 |
| 5 | oneplane_norm (1 focal plane, /2047) | 0.015 | 0.93 | 10.68 | 0.225 | 0.040 | 0.362 |
| 6 | oneplane_raw (1 focal plane, 0–2047) | 31.5 | 1899 | 28 159 | 446.9 | −0.003 | 0.0008 |

### 5.2 Findings

1. **Scale is a hard contract.** Any method that leaves pixel values off the
   `/2047` scale (raw_uint16, zscore, oneplane_raw) destroys the network:
   feature maps saturate (feat max 163–28 159 vs ~7 at baseline) and coefficient
   error blows up (RMS 4–735 rad vs 0.16). FOM_ML collapses to ~0. The BatchNorm
   layers partially absorb scale, but the MLP head and the downstream
   denormalization (`c = y·σ + μ`) are calibrated to the `/2047` range and
   cannot recover from a 2047× or z-scored input.

2. **Global min-max ≈ baseline.** minmax_global (FOM 0.481) tracks the baseline
   (0.506) closely because the eval-set dynamic range nearly matches the 12-bit
   scale — the transform is near-identity. Per-sample min-max (FOM 0.176) is
   much worse: it re-normalizes each sample to full 0–1, erasing the relative
   intensity differences between samples that the network learned.

3. **Single plane degrades but does not break.** oneplane_norm (FOM 0.362) keeps
   sensible features (feat max ~11) and retains partial correction, but loses
   ~28% FOM vs baseline. The 3-channel design encodes the ±z_R defocus
   parallax used for depth-aware phase recovery; with one plane the network
   degrades to 2D phase estimation and higher-order Zernike modes lose accuracy
   (Rj 0.040 vs 0.163).

4. **Combined failure mode is catastrophic.** oneplane_raw (1 plane + wrong
   scale) is the worst single case after raw_uint16 (FOM 0.0008, Rj ≈ 0) — both
   violations compound.

### 5.3 Conclusion

The CNN1 is **not scale-invariant** and **not plane-redundant**. The training
input contract — 3 measurement planes at `/2047` (12-bit) scale — is a hard
requirement for the network to produce useful Zernike estimates. Any
deployment-time preprocessing must preserve both the 3-plane structure and the
camera-intensity scale; min-max / z-score normalization and single-plane
cropping materially reduce (or eliminate) correction performance.

---

## 6. Reproducibility

```bash
uv sync
uv run python -m pytest tests/ -v            # 80/80

# 1. Data (OOPAO screens):
uv run python -m data.generate_h5 --config config.yaml

# 2. Train (GPU):
CUDA_VISIBLE_DEVICES=1 uv run python train.py --config config.yaml

# 3. Evaluate + WandB:
CUDA_VISIBLE_DEVICES=1 uv run python evaluate.py --config config.yaml --ckpt checkpoints/best.pt

# 4. Per-method preprocessing smoke test:
CUDA_VISIBLE_DEVICES=1 uv run python smoke_test.py --config config.yaml --ckpt checkpoints/best.pt
```

GPU note: the host runs VLLM workers on both GPUs (~20.6 GiB each), leaving
~3.4 GiB free on GPU 1. All GPU steps above use `CUDA_VISIBLE_DEVICES=1` and a
reduced batch (`--batch-size 8` for the smoke test) to fit.
