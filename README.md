# beaconless-ao-sim

Reproduction and extension of DiComo et al., *"Beaconless adaptive optics for
atmospheric laser propagation with multi-plane convolutional neural network,"*
Opt. Express 33(15):31010 (2025), DOI 10.1364/OE.561077 — simulation,
multi-plane CNN training (CNN1/CNNL), and evaluation at demo scale, with the
turbulence-screen substrate re-based on the **OOPAO** library.

> **Full metrics, training curves, and the input-preprocessing smoke test:** see
> [`REPORT.md`](REPORT.md).

## Overview

The pipeline implements the paper's full chain (Algorithm 1 + Sec 2.4-2.7):

1. **Physics simulation** (`data/simulate.py`, `physics/`) — split-step
   propagation of a focused laser beam through 10 Kolmogorov phase screens
   (100 m apart, 1 km path), a diffraction-limited Gaussian beacon
   (waist `λL/D`), back-propagation of the beacon to the pupil with
   **analytic parabolic-defocus removal**, tip/tilt tracking from the
   conjugated beacon phase, and the 78-mode Zernike projection
   `Φ_Z78 = M_Z78(M⁺_Z78 Φ_beacon)` as the CNN target (Eqs 1-5, Algorithms B-C).
   Turbulence screens are drawn from the **OOPAO** library
   (`physics/oopao/`, `physics/oopao_backend.py`).
2. **Multi-plane imaging** — 3 measurement planes (defocus ±z_R about the
   focal plane, Eq 12), rough-surface scattering, 12-bit quantization
   (Eq 13), label z-scoring (Eq 14).
3. **CNN training** (`train.py`, `models/cnn.py`) — 3-stage CNN + 4-layer MLP
   head (Sec 2.6), Adam lr 1e-4, MSE on scaled Zernike modes, periodic
   in-simulation FOM evaluation.
4. **Evaluation** (`evaluate.py`, `utils/metrics.py`) — nPIB (Eq 6), SIB
   (Eq 7), FOM (Eq 8), gain (Eq 15), η (Eq 16), per-mode Pearson (Eq 17).
5. **Input-preprocessing smoke test** (`smoke_test.py`) — runs the trained
   CNN1 on 7 input-preprocessing methods (baseline `/2047`, raw uint16,
   per-sample / global min-max, z-score, single-plane, raw+single-plane) and
   reports Rj, FOM_ML, and feature-map statistics per method.

## OOPAO integration

The turbulence-screen generator is re-based on
[OOPAO](https://github.com/cheritier/OOPAO) (vendored into `physics/oopao/`).
`OopaoScreenBackend` draws per-layer von-Karman screens, center-crops them to
the 512×512 pupil grid, and amplitude-rescales each layer to the target
per-slab r0 (`r0_slab = r0_path · n^(3/5)`). It is selected via the
`physical.beam_source` config switch (`soapy | aotools | oopao`; default
`oopao`). The custom FFT split-step propagator, Algorithm-1 beacon back-prop,
and multi-plane imaging are retained — OOPAO supplies the turbulence + pupil +
Zernike basis only. OOPAO screens are statistically equivalent to the aotools
path (per-slab OPD std ratio ≈ 0.84) and deterministic per seed.

## Layout

```
data/simulate.py          Algorithm 1 pipeline (screens, beacon, FOM legs, dataset)
data/generate_h5.py       CLI: two-pass HDF5 dataset writer
physics/oopao/            vendored OOPAO modules (Atmosphere, Telescope, Source, Zernike, ...)
physics/oopao_backend.py  OopaoScreenBackend (OOPAO turbulence, r0 rescale)
physics/                  zernike_aotools, screens_soapy, propagation_fft, scattering
models/cnn.py             CNN1 / CNNL architectures
train.py                  training loop (DDP-capable, gradient accumulation)
evaluate.py               eval/eval-retrace CLI + WandB figures
smoke_test.py             per-method input-preprocessing smoke test + WandB
utils/metrics.py          Eqs 6-8, 15-17 metrics
tests/                    80+ unit tests (physics, metrics, models, data schema)
config.yaml               paper-verbatim Table 1 + demo-scale data/model/train
REPORT.md                 full pipeline report (data → model → train → eval → smoke)
```

## Quick start

```bash
uv sync                      # create .venv from pyproject.toml / uv.lock
uv run python -m pytest tests/ -v

# 1. Generate the dataset (Algorithm 1, two-pass, OOPAO screens):
uv run python -m data.generate_h5 --config config.yaml

# 2. Train (paper batch 32 via micro-batch 8 x 4 accumulation):
CUDA_VISIBLE_DEVICES=1 uv run python train.py --config config.yaml

# 3. Evaluate + WandB figures:
CUDA_VISIBLE_DEVICES=1 uv run python evaluate.py --config config.yaml --ckpt checkpoints/best.pt

# 4. Per-method input-preprocessing smoke test + WandB:
CUDA_VISIBLE_DEVICES=1 uv run python smoke_test.py --config config.yaml --ckpt checkpoints/best.pt
```

> GPU note: the demo host runs VLLM workers on both GPUs (~20.6 GiB each,
> ~3.4 GiB free on GPU 1). Use `CUDA_VISIBLE_DEVICES=1` and a reduced batch
> (`--batch-size 8` for the smoke test) to fit.

## Demo scale vs paper

The simulation parameters (Table 1) are reproduced verbatim. Data/model/train
sizes are reduced for a quick demo (`config.yaml` documents each deviation):

| quantity   | paper     | demo    |
|------------|-----------|---------|
| n_train    | 81,000    | 2,000   |
| n_test     | 9,000     | 400     |
| n_eval     | 1,000     | 100     |
| train steps| 11,500    | 3,000   |
| batch      | 32        | 32 (micro 8) |

## Headline results (OOPAO dataset)

| Leg | median FOM |
|-----|-----------|
| noao / track | 0.27–0.28 |
| beacon | 0.93 |
| z78 (78-mode ceiling) | 0.88 |
| **ML (CNN1)** | **0.49** |

gain g (Eq 15) = **1.72**, η (Eq 16) = **0.34**.

**Preprocessing smoke test** — the network is sensitive to its training-time
input contract. Baseline `/2047` (3-plane) gives FOM_ML ≈ 0.51 and Rj ≈ 0.16;
raw uint16 and z-score inputs saturate the CNN trunk (feature max ~10⁴–10⁵)
and collapse FOM_ML to ~0.001; single-plane inputs lose the defocus parallax
needed for depth sensing (FOM_ML ≈ 0.36, Rj ≈ 0.04). Global min-max is the
most robust alternative (FOM_ML ≈ 0.48). Full table in [`REPORT.md`](REPORT.md#51-methods--results).

## Notes

- The 78-mode ceiling `FOM_Z78` sits close to the tracking baseline for
  strong-turbulence samples: with `D/r0≈7.4` the back-propagated beacon has
  intensity nulls whose phase branch points no 78-mode phase conjugator can
  command. This is a real operating-regime limit, not a simulation artifact;
  it bounds the achievable CNN gain (`η`, Eq 16) on the demo set.

## WandB

- Project: https://wandb.ai/ywzhang909/beaconless-ao-sim
- Training: `sparkling-plasma-1` (`1jjq96ae`)
- Evaluation: `eoudxg1u`
- Preprocessing smoke: `094tewxf`
