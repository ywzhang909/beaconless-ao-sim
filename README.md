# beaconless-ao-sim

Reproduction of DiComo et al., *"Beaconless adaptive optics for atmospheric laser
propagation with multi-plane convolutional neural network,"* Opt. Express
33(15):31010 (2025), DOI 10.1364/OE.561077 — simulation, multi-plane CNN
training (CNN1/CNNL), and evaluation at demo scale.

## Overview

The pipeline implements the paper's full chain (Algorithm 1 + Sec 2.4-2.7):

1. **Physics simulation** (`data/simulate.py`, `physics/`) — split-step
   propagation of a focused laser beam through 10 Kolmogorov phase screens
   (100 m apart, 1 km path), a diffraction-limited Gaussian beacon
   (waist `λL/D`), back-propagation of the beacon to the pupil with
   **analytic parabolic-defocus removal**, tip/tilt tracking from the
   conjugated beacon phase, and the 78-mode Zernike projection
   `Φ_Z78 = M_Z78(M⁺_Z78 Φ_beacon)` as the CNN target (Eqs 1-5, Algorithms B-C).
2. **Multi-plane imaging** — 3 measurement planes (defocus ±z_R about the
   focal plane, Eq 12), rough-surface scattering, 12-bit quantization
   (Eq 13), label z-scoring (Eq 14).
3. **CNN training** (`train.py`, `models/cnn.py`) — 3-stage CNN + 4-layer MLP
   head (Sec 2.6), Adam lr 1e-4, MSE on scaled Zernike modes, periodic
   in-simulation FOM evaluation.
4. **Evaluation** (`evaluate.py`, `utils/metrics.py`) — nPIB (Eq 6), SIB
   (Eq 7), FOM (Eq 8), gain (Eq 15), η (Eq 16), per-mode Pearson (Eq 17).

## Layout

```
data/simulate.py      Algorithm 1 pipeline (screens, beacon, FOM legs, dataset)
data/generate_h5.py   CLI: two-pass HDF5 dataset writer
physics/              zernike_aotools, screens_soapy, propagation_fft, scattering
models/cnn.py         CNN1 / CNNL architectures
train.py              training loop (DDP-capable, gradient accumulation)
evaluate.py           eval/eval-retrace CLI + WandB figures
utils/metrics.py      Eqs 6-8, 15-17 metrics
tests/                80+ unit tests (physics, metrics, models, data schema)
config.yaml           paper-verbatim Table 1 + demo-scale data/model/train
```

## Quick start

```bash
uv sync                      # create .venv from pyproject.toml / uv.lock
uv run python -m pytest tests/ -v

# 1. Generate the dataset (Algorithm 1, two-pass):
uv run python -m data.generate_h5 --config config.yaml

# 2. Train (paper batch 32 via micro-batch 8 x 4 accumulation):
uv run python train.py --config config.yaml

# 3. Evaluate + WandB figures:
uv run python evaluate.py --config config.yaml --ckpt checkpoints/best.pt
```

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

## Notes

- The 78-mode ceiling `FOM_Z78` sits close to the tracking baseline for
  strong-turbulence samples: with `D/r0≈7.4` the back-propagated beacon has
  intensity nulls whose phase branch points no 78-mode phase conjugator can
  command. This is a real operating-regime limit, not a simulation artifact;
  it bounds the achievable CNN gain (`η`, Eq 16) on the demo set.