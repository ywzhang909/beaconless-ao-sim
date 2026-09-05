# AGENTS.md — Project conventions for the Kilo agent

## Project

beaconless-ao-sim: GPU-optional reproduction of DiComo et al., "Beaconless
adaptive optics for atmospheric laser propagation with multi-plane
convolutional neural network", *Opt. Express* 33(15):31010 (2025).

## Layout

| Path | Purpose |
|---|---|
| `config.yaml` | Top-level config (paper-faithful Table 1, with `[DEMO]` flags for the smaller run) |
| `physics/` | Sim modules: `config.py` (SimConfig + load_config), `engine.py` (PhysicsEngine / MeasurementSource ABCs), `propagation_fft.py` (Propagator with split_step + zero-padded Fresnel), `screens_soapy.py` (aotools path), `oopao_backend.py` (OOPAO path), `scattering.py` (Lambertian roughness), `zernike_aotools.py` (78-order Noll basis) |
| `data/simulate.py` | Algorithm 1 single-pass pipeline (`simulate_sample`, `simulate_sample_fom`, `generate_dataset`) |
| `data/generate_h5.py` | Single-process CLI (`python -m data.generate_h5 --config config.yaml`) |
| `data/generate_full.py` | Chunked CLI for very large runs (--launch / --chunk N / --merge) |
| `data/dataset_summary.py` | Quick H5 summary CLI |
| `scripts/launch_chunks.py` | Convenience launcher for N parallel chunk workers |
| `scripts/逐过程仿真.py` | Step-by-step demo (sections 1-8) with matplotlib |
| `train.py` / `evaluate.py` | CNN training / evaluation |
| `tests/` | pytest suite (most are pre-existing) |

## Conventions

- **No comments unless asked.** Edit only with `Edit` / `Write`, never add commentary.
- **Run on Windows, Python 3.12, uv venv at `.venv`.** Use `.venv\Scripts\python.exe` for all runs.
- **Beam source is OOPAO** (`beam_source: "oopao"` in `config.yaml`).
- **Workers=1 is required on Windows** because OOPAO's `Telescope` / `Source` carry C-level state that is not picklable. `generate_dataset` auto-skips the multiprocessing Pool when `workers==1`.
- **Per-sample cost ~75 s on CPU** (N=512, n_roughness=10, largest zero-padded Fresnel N_pad=6125). The `_imaging` step accounts for ~95 % of that.

## Generate-data recipes

### 10-sample smoke test (~15 min)

```bash
python -m data.generate_h5 --config config_10samples.yaml
```

Writes `data/beaconless_demo_10samples.h5` (~16 MB). Validates Algorithm 1
end-to-end. Expected median FOMs: noao 0.47, beacon 0.98, z78 0.93.

### Full demo (2000/400/100) in 4 parallel chunks (~28 h)

```bash
python scripts/launch_chunks.py --config config_demo_full.yaml --n_chunks 4
```

Each chunk writes `data/chunks/chunk_NNNN.h5`; once all are done, merge:

```bash
python -m data.generate_full \
    --config config_demo_full.yaml --merge \
    --out_dir data/chunks --final_h5 data/beaconless_demo_full.h5
```

See `data/chunks/README.md` for details.

## Verifying a dataset

```bash
python data/dataset_summary.py data/<file>.h5
```

## H5 schema

`(N_total, 3, 512, 512) uint16 images`, `(N_total, 78) float32 labels`,
`(N_total,) float32 fom_{noao,track,beacon,z78}`, plus `seeds`, `L`,
`{train,test,eval}_idx`, `mu`, `sigma`, `scale_p`, `vacuum_intensity`.

## Pre-existing issues (do not touch without asking)

- `data/simulate.py` has ~20 LSP errors related to property overrides on
  `PhysicsEngine` and a `numpy.ndarray` vs `float` mismatch in
  `_unwrap_flood_fill_nb`. These are pyright / basedpyright quirks; the
  files run correctly at runtime.
- `data/simulate.py:530` (`_wrap_diff`) sees an `ndarray` from a numba
  fast-path; numba types resolve at runtime, not at type-check time.
- h5py `Group` / `Dataset` / `Datatype` LSP errors are all from a too-strict
  type stub; runtime is fine.
