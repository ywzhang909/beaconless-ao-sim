# Full-Demo Chunked Dataset Generation

## Overview

This directory contains the **Algorithm 1** (DiComo et al., 2025) full-demo
dataset generation pipeline split into 4 parallel chunks, then merged into a
single HDF5.

The 10-sample smoke test (`data/beaconless_demo_10samples.h5`) confirmed the
pipeline is correct: median FOMs are
- noao 0.47, track 0.47, **beacon 0.98**, **z78 0.93** —
  matching the paper's expected ordering and magnitudes.

## Files

| Path | Purpose |
|---|---|
| `config_demo_full.yaml` | Full-demo config (n_train=2000, n_test=400, n_eval=100) |
| `data/generate_full.py` | Chunked CLI (`--launch` / `--chunk N` / `--merge`) |
| `scripts/launch_chunks.py` | Convenience: launch N parallel chunk workers |
| `data/dataset_summary.py` | CLI to print the H5 summary |
| `data/chunks/chunk_0000.h5` … `chunk_0003.h5` | Per-chunk outputs (4 × 625 samples) |
| `data/chunks/logs/chunk_*.log` | Per-chunk worker logs |
| `data/beaconless_demo_10samples.h5` | 10-sample smoke-test output (16.8 MB) |

## How it was run

```bash
# Smoke test (10 samples, ~15 min)
python -m data.generate_h5 --config config_10samples.yaml

# Full demo in 4 parallel chunks (~28 h, in background)
python scripts/launch_chunks.py --config config_demo_full.yaml --n_chunks 4
```

## When chunks complete — merge

After `scripts/launch_chunks.py` returns (or when all `chunk_*.h5` are
present), run:

```bash
python -m data.generate_full \
    --config config_demo_full.yaml \
    --merge \
    --out_dir data/chunks \
    --final_h5 data/beaconless_demo_full.h5
```

This concatenates the 4 chunk H5s into the canonical full-demo dataset
(`data/beaconless_demo_full.h5`, ~4.2 GB for 2500 samples).

## Verifying the merged dataset

```bash
python data/dataset_summary.py data/beaconless_demo_full.h5
```

Expected: `N_total = 2500 (train=2000, test=400, eval=100)` and median FOMs
matching the smoke test (beacon ~0.98, z78 ~0.93).

## Performance

- Per sample: ~75 s in-process (single-threaded CPU)
- The bottleneck is `_imaging`: 10 roughness realizations × 3 imaging planes,
  with the largest zero-padded Fresnel `N_pad ≈ 6125` (300 MB complex64 per
  FFT, ~5 s per FFT on CPU).
- 4 chunks parallel: 4 × 625 samples × 75 s ≈ 13 h per chunk, all run in
  parallel.
- Single-process full run (91000 samples) is not feasible on a single
  workstation: ~87 days CPU.

## Why a chunked script?

`data.simulate.generate_dataset` uses a `multiprocessing.Pool` which on
Windows is `spawn`-based. The OOPAO `Telescope` / `Source` carry C-level
state that is not picklable, so the Pool cannot pass the engine to workers.
We therefore launch **independent** Python processes (each with its own
OOPAO instance), partition the 2500 indices into 4 disjoint ranges, and
write per-chunk HDF5 files. A final merge step concatenates the chunks.
