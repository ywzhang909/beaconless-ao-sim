"""CLI for generating the beaconless AO HDF5 dataset.

Usage::

    python -m data.generate_h5 --config config.yaml [-n | --dry]

Reads the YAML configuration, runs :func:`data.simulate.generate_dataset`, and
prints a summary (splits, per-plane max, label mu/std, median FOMs).
"""

from __future__ import annotations

import argparse
import json
import time

import h5py
import numpy as np
import yaml

from data.simulate import generate_dataset


def _load_cfg(path: str) -> dict:
    """Load and return the YAML configuration dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _print_summary(h5_path: str) -> None:
    """Print a summary of the generated HDF5 dataset."""
    with h5py.File(h5_path, "r") as f:
        n_total = f["images"].shape[0]
        n_train = f["train_idx"].shape[0]
        n_test = f["test_idx"].shape[0]
        n_eval = f["eval_idx"].shape[0]
        N = f["images"].shape[2]

        images = f["images"][:]
        labels = f["labels"][:]
        scale_p = f["scale_p"][:]
        mu = f["mu"][:]
        sigma = f["sigma"][:]

        fom_noao = f["fom_noao"][:]
        fom_track = f["fom_track"][:]
        fom_beacon = f["fom_beacon"][:]
        fom_z78 = f["fom_z78"][:]

    print("=" * 60)
    print(f"HDF5 dataset: {h5_path}")
    print(f"  N_total = {n_total}  (train={n_train}, test={n_test}, eval={n_eval})")
    print(f"  images shape = (N_total, 3, {N}, {N}) uint16")
    print(f"  labels shape = (N_total, 78) float32")
    print("-" * 60)
    print("Per-plane max (quantized, /2047):")
    for p in range(3):
        print(f"  plane {p}: max={images[:, p].max()}, scale_p={scale_p[p]:.4g}")
    print("-" * 60)
    print("Label mu/std (first 8 modes):")
    for j in range(min(8, 78)):
        print(f"  mode {j}: mu={mu[j]:+.4f}  sigma={sigma[j]:.4f}")
    print("-" * 60)
    print("Median FOMs:")
    print(f"  noao   : {np.median(fom_noao):.4f}")
    print(f"  track  : {np.median(fom_track):.4f}")
    print(f"  beacon : {np.median(fom_beacon):.4f}")
    print(f"  z78    : {np.median(fom_z78):.4f}")
    print("=" * 60)


def main() -> None:
    """Entry point for the ``python -m data.generate_h5`` CLI."""
    parser = argparse.ArgumentParser(
        description="Generate the beaconless AO HDF5 dataset (Algorithm 1)."
    )
    parser.add_argument("--config", required=True, help="Path to the YAML config.")
    parser.add_argument(
        "-n",
        "--dry",
        action="store_true",
        help="Dry run: print the configuration and planned dataset, do not generate.",
    )
    args = parser.parse_args()

    cfg = _load_cfg(args.config)

    if args.dry:
        d = cfg["data"]
        p = cfg["physical"]
        n_total = d["n_train"] + d["n_test"] + d["n_eval"]
        print("DRY RUN (no generation):")
        print(f"  config: {args.config}")
        print(f"  N={p['N']}, L={p['L']} m, cn2={p['cn2']:.3e}")
        print(
            f"  splits: train={d['n_train']}, test={d['n_test']}, "
            f"eval={d['n_eval']}  (N_total={n_total})"
        )
        print(f"  workers={d['workers']}, master_seed={d['master_seed']}")
        print(f"  h5_path={d['h5_path']}")
        print(f"  config_json={json.dumps(cfg)[:120]}...")
        return

    t0 = time.time()
    h5_path = generate_dataset(cfg)
    elapsed = time.time() - t0
    print(f"\nGeneration completed in {elapsed:.1f} s -> {h5_path}")
    _print_summary(h5_path)


if __name__ == "__main__":
    main()
