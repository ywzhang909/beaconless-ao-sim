"""Data generation for the beaconless AO simulation (Algorithm 1).

This package reproduces the training-data generation pipeline of DiComo et al.,
Opt. Express 33(15):31010 (2025): per-sample simulation (``simulate``) and the
HDF5 dataset writer (``generate_h5``).
"""

from data.simulate import (
    SimSample,
    bucket_mask_nd,
    generate_dataset,
    physics_from_cfg,
    simulate_sample,
    simulate_sample_fom,
    vacuum_intensity,
)

__all__ = [
    "SimSample",
    "physics_from_cfg",
    "simulate_sample",
    "simulate_sample_fom",
    "vacuum_intensity",
    "bucket_mask_nd",
    "generate_dataset",
]
