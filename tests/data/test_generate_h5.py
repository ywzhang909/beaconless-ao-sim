"""Tests for data.generate_h5 / data.simulate.generate_dataset.

TDD: these tests are written first (RED), then the modules are implemented
(GREEN). Verifies the pinned HDF5 schema, quantization (Eq 13), and the
train-only computation of mu/sigma (Eq 14) and scale_p.
"""

import json
import os

import h5py
import numpy as np
import pytest

from data.simulate import generate_dataset, physics_from_cfg, simulate_sample


def tiny_cfg(tmp_path) -> dict:
    """Small config: N=64, n_train=4, n_test=2, n_eval=2, workers=2."""
    return {
        "physical": {
            "cn2": 8.13e-15,
            "l0_sim": 0.01,
            "L0": 100.0,
            "L": 1000.0,
            "wavelength": 800e-9,
            "Dscope": 0.30,
            "rspot": 0.075,
            "focal": 1000.0,
            "N": 64,
            "box_size": 0.30,
            "n_screens": 2,
            "screen_sep": 500.0,
            "n_roughness": 2,
            "roughness_seed": 42,
            "beam_source": "aotools",
            "screen_pool": 0,
        },
        "imaging": {
            "zR_APWS": None,
            "f_obj": None,
            "plane_offset_frac": [0.0, 1.0, 2.0],
        },
        "bucket": {"diameter_frac": 2.5},
        "data": {
            "n_train": 4,
            "n_test": 2,
            "n_eval": 2,
            "master_seed": 20250830,
            "workers": 2,
            "h5_path": str(tmp_path / "test.h5"),
        },
    }


def test_generate_dataset_schema(tmp_path):
    """Full pipeline writes a file matching the pinned HDF5 schema."""
    cfg = tiny_cfg(tmp_path)
    h5_path = generate_dataset(cfg)
    assert os.path.exists(h5_path)

    N = cfg["physical"]["N"]
    n_train, n_test, n_eval = 4, 2, 2
    N_total = n_train + n_test + n_eval
    L = cfg["physical"]["L"]

    with h5py.File(h5_path, "r") as f:
        # shapes / dtypes
        assert f["images"].shape == (N_total, 3, N, N)
        assert f["images"].dtype == np.uint16
        assert f["labels"].shape == (N_total, 78)
        assert f["labels"].dtype == np.float32
        assert f["fom_noao"].shape == (N_total,)
        assert f["fom_track"].shape == (N_total,)
        assert f["fom_beacon"].shape == (N_total,)
        assert f["fom_z78"].shape == (N_total,)
        assert f["seeds"].shape == (N_total,)
        assert f["seeds"].dtype == np.int64
        assert f["L"].shape == (N_total,)
        assert f["L"].dtype == np.float32
        assert np.all(f["L"][:] == L)
        assert f["train_idx"].shape == (n_train,)
        assert f["test_idx"].shape == (n_test,)
        assert f["eval_idx"].shape == (n_eval,)
        assert f["mu"].shape == (78,)
        assert f["sigma"].shape == (78,)
        assert f["scale_p"].shape == (3,)
        assert f["vacuum_intensity"].shape == (N, N)
        assert f["vacuum_intensity"].dtype == np.float32
        assert "config_json" in f.attrs
        cfg_round = json.loads(f.attrs["config_json"])
        assert cfg_round["physical"]["N"] == N

        images = f["images"][:]
        labels = f["labels"][:]

        # quantization: uint16 <= 2047
        assert images.max() <= 2047

        # idx arrays partition 0..N_total-1
        idx = np.concatenate([f["train_idx"][:], f["test_idx"][:], f["eval_idx"][:]])
        np.testing.assert_array_equal(np.sort(idx), np.arange(N_total))

        # labels finite
        assert np.all(np.isfinite(labels))

        # mu/sigma computed ONLY on train_idx
        train_labels = labels[f["train_idx"][:]]
        np.testing.assert_allclose(f["mu"][:], train_labels.mean(axis=0), rtol=1e-5)
        np.testing.assert_allclose(f["sigma"][:], train_labels.std(axis=0), rtol=1e-5)

        # scale_p = max over TRAIN raw images only
        shared = physics_from_cfg(cfg)
        raw = np.zeros((N_total, 3, N, N), dtype=np.float32)
        for i in range(N_total):
            seed = cfg["data"]["master_seed"] + i
            s = simulate_sample(seed, cfg, shared=shared)
            raw[i] = s.images
        train_raw = raw[f["train_idx"][:]]
        np.testing.assert_allclose(
            f["scale_p"][:], train_raw.max(axis=(0, 2, 3)), rtol=1e-5
        )

        # max(images[:,p])/2047 close to max_raw/scale_p (Eq 13); scale_p is the
        # TRAIN-only per-plane max, so test/eval samples exceeding it clip at
        # 2047 (ratio saturates at 1.0).
        for p in range(3):
            for i in range(N_total):
                ratio = images[i, p].max() / 2047
                expected = min(raw[i, p].max() / f["scale_p"][p], 1.0)
                assert ratio == pytest.approx(expected, rel=1e-3)
