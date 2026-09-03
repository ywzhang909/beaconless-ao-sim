"""Tests for the CNNL (variable-propagation-length) plumbing.

Verifies that the per-sample propagation distance ``L`` is threaded from the
HDF5 dataset through the training loop / sim-eval / evaluate.predict into the
CNNL model's length head, normalized to ``[1.0, 2.6]`` (raw metres / 1000).

These tests are fast and hermetic: they exercise the wiring only (dataset
``__getitem__`` + a single CPU forward pass), never full training or simulation.
"""

from __future__ import annotations

import os

import h5py
import numpy as np
import pytest
import torch

import train
from models.cnn import CNNL

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")


@pytest.fixture
def cnnl_h5(tmp_path):
    """Tiny H5 with the pinned schema including /L (per-sample metres)."""
    path = tmp_path / "cnnl.h5"
    n_total, n_modes, N = 16, 78, 32
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        f.create_dataset(
            "/images",
            data=rng.integers(0, 2048, (n_total, 3, N, N)).astype(np.uint16),
        )
        f.create_dataset(
            "/labels", data=rng.normal(size=(n_total, n_modes)).astype(np.float32)
        )
        f.create_dataset(
            "/fom_track", data=rng.random(n_total).astype(np.float32)
        )
        f.create_dataset(
            "/fom_z78", data=rng.random(n_total).astype(np.float32)
        )
        f.create_dataset("/seeds", data=np.arange(n_total, dtype=np.int64))
        L = rng.uniform(1000.0, 2600.0, n_total).astype(np.float32)
        f.create_dataset("/L", data=L)
        f.create_dataset("/train_idx", data=np.arange(n_total, dtype=np.int64))
        f.create_dataset("/test_idx", data=np.arange(n_total, dtype=np.int64))
        f.create_dataset("/eval_idx", data=np.arange(n_total, dtype=np.int64))
        f.create_dataset("/mu", data=np.zeros(n_modes, dtype=np.float32))
        f.create_dataset("/sigma", data=np.ones(n_modes, dtype=np.float32))
        f.create_dataset("/scale_p", data=np.ones(3, dtype=np.float32))
        f.create_dataset(
            "/vacuum_intensity", data=np.ones((N, N), dtype=np.float32)
        )
        f.attrs["config_json"] = "{}"
    return path


def test_dataset_returns_L(cnnl_h5):
    """__getitem__ returns an ``L`` key matching the raw /L value in metres."""
    ds = train.BeaconlessH5Dataset(str(cnnl_h5), split="train")
    with h5py.File(str(cnnl_h5), "r") as f:
        L0 = f["/L"][0].astype(np.float32)
    sample = ds[0]
    assert "L" in sample
    assert sample["L"].dtype == torch.float32
    assert sample["L"].shape == ()
    assert float(sample["L"]) == pytest.approx(float(L0))


def test_cnnl_forward_with_normalized_length():
    """CNNL forward with the exact normalized length tensor used by the
    training loop produces the expected (B, n_modes) output shape."""
    torch.manual_seed(0)
    model = CNNL(n_modes=78, channels=(8, 16, 32), pool_size=18,
                 mlp_width=64, mlp_depth=4, dropout=0.0)
    images = torch.randn(2, 3, 32, 32)
    raw_L = torch.tensor([1000.0, 2600.0], dtype=torch.float32)
    length_norm = raw_L / 1000.0
    with torch.inference_mode():
        out = model(images, length_norm)
    assert out.shape == (2, 78)
    assert torch.isfinite(out).all()


def test_is_cnnl_helper(cnnl_h5):
    """_is_cnnl distinguishes CNNL from CNN1 and unwraps a DDP-style wrapper."""
    from models.cnn import CNN1

    cnnl = CNNL(n_modes=78, channels=(8, 16, 32), pool_size=18,
                mlp_width=64, mlp_depth=4, dropout=0.0)
    cnn1 = CNN1(n_modes=78, channels=(8, 16, 32), pool_size=18,
                mlp_width=64, mlp_depth=4, dropout=0.0)
    assert train._is_cnnl(cnnl) is True
    assert train._is_cnnl(cnn1) is False

    class _DDPLike:
        def __init__(self, module):
            self.module = module

    assert train._is_cnnl(_DDPLike(cnnl)) is True
    assert train._is_cnnl(_DDPLike(cnn1)) is False
