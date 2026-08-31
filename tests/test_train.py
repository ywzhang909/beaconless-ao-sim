"""Tests for train.py (TDD: written first, RED, then implemented GREEN).

Reproduces the training protocol of DiComo et al., Opt. Express 33(15):31010
(2025) Sec 2.6 (CNN1 config) at demo scale. All tests are CPU-only, fast, and
hermetic (no network, no real wandb runs, no real simulation).
"""

from __future__ import annotations

import copy
import os
import shutil
import subprocess
import sys
import types

import h5py
import numpy as np
import pytest
import torch
import yaml

import train

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _cpu_and_no_wandb(monkeypatch):
    """CPU-only + no network: force wandb disabled and hide CUDA from torch.

    The training loop is deterministic on CPU (torch.manual_seed); GPU runs are
    subject to cuDNN/atomic-op non-determinism, so the tests pin the CPU.
    """
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")


@pytest.fixture
def tiny_h5(tmp_path):
    """Craft a tiny HDF5 with the PINNED schema from data/generate_h5.py.

    - /images (64, 3, 32, 32) uint16, values in [0, 2047] (12-bit quantized)
    - /labels (64, 78) float32 raw radians
    - /fom_track, /fom_z78 (64,) float32
    - /seeds (64,) int64
    - /train_idx, /test_idx, /eval_idx int64
    - /mu (78,) zeros, /sigma (78,) ones  (normalization is identity)
    - /scale_p (3,) float32, /vacuum_intensity (32, 32) float32
    - attr config_json
    """
    path = tmp_path / "tiny.h5"
    n_total, n_modes, N = 64, 78, 32
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        f.create_dataset(
            "/images",
            data=rng.integers(0, 2048, size=(n_total, 3, N, N)).astype(np.uint16),
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


@pytest.fixture
def tiny_cfg(tmp_path, tiny_h5):
    """Full config.yaml with a tiny model / tiny training schedule monkeypatched in."""
    with open(os.path.join(REPO_ROOT, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    cfg["data"]["h5_path"] = str(tiny_h5)
    cfg["model"]["channels"] = [8, 16, 32]
    cfg["model"]["mlp_width"] = 64
    cfg["train"]["n_steps"] = 10
    cfg["train"]["batch_size"] = 8
    cfg["train"]["amp"] = False
    cfg["train"]["mixed_precision"] = False
    cfg["train"]["num_workers"] = 0
    cfg["train"]["log_every"] = 10
    cfg["train"]["sim_eval_every"] = 10**9  # never during the 10-step run
    cfg["train"]["ckpt_dir"] = str(tmp_path / "ckpt")
    return cfg


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
def test_dataset_normalization_matches_formula(tiny_h5):
    """target == (labels - mu) / sigma (Eq 14 normalization), raw labels kept."""
    ds = train.BeaconlessH5Dataset(str(tiny_h5), split="train")
    with h5py.File(str(tiny_h5), "r") as f:
        labels0 = f["/labels"][0].astype(np.float32)
        mu = f["/mu"][:].astype(np.float32)
        sigma = f["/sigma"][:].astype(np.float32)
    sample = ds[0]
    expected = (labels0 - mu) / sigma
    assert sample["target"].shape == (78,)
    assert sample["target"].dtype == torch.float32
    assert np.allclose(sample["target"].numpy(), expected)
    # raw labels preserved for sim eval
    assert np.allclose(sample["labels_raw"].numpy(), labels0)
    # seed preserved
    assert sample["seed"].item() == 0


def test_dataset_images_in_unit_range(tiny_h5):
    """images = uint16 / 2047.0 -> (3, N, N) float32 in [0, 1]."""
    ds = train.BeaconlessH5Dataset(str(tiny_h5), split="train")
    sample = ds[0]
    assert sample["images"].shape == (3, 32, 32)
    assert sample["images"].dtype == torch.float32
    assert float(sample["images"].min()) >= 0.0
    assert float(sample["images"].max()) <= 1.0


def test_dataset_subset_is_train_idx(tiny_h5):
    """Dataset length == len(/train_idx); __getitem__ indexes through it."""
    ds = train.BeaconlessH5Dataset(str(tiny_h5), split="train")
    with h5py.File(str(tiny_h5), "r") as f:
        assert len(ds) == len(f["/train_idx"])
    # eval split uses /eval_idx
    ds_eval = train.BeaconlessH5Dataset(str(tiny_h5), split="eval")
    with h5py.File(str(tiny_h5), "r") as f:
        assert len(ds_eval) == len(f["/eval_idx"])


def test_dataloader_batch_shapes(tiny_h5):
    """DataLoader collates the dict into batched tensors."""
    ds = train.BeaconlessH5Dataset(str(tiny_h5), split="train")
    loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False)
    batch = next(iter(loader))
    assert batch["images"].shape == (8, 3, 32, 32)
    assert batch["target"].shape == (8, 78)
    assert batch["seed"].shape == (8,)
    assert batch["labels_raw"].shape == (8, 78)
    assert float(batch["images"].min()) >= 0.0
    assert float(batch["images"].max()) <= 1.0


# --------------------------------------------------------------------------- #
# attach_run
# --------------------------------------------------------------------------- #
def test_attach_run_sets_paths(tmp_path, tiny_cfg):
    """attach_run sets ckpt_dir / h5_path / device / rank / amp on the cfg."""
    cfg = copy.deepcopy(tiny_cfg)
    cfg["train"]["ckpt_dir"] = str(tmp_path / "ckpt")
    out = train.attach_run(cfg, ckpt_dir=str(tmp_path / "ckpt"))
    assert out is cfg  # mutates and returns the same dict
    run = cfg["run"]
    assert run["ckpt_dir"] == os.path.abspath(str(tmp_path / "ckpt"))
    assert run["h5_path"] == os.path.abspath(str(tiny_cfg["data"]["h5_path"]))
    assert run["rank"] == 0
    assert run["world_size"] == 1
    assert run["is_distributed"] is False
    assert run["amp"] is False  # amp=False in tiny cfg
    assert run["device"] in ("cuda", "cpu")


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #
def test_loss_decreases_over_steps(tmp_path, tiny_cfg):
    """10 training steps on random data must reduce the MSE loss."""
    cfg = copy.deepcopy(tiny_cfg)
    cfg["train"]["ckpt_dir"] = str(tmp_path / "ckpt")
    result = train.train(cfg)
    assert result["step"] == 10
    assert len(result["losses"]) == 10
    assert result["losses"][-1] < result["losses"][0]
    assert np.isfinite(result["losses"]).all()


def test_checkpoint_file_with_expected_keys(tmp_path, tiny_cfg):
    """last.pt appears with {model_state, optimizer_state, scaler_state, step, cfg}."""
    cfg = copy.deepcopy(tiny_cfg)
    cfg["train"]["ckpt_dir"] = str(tmp_path / "ckpt")
    train.train(cfg)
    ckpt_path = tmp_path / "ckpt" / "last.pt"
    assert ckpt_path.exists()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    assert set(ckpt.keys()) == {
        "model_state",
        "optimizer_state",
        "scaler_state",
        "step",
        "cfg",
    }
    assert ckpt["step"] == 10
    # best.pt fallback (lowest running train loss, no sim evals yet)
    assert (tmp_path / "ckpt" / "best.pt").exists()


def test_deterministic_rerun_identical_losses(tmp_path, tiny_cfg):
    """Same seed -> identical loss trajectory across two independent runs."""
    cfg1 = copy.deepcopy(tiny_cfg)
    cfg1["train"]["ckpt_dir"] = str(tmp_path / "ckpt1")
    res1 = train.train(cfg1)
    cfg2 = copy.deepcopy(tiny_cfg)
    cfg2["train"]["ckpt_dir"] = str(tmp_path / "ckpt2")
    res2 = train.train(cfg2)
    assert res1["losses"] == pytest.approx(res2["losses"], abs=1e-12)


def test_wandb_disabled_no_crash(tmp_path, tiny_cfg):
    """WANDB_MODE=disabled -> train runs to completion with no network."""
    os.environ["WANDB_MODE"] = "disabled"
    cfg = copy.deepcopy(tiny_cfg)
    cfg["train"]["ckpt_dir"] = str(tmp_path / "ckpt")
    result = train.train(cfg)
    assert result["step"] == 10
    assert result["final_loss"] is not None


# --------------------------------------------------------------------------- #
# Sim-eval path
# --------------------------------------------------------------------------- #
def test_sim_eval_warning_when_module_missing(monkeypatch, tiny_cfg):
    """data.simulate unavailable -> evaluate_sim_fom returns None (warning path)."""
    # Force the lazy import to fail regardless of whether data/simulate.py exists.
    monkeypatch.setitem(sys.modules, "data.simulate", None)
    model = train.build_model(tiny_cfg)
    result = train.evaluate_sim_fom(
        model, tiny_cfg, device=torch.device("cpu"), use_pool=False
    )
    assert result is None


def test_sim_eval_invokes_stub(monkeypatch, tiny_cfg, tiny_h5):
    """Stubbed simulate_sample_fom is invoked sim_eval_n times and metrics logged."""
    pytest.importorskip("data.simulate")
    calls: list = []

    def stub(seed, cfg, coeffs, *, shared=None):
        calls.append((int(seed), np.asarray(coeffs, dtype=np.float64).copy()))
        return 0.5

    fake = types.ModuleType("data.simulate")
    fake.simulate_sample_fom = stub
    monkeypatch.setitem(sys.modules, "data.simulate", fake)

    cfg = copy.deepcopy(tiny_cfg)
    cfg["train"]["sim_eval_n"] = 4
    model = train.build_model(cfg)
    result = train.evaluate_sim_fom(
        model, cfg, device=torch.device("cpu"), use_pool=False
    )
    assert result is not None
    assert len(calls) == 4
    assert result["sim/n_eval"] == 4
    assert result["sim/median_fom_ml"] == pytest.approx(0.5)
    assert result["sim/median_fom_track"] > 0.0
    assert "sim/gain" in result
    assert "sim/eta" in result
    # denormalized coefficients: c_pred = y_pred * sigma + mu (sigma=1, mu=0)
    for _, coeffs in calls:
        assert coeffs.shape == (78,)


# --------------------------------------------------------------------------- #
# DDP smoke (optional)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    shutil.which("torchrun") is None, reason="torchrun not available"
)
def test_ddp_smoke(tmp_path, tiny_h5):
    """torchrun --nproc_per_node=2 train.py completes and writes last.pt."""
    with open(os.path.join(REPO_ROOT, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    cfg["data"]["h5_path"] = str(tiny_h5)
    cfg["model"]["channels"] = [8, 16, 32]
    cfg["model"]["mlp_width"] = 64
    cfg["train"]["n_steps"] = 4
    cfg["train"]["batch_size"] = 8
    cfg["train"]["amp"] = False
    cfg["train"]["num_workers"] = 0
    cfg["train"]["log_every"] = 2
    cfg["train"]["sim_eval_every"] = 10**9
    cfg["train"]["ckpt_dir"] = str(tmp_path / "ckpt_ddp")
    cfg_path = tmp_path / "ddp_cfg.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)

    env = dict(os.environ)
    env["WANDB_MODE"] = "disabled"
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        "2",
        "--standalone",
        "train.py",
        "--config",
        str(cfg_path),
        "--no-wandb",
    ]
    subprocess.run(
        cmd, cwd=REPO_ROOT, env=env, timeout=600, check=True
    )
    assert (tmp_path / "ckpt_ddp" / "last.pt").exists()