"""Tests for evaluate.py.

TDD: these tests are written first (RED), then evaluate.py is implemented (GREEN).

The evaluation protocol follows DiComo et al., "Beaconless adaptive optics using
a convolutional neural network for wavefront sensing," Opt. Express 33(15):31010
(2025), Sec 2.7 (evaluation protocol) and Figs 5-6 (per-mode Pearson Rj, FOM
scatter). All tests are CPU-only and fast.
"""

from __future__ import annotations

import builtins
import copy
import json
import os
import sys
import types

import h5py
import numpy as np
import pytest
import torch

from models.cnn import CNN1
from physics.config import SimConfig


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _make_tiny_h5(path, n=32, n_modes=78, N=32):
    """Build a tiny h5 file matching the pinned schema from data/generate_h5.py.

    Schema (pinned): /images (N_total,3,N,N) uint16, /labels (N_total,78)
    float32 RAW radians, /fom_noao /fom_track /fom_beacon /fom_z78 (N_total,)
    float32, /seeds (N_total,) int64, /train_idx /test_idx /eval_idx,
    /mu /sigma (78,) float32 (TRAIN-split), /scale_p (3,),
    /vacuum_intensity (N,N), attr config_json.
    """
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        f.create_dataset(
            "images",
            data=rng.integers(0, 2048, (n, 3, N, N), dtype=np.uint16),
        )
        f.create_dataset(
            "labels",
            data=rng.standard_normal((n, n_modes)).astype(np.float32),
        )
        foms = rng.uniform(0.1, 1.0, n).astype(np.float32)
        f.create_dataset("fom_noao", data=foms)
        f.create_dataset("fom_track", data=foms)
        f.create_dataset("fom_beacon", data=foms)
        f.create_dataset("fom_z78", data=foms)
        f.create_dataset("seeds", data=(np.arange(n) + 1000).astype(np.int64))
        f.create_dataset("train_idx", data=np.arange(0, n // 2, dtype=np.int64))
        f.create_dataset(
            "test_idx", data=np.arange(n // 2, 3 * n // 4, dtype=np.int64)
        )
        f.create_dataset(
            "eval_idx", data=np.arange(3 * n // 4, n, dtype=np.int64)
        )
        f.create_dataset("mu", data=np.zeros(n_modes, dtype=np.float32))
        f.create_dataset("sigma", data=np.ones(n_modes, dtype=np.float32))
        f.create_dataset("scale_p", data=np.ones(3, dtype=np.float32))
        f.create_dataset(
            "vacuum_intensity", data=np.ones((N, N), dtype=np.float32)
        )
        f.attrs["config_json"] = json.dumps({"model": {"n_modes": n_modes}})


def _make_cfg(tmp_path, h5_path):
    """Build a minimal config with a tiny CNN1 variant."""
    return SimConfig.from_dict({
        "physical": {
            "N": 32,
            "box_size": 0.3,
            "L": 1000.0,
            "wavelength": 800e-9,
            "Dscope": 0.3,
        },
        "bucket": {"diameter_frac": 2.5},
        "data": {
            "h5_path": str(h5_path),
            "workers": 1,
            "n_eval": 8,
        },
        "model": {
            "name": "CNN1",
            "n_modes": 78,
            "channels": [8, 16, 32],
            "kernel": 3,
            "stride": 1,
            "padding": 0,
            "mlp_width": 64,
            "mlp_depth": 4,
            "pool_size": 18,
            "length_head": False,
            "dropout": 0.0,
        },
        "eval": {
            "out_dir": str(tmp_path / "results"),
            "bucket_mask_px": None,
            "plot_every": 20,
        },
        "wandb": {
            "project": "beaconless-ao-sim",
            "entity": "ywzhang909",
            "run_name": None,
            "tags": [],
            "notes": "",
        },
    })


def _train_tiny_model(cfg, images, labels, steps=10):
    """Train a tiny CNN1 (channels [8,16,32], mlp_width 64) for a few steps.

    train.py does not exist yet, so the model is built directly from cfg.model
    via models.cnn.CNN1 (the same constructor train.py will use).
    """
    m = cfg.model
    model = CNN1(
        n_modes=m.n_modes,
        channels=tuple(m.channels),
        pool_size=m.pool_size,
        mlp_width=m.mlp_width,
        mlp_depth=m.mlp_depth,
        dropout=m.dropout,
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    x = torch.from_numpy(images.astype(np.float32) / 2047.0)
    y = torch.from_numpy(labels.astype(np.float32))
    for _ in range(steps):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
    return model


def _make_fom_stub(seed_to_fom_track):
    """Stub for data.simulate.simulate_sample_fom.

    ``fom_ml = 0.5 + 0.3 * fom_track(seed)`` so the ML FOM is a deterministic
    function of the tracking FOM (gain/eta are well-defined).
    """

    def _stub(seed, cfg, coeffs):
        return 0.5 + 0.3 * seed_to_fom_track[int(seed)]

    return _stub


def _inject_simulate_stub(monkeypatch, fom_fn):
    """Inject a simulate_sample_fom stub without touching data.simulate if absent.

    If ``data.simulate`` is importable (the parallel agent built it), the real
    module's ``simulate_sample_fom`` is monkeypatched. Otherwise a fake module
    is injected into ``sys.modules`` so evaluate.py's lazy import finds it.
    """
    try:
        import data.simulate  # noqa: F401
    except ImportError:
        fake = types.ModuleType("data.simulate")
        fake.simulate_sample_fom = fom_fn
        monkeypatch.setitem(sys.modules, "data.simulate", fake)
    else:
        monkeypatch.setattr(data.simulate, "simulate_sample_fom", fom_fn)


@pytest.fixture(scope="module")
def tiny_env(tmp_path_factory):
    """Build the tiny h5 + trained ckpt once per module."""
    tmp = tmp_path_factory.mktemp("eval")
    h5_path = tmp / "tiny.h5"
    _make_tiny_h5(h5_path)
    cfg = _make_cfg(tmp, h5_path)
    with h5py.File(h5_path, "r") as f:
        images = f["images"][:]
        labels = f["labels"][:]
    model = _train_tiny_model(cfg, images, labels, steps=10)
    ckpt_path = tmp / "best.pt"
    torch.save(
        {"model_state": model.state_dict(), "optimizer_state": {}, "step": 10},
        ckpt_path,
    )
    return {"tmp": tmp, "h5_path": h5_path, "ckpt_path": ckpt_path, "cfg": cfg}


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_evaluate_smoke(tiny_env, monkeypatch):
    """End-to-end: tiny h5 + tiny ckpt -> results.json + 4 non-empty PNGs."""
    import evaluate

    cfg = tiny_env["cfg"]
    with h5py.File(tiny_env["h5_path"], "r") as f:
        seeds = f["seeds"][:]
        fom_track = f["fom_track"][:]
    seed_to_fom = dict(zip(seeds.tolist(), fom_track.tolist()))
    _inject_simulate_stub(monkeypatch, _make_fom_stub(seed_to_fom))

    results = evaluate.main(cfg, str(tiny_env["ckpt_path"]), no_wandb=True)

    out_dir = tiny_env["tmp"] / "results"
    results_path = out_dir / "results.json"
    assert results_path.exists()
    with open(results_path) as f:
        data = json.load(f)
    for key in (
        "cfg_json",
        "n_eval",
        "median_fom",
        "mean_fom",
        "gain",
        "eta",
        "Rj_mean",
        "Rj",
        "per_sample",
    ):
        assert key in data, f"missing key {key}"
    assert data["gain"] is not None
    assert data["eta"] is not None
    assert len(data["Rj"]) == 78
    assert data["n_eval"] == 8
    assert len(data["per_sample"]) == 8
    for name in (
        "fig_Rj_per_mode.png",
        "fig_pred_vs_true.png",
        "fig_FOM_scatter.png",
        "fig_samples.png",
    ):
        p = out_dir / name
        assert p.exists(), f"missing {name}"
        assert os.path.getsize(p) > 1000, f"{name} too small"


def test_evaluate_without_sim(tiny_env, monkeypatch):
    """Graceful fallback: FOM_ML fields null and plots exclude ML when sim missing."""
    import evaluate

    cfg = copy.deepcopy(tiny_env["cfg"])
    cfg.eval.out_dir = str(tiny_env["tmp"] / "results_nosim")

    real_import = builtins.__import__

    def _blocked(name, *args, **kwargs):
        if name == "data.simulate" or name.startswith("data.simulate."):
            raise ImportError("data.simulate not built")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked)
    monkeypatch.delitem(sys.modules, "data.simulate", raising=False)

    results = evaluate.main(cfg, str(tiny_env["ckpt_path"]), no_wandb=True)
    assert results["median_fom"]["ml"] is None
    assert results["gain"] is None
    assert results["eta"] is None
    out_dir = tiny_env["tmp"] / "results_nosim"
    assert (out_dir / "results.json").exists()
    assert (out_dir / "fig_FOM_scatter.png").exists()
    assert (out_dir / "fig_samples.png").exists()


def test_mode_pearson_known_correlation():
    """Linear y=2x -> R=1 via evaluate.compute_metrics (Eq 17)."""
    import evaluate

    x = np.linspace(-1, 1, 50)
    c_pred = np.stack([x] * 78, axis=1)
    c_true = np.stack([2 * x] * 78, axis=1)
    foms = {
        k: np.linspace(0.1, 1.0, 50)
        for k in ("fom_noao", "fom_track", "fom_beacon", "fom_z78")
    }
    fom_ml = np.linspace(0.2, 1.0, 50)
    metrics = evaluate.compute_metrics(c_pred, c_true, foms, fom_ml)
    assert metrics["Rj"].shape == (78,)
    assert np.allclose(metrics["Rj"], 1.0)


def test_figures_render_minimal(tmp_path):
    """All 4 figure functions render with minimal data (3 samples, 78 modes)."""
    import evaluate

    out = tmp_path / "figs"
    out.mkdir()
    rng = np.random.default_rng(0)
    c_pred = rng.standard_normal((3, 78))
    c_true = rng.standard_normal((3, 78))
    foms = {
        k: rng.uniform(0.1, 1.0, 3)
        for k in ("fom_noao", "fom_track", "fom_beacon", "fom_z78")
    }
    fom_ml = rng.uniform(0.1, 1.0, 3)
    metrics = evaluate.compute_metrics(c_pred, c_true, foms, fom_ml)

    paths = [
        evaluate.plot_fig5(metrics["Rj"], str(out)),
        evaluate.plot_pred_vs_true(c_pred, c_true, str(out)),
        evaluate.plot_fig6(foms, fom_ml, metrics, str(out)),
        evaluate.plot_samples(
            rng.integers(0, 2048, (3, 3, 32, 32), dtype=np.uint16),
            None,
            str(out),
        ),
    ]
    for p in paths:
        assert os.path.getsize(p) > 1000