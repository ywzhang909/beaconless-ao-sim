"""Evaluation protocol (Sec 2.7) and figures (Figs 5, 6) of DiComo et al.

Reproduces the evaluation protocol of G. P. DiComo et al., "Beaconless adaptive
optics using a convolutional neural network for wavefront sensing," Opt. Express
33(15):31010 (2025):

- Sec 2.7 evaluation protocol: load a trained CNN1 checkpoint, run batch
  inference on the held-out eval split, denormalize the mode coefficients
  (inverse of Eq 14), evaluate the predicted phase in the physics simulation to
  obtain ``FOM_ML``, and report the aggregate metrics.
- Eq 15 (gain): ``g = FOM_ML / FOM_track`` -- FOM gain factor relative to the
  tracking-only solution.
- Eq 16 (eta): ``eta = (FOM_ML - FOM_track) / (FOM_Z78 - FOM_track)`` -- CNN
  effectiveness, the fraction of the possible FOM gain realized.
- Eq 17 (Rj): per-Noll-mode Pearson correlation between the ML prediction and
  the test-set amplitude.
- Fig 5: per-mode Pearson ``Rj`` bar chart (modes 1..78).
- Fig 6: FOM scatter, ``x = FOM_track`` vs ``y = FOM_noAO / FOM_Z78 / FOM_ML``.

CLI::

    uv run python evaluate.py --config config.yaml --ckpt checkpoints/best.pt \\
        [--no-wandb] [--limit N]

Outputs (in ``cfg.eval.out_dir``, default ``results/``):

- ``results.json`` -- full metric report (schema documented in :func:`build_results`).
- ``fig_Rj_per_mode.png``  -- Fig 5 style per-mode Pearson bar chart.
- ``fig_pred_vs_true.png`` -- prediction-vs-truth scatter for representative modes.
- ``fig_FOM_scatter.png``  -- Fig 6 style FOM scatter with gain/eta annotation.
- ``fig_samples.png``      -- montage of 3 eval samples (measurement planes +
  object-plane images when the simulation module is available).

``data.simulate`` is imported lazily (never at module top). If it is missing,
``FOM_ML`` is ``None``, the ML fields of ``results.json`` are ``null``, and the
figures exclude the ML series -- the evaluation still completes.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
from typing import Any, Optional

import h5py
import numpy as np
import torch
import yaml
from tqdm import tqdm

import matplotlib

matplotlib.use("Agg")

from models.cnn import CNN1, CNN1Freq, CNN1Star, CNNL  # noqa: E402
from utils.metrics import eta, gain, mode_pearson  # noqa: E402
from utils.wandb_utils import (  # noqa: E402
    finish_wandb,
    init_wandb,
    log_figure,
    log_metrics,
)

# Multiprocessing worker state (set once per worker process by _init_worker).
_WORKER_CFG: dict = {}


# --------------------------------------------------------------------------- #
# Config plumbing
# --------------------------------------------------------------------------- #
def _derive_bucket_mask_px(cfg: dict) -> float:
    """Derive the FOM bucket diameter in pixels from the config geometry.

    ``D_bucket = diameter_frac * L * lambda / D_telescope`` (Eq 6 geometry,
    config.yaml ``bucket.diameter_frac``), converted to pixels with the grid
    scale ``box_size / N``.
    """
    phys = cfg["physical"]
    bucket = cfg["bucket"]
    # Coerce to float: PyYAML parses "800e-9" (no decimal point) as a string.
    d_bucket = (
        float(bucket["diameter_frac"])
        * float(phys["L"])
        * float(phys["wavelength"])
        / float(phys["Dscope"])
    )
    px_scale = float(phys["box_size"]) / float(phys["N"])
    return float(d_bucket / px_scale)


def attach_eval(cfg: dict, ckpt_path: str) -> dict:
    """Return a copy of ``cfg`` with evaluation fields attached.

    Adds ``cfg.eval.out_dir`` (default ``"results"``), ``cfg.eval.ckpt_path``
    and ``cfg.eval.bucket_mask_px`` (derived from the bucket geometry when the
    config leaves it ``null``).
    """
    cfg = dict(cfg)
    cfg["eval"] = dict(cfg.get("eval", {}))
    cfg["eval"]["out_dir"] = cfg["eval"].get("out_dir", "results")
    cfg["eval"]["ckpt_path"] = ckpt_path
    if cfg["eval"].get("bucket_mask_px") is None:
        cfg["eval"]["bucket_mask_px"] = _derive_bucket_mask_px(cfg)
    return cfg


# --------------------------------------------------------------------------- #
# Model + checkpoint
# --------------------------------------------------------------------------- #
def build_model(cfg: dict) -> torch.nn.Module:
    """Build the CNN exactly as train.py does (models.cnn, ``cfg.model``).

    ``CNN1`` for the fixed-propagation-length network; ``CNNL`` when
    ``cfg.model.length_head`` is true.
    """
    m = cfg["model"]
    kwargs: dict[str, Any] = {
        "n_modes": m["n_modes"],
        "mlp_width": m.get("mlp_width", 512),
        "mlp_depth": m.get("mlp_depth", 4),
        "dropout": m.get("dropout", 0.0),
    }
    if "channels" in m:
        kwargs["channels"] = tuple(m["channels"])
        kwargs.setdefault("pool_size", m.get("pool_size", 18))
    if m.get("length_head", False):
        return CNNL(**kwargs)
    if m.get("name") == "CNN1Freq":
        freq_kwargs = {
            "freq_pool": m.get("freq_pool", 8),
            "freq_refine_ch": m.get("freq_refine_ch", 16),
        }
        return CNN1Freq(**kwargs, **freq_kwargs)
    if m.get("name") == "CNN1Star":
        star_kwargs = {
            "n_modes": m["n_modes"],
            "mlp_width": m.get("mlp_width", 512),
            "mlp_depth": m.get("mlp_depth", 4),
            "dropout": m.get("dropout", 0.0),
            "base_dim": m.get("base_dim", 32),
            "depths": tuple(int(d) for d in m.get("depths", (1, 1, 2))),
            "mlp_ratio": m.get("mlp_ratio", 4),
            "use_se": m.get("use_se", False),
            "se_reduction": m.get("se_reduction", 16),
            "pool_size": m.get("pool_size", 12),
            "kernel": m.get("kernel", 3),
        }
        return CNN1Star(**star_kwargs)
    return CNN1(**kwargs)


def load_checkpoint(ckpt_path: str, model: torch.nn.Module) -> dict:
    """Load a train.py checkpoint into ``model`` and return the raw dict.

    The train.py checkpoint contract (tests/test_train.py) is a dict with keys
    ``model_state`` / ``optimizer_state`` / ``step``.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    return ckpt


# --------------------------------------------------------------------------- #
# Data loading + inference
# --------------------------------------------------------------------------- #
def load_h5(cfg: dict) -> dict:
    """Load the pinned h5 schema produced by data/generate_h5.py.

    Datasets: ``/images`` (N_total,3,N,N) uint16, ``/labels`` (N_total,78)
    float32 RAW radians, ``/fom_noao`` ``/fom_track`` ``/fom_beacon``
    ``/fom_z78`` (N_total,) float32, ``/seeds`` (N_total,) int64,
    ``/train_idx`` ``/test_idx`` ``/eval_idx``, ``/mu`` ``/sigma`` (78,)
    float32 (TRAIN-split), ``/scale_p`` (3,), ``/vacuum_intensity`` (N,N).
    """
    path = cfg["data"]["h5_path"]
    with h5py.File(path, "r") as f:
        return {
            "images": f["images"][:],
            "labels": f["labels"][:],
            "fom_noao": f["fom_noao"][:],
            "fom_track": f["fom_track"][:],
            "fom_beacon": f["fom_beacon"][:],
            "fom_z78": f["fom_z78"][:],
            "seeds": f["seeds"][:],
            "train_idx": f["train_idx"][:],
            "test_idx": f["test_idx"][:],
            "eval_idx": f["eval_idx"][:],
            "mu": f["mu"][:],
            "sigma": f["sigma"][:],
            "scale_p": f["scale_p"][:],
            "vacuum_intensity": f["vacuum_intensity"][:],
        }


def predict(
    model: torch.nn.Module,
    images: np.ndarray,
    eval_idx: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    """Batch inference on the eval subset, returning denormalized coefficients.

    Images are normalized as ``float32 / 2047`` (12-bit camera scale) and fed
    through the model in ``no_grad`` mode. The normalized prediction is
    denormalized with the TRAIN-split statistics (inverse of Eq 14):

    ``c_pred = y_pred * sigma + mu``.
    """
    model.eval()
    n_eval = len(eval_idx)
    n_modes = int(model.n_modes)
    c_pred = np.zeros((n_eval, n_modes), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, n_eval, batch_size):
            idx = eval_idx[i : i + batch_size]
            batch = images[idx].astype(np.float32) / 2047.0
            batch_t = torch.from_numpy(batch).to(device)
            out = model(batch_t).cpu().numpy()
            c_pred[i : i + batch_size] = out
    c_pred = c_pred * sigma + mu
    return c_pred


# --------------------------------------------------------------------------- #
# FOM_ML via the physics simulation (lazy import, multiprocessing)
# --------------------------------------------------------------------------- #
def _init_worker(cfg: dict) -> None:
    """Pool initializer: stash the config for the worker processes."""
    global _WORKER_CFG
    _WORKER_CFG = cfg


def _fom_worker(args: tuple) -> float:
    """Compute ``FOM_ML`` for one ``(seed, coeffs)`` pair in a worker process.

    ``data.simulate`` is imported lazily inside the worker so the module is
    only required when FOM_ML is actually computed. The pinned API is
    ``simulate_sample_fom(seed=..., cfg=..., coeffs=...)``.
    """
    seed, coeffs = args
    from data.simulate import simulate_sample_fom

    return simulate_sample_fom(seed=int(seed), cfg=_WORKER_CFG, coeffs=coeffs)


def compute_fom_ml(
    seeds: np.ndarray, c_pred: np.ndarray, cfg: dict
) -> Optional[np.ndarray]:
    """Compute ``FOM_ML`` for every eval sample via data.simulate.

    Imports ``data.simulate`` lazily. If the module is missing, prints a clear
    error and returns ``None`` (the caller then writes ``null`` FOM_ML fields
    and the figures exclude the ML series).

    Uses a ``multiprocessing.Pool`` with ``min(cfg.data.workers, 16)`` workers
    (serial loop when ``workers <= 1`` so monkeypatched stubs keep working).
    """
    try:
        from data.simulate import simulate_sample_fom  # noqa: F401
    except ImportError:
        print("data.simulate not built — run data/generate_h5.py first")
        return None

    workers = min(int(cfg["data"].get("workers", 1)), 16)
    items = [(int(s), c) for s, c in zip(seeds, c_pred)]

    if workers <= 1:
        foms = [
            simulate_sample_fom(seed=seed, cfg=cfg, coeffs=coeffs)
            for seed, coeffs in tqdm(items, desc="FOM_ML (serial)")
        ]
        return np.asarray(foms, dtype=float)

    with multiprocessing.Pool(
        workers, initializer=_init_worker, initargs=(cfg,)
    ) as pool:
        foms = list(
            tqdm(pool.imap(_fom_worker, items), total=len(items), desc="FOM_ML")
        )
    return np.asarray(foms, dtype=float)


# --------------------------------------------------------------------------- #
# Metrics (Eqs 15-17)
# --------------------------------------------------------------------------- #
def _get_fom(foms: dict, name: str) -> np.ndarray:
    """Fetch a FOM array by short name, accepting ``fom_``-prefixed keys too."""
    if name in foms:
        return np.asarray(foms[name], dtype=float)
    return np.asarray(foms["fom_" + name], dtype=float)


def compute_metrics(
    c_pred: np.ndarray,
    c_true: np.ndarray,
    foms: dict,
    fom_ml: Optional[np.ndarray],
) -> dict:
    """Compute the Sec 2.7 aggregate metrics.

    - ``Rj`` (Eq 17): per-mode Pearson correlation over the eval set.
    - median/mean FOM per system (noao, track, beacon, z78, ml).
    - ``gain`` (Eq 15): ``median(FOM_ML) / median(FOM_track)``.
    - ``eta`` (Eq 16): ``(median(FOM_ML) - median(FOM_track)) /
      (median(FOM_Z78) - median(FOM_track))``.

    When ``fom_ml`` is ``None`` the ML fields are ``None``.
    """
    Rj = mode_pearson(c_pred, c_true)
    median_fom = {
        name: float(np.median(_get_fom(foms, name)))
        for name in ("noao", "track", "beacon", "z78")
    }
    mean_fom = {
        name: float(np.mean(_get_fom(foms, name)))
        for name in ("noao", "track", "beacon", "z78")
    }
    if fom_ml is not None:
        median_fom["ml"] = float(np.median(fom_ml))
        mean_fom["ml"] = float(np.mean(fom_ml))
        g = gain(median_fom["ml"], median_fom["track"])
        e = eta(median_fom["ml"], median_fom["track"], median_fom["z78"])
    else:
        median_fom["ml"] = None
        mean_fom["ml"] = None
        g = None
        e = None
    return {
        "median_fom": median_fom,
        "mean_fom": mean_fom,
        "gain": g,
        "eta": e,
        "Rj_mean": float(Rj.mean()),
        "Rj": Rj,
    }


def build_results(
    cfg: dict,
    eval_idx: np.ndarray,
    data: dict,
    foms: dict,
    fom_ml: Optional[np.ndarray],
    metrics: dict,
) -> dict:
    """Assemble the ``results.json`` dict.

    Schema::

        {"cfg_json": "...", "n_eval": N, "median_fom": {"noao":..,"track":..,
         "beacon":..,"z78":..,"ml":..}, "mean_fom": {...}, "gain": x, "eta": x,
         "Rj_mean": x, "Rj": [78 floats],
         "per_sample": [{"idx":i,"seed":s,"fom_track":..,"fom_ml":..,"fom_z78":..}, ...]}
    """
    per_sample = []
    for i, idx in enumerate(eval_idx):
        per_sample.append(
            {
                "idx": int(idx),
                "seed": int(data["seeds"][idx]),
                "fom_track": float(_get_fom(foms, "track")[i]),
                "fom_ml": float(fom_ml[i]) if fom_ml is not None else None,
                "fom_z78": float(_get_fom(foms, "z78")[i]),
            }
        )
    return {
        "cfg_json": json.dumps(cfg, sort_keys=True),
        "n_eval": int(len(eval_idx)),
        "median_fom": metrics["median_fom"],
        "mean_fom": metrics["mean_fom"],
        "gain": metrics["gain"],
        "eta": metrics["eta"],
        "Rj_mean": metrics["Rj_mean"],
        "Rj": [float(x) for x in metrics["Rj"]],
        "per_sample": per_sample,
    }


# --------------------------------------------------------------------------- #
# Figures (Figs 5, 6 + prediction scatter + samples montage)
# --------------------------------------------------------------------------- #
def plot_fig5(Rj: np.ndarray, out_dir: str) -> str:
    """Fig 5 style: per-mode Pearson ``Rj`` (Eq 17) bar chart, modes 1..78.

    A dashed line marks ``R = 0``.
    """
    import matplotlib.pyplot as plt

    modes = np.arange(1, len(Rj) + 1)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(modes, Rj, color="steelblue", edgecolor="none", width=0.9)
    ax.axhline(0.0, color="k", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Noll mode index")
    ax.set_ylabel("Pearson R$_j$ (Eq 17)")
    ax.set_title("Per-mode prediction correlation (Fig 5)")
    ax.set_xlim(0, len(Rj) + 1)
    ax.set_ylim(-1.05, 1.05)
    fig.tight_layout()
    path = os.path.join(out_dir, "fig_Rj_per_mode.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_pred_vs_true(
    c_pred: np.ndarray,
    c_true: np.ndarray,
    out_dir: str,
    modes: tuple = (1, 2, 3, 4, 5, 9, 10, 20, 40),
) -> str:
    """Prediction-vs-truth scatter for representative Noll modes (3x3 grid).

    Each subplot shows the 45-degree line and the per-mode Pearson ``R``
    (Eq 17). ``modes`` are 0-based column indices.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    for ax, j in zip(axes.ravel(), modes):
        p = c_pred[:, j]
        t = c_true[:, j]
        ax.scatter(t, p, s=8, alpha=0.5, color="steelblue")
        lo = float(min(t.min(), p.min()))
        hi = float(max(t.max(), p.max()))
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8)
        r = float(np.corrcoef(t, p)[0, 1])
        ax.set_title(f"Noll {j + 1}: R = {r:.3f}")
        ax.set_xlabel("true (rad)")
        ax.set_ylabel("pred (rad)")
    fig.suptitle("Prediction vs truth per Noll mode (Eq 17)", y=1.0)
    fig.tight_layout()
    path = os.path.join(out_dir, "fig_pred_vs_true.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_fig6(
    foms: dict,
    fom_ml: Optional[np.ndarray],
    metrics: dict,
    out_dir: str,
) -> str:
    """Fig 6 style: FOM scatter, ``x = FOM_track`` vs the other systems.

    One color per system: no-AO (gray), Z78 upper bound (green), ML (red).
    A dashed ``y = x`` line marks the tracking-only baseline. The title
    annotates the median gain ``g`` (Eq 15) and effectiveness ``eta`` (Eq 16).
    When ``fom_ml`` is ``None`` the ML series is excluded.
    """
    import matplotlib.pyplot as plt

    x = _get_fom(foms, "track")
    fom_noao = _get_fom(foms, "noao")
    fom_z78 = _get_fom(foms, "z78")
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(x, fom_noao, s=10, alpha=0.5, color="gray", label="no AO")
    ax.scatter(x, fom_z78, s=10, alpha=0.5, color="green", label="Z78 (upper bound)")
    if fom_ml is not None:
        ax.scatter(x, fom_ml, s=10, alpha=0.5, color="red", label="ML (CNN)")
    lo = float(min(x.min(), fom_noao.min(), fom_z78.min()))
    hi = float(max(x.max(), fom_noao.max(), fom_z78.max()))
    if fom_ml is not None:
        lo = min(lo, float(fom_ml.min()))
        hi = max(hi, float(fom_ml.max()))
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, label="y = x (tracking)")
    ax.set_xlabel("FOM$_{track}$")
    ax.set_ylabel("FOM")
    title = "FOM scatter (Fig 6)"
    if metrics["gain"] is not None:
        title += f"\ngain g = {metrics['gain']:.3f} (Eq 15),  eta = {metrics['eta']:.3f} (Eq 16)"
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    path = os.path.join(out_dir, "fig_FOM_scatter.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _try_get_sim_images(
    seed: int, cfg: dict, coeffs: np.ndarray
) -> Optional[dict]:
    """Best-effort retrieval of object-plane images from data.simulate.

    Uses the public ``simulate_sample(seed, cfg, correction_coeffs=coeffs)``
    API to obtain ``I_obj_track`` and ``I_vac``. ``I_obj_ml`` (the ML-corrected
    object-plane intensity) is not exposed by ``SimSample``, so it is computed
    by re-propagating the ML beam phase through the shared state; on any failure
    it falls back to ``I_obj_track``. Returns ``None`` when the simulation
    module is unavailable.
    """
    try:
        from data.simulate import simulate_sample
    except ImportError:
        return None
    try:
        sample = simulate_sample(
            seed=int(seed), cfg=cfg, correction_coeffs=coeffs
        )
    except Exception:
        return None
    out = {
        "I_obj_track": np.asarray(sample.I_obj_track),
        "I_vac": np.asarray(sample.I_vac),
    }
    try:
        from data.simulate import (
            _beacon_phase_conj,
            _get_shared,
            _make_screens,
            _tracking,
        )

        shared = _get_shared(cfg)
        screens = _make_screens(int(seed), cfg, shared)
        phi_conj, _ = _beacon_phase_conj(int(seed), cfg, shared, screens)
        phi_track, _ = _tracking(shared, phi_conj)
        phi_ml = (
            shared.phi_focus
            + phi_track
            + shared.zern.zernike_to_phase(coeffs)
        )
        E_obj = shared.prop.split_step(
            (shared.E0 * np.exp(1j * phi_ml)).astype(np.complex64),
            screens,
            shared.dz,
        )
        out["I_obj_ml"] = (np.abs(E_obj) ** 2).astype(np.float32)
    except Exception:
        out["I_obj_ml"] = np.asarray(sample.I_obj_track)
    return out


def plot_samples(
    images_eval: np.ndarray,
    sim_images: Optional[list],
    out_dir: str,
    n_samples: int = 3,
) -> str:
    """Montage of eval samples: rows = samples, cols = 6.

    Columns: ``plane0, plane1, plane2`` measurement-plane images (log10+1
    scale) followed by ``I_obj_track``, ``I_obj_ml`` (ML-corrected) and
    ``I_vac``. When ``sim_images`` is ``None`` (simulation module missing) the
    object-plane panels render as gray "sim N/A" placeholders.
    """
    import matplotlib.pyplot as plt

    K = images_eval.shape[0]
    if K > n_samples:
        idx = np.linspace(0, K - 1, n_samples).astype(int)
        images_eval = images_eval[idx]
        if sim_images is not None:
            sim_images = [sim_images[i] for i in idx]

    n_rows = images_eval.shape[0]
    fig, axes = plt.subplots(n_rows, 6, figsize=(18, 3 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]

    for i in range(n_rows):
        for p in range(3):
            ax = axes[i, p]
            img = images_eval[i, p]
            im = ax.imshow(np.log10(img.astype(np.float64) + 1.0), cmap="inferno")
            ax.set_title(f"plane{p} (log)")
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        for c, key in enumerate(("I_obj_track", "I_obj_ml", "I_vac")):
            ax = axes[i, 3 + c]
            ax.set_xticks([])
            ax.set_yticks([])
            if sim_images is not None and sim_images[i] is not None and key in sim_images[i]:
                img = np.asarray(sim_images[i][key], dtype=np.float64)
                im = ax.imshow(img, cmap="inferno")
                ax.set_title(key)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            else:
                ax.set_facecolor("0.85")
                ax.text(
                    0.5,
                    0.5,
                    "sim\nN/A",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=10,
                )
                ax.set_title(key)

    fig.suptitle("Eval samples: measurement planes (log) | object-plane images", y=1.0)
    fig.tight_layout()
    path = os.path.join(out_dir, "fig_samples.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# WandB
# --------------------------------------------------------------------------- #
def log_to_wandb(
    cfg: dict,
    metrics: dict,
    figs: dict,
    results_path: str,
    no_wandb: bool,
) -> None:
    """Log metrics, figures and the results artifact to WandB (never raises).

    Uses utils.wandb_utils (init_wandb / log_metrics / log_figure /
    finish_wandb). ``--no-wandb`` disables logging entirely; ``WANDB_MODE``
    environment variables (e.g. ``disabled``) are honoured by wandb itself.
    """
    if no_wandb:
        return
    wcfg = cfg.get("wandb", {})
    run = init_wandb(
        cfg,
        run_name=wcfg.get("run_name") or "evaluation",
        project=wcfg.get("project", "beaconless-ao-sim"),
        tags=wcfg.get("tags"),
        job_type="evaluation",
    )
    if run is None:
        return
    log_metrics(
        run,
        {
            "gain": metrics["gain"],
            "eta": metrics["eta"],
            "Rj_mean": metrics["Rj_mean"],
            **{f"median_fom/{k}": v for k, v in metrics["median_fom"].items()},
        },
    )
    for name, fig in figs.items():
        log_figure(run, fig, name)
    try:
        import wandb

        artifact = wandb.Artifact("results", type="results")
        artifact.add_file(results_path)
        run.log_artifact(artifact)
    except Exception:
        pass
    finish_wandb(run)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def print_report(metrics: dict, n_eval: int) -> None:
    """Print the Sec 2.7 aggregate report table to stdout."""
    mf = metrics["median_fom"]
    print("\n=== Evaluation report (Sec 2.7) ===")
    print(f"n_eval            : {n_eval}")
    for k in ("noao", "track", "beacon", "z78", "ml"):
        v = mf.get(k)
        print(f"median FOM {k:<7}: {v if v is not None else 'N/A'}")
    print(f"gain g (Eq 15)    : {metrics['gain'] if metrics['gain'] is not None else 'N/A'}")
    print(f"eta (Eq 16)       : {metrics['eta'] if metrics['eta'] is not None else 'N/A'}")
    print(f"mean Rj (Eq 17)   : {metrics['Rj_mean']:.4f}")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def main(
    cfg: dict,
    ckpt_path: str,
    no_wandb: bool = False,
    limit: Optional[int] = None,
) -> dict:
    """Run the full Sec 2.7 evaluation protocol.

    Loads the checkpoint, runs batch inference on the eval split, computes the
    aggregate metrics (Eqs 15-17), writes ``results.json`` and the four figures
    (Figs 5, 6 + prediction scatter + samples montage), logs to WandB unless
    ``no_wandb``, and prints the report table. Returns the results dict.
    """
    cfg = attach_eval(cfg, ckpt_path)
    out_dir = cfg["eval"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    # Model + checkpoint.
    model = build_model(cfg)
    load_checkpoint(ckpt_path, model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    # Data.
    data = load_h5(cfg)
    eval_idx = np.asarray(data["eval_idx"], dtype=np.int64)
    if limit is not None:
        eval_idx = eval_idx[: int(limit)]

    # Inference + ground truth.
    c_pred = predict(
        model,
        data["images"],
        eval_idx,
        data["mu"],
        data["sigma"],
        device,
        batch_size=int(cfg["eval"].get("batch_size", 32)),
    )
    c_true = np.asarray(data["labels"], dtype=np.float64)[eval_idx]

    # FOMs.
    foms = {
        k: np.asarray(data[k], dtype=np.float64)[eval_idx]
        for k in ("fom_noao", "fom_track", "fom_beacon", "fom_z78")
    }
    fom_ml = compute_fom_ml(data["seeds"][eval_idx], c_pred, cfg)

    # Metrics + results.json.
    metrics = compute_metrics(c_pred, c_true, foms, fom_ml)
    results = build_results(cfg, eval_idx, data, foms, fom_ml, metrics)
    results_path = os.path.join(out_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Figures.
    figs = {
        "fig_Rj_per_mode.png": plot_fig5(metrics["Rj"], out_dir),
        "fig_pred_vs_true.png": plot_pred_vs_true(c_pred, c_true, out_dir),
        "fig_FOM_scatter.png": plot_fig6(foms, fom_ml, metrics, out_dir),
    }
    # Samples montage: 3 representative eval samples.
    n_plot = min(3, len(eval_idx))
    plot_idx = np.linspace(0, len(eval_idx) - 1, n_plot).astype(int)
    sim_images = None
    if fom_ml is not None:
        sim_images = [
            _try_get_sim_images(
                int(data["seeds"][eval_idx[i]]), cfg, c_pred[i]
            )
            for i in plot_idx
        ]
    figs["fig_samples.png"] = plot_samples(
        data["images"][eval_idx[plot_idx]], sim_images, out_dir
    )

    # WandB.
    log_to_wandb(cfg, metrics, figs, results_path, no_wandb)

    # Report.
    print_report(metrics, len(eval_idx))
    return results


def main_cli(argv: Optional[list] = None) -> dict:
    """Command-line entry point.

    ``evaluate.py --config config.yaml --ckpt checkpoints/best.pt
    [--no-wandb] [--limit N]``
    """
    parser = argparse.ArgumentParser(
        description="Evaluate a trained CNN on the Sec 2.7 protocol "
        "(DiComo et al., Opt. Express 33(15):31010, 2025)."
    )
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--ckpt", required=True, help="Path to the train.py checkpoint")
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB logging")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N eval samples")
    args = parser.parse_args(argv)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    return main(cfg, args.ckpt, no_wandb=args.no_wandb, limit=args.limit)


if __name__ == "__main__":
    main_cli()