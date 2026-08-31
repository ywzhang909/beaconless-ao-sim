"""Input-preprocessing smoke test for the trained CNN1.

Runs the trained CNN1 on a fixed set of input-preprocessing methods to probe
how the network responds when its training-time input contract (uint16 / 2047,
3 planes) is varied. Each method is tested independently ("one by one").

Methods
-------
- ``baseline_norm``    : the training-time contract -- all 3 planes, ``uint16/2047``.
- ``raw_uint16``       : all 3 planes, raw 0..2047 integer scale (no /2047).
- ``minmax_sample``    : all 3 planes, per-sample min-max -> [0,1].
- ``minmax_global``    : all 3 planes, global (eval-set) min-max -> [0,1].
- ``zscore``           : all 3 planes, per-plane z-score -> zero mean, unit var.
- ``oneplane_norm``    : single focal plane (idx 1) replicated to 3 ch, ``/2047``.
- ``oneplane_raw``     : single focal plane replicated, raw uint16 scale.

For each method we log to WandB: input stats, CNN-trunk feature-map stats
(reveals saturation / dead features), per-mode Pearson Rj, RMS coefficient
error, the physics-simulation FOM_ML, an input-plane grid, and a summary table.

Usage:
    CUDA_VISIBLE_DEVICES=1 uv run python smoke_test.py --config config.yaml \
        --ckpt checkpoints/best.pt --n-samples 50 --batch-size 8
"""

from __future__ import annotations

import argparse
import json
from typing import Optional

import h5py
import numpy as np
import torch
import torch.nn as nn
from PIL import Image as PILImage
from torchvision.utils import make_grid

import wandb
import yaml

from evaluate import (
    attach_eval,
    build_model,
    compute_fom_ml,
    load_checkpoint,
)
from models.cnn import count_parameters

IMAGE_MAX = 2047.0


def load_raw_h5(path: str) -> dict:
    with h5py.File(path, "r") as f:
        return {
            "images": f["images"][:],
            "labels": f["labels"][:],
            "eval_idx": f["eval_idx"][:],
            "seeds": f["seeds"][:],
            "mu": f["mu"][:],
            "sigma": f["sigma"][:],
        }


def replicate_planes(single: np.ndarray) -> np.ndarray:
    """(B, N, N) -> (B, 3, N, N) by replicating the single plane to 3 channels."""
    return np.stack([single, single, single], axis=1)


def make_inputs(images: np.ndarray, eval_idx: np.ndarray, method: str) -> np.ndarray:
    """Build (B, 3, N, N) float32 inputs for a preprocessing method.

    The 3 planes are the raw uint16 intensities; a per-sample / per-global
    statistic is derived from all 3 planes of the eval set for the min-max and
    z-score variants so the transform is defined over the whole measurement.
    """
    imgs = images[eval_idx].astype(np.float32)  # (B, 3, N, N)
    if method == "baseline_norm":
        return imgs / IMAGE_MAX
    if method == "raw_uint16":
        return imgs
    if method == "minmax_sample":
        lo = imgs.min(axis=(1, 2, 3), keepdims=True)
        hi = imgs.max(axis=(1, 2, 3), keepdims=True)
        return (imgs - lo) / (hi - lo + 1e-8)
    if method == "minmax_global":
        lo = imgs.min()
        hi = imgs.max()
        return (imgs - lo) / (hi - lo + 1e-8)
    if method == "zscore":
        mu = imgs.mean()
        sd = imgs.std()
        return (imgs - mu) / (sd + 1e-8)
    if method == "oneplane_norm":
        return replicate_planes(imgs[:, 1] / IMAGE_MAX)
    if method == "oneplane_raw":
        return replicate_planes(imgs[:, 1])
    raise ValueError(f"unknown method {method}")


def trunk_features(model: nn.Module, x: torch.Tensor) -> np.ndarray:
    """(n_channels,) array of [mean, std, max] over each pooled feature map."""
    with torch.no_grad():
        pooled = model.avgpool(model.features(x))
    stats = []
    for c in range(pooled.shape[1]):
        ch = pooled[:, c].cpu().numpy()
        stats.append([float(ch.mean()), float(ch.std()), float(ch.max())])
    return np.asarray(stats)


def pearson_rj(c_pred: np.ndarray, c_true: np.ndarray) -> np.ndarray:
    """Per-mode Pearson correlation Rj (Eq 17)."""
    rj = np.zeros(c_pred.shape[1], dtype=np.float64)
    for j in range(c_pred.shape[1]):
        p = c_pred[:, j].astype(np.float64)
        t = c_true[:, j].astype(np.float64)
        sp, st = p - p.mean(), t - t.mean()
        denom = np.sqrt((sp**2).sum() * (st**2).sum())
        rj[j] = float((sp * st).sum() / denom) if denom > 0 else 0.0
    return rj


def rms_err(c_pred: np.ndarray, c_true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((c_pred - c_true) ** 2)))


def summarize(inputs: np.ndarray, feats: np.ndarray, c_pred: np.ndarray,
              c_true: np.ndarray, rj: np.ndarray) -> dict:
    in_flat = inputs.reshape(inputs.shape[0], -1)
    return {
        "input_mean": float(in_flat.mean()),
        "input_std": float(in_flat.std()),
        "input_max": float(in_flat.max()),
        "input_min": float(in_flat.min()),
        "feat_mean_abs": float(np.abs(feats[:, 0]).mean()),
        "feat_std_mean": float(feats[:, 1].mean()),
        "feat_max_mean": float(feats[:, 2].mean()),
        "pred_rms": rms_err(c_pred, c_true),
        "rj_mean": float(rj.mean()),
        "rj_std": float(rj.std()),
    }


def plane_grid(inputs: np.ndarray, device: torch.device, method: str) -> wandb.Image:
    plane = torch.from_numpy(inputs[:, 0]).unsqueeze(1).to(device)
    plane_rgb = torch.cat([plane, plane, plane], dim=1)
    grid = make_grid(plane_rgb, nrow=5, normalize=False).cpu().numpy()
    if grid.ndim == 3:
        grid = np.moveaxis(grid, 0, -1)
    grid = np.clip((grid - grid.min()) / max(grid.max() - grid.min(), 1e-8), 0, 1)
    return wandb.Image(PILImage.fromarray((grid * 255).astype(np.uint8), mode="RGB"),
                       caption=method)


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--run-name", default="preprocessing-smoke-test")
    args = parser.parse_args(argv)

    cfg = yaml.safe_load(open(args.config))
    cfg = attach_eval(cfg, args.ckpt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg)
    load_checkpoint(args.ckpt, model)
    model.to(device).eval()

    data = load_raw_h5(cfg["data"]["h5_path"])
    eval_idx = data["eval_idx"][: args.n_samples]
    c_true = np.asarray(data["labels"], dtype=np.float32)[eval_idx]
    mu, sigma = data["mu"], data["sigma"]

    if not args.no_wandb:
        wandb.init(
            project=cfg["wandb"]["project"],
            entity=cfg["wandb"]["entity"],
            name=args.run_name,
            config={
                "ckpt": args.ckpt,
                "n_samples": args.n_samples,
                "n_params": count_parameters(model),
                "description": "CNN1 per-method input-preprocessing smoke test",
            },
        )

    methods = [
        "baseline_norm", "raw_uint16", "minmax_sample", "minmax_global",
        "zscore", "oneplane_norm", "oneplane_raw",
    ]
    all_results = {}
    bs = max(1, args.batch_size)

    for method in methods:
        inputs = make_inputs(data["images"], eval_idx, method)

        # Batched forward to bound peak VRAM on the constrained GPU.
        y_chunks, feat_chunks = [], []
        for i in range(0, len(eval_idx), bs):
            x = torch.from_numpy(inputs[i : i + bs]).to(device)
            with torch.no_grad():
                y_chunks.append(model(x).cpu().numpy())
            feat_chunks.append(trunk_features(model, x))
            del x
        y_norm = np.concatenate(y_chunks, axis=0)
        feats = np.concatenate(feat_chunks, axis=0)

        c_pred = y_norm * sigma + mu
        rj = pearson_rj(c_pred, c_true)
        scalars = summarize(inputs, feats, c_pred, c_true, rj)
        fom_ml = compute_fom_ml(data["seeds"][eval_idx], c_pred, cfg)
        scalars["fom_ml_median"] = float(np.median(fom_ml))

        all_results[method] = scalars
        print(f"[{method}] " + json.dumps(scalars, indent=2))

        if not args.no_wandb:
            prefix = method.replace("_", "/")
            for k, v in scalars.items():
                wandb.log({f"{prefix}/{k}": v})
            wandb.log({f"{prefix}/rj_per_mode": wandb.Histogram(rj, num_bins=40)})
            wandb.log({f"{prefix}/input_plane_grid": plane_grid(inputs, device, method)})
            wandb.log({
                f"{prefix}/feat_stats_by_channel": wandb.plot.line_series(
                    xs=list(range(feats.shape[1])),
                    ys=[feats[:, 0], feats[:, 1], feats[:, 2]],
                    keys=["mean", "std", "max"],
                    title=f"{method} feature stats",
                    xname="channel",
                )
            })
            table = wandb.Table(
                data=list(zip(c_true[:, 1], c_pred[:, 1])), columns=["true", "pred"]
            )
            wandb.log({
                f"{prefix}/pred_vs_true_tilt": wandb.plot.scatter(
                    table=table, x="true", y="pred",
                    title=f"{method} pred vs true (mode 2, tilt)",
                )
            })

    if not args.no_wandb:
        wandb.log({"summary/all_methods": wandb.Table(
            data=[[m, r["input_mean"], r["pred_rms"], r["rj_mean"], r["fom_ml_median"]]
                  for m, r in all_results.items()],
            columns=["method", "input_mean", "pred_rms", "rj_mean", "fom_ml_median"],
        )})
        wandb.run.finish()

    print("SMOKE TEST DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
