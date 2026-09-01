"""Training script for the beaconless AO CNN.

Reproduces the training protocol of DiComo et al., "Beaconless adaptive optics
for atmospheric laser propagation with multi-plane convolutional neural
network," Opt. Express 33(15):31010 (2025), Sec 2.6 (CNN1 config), at demo
scale.

Training hyperparameters (Table 1 / Sec 2.6 of the paper):
- Optimizer: Adam (Torch implementation), lr = 1e-4, betas = (0.9, 0.999)
  ("standard momentum values").
- Loss: MSE between the predicted and the actual *scaled* Zernike modes
  (Eq 14 normalization: ``(a - mu) / sigma``).
- Batch size: 32 training pairs per step.
- Every ``sim_eval_every`` steps the current model is evaluated in simulation
  and its FOM compared against the tracking-only solution (Sec 2.6: "every 100
  batches, predicted phase evaluated in simulation").

CLI::

    uv run python train.py --config config.yaml [--ckpt-dir ...] [--no-wandb] [--resume ...]

Single-GPU by default; DDP is enabled automatically when launched via
``torchrun --nproc_per_node=N train.py --config ...`` (LOCAL_RANK/WORLD_SIZE
are detected from the environment).
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import sys
from typing import Any, Callable, Optional

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from models.cnn import CNN1, CNN1Freq, CNN1Star, CNNL
from utils import wandb_utils
from utils.metrics import eta, gain

# h5py chunk-cache for random-access image reads during training (per handle).
H5_CACHE_NBYTES = 256 * 1024 * 1024
H5_CACHE_NSLOTS = 1_000_000

# 12-bit camera quantization: uint16 intensities live in [0, 2047].
IMAGE_MAX = 2047.0


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def attach_run(cfg: dict, *, ckpt_dir: Optional[str] = None, no_wandb: bool = False) -> dict:
    """Attach run-time paths and DDP/device info to ``cfg`` (mutates and returns it).

    Sets ``cfg["run"]`` with:
    - IO paths: ``ckpt_dir``, ``out_dir``, ``h5_path`` (absolute).
    - Device / DDP: ``device``, ``rank``, ``world_size``, ``is_distributed``.
    - Training: ``amp`` (autocast enabled only on CUDA), ``no_wandb``.

    Parameters
    ----------
    cfg : dict
        YAML-loaded configuration (must contain ``data.h5_path``,
        ``train.ckpt_dir``, ``train.amp``).
    ckpt_dir : str, optional
        Overrides ``cfg["train"]["ckpt_dir"]`` (CLI ``--ckpt-dir``).
    no_wandb : bool, optional
        ``True`` disables wandb entirely (CLI ``--no-wandb``).

    Returns
    -------
    dict
        The same ``cfg`` dict, mutated in place.
    """
    run = cfg.setdefault("run", {})
    run["ckpt_dir"] = os.path.abspath(
        ckpt_dir or cfg["train"].get("ckpt_dir", "checkpoints")
    )
    run["out_dir"] = os.path.abspath(cfg.get("eval", {}).get("out_dir", "results"))
    run["h5_path"] = os.path.abspath(cfg["data"]["h5_path"])
    run["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    run["rank"] = int(os.environ.get("LOCAL_RANK", "0"))
    run["world_size"] = int(os.environ.get("WORLD_SIZE", "1"))
    run["is_distributed"] = run["world_size"] > 1
    run["amp"] = bool(cfg["train"].get("amp", False)) and run["device"] == "cuda"
    run["no_wandb"] = bool(no_wandb) or os.environ.get("WANDB_MODE") == "disabled"
    return cfg


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class BeaconlessH5Dataset(torch.utils.data.Dataset):
    """Reads the beaconless AO HDF5 dataset (schema from ``data/generate_h5.py``).

    Pinned schema
    -------------
    - ``/images``  ``(N_total, 3, N, N)`` uint16, 12-bit quantized intensities.
    - ``/labels``  ``(N_total, 78)`` float32 raw Zernike coefficients (radians).
    - ``/fom_*``   ``(N_total,)`` float32.
    - ``/seeds``   ``(N_total,)`` int64.
    - ``/train_idx``, ``/test_idx``, ``/eval_idx`` int64 index arrays.
    - ``/mu``, ``/sigma`` ``(78,)`` float32 computed over the TRAIN split.
    - ``/scale_p`` ``(3,)`` float32; ``/vacuum_intensity`` ``(N, N)`` float32.
    - attribute ``config_json``.

    Sample ``i`` returns a dict:
    - ``images``: ``(3, N, N)`` float32 in ``[0, 1]`` (``uint16 / 2047.0``).
    - ``target``: ``(78,)`` float32 ``= (labels - mu) / sigma`` (Eq 14).
    - ``seed``: int64 scalar (raw turbulence seed, needed for sim eval).
    - ``labels_raw``: ``(78,)`` float32 raw radians (needed for sim eval).

    The HDF5 handle is opened lazily per process so the dataset pickles cleanly
    to DataLoader workers (each worker gets its own handle + chunk cache).
    """

    def __init__(self, h5_path: str, split: str = "train") -> None:
        self.h5_path = h5_path
        self.split = split
        self._f: Optional[h5py.File] = None
        with h5py.File(h5_path, "r") as f:
            self.indices = f[f"/{split}_idx"][:].astype(np.int64)
            self.mu = f["/mu"][:].astype(np.float32)
            self.sigma = f["/sigma"][:].astype(np.float32)
        # Guard against zero-variance modes (sigma == 0 -> division by zero).
        self._sigma_safe = np.where(self.sigma == 0.0, 1.0, self.sigma)

    def _file(self) -> h5py.File:
        if self._f is None:
            self._f = h5py.File(
                self.h5_path,
                "r",
                rdcc_nbytes=H5_CACHE_NBYTES,
                rdcc_nslots=H5_CACHE_NSLOTS,
            )
        return self._f

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        idx = int(self.indices[i])
        f = self._file()
        images = f["/images"][idx].astype(np.float32) / IMAGE_MAX
        labels_raw = f["/labels"][idx].astype(np.float32)
        target = (labels_raw - self.mu) / self._sigma_safe
        seed = int(f["/seeds"][idx])
        return {
            "images": torch.from_numpy(images),
            "target": torch.from_numpy(target),
            "seed": torch.as_tensor(seed, dtype=torch.int64),
            "labels_raw": torch.from_numpy(labels_raw),
        }


def _worker_init_fn(base_seed: int) -> Callable[[int], None]:
    """Return a DataLoader worker_init_fn seeding torch/numpy per worker."""

    def _init(worker_id: int) -> None:
        torch.manual_seed(base_seed + worker_id)
        np.random.seed(base_seed + worker_id)

    return _init


# --------------------------------------------------------------------------- #
# Model / optimizer
# --------------------------------------------------------------------------- #
def build_model(cfg: dict) -> nn.Module:
    """Build the CNN1 (or CNNL) model from ``cfg["model"]``.

    Architecture per Sec 2.6: 3-stage CNN (3x3 conv, stride 1, padding 0,
    BatchNorm2d, ReLU, 2x2 MaxPool) -> AdaptiveAvgPool2d((18, 18)) -> flatten
    -> MLP of 4 hidden layers of 512 ReLU neurons -> ``n_modes`` outputs.
    """
    m = cfg["model"]
    name = m.get("name", "CNN1")
    kwargs: dict[str, Any] = dict(
        n_modes=int(m["n_modes"]),
        mlp_width=int(m.get("mlp_width", 512)),
        mlp_depth=int(m.get("mlp_depth", 4)),
        dropout=float(m.get("dropout", 0.0)),
    )
    if "channels" in m:
        kwargs["channels"] = tuple(int(c) for c in m["channels"])
        kwargs.setdefault("pool_size", int(m.get("pool_size", 18)))
    if name == "CNN1":
        return CNN1(**kwargs)
    if name == "CNNL":
        return CNNL(**kwargs)
    if name == "CNN1Freq":
        freq_kwargs = dict(
            freq_pool=int(m.get("freq_pool", 8)),
            freq_refine_ch=int(m.get("freq_refine_ch", 16)),
        )
        return CNN1Freq(**kwargs, **freq_kwargs)
    if name == "CNN1Star":
        # CNN1Star uses base_dim/depths instead of channels, and its own
        # pool_size default (12), so it gets only the shared MLP kwargs.
        star_kwargs = dict(
            n_modes=int(m["n_modes"]),
            mlp_width=int(m.get("mlp_width", 512)),
            mlp_depth=int(m.get("mlp_depth", 4)),
            dropout=float(m.get("dropout", 0.0)),
            base_dim=int(m.get("base_dim", 32)),
            depths=tuple(int(d) for d in m.get("depths", (1, 1, 2))),
            mlp_ratio=int(m.get("mlp_ratio", 4)),
            use_se=bool(m.get("use_se", False)),
            se_reduction=int(m.get("se_reduction", 16)),
            pool_size=int(m.get("pool_size", 12)),
            kernel=int(m.get("kernel", 3)),
        )
        return CNN1Star(**star_kwargs)
    raise ValueError(f"unknown model name {name!r}")


def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    """Adam optimizer per Table 1: lr 1e-4, betas (0.9, 0.999)."""
    t = cfg["train"]
    return torch.optim.Adam(
        model.parameters(),
        lr=float(t.get("lr", 1e-4)),
        betas=(float(t.get("beta1", 0.9)), float(t.get("beta2", 0.999))),
    )


def compute_grad_norm(model: nn.Module) -> float:
    """Global L2 norm of all parameter gradients (0.0 if none)."""
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().float().norm().item() ** 2)
    return float(np.sqrt(total))


# --------------------------------------------------------------------------- #
# Checkpoints
# --------------------------------------------------------------------------- #
def sync_barrier(is_distributed: bool) -> None:
    """Collective barrier called by ALL ranks at the same loop point.

    DDP correctness: barriers must never sit inside a rank-0-only branch, or
    the other ranks race ahead and deadlock on the next gradient all-reduce.
    """
    if is_distributed:
        torch.distributed.barrier()


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    step: int,
    cfg: dict,
    *,
    rank: int,
    is_distributed: bool,
) -> None:
    """Save ``{model_state, optimizer_state, scaler_state, step, cfg}``.

    Only rank 0 writes. The caller is responsible for surrounding this with
    :func:`sync_barrier` calls so every rank reaches the save point together.
    """
    if rank == 0:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        state = {
            "model_state": (
                model.module.state_dict() if is_distributed else model.state_dict()
            ),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "step": int(step),
            "cfg": cfg,
        }
        torch.save(state, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    *,
    is_distributed: bool,
) -> int:
    """Load a checkpoint onto ``device``; returns the saved step.

    DDP: each rank loads the shared file with ``map_location`` to its device.
    """
    ckpt = torch.load(path, map_location=device)
    target = model.module if is_distributed else model
    target.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scaler.load_state_dict(ckpt["scaler_state"])
    return int(ckpt["step"])


# --------------------------------------------------------------------------- #
# Simulation FOM evaluation (lazy import of data.simulate)
# --------------------------------------------------------------------------- #
# Worker-global state for the persistent sim-eval Pool: each worker builds the
# physics stack once (physics_from_cfg) and reuses it across all eval calls.
_SIM_WORKER: dict[str, Any] = {"cfg": None, "shared": None}


def _sim_worker_init(cfg: dict) -> None:
    """Pool initializer: build the physics stack once per worker."""
    _SIM_WORKER["cfg"] = cfg
    try:
        from data.simulate import physics_from_cfg

        _SIM_WORKER["shared"] = physics_from_cfg(cfg)
    except Exception:
        # simulate_sample_fom(shared=None) builds physics internally.
        _SIM_WORKER["shared"] = None


def _sim_worker_eval(task: tuple) -> float:
    """Pool worker: run one simulation FOM evaluation."""
    seed, coeffs, cfg = task
    from data.simulate import simulate_sample_fom

    return simulate_sample_fom(
        seed=seed, cfg=cfg, coeffs=coeffs, shared=_SIM_WORKER["shared"]
    )


class SimEvaluator:
    """Runs ``simulate_sample_fom`` calls in a persistent multiprocessing Pool.

    The Pool is created lazily on first use and reused across all eval calls
    ("reuse_workers"); each worker builds ``physics_from_cfg`` once via the
    initializer. Use ``spawn`` by default (safe with torch); ``fork`` may be
    selected via ``cfg["train"]["sim_eval_context"]`` on Linux.
    """

    def __init__(self, cfg: dict, processes: Optional[int] = None) -> None:
        self.cfg = cfg
        self.processes = processes or int(cfg["train"].get("sim_eval_workers", 4))
        self._pool: Optional[multiprocessing.pool.Pool] = None

    def _ensure(self) -> None:
        if self._pool is not None:
            return
        context = self.cfg["train"].get("sim_eval_context", "spawn")
        ctx = multiprocessing.get_context(context)
        self._pool = ctx.Pool(
            processes=self.processes,
            initializer=_sim_worker_init,
            initargs=(self.cfg,),
        )

    def run(self, seeds: np.ndarray, coeffs_list: list[np.ndarray]) -> list[float]:
        """Evaluate ``simulate_sample_fom(seed, cfg, coeffs)`` for each sample."""
        self._ensure()
        assert self._pool is not None
        tasks = [
            (int(s), np.asarray(c, dtype=np.float64), self.cfg)
            for s, c in zip(seeds, coeffs_list)
        ]
        return list(self._pool.map(_sim_worker_eval, tasks))

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None


def evaluate_sim_fom(
    model: nn.Module,
    cfg: dict,
    *,
    device: Optional[torch.device] = None,
    use_pool: bool = True,
    evaluator: Optional[SimEvaluator] = None,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Optional[dict[str, float]]:
    """Evaluate the current model on the eval split via simulation FOM.

    Picks the first ``cfg.train.sim_eval_n`` samples of ``/eval_idx``, predicts
    denormalized coefficients ``c_pred = y_pred * sigma + mu``, and runs
    ``data.simulate.simulate_sample_fom(seed, cfg, coeffs)`` for each. Returns
    a metrics dict (median FOM, gain vs tracking, eta vs z78) or ``None`` if
    ``data.simulate`` is unavailable (lazy import; logs a warning and skips).

    Parameters
    ----------
    model : nn.Module
        The trained model (predictions are made in eval mode).
    cfg : dict
        Full config; ``cfg["data"]["h5_path"]`` locates the dataset.
    device : torch.device, optional
        Device for predictions (defaults to ``cfg["run"]["device"]`` or CPU).
    use_pool : bool
        ``True`` runs the simulation calls through the persistent Pool;
        ``False`` calls ``simulate_sample_fom`` in-process (test mode).
    evaluator : SimEvaluator, optional
        Reusable Pool wrapper; created lazily if ``None``.
    log_fn : callable, optional
        Receives the "sim eval unavailable" warning (defaults to ``print``).
    """
    if device is None:
        device = torch.device(cfg.get("run", {}).get("device", "cpu"))
    if log_fn is None:
        log_fn = print

    # Lazy import: data/simulate.py may not exist yet (parallel agent).
    try:
        from data.simulate import simulate_sample_fom
    except ImportError:
        log_fn("sim eval unavailable (data.simulate not built); skipping")
        return None

    run = cfg.get("run", {})
    h5_path = run.get("h5_path") or cfg["data"]["h5_path"]
    sim_eval_n = int(cfg["train"].get("sim_eval_n", 8))
    amp = bool(run.get("amp", False))

    with h5py.File(h5_path, "r") as f:
        eval_idx = f["/eval_idx"][:sim_eval_n].astype(np.int64)
        seeds = f["/seeds"][eval_idx].astype(np.int64)
        mu = f["/mu"][:].astype(np.float32)
        sigma = f["/sigma"][:].astype(np.float32)
        fom_track = f["/fom_track"][eval_idx].astype(np.float64)
        fom_z78 = f["/fom_z78"][eval_idx].astype(np.float64)
        images = f["/images"][eval_idx].astype(np.float32) / IMAGE_MAX
        labels_raw = f["/labels"][eval_idx].astype(np.float32)

    # Predict denormalized coefficients for each eval sample.
    model.eval()
    coeffs_list: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(len(eval_idx)):
            img = torch.from_numpy(images[i]).unsqueeze(0).to(device)
            with torch.autocast("cuda", enabled=amp):
                y_pred = model(img)
            y_pred = y_pred.float().cpu().numpy()[0]
            c_pred = y_pred * sigma + mu
            coeffs_list.append(c_pred.astype(np.float64))
    model.train()

    if use_pool:
        if evaluator is None:
            evaluator = SimEvaluator(cfg)
        fom_ml = evaluator.run(seeds, coeffs_list)
    else:
        fom_ml = [
            simulate_sample_fom(seed=int(s), cfg=cfg, coeffs=c)
            for s, c in zip(seeds, coeffs_list)
        ]

    fom_ml = np.asarray(fom_ml, dtype=np.float64)
    med_ml = float(np.median(fom_ml))
    med_track = float(np.median(fom_track))
    med_z78 = float(np.median(fom_z78))
    metrics: dict[str, float] = {
        "sim/median_fom_ml": med_ml,
        "sim/median_fom_track": med_track,
        "sim/median_fom_z78": med_z78,
        "sim/gain": gain(med_ml, med_track),
        "sim/eta": eta(med_ml, med_track, med_z78),
        "sim/n_eval": float(len(fom_ml)),
    }

    # Optional live baseline sanity: z78-via-simulation through the same Pool.
    if cfg["train"].get("sim_eval_z78", False):
        z78_coeffs = [np.asarray(lr, dtype=np.float64) for lr in labels_raw]
        if use_pool:
            fom_z78_sim = evaluator.run(seeds, z78_coeffs)
        else:
            fom_z78_sim = [
                simulate_sample_fom(seed=int(s), cfg=cfg, coeffs=c)
                for s, c in zip(seeds, z78_coeffs)
            ]
        metrics["sim/median_fom_z78_sim"] = float(np.median(fom_z78_sim))

    return metrics


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #
def train(cfg: dict) -> dict[str, Any]:
    """Run the training loop; returns a summary dict.

    Summary keys: ``losses`` (per-step MSE), ``step``, ``final_loss``,
    ``final_fom_ml`` / ``final_gain`` / ``final_eta`` (last sim eval, if any),
    ``best_fom``.

    The loop iterates epochs over the shuffled training set until
    ``cfg.train.n_steps`` steps are exhausted. Checkpoints: ``last.pt`` every
    log cycle, ``best.pt`` on median sim-FOM improvement (fallback: lowest
    running train loss before the first sim eval).
    """
    if "run" not in cfg:
        attach_run(cfg)
    run = cfg["run"]
    rank = int(run["rank"])
    world_size = int(run["world_size"])
    is_dist = bool(run["is_distributed"])
    device = torch.device(run["device"])
    amp = bool(run["amp"])
    t = cfg["train"]
    n_steps = int(t["n_steps"])
    batch_size = int(t["batch_size"])
    micro_batch_size = int(t.get("micro_batch_size", batch_size))
    grad_accum = max(1, batch_size // micro_batch_size)
    log_every = int(t.get("log_every", 10))
    sim_eval_every = int(t.get("sim_eval_every", 500))
    num_workers = int(t.get("num_workers", 4))
    seed = int(t.get("seed", 0))

    # --- DDP init (auto-detected from torchrun env) ---
    if is_dist:
        backend = "nccl" if device.type == "cuda" else "gloo"
        if device.type == "cuda":
            torch.distributed.init_process_group(backend=backend, device_id=rank)
            torch.cuda.set_device(rank)
        else:
            torch.distributed.init_process_group(backend=backend)

    # --- Determinism: seed per rank ---
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)

    # --- Data ---
    train_ds = BeaconlessH5Dataset(run["h5_path"], split="train")
    sampler = None
    shuffle = True
    if is_dist:
        sampler = torch.utils.data.DistributedSampler(
            train_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=seed + rank,
        )
        shuffle = False
    loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=micro_batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        worker_init_fn=_worker_init_fn(seed + rank),
    )

    # --- Model / optimizer / scaler ---
    model = build_model(cfg).to(device)
    if t.get("channels_last", False) and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    if t.get("compile", False):
        model = torch.compile(model)
    if is_dist:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[rank] if device.type == "cuda" else None,
        )
    optimizer = build_optimizer(model, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    # --- Resume ---
    step = 0
    if run.get("resume"):
        step = load_checkpoint(
            run["resume"],
            model,
            optimizer,
            scaler,
            device,
            is_distributed=is_dist,
        )
        sync_barrier(is_dist)
        if rank == 0:
            print(f"[train] resumed from {run['resume']} at step {step}")

    # --- WandB (rank 0 only) ---
    wandb_run = None
    if rank == 0 and not run["no_wandb"]:
        w = cfg.get("wandb", {})
        if w.get("entity"):
            os.environ["WANDB_ENTITY"] = str(w["entity"])
        wandb_run = wandb_utils.init_wandb(
            config=cfg,
            run_name=w.get("run_name"),
            project=w.get("project", "beaconless-ao-sim"),
            tags=w.get("tags"),
            job_type="training",
        )

    def _log(metrics: dict[str, float], s: int) -> None:
        wandb_utils.log_metrics(wandb_run, metrics, step=s)

    # --- Checkpoint paths ---
    ckpt_dir = run["ckpt_dir"]
    last_path = os.path.join(ckpt_dir, "last.pt")
    best_path = os.path.join(ckpt_dir, "best.pt")

    # --- Loop state ---
    losses: list[float] = []
    best_fom = -float("inf")
    best_loss = float("inf")
    has_sim_eval = False
    last_eval: Optional[dict[str, float]] = None
    evaluator: Optional[SimEvaluator] = None

    def _save_last() -> None:
        save_checkpoint(
            last_path, model, optimizer, scaler, step, cfg,
            rank=rank, is_distributed=is_dist,
        )

    def _save_best() -> None:
        save_checkpoint(
            best_path, model, optimizer, scaler, step, cfg,
            rank=rank, is_distributed=is_dist,
        )

    epoch = 0
    accum_count = 0
    accum_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    while step < n_steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            if step >= n_steps:
                break

            images = batch["images"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            if t.get("channels_last", False) and device.type == "cuda":
                images = images.to(memory_format=torch.channels_last)

            # Micro-batch forward/backward; gradient is scaled by 1/grad_accum
            # so the optimizer step uses the mean gradient over the effective
            # batch (paper batch 32), independent of the VRAM-limited
            # micro-batch size.
            with torch.autocast("cuda", enabled=amp):
                pred = model(images)
                mse = F.mse_loss(pred, target)
            scaler.scale(mse / grad_accum).backward()

            accum_loss += float(mse.detach().float().item())
            accum_count += 1
            if accum_count < grad_accum:
                continue

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            loss_val = accum_loss / grad_accum
            accum_loss = 0.0
            accum_count = 0
            losses.append(loss_val)
            step += 1

            # --- Logging + last.pt checkpoint every log cycle ---
            # All ranks synchronize here (they are aligned after the DDP
            # gradient all-reduce); only rank 0 logs and writes files.
            if step % log_every == 0:
                if rank == 0:
                    lr = float(optimizer.param_groups[0]["lr"])
                    gn = compute_grad_norm(model)
                    _log(
                        {
                            "train/loss": loss_val,
                            "train/lr": lr,
                            "train/grad_norm": gn,
                        },
                        step,
                    )
                sync_barrier(is_dist)
                _save_last()
                if rank == 0 and not has_sim_eval and loss_val < best_loss:
                    best_loss = loss_val
                    _save_best()
                sync_barrier(is_dist)

            # --- Simulation FOM eval (rank 0 only, all ranks wait) ---
            if step % sim_eval_every == 0:
                sync_barrier(is_dist)
                if rank == 0:
                    if evaluator is None:
                        evaluator = SimEvaluator(cfg)
                    eval_metrics = evaluate_sim_fom(
                        model, cfg, device=device, evaluator=evaluator
                    )
                    if eval_metrics is not None:
                        has_sim_eval = True
                        last_eval = eval_metrics
                        _log(eval_metrics, step)
                        med = float(eval_metrics["sim/median_fom_ml"])
                        if med > best_fom:
                            best_fom = med
                            _save_best()
                        print(
                            f"[train] step {step}: sim FOM={med:.4f} "
                            f"gain={eval_metrics['sim/gain']:.3f} "
                            f"eta={eval_metrics['sim/eta']:.3f}"
                        )
                sync_barrier(is_dist)
        epoch += 1

    # --- Final checkpoint + cleanup ---
    sync_barrier(is_dist)
    _save_last()
    sync_barrier(is_dist)
    if evaluator is not None:
        evaluator.close()
    if is_dist:
        torch.distributed.destroy_process_group()

    summary: dict[str, Any] = {
        "losses": losses,
        "step": step,
        "final_loss": losses[-1] if losses else None,
        "final_fom_ml": last_eval["sim/median_fom_ml"] if last_eval else None,
        "final_gain": last_eval["sim/gain"] if last_eval else None,
        "final_eta": last_eval["sim/eta"] if last_eval else None,
        "best_fom": best_fom if has_sim_eval else None,
    }
    if rank == 0:
        print("=" * 64)
        print(f"[train] complete: {step} steps")
        if losses:
            print(f"[train] final train loss: {losses[-1]:.6f}")
        if last_eval is not None:
            print(
                f"[train] final sim-eval: median FOM={last_eval['sim/median_fom_ml']:.4f} "
                f"gain={last_eval['sim/gain']:.3f} eta={last_eval['sim/eta']:.3f}"
            )
        print(f"[train] checkpoints: {ckpt_dir}")
        print("=" * 64)
    return summary


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point: ``train.py --config config.yaml [--ckpt-dir ...]``."""
    parser = argparse.ArgumentParser(
        description=(
            "Train the beaconless AO CNN (DiComo et al., Opt. Express "
            "33(15):31010 (2025), Sec 2.6)."
        )
    )
    parser.add_argument("--config", required=True, help="path to config.yaml")
    parser.add_argument("--ckpt-dir", default=None, help="override train.ckpt_dir")
    parser.add_argument("--no-wandb", action="store_true", help="disable wandb")
    parser.add_argument("--resume", default=None, help="checkpoint to resume from")
    args = parser.parse_args(argv)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    attach_run(cfg, ckpt_dir=args.ckpt_dir, no_wandb=args.no_wandb)
    if args.resume:
        cfg["run"]["resume"] = os.path.abspath(args.resume)

    train(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())