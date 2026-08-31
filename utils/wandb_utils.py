"""WandB logging helpers for the beaconless AO simulation.

All functions are defensive: they never raise, and every wandb call is wrapped
in try/except so that wandb failures can never crash training. When ``run`` is
``None`` every function is a no-op.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

try:
    import wandb
except Exception:  # pragma: no cover - wandb is a hard dependency
    wandb = None  # type: ignore[assignment]

try:
    import torch
except Exception:  # pragma: no cover - torch is a hard dependency
    torch = None  # type: ignore[assignment]


def build_montage(images: np.ndarray, n_cols: int = 3) -> np.ndarray:
    """Tile a batch of images into a single uint8 grid with 1px gray padding.

    Args:
        images: Array of shape ``(B, H, W)`` (float or uint8) or ``(B, C, H, W)``.
            Each channel is normalized independently to ``[0, 255]``.
        n_cols: Number of grid columns.

    Returns:
        uint8 array of shape ``(grid_H, grid_W)`` for grayscale input or
        ``(grid_H, grid_W, 3)`` for RGB input, where each cell is ``W`` px wide
        and there is a 1px gray (128) border between and around every cell.
    """
    images = np.asarray(images)
    if images.ndim == 3:
        # (B, H, W) grayscale
        b, h, w = images.shape
        cells = images[:, None, :, :]  # (B, 1, H, W)
    elif images.ndim == 4:
        # (B, C, H, W)
        b, c, h, w = images.shape
        cells = images
    else:
        raise ValueError(f"images must be (B, H, W) or (B, C, H, W), got {images.shape}")

    # Normalize each channel independently to [0, 255].
    cells = cells.astype(np.float64)
    for i in range(cells.shape[0]):
        for ch in range(cells.shape[1]):
            chan = cells[i, ch]
            lo, hi = float(chan.min()), float(chan.max())
            if hi > lo:
                cells[i, ch] = (chan - lo) / (hi - lo) * 255.0
            else:
                cells[i, ch] = 0.0

    n_rows = int(np.ceil(b / n_cols))
    grid_h = n_rows * h + (n_rows + 1)
    grid_w = n_cols * w + (n_cols + 1)
    n_ch = cells.shape[1]

    # Move channel axis to the last position: (B, H, W, C).
    cells = np.transpose(cells, (0, 2, 3, 1))

    grid = np.full((grid_h, grid_w, n_ch), 128.0, dtype=np.float64)

    for idx in range(b):
        r, c = divmod(idx, n_cols)
        y0 = r * h + (r + 1)
        x0 = c * w + (c + 1)
        grid[y0 : y0 + h, x0 : x0 + w, :] = cells[idx]

    if n_ch == 1:
        grid = grid[..., 0]

    return np.clip(np.round(grid), 0, 255).astype(np.uint8)


def init_wandb(
    config: dict,
    run_name: str,
    project: str = "beaconless-ao-sim",
    group: Optional[str] = None,
    tags: Optional[list] = None,
    job_type: str = "training",
    mode: Optional[str] = None,
) -> Any:
    """Initialize a wandb run, falling back to offline or None on any failure.

    Attempts ``wandb.login(anonymous='never', force=False)`` first (auth via
    ``~/.netrc``). If login raises or returns False, mode is forced to
    ``'offline'``. Never raises.

    Returns:
        A wandb Run, or ``None`` if wandb is unavailable / init fails.
    """
    if wandb is None:
        return None

    effective_mode = mode
    try:
        login_ok = wandb.login(anonymous="never", force=False)
        if not login_ok:
            effective_mode = "offline"
    except Exception:
        effective_mode = "offline"

    try:
        return wandb.init(
            project=project,
            name=run_name,
            group=group,
            tags=tags,
            job_type=job_type,
            config=config,
            mode=effective_mode,
        )
    except Exception:
        return None


def log_gpu_stats(run: Any, step: Optional[int] = None) -> None:
    """Log GPU utilization and memory usage if CUDA is available and run is set."""
    if run is None or torch is None:
        return
    if not torch.cuda.is_available():
        return
    try:
        util = torch.cuda.utilization()
        mem_used = torch.cuda.memory_allocated() / (1024**3)
        mem_alloc = torch.cuda.memory_reserved() / (1024**3)
        run.log(
            {
                "gpu/util_%": util,
                "gpu/mem_used_GB": mem_used,
                "gpu/mem_alloc_GB": mem_alloc,
                **({"step": step} if step is not None else {}),
            }
        )
    except Exception:
        pass


def log_metrics(run: Any, metrics: dict, step: Optional[int] = None) -> None:
    """Log a metrics dict to the run, optionally with a step."""
    if run is None:
        return
    try:
        run.log({"step": step, **metrics} if step is not None else metrics)
    except Exception:
        pass


def log_figure(run: Any, fig: Any, name: str, step: Optional[int] = None) -> None:
    """Log a matplotlib Figure as a wandb.Image. Never raises."""
    if run is None or fig is None:
        return
    try:
        run.log({name: wandb.Image(fig), **({"step": step} if step is not None else {})})
    except Exception:
        pass


def log_montage(
    run: Any,
    images: np.ndarray,
    name: str,
    n_cols: int = 3,
    step: Optional[int] = None,
) -> None:
    """Build a montage from images and log it as a wandb.Image. Never raises."""
    if run is None:
        return
    try:
        montage = build_montage(images, n_cols=n_cols)
        run.log({name: wandb.Image(montage), **({"step": step} if step is not None else {})})
    except Exception:
        pass


def finish_wandb(run: Any) -> None:
    """Finish a wandb run if it is set. Never raises."""
    if run is None:
        return
    try:
        run.finish()
    except Exception:
        pass
