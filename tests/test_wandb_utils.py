"""Tests for utils.wandb_utils.

TDD: these tests are written first (RED), then the module is implemented (GREEN).
All tests are hermetic — no network access, no real online wandb runs.
"""

import numpy as np
import pytest

from utils.wandb_utils import (
    build_montage,
    finish_wandb,
    init_wandb,
    log_figure,
    log_gpu_stats,
    log_metrics,
)


def test_init_offline_returns_run():
    """init_wandb with mode='offline' returns a wandb.Run; finish does not raise."""
    run = init_wandb({"lr": 1e-4}, "t", mode="offline")
    assert run is not None
    finish_wandb(run)  # must not raise


def test_log_metrics_none_run():
    """log_metrics with run=None is a no-op and does not raise."""
    log_metrics(None, {"a": 1})


def test_log_figure_none():
    """log_figure with run=None and fig=None is a no-op and does not raise."""
    log_figure(None, None, "x")


def test_build_montage_shape():
    """4 images of (16,16) with n_cols=2 tile to (35,35) with 1px gray borders."""
    images = np.zeros((4, 16, 16), dtype=np.float32)
    montage = build_montage(images, n_cols=2)
    # 2 columns of 16px + 3 borders of 1px = 2*16 + 3 = 35
    assert montage.shape == (35, 35)
    assert montage.dtype == np.uint8


def test_log_gpu_stats_none():
    """log_gpu_stats with run=None is a no-op and does not raise."""
    log_gpu_stats(None)


def test_build_montage_rgb_shape():
    """(B, C, H, W) input with C=3 produces a 3-channel montage."""
    images = np.zeros((4, 3, 16, 16), dtype=np.uint8)
    montage = build_montage(images, n_cols=2)
    assert montage.shape == (35, 35, 3)
    assert montage.dtype == np.uint8


def test_build_montage_padding_value():
    """Border pixels are gray (128) and distinct from black image content."""
    images = np.zeros((4, 16, 16), dtype=np.float32)
    montage = build_montage(images, n_cols=2)
    # top-left border pixel should be gray padding
    assert montage[0, 0] == 128
    # interior of first cell should be black (normalized 0)
    assert montage[1, 1] == 0
