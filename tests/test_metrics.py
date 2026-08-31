"""Tests for utils.metrics.

TDD: these tests are written first (RED), then the module is implemented (GREEN).
Metric definitions follow DiComo et al., Opt. Express 33(15):31010 (2025), Eqs 6-8, 15-17.
"""

import numpy as np
import pytest

from utils.metrics import (
    FOM,
    SIB,
    bucket_mask,
    eta,
    gain,
    mode_pearson,
    nPIB,
)


# --------------------------------------------------------------------------- #
# bucket_mask
# --------------------------------------------------------------------------- #
def test_bucket_mask_center_true():
    """Center cell of the mask is always included."""
    mask = bucket_mask(11, diameter_px=5.0)
    c = 5
    assert mask[c, c]


def test_bucket_mask_corner_false():
    """Corner cell (0,0) is excluded for a small diameter."""
    mask = bucket_mask(11, diameter_px=5.0)
    assert not mask[0, 0]


def test_bucket_mask_diameter3_nine_cells():
    """d=3 (r=1.5): the 3x3 block around center (all cells within r=1.5) = 9 cells.

    Center is at (5,5) for N=11. Orthogonal neighbors are at distance 1.0 and
    diagonal neighbors at distance sqrt(2) ~ 1.414; both are <= 1.5, so the
    full 3x3 block (9 cells) is included by the euclidean-disk rule.
    """
    mask = bucket_mask(11, diameter_px=3.0)
    assert mask.sum() == 9


def test_bucket_mask_area_approx_pi():
    """Mask area approximates pi*(d/2)^2 for d=20 (within 5%)."""
    d = 20.0
    mask = bucket_mask(101, diameter_px=d)
    area = mask.sum()
    expected = np.pi * (d / 2.0) ** 2
    assert abs(area - expected) / expected < 0.05


# --------------------------------------------------------------------------- #
# nPIB
# --------------------------------------------------------------------------- #
def test_nPIB_uniform_ratio_one():
    """Uniform I and I_vac over the same mask -> ratio of areas = 1.0."""
    N = 21
    mask = bucket_mask(N, diameter_px=7.0)
    I = np.ones((N, N))
    I_vac = np.ones((N, N))
    assert nPIB(I, I_vac, mask) == pytest.approx(1.0)


def test_nPIB_all_in_bucket_greater_than_one():
    """All power inside bucket for I but spread for I_vac -> nPIB > 1."""
    N = 21
    mask = bucket_mask(N, diameter_px=7.0)
    I = np.zeros((N, N))
    I[mask] = 1.0  # all power in bucket
    I_vac = np.ones((N, N))  # power spread everywhere
    assert nPIB(I, I_vac, mask) > 1.0


# --------------------------------------------------------------------------- #
# SIB
# --------------------------------------------------------------------------- #
def test_SIB_scaling():
    """I = 2*I_vac everywhere -> SIB = 2."""
    N = 21
    mask = bucket_mask(N, diameter_px=7.0)
    I_vac = np.ones((N, N))
    I = 2.0 * I_vac
    assert SIB(I, I_vac, mask) == pytest.approx(2.0)


# --------------------------------------------------------------------------- #
# FOM
# --------------------------------------------------------------------------- #
def test_FOM_composition():
    """FOM = sqrt(nPIB * SIB); all-in-bucket for both I and I_vac -> ~1."""
    N = 21
    mask = bucket_mask(N, diameter_px=7.0)
    I = np.zeros((N, N))
    I[mask] = 1.0
    I_vac = np.zeros((N, N))
    I_vac[mask] = 1.0  # vacuum beam also fully in bucket -> normalized
    fom = FOM(I, I_vac, mask)
    assert fom == pytest.approx(np.sqrt(nPIB(I, I_vac, mask) * SIB(I, I_vac, mask)))
    assert fom == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# gain
# --------------------------------------------------------------------------- #
def test_gain_basic():
    """gain(2,1)=2; gain(0.5,1)=0.5; gain(any,0)=0."""
    assert gain(2.0, 1.0) == pytest.approx(2.0)
    assert gain(0.5, 1.0) == pytest.approx(0.5)
    assert gain(3.0, 0.0) == 0.0
    assert gain(3.0, -1.0) == 0.0


# --------------------------------------------------------------------------- #
# eta
# --------------------------------------------------------------------------- #
def test_eta_basic():
    """eta(0.8, 0.2, 0.9) = (0.6)/(0.7) = 0.857..."""
    assert eta(0.8, 0.2, 0.9) == pytest.approx(0.6 / 0.7)


def test_eta_no_work():
    """eta(0.2, 0.2, 0.9) = 0 (ML equals tracking-only -> no work)."""
    assert eta(0.2, 0.2, 0.9) == 0.0


def test_eta_zero_denominator():
    """eta(x, 0.2, 0.2) = 0 (denominator 0, numerator 0)."""
    assert eta(0.5, 0.2, 0.2) == 0.0


def test_eta_ceiling_below_tracking():
    """eta(x, 0.8, 0.2) = 0 (78-mode ceiling below tracking -> degenerate)."""
    assert eta(0.5, 0.8, 0.2) == 0.0
    assert eta(0.9, 0.8, 0.2) == 0.0


# --------------------------------------------------------------------------- #
# mode_pearson
# --------------------------------------------------------------------------- #
def test_mode_pearson_perfect_positive():
    """Perfect linear correlation (y=2x) -> 1.0."""
    x = np.linspace(-1, 1, 50)
    ml = np.stack([x, x], axis=1)
    test = np.stack([2 * x, 2 * x], axis=1)
    r = mode_pearson(ml, test)
    assert r.shape == (2,)
    assert np.allclose(r, 1.0)


def test_mode_pearson_perfect_negative():
    """Anti-correlation -> -1.0."""
    x = np.linspace(-1, 1, 50)
    ml = np.stack([x], axis=1)
    test = np.stack([-x], axis=1)
    r = mode_pearson(ml, test)
    assert r.shape == (1,)
    assert np.allclose(r, -1.0)


def test_mode_pearson_uncorrelated():
    """Uncorrelated random with seed -> |R| < 0.3."""
    rng = np.random.default_rng(42)
    ml = rng.normal(size=(200, 1))
    test = rng.normal(size=(200, 1))
    r = mode_pearson(ml, test)
    assert abs(r[0]) < 0.3


def test_mode_pearson_zero_variance():
    """Zero-variance column -> 0.0."""
    ml = np.ones((10, 1))
    test = np.linspace(0, 1, 10)[:, None]
    r = mode_pearson(ml, test)
    assert r[0] == 0.0


def test_mode_pearson_shape():
    """Input (K,3) -> length-3 output."""
    rng = np.random.default_rng(7)
    ml = rng.normal(size=(100, 3))
    test = rng.normal(size=(100, 3))
    r = mode_pearson(ml, test)
    assert r.shape == (3,)
