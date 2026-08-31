"""Tests for physics.screens_soapy.

TDD: these tests are written first (RED), then the module is implemented (GREEN).
"""

import numpy as np
import pytest

from physics.screens_soapy import SoapyPhaseScreenGenerator, compute_r0


def test_compute_r0_units_and_value():
    """compute_r0 returns a positive value in metres for typical parameters."""
    lam = 500e-9
    Cn2 = 1e-14
    L = 1000.0
    r0 = compute_r0(lam, Cn2, L)
    assert np.isfinite(r0)
    assert r0 > 0
    # r0 = (0.423 * k**2 * Cn2 * L)**(-3/5), k = 2*pi/lam
    k = 2 * np.pi / lam
    expected = (0.423 * k**2 * Cn2 * L) ** (-3 / 5)
    assert r0 == pytest.approx(expected)


def test_structure_function_theory_ratio():
    """Structure function at lag=r0 should be ~6.88 (within SH overshoot bounds)."""
    N = 256
    dx = 0.03 / N
    r0 = 30 * dx  # ~3.5mm
    lag = 30  # px
    gen = SoapyPhaseScreenGenerator(
        N=N, dx=dx, L0=100.0, l0=0.005, lam=500e-9, generator="soapy"
    )
    pool = gen.make_pool(1, r0, seed=0)
    screen = pool[0]
    sf = SoapyPhaseScreenGenerator.structure_function(screen, lag)
    theory = 6.88 * (lag * dx / r0) ** (5 / 3)  # = 6.88
    ratio = sf / theory
    assert 0.3 < ratio < 1.5, f"sf/theory={ratio} out of [0.3, 1.5]"


def test_pool_shape():
    """make_pool returns (n_pool, N, N) float32 finite screens."""
    N = 256
    dx = 0.03 / N
    gen = SoapyPhaseScreenGenerator(
        N=N, dx=dx, L0=100.0, l0=0.005, lam=500e-9, generator="soapy"
    )
    pool = gen.make_pool(5, r0=30 * dx, seed=42)
    assert pool.shape == (5, N, N)
    assert pool.dtype == np.float32
    assert np.all(np.isfinite(pool))


def test_slide_window():
    """slide_window returns the exact requested slice."""
    N = 64
    dx = 0.03 / N
    gen = SoapyPhaseScreenGenerator(
        N=N, dx=dx, L0=100.0, l0=0.005, lam=500e-9, generator="soapy"
    )
    pool = gen.make_pool(5, r0=30 * dx, seed=1)
    window = gen.slide_window(pool, 3, 1)
    assert np.array_equal(window, pool[1:4])


def test_slide_window_bounds():
    """slide_window raises ValueError when the window exceeds the pool."""
    N = 64
    dx = 0.03 / N
    gen = SoapyPhaseScreenGenerator(
        N=N, dx=dx, L0=100.0, l0=0.005, lam=500e-9, generator="soapy"
    )
    pool = gen.make_pool(5, r0=30 * dx, seed=1)
    with pytest.raises(ValueError):
        gen.slide_window(pool, 3, len(pool) - 2)


def test_l0_zero_guard():
    """l0=0.0 must raise ValueError (aotools divides by l0) BEFORE calling aotools."""
    N = 64
    dx = 0.03 / N
    with pytest.raises(ValueError, match="l0"):
        SoapyPhaseScreenGenerator(
            N=N, dx=dx, L0=100.0, l0=0.0, lam=500e-9, generator="soapy"
        )
