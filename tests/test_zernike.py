"""Tests for the Zernike basis (physics.zernike_aotools.ZernikeBasis)."""

import numpy as np
import pytest

from physics.zernike_aotools import ZernikeBasis


def test_zernike_array_shape():
    """ZernikeBasis(64, 10).Z must have shape (10, 64, 64)."""
    basis = ZernikeBasis(64, 10)
    assert basis.Z.shape == (10, 64, 64)


def test_mask_pixels():
    """In-circle mask pixel count must be ~pi*(N/2)^2 within 3%."""
    N = 64
    basis = ZernikeBasis(N, 10)
    assert basis.mask.shape == (N, N)
    assert basis.mask.dtype == bool
    assert basis.mask.sum() == pytest.approx(np.pi * (N / 2) ** 2, rel=0.03)


def test_roundtrip():
    """zernike_to_phase then phase_to_zernike must recover the coefficients."""
    rng = np.random.default_rng(0)
    basis = ZernikeBasis(64, 10)
    c = rng.standard_normal(10)
    phase = basis.zernike_to_phase(c)
    fitted = basis.phase_to_zernike(phase)
    assert fitted.shape == (10,)
    assert fitted.dtype == np.float64
    np.testing.assert_allclose(fitted, c, rtol=1e-2)


def test_reconstruct():
    """reconstruct(c) must equal phase[mask] for phase = zernike_to_phase(c)."""
    rng = np.random.default_rng(1)
    basis = ZernikeBasis(64, 10)
    c = rng.standard_normal(10)
    phase = basis.zernike_to_phase(c)
    np.testing.assert_allclose(basis.reconstruct(c), phase[basis.mask], rtol=1e-12)


def test_tip_tilt_separation():
    """Tilt-only phase must fit to dominant index 1; keep_modes preserves it."""
    basis = ZernikeBasis(64, 10)
    tilt = basis.Z[1]  # Noll 2 (tilt) -> aotools row index 1
    coeffs = basis.phase_to_zernike(tilt)
    assert np.argmax(np.abs(coeffs)) == 1
    proj = basis.project_coeffs(coeffs, np.array([0, 1, 2]))
    np.testing.assert_allclose(proj[1], coeffs[1], rtol=1e-12)
    assert proj[3] == 0.0
    assert proj.shape == coeffs.shape