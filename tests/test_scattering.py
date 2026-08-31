"""Tests for physics.scattering.

TDD: these tests are written first (RED), then the module is implemented (GREEN).
"""

import numpy as np
import pytest

from physics.scattering import LambertianScatterer, random_roughness_phase


class StubPropagator:
    """Stub propagator whose split_step returns the field unchanged."""

    def split_step(self, E, screens_back, dz):
        return E


def test_scatter_energy():
    """Averaging random-phase realizations cancels speckle (modulus << 1).

    The mean of ``n_roughness`` unit-magnitude random-phase vectors has
    expected modulus ``sqrt(pi / (4 * n_roughness))`` ~ 0.4 for n=5, i.e. the
    incoherent average does NOT keep modulus ~1 (random phases cancel).
    """
    N = 64
    I = np.ones((N, N))
    stub = StubPropagator()
    E_back = LambertianScatterer(n_roughness=5).scatter_and_backprop(
        I, stub, [], 1.0
    )
    assert E_back.shape == (N, N)
    assert np.iscomplexobj(E_back)
    assert E_back.dtype == np.complex128
    mean_mod = np.abs(E_back).mean()
    assert 0.1 < mean_mod < 0.6


def test_phase_uniform():
    """random_roughness_phase is uniform in [0, 2pi) with expected std."""
    phi = random_roughness_phase((256, 256), seed=5)
    assert phi.shape == (256, 256)
    assert phi.dtype == np.float32
    assert np.all(phi >= 0.0)
    assert np.all(phi < 2 * np.pi)
    expected_std = np.pi / np.sqrt(3)
    assert phi.std() == pytest.approx(expected_std, rel=0.05)
    # deterministic with the same seed
    phi2 = random_roughness_phase((256, 256), seed=5)
    assert np.array_equal(phi, phi2)


def test_quantize_max():
    """quantize12 maps [0, 0.5, 1.0] to [0, 1024, 2047] as uint16."""
    out = LambertianScatterer.quantize12(np.array([0.0, 0.5, 1.0]), 1.0)
    assert out.dtype == np.uint16
    assert np.array_equal(out, np.array([0, 1024, 2047], dtype=np.uint16))


def test_quantize_clip():
    """Values above max_val clip to the top of the 12-bit range."""
    out = LambertianScatterer.quantize12(np.array([5.0]), 1.0)
    assert out.dtype == np.uint16
    assert out[0] == 2047


def test_quantize_zero_guard():
    """max_val <= 0 raises ValueError."""
    with pytest.raises(ValueError):
        LambertianScatterer.quantize12(np.ones((4, 4)), 0.0)
