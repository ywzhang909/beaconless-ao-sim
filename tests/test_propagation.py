"""Tests for physics.propagation_fft.Propagator."""

import numpy as np
import pytest

from physics.propagation_fft import Propagator, rayleigh_range

N = 64
DX = 0.01
LAM = 800e-9


@pytest.fixture(scope="module")
def prop():
    return Propagator(N=N, dx=DX, lam=LAM)


def test_propagate_power_conservation(prop):
    rng = np.random.default_rng(0)
    E = (rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))).astype(
        np.complex64
    )
    P_in = np.sum(np.abs(E) ** 2)
    E2 = prop.propagate(E, 50.0)
    P_out = np.sum(np.abs(E2) ** 2)
    assert abs(P_in - P_out) / P_in < 0.02


def test_split_step_no_screens_equals_propagate(prop):
    rng = np.random.default_rng(1)
    E = (rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))).astype(
        np.complex64
    )
    screens = [np.zeros((N, N), dtype=np.float32)] * 3
    E_split = prop.split_step(E, screens, dz=10.0)
    E_direct = prop.propagate(E, 30.0)
    rel = np.linalg.norm(E_split - E_direct) / np.linalg.norm(E_direct)
    assert rel < 1e-3


def test_conservation_imaging(prop):
    aperture = np.zeros((N, N), dtype=np.float32)
    aperture[N // 4 : 3 * N // 4, N // 4 : 3 * N // 4] = 1.0
    E = aperture.astype(np.complex64)
    I = prop.angular_spectrum_intensity(E, 500.0)
    P_in = np.sum(np.abs(E) ** 2)
    P_out = np.sum(I)
    assert abs(P_in - P_out) / P_in < 0.02


def test_rayleigh():
    w0 = 0.075
    lam = 800e-9
    expected = np.pi * w0**2 / lam
    got = rayleigh_range(w0, lam)
    assert abs(got - expected) / expected < 0.01


def test_dtype_outputs(prop):
    E = np.ones((N, N), dtype=np.complex64)
    E2 = prop.propagate(E, 10.0)
    assert E2.dtype == np.complex64
    I = prop.angular_spectrum_intensity(E, 10.0)
    assert I.dtype == np.float32
