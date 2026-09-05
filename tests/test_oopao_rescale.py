"""Regression tests for the OOPAO reference-screen wavelength rescale.

Locks the correction of the per-layer amplitude rescale: OOPAO generates
layers at ``_LAM_REF_500`` (500 nm), the simulation applies them at ``lam``,
and the old ``(r0_slab/r0_ref)**(5/6)`` formula ignored both the wavelength
conversion and the r0-ratio direction. ``_rescale_for`` must be monotonic
decreasing in ``r0_slab`` (larger r0 => weaker turbulence) and must recover
the theory per-slab phase std after the ``lam_ref/lam`` conversion.
"""

import math

import numpy as np
import pytest

from physics.oopao_backend import _CAL_REF, _LAM_REF_500, _rescale_for

D = 0.30
RAW_STD_REF = _CAL_REF * (D / 0.15) ** (5.0 / 6.0)  # ~1.103 rad @ 500 nm


def theory_std(r0_slab: float) -> float:
    return math.sqrt(1.03) * (D / r0_slab) ** (5.0 / 6.0)


def test_rescale_applied_phase_matches_theory():
    """M * raw(500nm) * (lam_ref/lam) equals the Kolmogorov per-slab std."""
    lam = 800e-9
    for r0_slab in (0.16, 0.20, 0.25):
        applied = _rescale_for(r0_slab, lam) * RAW_STD_REF * (_LAM_REF_500 / lam)
        assert applied == pytest.approx(theory_std(r0_slab), rel=1e-9)


def test_rescale_is_monotonic_decreasing_in_r0_slab():
    """Larger r0_slab (weaker turbulence) must give a smaller multiplier."""
    lam = 800e-9
    lo = _rescale_for(0.16, lam)
    hi = _rescale_for(0.25, lam)
    assert hi < lo
    assert abs((lo / hi) - (0.25 / 0.16) ** (5.0 / 6.0)) < 1e-12


def test_wavelength_conversion_factor():
    """Rescale scales linearly with lam / lam_ref (phase ~ 1/lam)."""
    base = _rescale_for(0.16, 500e-9)
    at_800 = _rescale_for(0.16, 800e-9)
    assert at_800 / base == pytest.approx(800e-9 / 500e-9)
    assert at_800 / base == pytest.approx(1.6)