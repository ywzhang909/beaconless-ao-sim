"""Tests for data.simulate (Algorithm 1 reproduction from DiComo et al. 2025).

TDD: these tests are written first (RED), then data/simulate.py is implemented
(GREEN). The physics-sanity ordering test uses N=128 because at N=64 the
point-source beacon back-propagation produces a spherical-wavefront phase whose
per-pixel gradient exceeds pi, which breaks the 2D ``np.unwrap`` (a resolution
limit, not a sign error -- the ordering is verified at N=128).
"""

import numpy as np
import pytest

from data.simulate import (
    SimSample,
    bucket_mask_nd,
    physics_from_cfg,
    simulate_sample,
    simulate_sample_fom,
    vacuum_intensity,
)
from physics.config import SimConfig


def make_cfg(N: int = 64, n_screens: int = 2, screen_sep: float = 500.0) -> SimConfig:
    """Build a small, fast configuration for tests (paper-faithful values)."""
    return SimConfig.from_dict({
        "physical": {
            "cn2": 8.13e-15,
            "l0_sim": 0.01,
            "L0": 100.0,
            "L": 1000.0,
            "wavelength": 800e-9,
            "Dscope": 0.30,
            "rspot": 0.075,
            "focal": 1000.0,
            "N": N,
            "box_size": 0.30,
            "n_screens": n_screens,
            "screen_sep": screen_sep,
            "n_roughness": 2,
            "roughness_seed": 42,
            "beam_source": "aotools",
            "screen_pool": 0,
        },
        "imaging": {
            "zR_APWS": None,
            "f_obj": None,
            "plane_offset_frac": [0.0, 1.0, 2.0],
        },
        "bucket": {"diameter_frac": 2.5},
        "data": {
            "n_train": 4,
            "n_test": 2,
            "n_eval": 2,
            "master_seed": 20250830,
            "workers": 2,
            "h5_path": "/tmp/test.h5",
        },
    })


@pytest.fixture(scope="module")
def shared():
    return physics_from_cfg(make_cfg())


def test_physics_from_cfg_tuple(shared):
    """physics_from_cfg returns (Propagator, ZernikeBasis, bucket_mask, dz)."""
    prop, zern, mask, dz = shared
    assert prop.N == 64
    assert zern.n_modes == 78
    assert mask.shape == (64, 64)
    assert mask.dtype == bool
    assert dz == 500.0


def test_simulate_determinism():
    """simulate_sample(seed, cfg) twice -> EXACT identical outputs."""
    cfg = make_cfg()
    s1 = simulate_sample(0, cfg)
    s2 = simulate_sample(0, cfg)
    np.testing.assert_array_equal(s1.images, s2.images)
    np.testing.assert_array_equal(s1.labels, s2.labels)
    np.testing.assert_array_equal(s1.I_vac, s2.I_vac)
    np.testing.assert_array_equal(s1.I_obj_track, s2.I_obj_track)
    np.testing.assert_array_equal(s1.track_slopes, s2.track_slopes)
    assert s1.fom_noao == s2.fom_noao
    assert s1.fom_track == s2.fom_track
    assert s1.fom_beacon == s2.fom_beacon
    assert s1.fom_z78 == s2.fom_z78


def test_screens_deterministic():
    """Screens from the same seed are identical (aotools fallback determinism)."""
    from data.simulate import _make_screens

    cfg = make_cfg()
    shared = physics_from_cfg(cfg)
    a = _make_screens(0, cfg, shared)
    b = _make_screens(0, cfg, shared)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (2, 64, 64)
    assert a.dtype == np.float32
    assert np.all(np.isfinite(a))


def test_shapes_types():
    """images (3,N,N) float32 finite non-negative; labels (78,); foms in (0,2]."""
    cfg = make_cfg()
    N = cfg.physical.N
    s = simulate_sample(0, cfg)
    assert isinstance(s, SimSample)
    assert s.images.shape == (3, N, N)
    assert s.images.dtype == np.float32
    assert np.all(np.isfinite(s.images))
    assert np.all(s.images >= 0)
    assert s.labels.shape == (78,)
    assert s.labels.dtype == np.float64
    assert np.all(np.isfinite(s.labels))
    for f in (s.fom_noao, s.fom_track, s.fom_beacon, s.fom_z78):
        assert 0.0 < f <= 2.0
    assert s.I_vac.shape == (N, N)
    assert s.I_vac.dtype == np.float32
    assert s.I_obj_track.shape == (N, N)
    assert s.I_obj_track.dtype == np.float32
    assert s.track_slopes.shape == (2,)
    assert s.track_slopes.dtype == np.float64
    for ph in (s.phase_conj, s.phase_track, s.phase_beacon, s.phase_z78):
        assert ph.shape == (N, N)
        assert ph.dtype == np.float64
    assert set(s.beam_phases.keys()) == {"noao", "track", "beacon", "z78"}
    assert s.fom_ml is None


def test_fom_ml_filled():
    """Passing correction_coeffs fills fom_ml and adds the 'ml' beam phase."""
    cfg = make_cfg()
    coeffs = np.zeros(78)
    s = simulate_sample(0, cfg, correction_coeffs=coeffs)
    assert s.fom_ml is not None
    assert 0.0 < s.fom_ml <= 2.0
    assert "ml" in s.beam_phases
    assert s.beam_phases["ml"].shape == (64, 64)


def test_physics_sanity_fom_ordering():
    """Median FOM_beacon >= FOM_z78 >= FOM_track >= FOM_noao (sign check).

    Uses N=128 so the point-source beacon phase unwraps correctly (see module
    docstring). If this ordering fails the phase sign convention is wrong.
    """
    cfg = make_cfg(N=128)
    fom_noao, fom_track, fom_beacon, fom_z78 = [], [], [], []
    for seed in range(4):
        s = simulate_sample(seed, cfg)
        fom_noao.append(s.fom_noao)
        fom_track.append(s.fom_track)
        fom_beacon.append(s.fom_beacon)
        fom_z78.append(s.fom_z78)
    assert np.median(fom_beacon) >= np.median(fom_z78)
    assert np.median(fom_z78) >= np.median(fom_track)
    assert np.median(fom_track) >= np.median(fom_noao)


def test_bucket_mask_diameter():
    """diameter_px ~ 11.4 for the full N=512 config; mask matches pixel count."""
    N = 512
    box = 0.30
    lam = 800e-9
    L = 1000.0
    Dscope = 0.30
    frac = 2.5
    D_bucket = frac * L * lam / Dscope
    diameter_px = D_bucket / (box / N)
    assert diameter_px == pytest.approx(11.4, rel=0.05)
    mask = bucket_mask_nd(diameter_px, N)
    assert mask.shape == (N, N)
    assert mask.dtype == bool
    y, x = np.mgrid[0:N, 0:N]
    xc = yc = (N - 1) / 2.0
    r2 = (x - xc) ** 2 + (y - yc) ** 2
    expected = np.sum(r2 <= (diameter_px / 2.0) ** 2)
    assert mask.sum() == expected


def test_vacuum_intensity():
    """|propagate|^2 of the focused Gaussian in vacuum peaks near center, finite."""
    cfg = make_cfg()
    N = cfg.physical.N
    I_vac = vacuum_intensity(cfg)
    assert I_vac.shape == (N, N)
    assert I_vac.dtype == np.float32
    assert np.all(np.isfinite(I_vac))
    assert np.all(I_vac >= 0)
    assert np.isfinite(I_vac.sum())
    c = N // 2
    assert I_vac[c, c] == I_vac.max()


def test_simulate_sample_fom_matches_ml_leg():
    """simulate_sample_fom reproduces the ml FOM of simulate_sample."""
    cfg = make_cfg()
    coeffs = np.random.default_rng(0).standard_normal(78)
    s = simulate_sample(0, cfg, correction_coeffs=coeffs)
    fom_fast = simulate_sample_fom(0, cfg, coeffs)
    assert fom_fast == pytest.approx(s.fom_ml, rel=1e-6)
