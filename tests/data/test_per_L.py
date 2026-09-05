"""Tests for the per-sample propagation distance ``L`` (CNNL regime).

These are written RED first (they fail against the current constant-``L``
implementation) and then made GREEN by ``data/simulate.py`` gaining a
``SimSampleState`` that recomputes the ``L``-derived quantities (focus phase,
vacuum reference, imaging geometry, bucket, screens) per sample.

The CNNL regime draws a per-sample propagation distance ``L_i ~ U[L_min, L_max]``
deterministically from the sample seed, plus a per-sample Rytov variance
``sigma2_i ~ U[s2_min, s2_max]`` (diComo et al. 2025, CNNL). When
``L_random`` is ``False`` (default, CNN1) every sample is identical to the
legacy constant-``L`` path, so the existing dataset is bit-reproducible.
"""

import numpy as np
import pytest

from data.simulate import (
    SimSample,
    SimSampleState,
    compute_L,
    compute_rytov,
    make_state,
    simulate_sample,
    simulate_sample_fom,
)
from physics.config import SimConfig


def make_cfg(
    N: int = 128,
    n_screens: int = 2,
    screen_sep: float = 500.0,
    L_min: float = 500.0,
    L_max: float = 2600.0,
    L_random: bool = True,
    s2_min: float = 0.1,
    s2_max: float = 2.0,
) -> SimConfig:
    """Small fast config with per-sample L (CNNL) enabled."""
    return SimConfig.from_dict(
        {
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
                "L_min": L_min,
                "L_max": L_max,
                "L_random": L_random,
                "rytov_min": s2_min,
                "rytov_max": s2_max,
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
                "h5_path": "/tmp/test_perL.h5",
            },
        }
    )


# --------------------------------------------------------------------------- #
# Per-sample L / Rytov draws
# --------------------------------------------------------------------------- #
def test_compute_L_deterministic_and_in_range():
    """compute_L(seed, L_min, L_max) is deterministic and in [L_min, L_max]."""
    cfg = make_cfg()
    p = cfg.physical
    L1 = compute_L(1234, p)
    L2 = compute_L(1234, p)
    assert L1 == L2
    assert p.L_min <= L1 <= p.L_max
    # Different seeds -> (almost surely) different L.
    L3 = compute_L(1235, p)
    assert L1 != L3


def test_compute_L_fixed_mode_equals_scalar_L():
    """With L_random=False, compute_L returns the fixed p.L (CNN1 backward-compat)."""
    cfg = make_cfg(L_random=False)
    p = cfg.physical
    assert compute_L(0, p) == p.L
    assert compute_L(99, p) == p.L


def test_compute_rytov_deterministic_and_in_range():
    """compute_rytov is deterministic and in [rytov_min, rytov_max]."""
    cfg = make_cfg()
    p = cfg.physical
    s1 = compute_rytov(777, p)
    s2 = compute_rytov(777, p)
    assert s1 == s2
    assert p.rytov_min <= s1 <= p.rytov_max


# --------------------------------------------------------------------------- #
# SimSampleState
# --------------------------------------------------------------------------- #
def test_make_state_fields_and_shapes():
    """make_state builds the per-L derived fields with correct shapes."""
    cfg = make_cfg()
    shared = _base_shared(cfg)
    L = compute_L(0, cfg.physical)
    st = make_state(cfg, shared, L)
    assert isinstance(st, SimSampleState)
    assert st.L == L
    assert st.phi_focus.shape == (cfg.physical.N, cfg.physical.N)
    assert st.I_vac.shape == (cfg.physical.N, cfg.physical.N)
    assert st.bucket_mask.dtype == bool
    assert st.plane_offsets.shape == (3,)
    # f_obj = 2 zR (not overridden in imaging)
    assert st.f_obj == pytest.approx(2.0 * st.zR_APWS)
    assert st.phi_focus.dtype == np.float64
    assert st.I_vac.dtype == np.float32


def test_make_state_phi_focus_uses_state_L():
    """phi_focus is evaluated with focal = state.L (not the base config focal)."""
    cfg = make_cfg()
    shared = _base_shared(cfg)
    L = 1300.0
    st = make_state(cfg, shared, L)
    k = 2.0 * np.pi / shared.lam
    expected = -k * shared.r2 / (2.0 * L)
    np.testing.assert_allclose(st.phi_focus, expected, rtol=1e-12, atol=0.0)


def test_make_state_screen_count_and_dz_track_L():
    """RED: n_screens and dz derive from state.L, not the fixed p.n_screens.

    Prior to the root-cause fix, make_state hard-coded ``n_screens = int(p.n_screens)``
    (10) while per-sample L spanned [L_min, L_max]. That meant xed split_step
    through 10 screens = 1000 m regardless of L, so a sample drawn at L=2599 m
    was focused at 1000 m and no correction branch could refocus it. This test
    pins n_screens = round(L / screen_sep) and dz = L / n_screens (exactly the
    per-L propagation distance the fresnel legs and imaging backprop use).
    """
    cfg = make_cfg()
    shared = _base_shared(cfg)
    p = cfg.physical
    st = make_state(cfg, shared, 2599.0)
    assert st.n_screens == int(round(2599.0 / p.screen_sep))
    assert st.dz == pytest.approx(2599.0 / st.n_screens)
    st_fixed = make_state(cfg, shared, p.L)
    assert st_fixed.n_screens == p.n_screens


def test_make_state_r0_slab_uses_screen_count():
    """RED: r0_slab = r0_path(L) * n^(3/5) with n = round(L/screen_sep)."""
    cfg = make_cfg()
    shared = _base_shared(cfg)
    p = cfg.physical
    for L in (1006.0, 2599.0):
        st = make_state(cfg, shared, L)
        from data.simulate import compute_r0

        n = int(round(L / p.screen_sep))
        r0_path = compute_r0(shared.lam, float(p.cn2), L)
        assert st.r0_slab == pytest.approx(r0_path * n ** (3.0 / 5.0))


def test_make_state_beacon_spherical_uses_state_L():
    """RED: get(zern) defocus removal must use state.L, not the fixed p.L.

    _beacon_phase_conj removes the converging-spherical phase with radius
    ``shared.L``. Prior to the fix the wrapper used ``float(p.L)`` (fixed 1000 m),
    leaving ~50 rad of residual defocus inside phi_conj for an L=2600 m sample.
    This pins the state's L as the defocus-removal radius.
    """
    cfg = make_cfg()
    shared = _base_shared(cfg)
    st = make_state(cfg, shared, 2599.0)
    assert st.L == 2599.0
    assert st.L == pytest.approx(st.focal)


def test_make_state_bucket_and_imaging_scale_with_L():
    """Larger L -> larger bucket diameter, larger f_obj (zR ~ r0^2 ~ 1/L^{...})."""
    cfg = make_cfg()
    shared = _base_shared(cfg)
    st_small = make_state(cfg, shared, 600.0)
    st_large = make_state(cfg, shared, 2400.0)
    # Bucket diameter px ~ L  (D_bucket = frac*L*lam/Dscope / dx)
    assert st_large.bucket_mask.sum() > st_small.bucket_mask.sum()
    # f_obj = 2 zR, zR = r0(L)^2/(pi lam); r0(L) ~ L^{-3/5}, so zR decreases with L
    assert st_large.f_obj < st_small.f_obj
    # Plane offsets are centered on f_obj: middle plane == f_obj
    assert st_large.plane_offsets[1] == pytest.approx(st_large.f_obj)
    assert st_small.plane_offsets[1] == pytest.approx(st_small.f_obj)


def test_state_reuses_base_propagator_and_grids():
    """make_state does NOT rebuild the expensive Propagator / Zernike / grids."""
    cfg = make_cfg()
    shared = _base_shared(cfg)
    st = make_state(cfg, shared, 1500.0)
    assert st.prop is shared.prop
    assert st.zern is shared.zern
    assert st.E0 is shared.E0
    assert st.G is shared.G
    assert st.r2 is shared.r2


def test_make_state_fixed_L_matches_base():
    """For the default config L == p.L, the state equals the base SharedSim."""
    cfg = make_cfg(L_random=False)
    shared = _base_shared(cfg)
    st = make_state(cfg, shared, L=cfg.physical.L)
    # phi_focus matches the base (base focal == p.L == state L)
    np.testing.assert_allclose(st.phi_focus, shared.phi_focus, rtol=1e-12)
    np.testing.assert_allclose(st.I_vac, shared.I_vac, rtol=1e-5)
    np.testing.assert_allclose(st.plane_offsets, shared.plane_offsets, rtol=1e-12)


# --------------------------------------------------------------------------- #
# simulate_sample threading L
# --------------------------------------------------------------------------- #
def test_simulate_sample_exposes_L_field():
    """SimSample now carries a per-sample L field."""
    cfg = make_cfg()
    s = simulate_sample(0, cfg)
    assert isinstance(s, SimSample)
    assert s.L is not None
    assert cfg.physical.L_min <= s.L <= cfg.physical.L_max


def test_simulate_sample_per_L_deterministic():
    """Two runs of the same seed give identical images, labels, L, and FOMs."""
    cfg = make_cfg()
    s1 = simulate_sample(5, cfg)
    s2 = simulate_sample(5, cfg)
    assert s1.L == s2.L
    np.testing.assert_array_equal(s1.images, s2.images)
    np.testing.assert_array_equal(s1.labels, s2.labels)
    assert s1.fom_noao == s2.fom_noao
    assert s1.fom_beacon == s2.fom_beacon


def test_simulate_sample_per_L_varies_across_seeds():
    """Different seeds give (almost surely) different L and different images."""
    cfg = make_cfg()
    a = simulate_sample(0, cfg)
    b = simulate_sample(1, cfg)
    assert a.L != b.L
    assert not np.array_equal(a.images, b.images)


def test_simulate_sample_fom_accepts_L():
    """simulate_sample_fom threads a per-sample L and returns a valid FOM."""
    cfg = make_cfg()
    coeffs = np.zeros(78)
    L = compute_L(0, cfg.physical)
    fom = simulate_sample_fom(0, cfg, coeffs, L=L)
    assert 0.0 < fom <= 2.0


# --------------------------------------------------------------------------- #
# Per-sample L plumbed through simulate_sample
# --------------------------------------------------------------------------- #
def test_simulate_sample_has_L_field_and_is_in_range():
    """SimSample gains an ``L`` field equal to the per-sample draw, in range."""
    cfg = make_cfg()
    p = cfg.physical
    s = simulate_sample(0, cfg)
    assert isinstance(s, SimSample)
    assert hasattr(s, "L")
    assert p.L_min <= s.L <= p.L_max
    # Deterministic: same seed -> same L
    s2 = simulate_sample(0, cfg)
    assert s.L == s2.L
    # Different seed -> different L (almost surely)
    s3 = simulate_sample(1, cfg)
    assert s.L != s3.L


def test_simulate_sample_fom_matches_ml_leg_with_L():
    """The fast FOM path matches the simulate_sample 'ml' leg under per-L."""
    cfg = make_cfg()
    coeffs = np.random.default_rng(0).standard_normal(78)
    s = simulate_sample(0, cfg, correction_coeffs=coeffs)
    fom_fast = simulate_sample_fom(0, cfg, coeffs, L=s.L)
    assert fom_fast == pytest.approx(s.fom_ml, rel=1e-6)


def test_generate_dataset_writes_per_row_L(tmp_path):
    """generate_dataset writes a per-row L (CNNL) into the H5 file.

    With L_random=True each sample gets its own propagation distance, so the
    stored f["L"] row must match the per-sample simulate_sample(seed) value
    and span the [L_min, L_max] range.
    """
    import h5py

    from data.simulate import generate_dataset

    cfg = make_cfg()
    cfg.data.h5_path = str(tmp_path / "perL.h5")
    p = cfg.physical
    n_total = cfg.data.n_train + cfg.data.n_test + cfg.data.n_eval
    h5_path = generate_dataset(cfg)

    with h5py.File(h5_path, "r") as f:
        L_rows = f["L"][:]
        seeds = f["seeds"][:]
        assert L_rows.shape == (n_total,)
        assert L_rows.dtype == np.float32
        assert np.all(L_rows >= p.L_min - 1e-3)
        assert np.all(L_rows <= p.L_max + 1e-3)
        # Distinct seeds must (almost surely) give distinct L values.
        assert len(np.unique(L_rows)) > 1
        # Each row must match the deterministic per-sample compute_L.
        for i in range(n_total):
            expected = compute_L(int(seeds[i]), p)
            assert float(L_rows[i]) == pytest.approx(expected, rel=1e-5)


def _base_shared(cfg: SimConfig):
    """Build the base SharedSim (legacy path) for make_state tests."""
    from data.simulate import _get_shared

    return _get_shared(cfg)
