"""Tests for the physics.engine / physics.measurement abstraction.

Covers the two seam interfaces behind the pipeline:
- ``physics.engine.PhysicsEngine`` (physics forward model)
- ``physics.engine.MeasurementSource`` (camera-images seam)

and their concrete implementations:
- ``data.simulate.SimulatedPhysicsEngine`` (the existing finite-difference sim)
- ``data.simulate.SimulatedMeasurementSource`` (rough-surface imaging)
- ``physics.engine.HardwareMeasurementSource`` (pre-acquired camera frames)
"""

import numpy as np
import pytest

from data.simulate import (
    SharedSim,
    SimulatedMeasurementSource,
    SimulatedPhysicsEngine,
    _get_shared,
    physics_from_cfg,
    simulate_sample,
)
from physics.engine import (
    HardwareMeasurementSource,
    MeasurementSource,
    PhysicsEngine,
)
from physics.config import SimConfig


def make_cfg(N: int = 64, n_screens: int = 2, screen_sep: float = 500.0) -> SimConfig:
    """Small, fast configuration matching tests/data/test_simulate.py."""
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
            "h5_path": "/tmp/test_engine.h5",
        },
    })


def test_engine_is_abstract():
    """PhysicsEngine / MeasurementSource are abstract (cannot be instantiated)."""
    with pytest.raises(TypeError):
        PhysicsEngine()  # type: ignore[abstract]
    with pytest.raises(TypeError):
        MeasurementSource()  # type: ignore[abstract]


def test_simulated_engine_implements_abc_methods():
    """SimulatedPhysicsEngine implements every PhysicsEngine ABC method."""
    for m in (
        "make_screens",
        "beacon_phase_conj",
        "track",
        "forward_fom",
        "phase_to_zernike",
        "zernike_to_phase",
    ):
        assert callable(getattr(SimulatedPhysicsEngine, m)), m


def test_simulated_engine_exposes_shared_state():
    """The engine's read-only state matches the underlying SharedSim."""
    cfg = make_cfg()
    shared = _get_shared(cfg)
    engine = SimulatedPhysicsEngine(cfg, shared)
    assert engine.N == shared.N
    assert engine.dx == shared.dx
    assert engine.lam == shared.lam
    assert engine.k == shared.k
    assert engine.dz == shared.dz
    assert engine.n_screens == 2
    assert engine.cn2 == pytest.approx(8.13e-15)
    assert engine.L0 == pytest.approx(100.0)
    assert engine.l0_sim == pytest.approx(0.01)
    for attr in (
        "pupil",
        "E0",
        "phi_focus",
        "I_vac",
        "r2",
        "X",
        "Y",
        "G",
        "zern",
        "f_obj",
        "plane_offsets",
    ):
        a = getattr(engine, attr)
        b = getattr(shared, attr)
        if isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b)
        else:
            assert a == b


def test_engine_default_equals_legacy_shared():
    """simulate_sample(engine=engine) matches simulate_sample(shared=shared)."""
    cfg = make_cfg()
    s_legacy = simulate_sample(0, cfg, shared=_get_shared(cfg))
    s_engine = simulate_sample(0, cfg, engine=SimulatedPhysicsEngine(cfg))
    np.testing.assert_array_equal(s_engine.images, s_legacy.images)
    np.testing.assert_array_equal(s_engine.labels, s_legacy.labels)
    np.testing.assert_array_equal(s_engine.I_obj_track, s_legacy.I_obj_track)
    for f in ("fom_noao", "fom_track", "fom_beacon", "fom_z78"):
        assert getattr(s_engine, f) == getattr(s_legacy, f)


def test_engine_from_shared_adapter():
    """SimulatedPhysicsEngine(cfg, shared) matches the default engine."""
    cfg = make_cfg()
    s_default = simulate_sample(0, cfg)
    s_adapter = simulate_sample(
        0, cfg, engine=SimulatedPhysicsEngine(cfg, physics_from_cfg(cfg))
    )
    np.testing.assert_array_equal(s_adapter.images, s_default.images)
    assert s_adapter.fom_beacon == s_default.fom_beacon


def test_simulated_measurement_source_acquires():
    """SimulatedMeasurementSource.acquire returns (3, N, N) images + I_obj."""
    cfg = make_cfg()
    N = cfg.physical.N
    engine = SimulatedPhysicsEngine(cfg)
    src = SimulatedMeasurementSource(engine, cfg)
    screens = engine.make_screens(0)
    phi_conj, _ = engine.beacon_phase_conj(0, screens)
    phi_track, _ = engine.track(phi_conj)
    images, I_obj = src.acquire(
        seed=0, sample_index=0, screens=screens, phi_track=phi_track
    )
    assert images.shape == (3, N, N)
    assert images.dtype == np.float32
    assert np.all(np.isfinite(images))
    assert I_obj is not None
    assert I_obj.shape == (N, N)


def test_hardware_source_reuses_frames():
    """HardwareMeasurementSource returns the provided frames, resampled + None I_obj."""
    N = 64
    rng = np.random.default_rng(7)
    frames = rng.random((3, N, N), dtype=np.float64).astype(np.float32) * 100.0
    src = HardwareMeasurementSource(frames, target_N=N)
    images, I_obj = src.acquire(
        seed=0, sample_index=0, screens=np.zeros((2, N, N)), phi_track=np.zeros((N, N))
    )
    assert images.shape == (3, N, N)
    assert images.dtype == np.float32
    np.testing.assert_allclose(images, frames, rtol=1e-6)
    assert I_obj is None


def test_hardware_source_resamples_smaller():
    """A smaller camera frame is centre-padded to the target grid."""
    N = 64
    rng = np.random.default_rng(7)
    frames = rng.random((3, 32, 32), dtype=np.float64).astype(np.float32)
    src = HardwareMeasurementSource(frames, target_N=N)
    images, _ = src.acquire(seed=0, sample_index=0, screens=None, phi_track=None)
    assert images.shape == (3, N, N)
    # centre-pad: the 32x32 block sits at the centre of the 64x64 grid
    assert images[:, 16:48, 16:48].mean() > 0.0
    assert images[:, :16, :16].mean() == 0.0


def test_hardware_source_single_frame_repeats():
    """A single (N, N) frame is repeated across the 3 planes."""
    N = 64
    rng = np.random.default_rng(3)
    frame = rng.random((N, N), dtype=np.float64).astype(np.float32)
    src = HardwareMeasurementSource(frame, target_N=N)
    images, I_obj = src.acquire(seed=0, sample_index=0, screens=None, phi_track=None)
    assert images.shape == (3, N, N)
    np.testing.assert_array_equal(images[0], images[1])
    np.testing.assert_array_equal(images[1], images[2])
    assert I_obj is None


def test_hardware_source_rejects_bad_shape():
    """Frame counts other than 1/3 planes raise ValueError."""
    N = 64
    with pytest.raises(ValueError):
        HardwareMeasurementSource(np.zeros((4, N, N)), target_N=N)
    with pytest.raises(ValueError):
        HardwareMeasurementSource(
            np.zeros((N, N)), target_N=N, repeat_single=False
        )


def test_engine_injected_sample_matches():
    """simulate_sample with a hardware measurement returns the camera frames."""
    cfg = make_cfg()
    N = cfg.physical.N
    rng = np.random.default_rng(1)
    frames = rng.random((3, N, N), dtype=np.float64).astype(np.float32)
    hw = HardwareMeasurementSource(frames, target_N=N)
    engine = SimulatedPhysicsEngine(cfg)
    s = simulate_sample(0, cfg, engine=engine, measurement=hw)
    # Physics labels / FOMs are unchanged; images come from the camera.
    np.testing.assert_allclose(s.images, frames, rtol=1e-6)
    assert s.fom_beacon > 0.0
    assert s.I_obj_track is None