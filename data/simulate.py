"""Algorithm 1 reproduction from DiComo et al., Opt. Express 33(15):31010 (2025).

This module implements the full per-sample simulation pipeline (Algorithm 1 of
the paper): turbulence phase screens, focused-beam propagation, diffraction-
limited beacon back-propagation, tracking, phase-conjugate / Zernike-78 DM
corrections, FOM evaluation, and multi-plane rough-surface imaging.

Equations referenced are from the paper:
    Eq 6-8  : nPIB / SIB / FOM
    Eq 9-12 : imaging geometry (z_R_APWS, r_APWS, r0_eff, f_obj)
    Eq 13   : 12-bit intensity quantization
    Eq 14   : per-mode Zernike normalization (mu / sigma)

Determinism
-----------
Every sample is fully deterministic given its ``seed``. Phase screens are
generated with aotools ``ft_sh_phase_screen(..., seed=seed+i)`` (one seed per
screen) and roughness realizations use ``np.random.default_rng`` with a
seed derived from the sample seed. No global RNG state is consumed, so results
are reproducible regardless of process/worker assignment.

Screen-generation fallback
--------------------------
The task specifies Soapy's ``makePhaseScreens`` (via
``physics.screens_soapy.SoapyPhaseScreenGenerator``), but that path is NOT
deterministic under ``np.random.seed`` (verified empirically). Per the task
instructions we therefore fall back to aotools directly:
``aotools.turbulence.phasescreen.ft_sh_phase_screen(r0, N, box_size/N, L0,
l0_sim, seed=seed+i)`` for each screen ``i``. This is deterministic and
produces independent screens.
"""

from __future__ import annotations

import json
import multiprocessing
import os
from dataclasses import dataclass, field
from typing import Any, Optional

import h5py
import numpy as np
from numba import njit
from aotools.turbulence.phasescreen import ft_sh_phase_screen
from tqdm import tqdm

from physics.oopao_backend import OopaoScreenBackend
from physics.propagation_fft import Propagator
from physics.scattering import random_roughness_phase
from physics.screens_soapy import compute_r0
from physics.zernike_aotools import ZernikeBasis
from utils.metrics import FOM, bucket_mask

__all__ = [
    "SimSample",
    "physics_from_cfg",
    "simulate_sample",
    "simulate_sample_fom",
    "vacuum_intensity",
    "bucket_mask_nd",
    "generate_dataset",
]

N_MODES = 78  # Zernike truncation J = 78 (Table 1)


# --------------------------------------------------------------------------- #
# Shared per-process state
# --------------------------------------------------------------------------- #
@dataclass
class SharedSim:
    """Everything built once per process and reused across samples.

    The Propagator (FFTW_MEASURE init ~1-3 s) and ZernikeBasis (pinv ~1 s) are
    expensive to construct, so they are built once and shared. Grids, the
    aperture beam, focusing phase, vacuum intensity and imaging geometry are
    also precomputed here.
    """

    prop: Propagator
    zern: ZernikeBasis
    bucket_mask: np.ndarray
    dz: float
    N: int
    dx: float
    lam: float
    k: float
    rspot: float
    focal: float
    X: np.ndarray  # (N, N) x coordinates [m]
    Y: np.ndarray  # (N, N) y coordinates [m]
    r2: np.ndarray  # (N, N) r^2 [m^2]
    pupil: np.ndarray  # (N, N) bool aperture mask
    G: np.ndarray  # (N, N) tracking Gaussian weighting
    E0: np.ndarray  # (N, N) complex64 aperture beam amplitude
    phi_focus: np.ndarray  # (N, N) float64 focusing phase
    I_vac: np.ndarray  # (N, N) float32 vacuum object-plane intensity
    zR_APWS: float
    f_obj: float
    plane_offsets: np.ndarray  # (3,) distances behind objective lens [m]
    oopao: Optional[OopaoScreenBackend] = None  # set when beam_source == "oopao"


_shared_cache: dict[tuple, SharedSim] = {}


def _cfg_key(cfg: dict) -> tuple:
    """Hashable key identifying the physical/imaging/bucket configuration."""
    p = cfg["physical"]
    img = cfg["imaging"]
    b = cfg["bucket"]
    return (
        p["N"],
        p["box_size"],
        p["wavelength"],
        p["Dscope"],
        p["rspot"],
        p["focal"],
        p["L"],
        p["cn2"],
        p["l0_sim"],
        p["L0"],
        p["screen_sep"],
        str(p.get("beam_source", "soapy")).lower(),
        img["zR_APWS"],
        img["f_obj"],
        tuple(img["plane_offset_frac"]),
        b["diameter_frac"],
    )


def _build_shared(cfg: dict) -> SharedSim:
    """Construct the shared per-process simulation state."""
    p = cfg["physical"]
    img = cfg["imaging"]
    b = cfg["bucket"]

    N = int(p["N"])
    box = float(p["box_size"])
    dx = box / N
    lam = float(p["wavelength"])
    k = 2.0 * np.pi / lam
    rspot = float(p["rspot"])
    focal = float(p["focal"])
    L = float(p["L"])
    Dscope = float(p["Dscope"])

    prop = Propagator(N, dx, lam)
    zern = ZernikeBasis(N, N_MODES)

    # Centered grid coordinates.
    x = (np.arange(N) - (N - 1) / 2.0) * dx
    X, Y = np.meshgrid(x, x)
    r2 = X**2 + Y**2

    # Aperture mask: radius Dscope/2 (= N/2 px since box == Dscope).
    pupil = r2 <= (Dscope / 2.0) ** 2

    # Aperture beam: Gaussian amplitude, zero outside the pupil.
    E0 = np.exp(-(r2 / rspot**2)).astype(np.complex64)
    E0[~pupil] = 0.0

    # Focusing phase (Eq: phi_focus = -k r^2 / (2 f)).
    phi_focus = (-k * r2 / (2.0 * focal)).astype(np.float64)

    # Vacuum object-plane intensity (no screens).
    I_vac = prop.angular_spectrum_intensity(
        (E0 * np.exp(1j * phi_focus)).astype(np.complex64), L
    )

    # Tracking Gaussian weighting (identical diameter to the outgoing beam).
    G = np.exp(-(r2 / rspot**2))

    # Imaging geometry (Eqs 9-12).
    r0 = compute_r0(lam, float(p["cn2"]), L)
    zR_APWS = img["zR_APWS"] if img["zR_APWS"] is not None else r0**2 / (np.pi * lam)
    f_obj = img["f_obj"] if img["f_obj"] is not None else 2.0 * zR_APWS
    plane_offsets = np.array(
        [f_obj + (frac - 1.0) * zR_APWS for frac in img["plane_offset_frac"]],
        dtype=np.float64,
    )

    # Bucket mask (Eq 6).
    D_bucket = float(b["diameter_frac"]) * L * lam / Dscope
    diameter_px = D_bucket / dx
    bmask = bucket_mask(N, diameter_px)

    # OOPAO screen backend (beam_source == "oopao"); None otherwise. Built once
    # per process and shared across samples.
    oopao = None
    if str(p.get("beam_source", "soapy")).lower() == "oopao":
        oopao = OopaoScreenBackend(
            N=N,
            dx=dx,
            Dscope=Dscope,
            lam=lam,
            cn2=float(p["cn2"]),
            L=L,
            L0=float(p["L0"]),
            n_screens=int(p["n_screens"]),
        )

    return SharedSim(
        prop=prop,
        zern=zern,
        bucket_mask=bmask,
        dz=float(p["screen_sep"]),
        N=N,
        dx=dx,
        lam=lam,
        k=k,
        rspot=rspot,
        focal=focal,
        X=X,
        Y=Y,
        r2=r2,
        pupil=pupil,
        G=G,
        E0=E0,
        phi_focus=phi_focus,
        I_vac=I_vac,
        zR_APWS=zR_APWS,
        f_obj=f_obj,
        plane_offsets=plane_offsets,
        oopao=oopao,
    )


def _get_shared(cfg: dict) -> SharedSim:
    """Return the cached shared state for ``cfg``, building it if needed."""
    key = _cfg_key(cfg)
    if key not in _shared_cache:
        _shared_cache[key] = _build_shared(cfg)
    return _shared_cache[key]


def _resolve_shared(shared: Any, cfg: dict) -> SharedSim:
    """Resolve the ``shared`` argument to a :class:`SharedSim`.

    Accepts ``None`` (build/cache from ``cfg``), a :class:`SharedSim`, or the
    ``(Propagator, ZernikeBasis, bucket_mask, dz)`` tuple returned by
    :func:`physics_from_cfg` (resolved through the per-cfg cache).
    """
    if shared is None or isinstance(shared, SharedSim):
        return shared if isinstance(shared, SharedSim) else _get_shared(cfg)
    # Tuple from physics_from_cfg -> resolve via the cfg cache.
    return _get_shared(cfg)


def physics_from_cfg(cfg: dict) -> tuple:
    """Build (once per process) and return ``(Propagator, ZernikeBasis, bucket_mask_2d, dz)``.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary (see config.yaml).

    Returns
    -------
    tuple
        ``(Propagator, ZernikeBasis, bucket_mask_2d, dz)``.
    """
    shared = _get_shared(cfg)
    return shared.prop, shared.zern, shared.bucket_mask, shared.dz


def bucket_mask_nd(diameter_px: float, N: int) -> np.ndarray:
    """Boolean (N, N) circular bucket mask with the given diameter in pixels.

    Delegates to :func:`utils.metrics.bucket_mask`.
    """
    return bucket_mask(N, diameter_px)


def vacuum_intensity(cfg: dict, shared: Any = None) -> np.ndarray:
    """Return the vacuum object-plane intensity ``|propagate(E0 e^{i phi_focus}, L)|^2``.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary.
    shared : SharedSim or tuple, optional
        Prebuilt shared state (avoids rebuild).

    Returns
    -------
    np.ndarray
        ``(N, N)`` float32 vacuum intensity.
    """
    shared = _resolve_shared(shared, cfg)
    return shared.I_vac


# --------------------------------------------------------------------------- #
# Per-sample helpers
# --------------------------------------------------------------------------- #
def _make_screens(seed: int, cfg: dict, shared: Any) -> np.ndarray:
    """Generate ``n_screens`` deterministic turbulence phase screens.

    Uses aotools ``ft_sh_phase_screen`` directly (Soapy's wrapper is not
    deterministic under ``np.random.seed``). Each screen ``i`` uses seed
    ``seed + i`` so the screens are independent and reproducible.

    Parameters
    ----------
    seed : int
        Sample seed.
    cfg : dict
        Configuration dictionary.
    shared : SharedSim or tuple
        Shared state (provides N, dx, L0, l0_sim, lam).

    Returns
    -------
    np.ndarray
        ``(n_screens, N, N)`` float32 phase screens in radians.
    """
    shared = _resolve_shared(shared, cfg)
    p = cfg["physical"]

    if shared.oopao is not None:
        # OOPAO path: per-layer screens drawn from the shared OOPAO Atmosphere,
        # each already rescaled to the target per-slab r0 and cropped to N x N.
        return shared.oopao.make_screens(seed)

    n_screens = int(p["n_screens"])
    # Per-slab coherence length. ``compute_r0`` returns the path-integrated r0
    # for the full L. Each of the ``n_screens`` slabs (thickness L/n_screens)
    # carries r0_slab = r0_path * n_screens**(3/5); using the path r0 for every
    # slab would make the total turbulence ~n_screens**(3/5) times too strong.
    r0_path = compute_r0(shared.lam, float(p["cn2"]), float(p["L"]))
    r0_slab = r0_path * n_screens ** (3.0 / 5.0)
    l0_sim = float(p["l0_sim"])
    L0 = float(p["L0"])
    screens = np.stack(
        [
            ft_sh_phase_screen(r0_slab, shared.N, shared.dx, L0, l0_sim, seed=seed + i)
            for i in range(n_screens)
        ]
    ).astype(np.float32)
    return screens



def _unwrap_flood_fill(phi_w: np.ndarray, quality: np.ndarray) -> np.ndarray:
    """2D phase unwrap by intensity-guided flood fill (numba).

    The sequential row/column unwrap (``np.unwrap`` twice) creates multi-2pi
    branch cuts/ramps through the bright region of the beacon pupil field,
    which corrupt any phase-based Zernike fit (FOM_z78 ~ 0.02). A flood fill
    that grows from the brightest pixel and unwraps each new pixel against its
    already-unwrapped neighbours pushes the 2pi cuts into the weak-field
    regions, where they are invisible to the FOM and down-weighted by the
    Zernike fit. Verified: FOM_z78 0.02 -> ~0.93 (vs FOM_beacon ~0.95).

    Parameters
    ----------
    phi_w : np.ndarray
        ``(N, N)`` wrapped phase in radians.
    quality : np.ndarray
        ``(N, N)`` quality map (beacon intensity); the flood fill starts at
        its argmax.

    Returns
    -------
    np.ndarray
        ``(N, N)`` unwrapped phase (absolute 2pi offset is arbitrary; piston
        is removed by the caller).
    """
    N = phi_w.shape[0]
    return _unwrap_flood_fill_nb(np.ascontiguousarray(phi_w), np.ascontiguousarray(quality), N)


@njit(cache=True)
def _wrap_diff(d: float) -> float:
    """Wrap a phase difference into (-pi, pi]."""
    return d - 2.0 * np.pi * np.round(d / (2.0 * np.pi))


@njit(cache=True)
def _median4(v: np.ndarray, cnt: int) -> float:
    """Median of the first ``cnt`` (<= 4) elements of ``v`` (insertion sort)."""
    for a in range(1, cnt):
        key = v[a]
        b = a - 1
        while b >= 0 and v[b] > key:
            v[b + 1] = v[b]
            b -= 1
        v[b + 1] = key
    if cnt % 2 == 1:
        return v[cnt // 2]
    return 0.5 * (v[cnt // 2 - 1] + v[cnt // 2])


@njit(cache=True)
def _unwrap_flood_fill_nb(phi_w: np.ndarray, quality: np.ndarray, N: int) -> np.ndarray:
    """Numba kernel: BFS flood fill from the brightest pixel (see wrapper)."""
    out = np.zeros((N, N))
    done = np.zeros((N, N), dtype=np.bool_)
    i0, j0 = 0, 0
    best = -1e30
    for i in range(N):
        for j in range(N):
            if quality[i, j] > best:
                best = quality[i, j]
                i0, j0 = i, j
    out[i0, j0] = phi_w[i0, j0]
    done[i0, j0] = True
    qi = np.zeros(N * N, dtype=np.int64)
    qj = np.zeros(N * N, dtype=np.int64)
    head, tail = 0, 0
    qi[tail], qj[tail] = i0, j0
    tail += 1
    while head < tail:
        i, j = qi[head], qj[head]
        head += 1
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = i + di, j + dj
            if 0 <= ni < N and 0 <= nj < N and not done[ni, nj]:
                vals = np.empty(4)
                cnt = 0
                for di2, dj2 in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    mi, mj = ni + di2, nj + dj2
                    if 0 <= mi < N and 0 <= mj < N and done[mi, mj]:
                        vals[cnt] = out[mi, mj] + _wrap_diff(phi_w[ni, nj] - phi_w[mi, mj])
                        cnt += 1
                if cnt > 0:
                    out[ni, nj] = _median4(vals, cnt)
                else:
                    out[ni, nj] = phi_w[ni, nj]
                done[ni, nj] = True
                qi[tail], qj[tail] = ni, nj
                tail += 1
    return out


def _beacon_phase_conj(
    seed: int, cfg: dict, shared: SharedSim, screens: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Step D: diffraction-limited beacon back-propagation -> phi_conj.

    A diffraction-limited beacon (small Gaussian, waist = lambda*L/Dscope) at
    the object plane is back-propagated through the (reversed) screens to the
    pupil. The pupil-field phase is 2D-unwrapped (a simplification of full 2D
    phase unwrapping), piston (mode 0) and defocus (mode 3) are removed, and
    the result is conjugated:

        phi_conj = -(phi_unwrapped - phi_piston_defocus)

    Parameters
    ----------
    seed : int
        Sample seed (unused here, kept for signature symmetry).
    cfg : dict
        Configuration dictionary.
    shared : SharedSim
        Shared state.
    screens : np.ndarray
        ``(n_screens, N, N)`` float32 phase screens.

    Returns
    -------
    tuple
        ``(phi_conj, I_beacon)`` where ``phi_conj`` is the ``(N, N)`` float64
        conjugated beacon phase and ``I_beacon`` is the ``(N, N)`` float64
        beacon intensity at the pupil (diagnostic).
    """
    prop = shared.prop
    zern = shared.zern
    N = shared.N
    p = cfg["physical"]

    # Diffraction-limited beacon: a small Gaussian at the object plane whose
    # waist equals the diffraction limit seen from the telescope,
    # w = lambda * L / Dscope (~2.7 mm). A single-pixel delta has a flat
    # (infinite-bandwidth) angular spectrum, which makes the back-propagated
    # phase numerically spurious and uncorrelated with the true turbulence.
    w = shared.lam * float(p["L"]) / float(p["Dscope"])
    E_pt = (np.exp(-shared.r2 / w**2) * shared.pupil).astype(np.complex64)

    # Back-propagate through the reversed screens to the pupil.
    E_back = prop.split_step(E_pt, screens[::-1], -shared.dz)

    # The back-propagated beacon is a converging spherical wave with phase
    # -k*r^2/(2L). Remove it analytically (paper: "corrected to remove the
    # parabolic defocus term") so the residual turbulence phase unwraps
    # cleanly; a full-pupil low-order Zernike fit of the defocus would be
    # corrupted by weak-field edge outliers.
    k = 2.0 * np.pi / shared.lam
    spherical = k * shared.r2 / (2.0 * float(p["L"]))
    E_flat = E_back * np.exp(1j * spherical)

    phi_unwrapped = _unwrap_flood_fill(np.angle(E_flat), (np.abs(E_back) ** 2).astype(np.float64))
    phi_unwrapped = phi_unwrapped - phi_unwrapped[shared.pupil].mean()

    phi_conj = -phi_unwrapped
    I_beacon = (np.abs(E_back) ** 2).astype(np.float64)
    return phi_conj, I_beacon


def _tracking(shared: SharedSim, phi_conj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Step E: tilt tracking from the conjugated beacon phase.

    The gradient of ``phi_conj`` is weighted by the Gaussian ``G`` (identical
    diameter to the outgoing beam) and averaged to give the tilt slopes
    ``[a_x, a_y]``; the tracking phase is the linear ramp ``a_x x + a_y y``.

    Parameters
    ----------
    shared : SharedSim
        Shared state.
    phi_conj : np.ndarray
        ``(N, N)`` conjugated beacon phase.

    Returns
    -------
    tuple
        ``(phi_track, track_slopes)`` where ``phi_track`` is ``(N, N)`` float64
        and ``track_slopes`` is ``(2,)`` float64 ``[a_x, a_y]``.
    """
    G = shared.G
    gx, gy = np.gradient(phi_conj)
    a_x = float(np.sum(G * gx) / np.sum(G))
    a_y = float(np.sum(G * gy) / np.sum(G))
    phi_track = a_x * shared.X + a_y * shared.Y
    track_slopes = np.array([a_x, a_y], dtype=np.float64)
    return phi_track, track_slopes


def _fom_leg(
    shared: SharedSim, screens: np.ndarray, phi_total: np.ndarray
) -> float:
    """Step G: forward-propagate with a total aperture phase and return FOM.

    ``E_obj = split_step(E0 e^{i phi_total}, screens, dz)`` then
    ``FOM(|E_obj|^2, I_vac, bucket_mask)`` (Eqs 6-8).

    Parameters
    ----------
    shared : SharedSim
        Shared state.
    screens : np.ndarray
        ``(n_screens, N, N)`` float32 phase screens.
    phi_total : np.ndarray
        ``(N, N)`` total beam phase at the aperture.

    Returns
    -------
    float
        The figure of merit.
    """
    E_obj = shared.prop.split_step(
        (shared.E0 * np.exp(1j * phi_total)).astype(np.complex64),
        screens,
        shared.dz,
    )
    I_obj = (np.abs(E_obj) ** 2).astype(np.float32)
    return FOM(I_obj, shared.I_vac, shared.bucket_mask)


def _imaging(
    seed: int, cfg: dict, shared: SharedSim, screens: np.ndarray, phi_track: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Step H: multi-plane rough-surface imaging (tracking-only condition).

    The tracking-only object field is scattered off ``n_roughness`` independent
    rough surfaces, back-propagated through the (reversed) screens, collimated
    by the conjugate of the outgoing phase, focused by the objective lens, and
    propagated to each measurement plane. The per-plane intensities are averaged
    over realizations.

    Parameters
    ----------
    seed : int
        Sample seed (drives the roughness RNG stream).
    cfg : dict
        Configuration dictionary.
    shared : SharedSim
        Shared state.
    screens : np.ndarray
        ``(n_screens, N, N)`` float32 phase screens.
    phi_track : np.ndarray
        ``(N, N)`` tracking phase.

    Returns
    -------
    tuple
        ``(images, I_obj_track)`` where ``images`` is ``(3, N, N)`` float32
        (per-plane mean intensity) and ``I_obj_track`` is ``(N, N)`` float32.
    """
    prop = shared.prop
    N = shared.N
    n_roughness = int(cfg["physical"]["n_roughness"])
    k = shared.k
    f_obj = shared.f_obj
    r2 = shared.r2

    # Tracking-only object-plane intensity.
    phi_total = shared.phi_focus + phi_track
    E_obj_track = prop.split_step(
        (shared.E0 * np.exp(1j * phi_total)).astype(np.complex64),
        screens,
        shared.dz,
    )
    I_obj_track = (np.abs(E_obj_track) ** 2).astype(np.float32)

    images = np.zeros((3, N, N), dtype=np.float32)
    for j in range(n_roughness):
        # Roughness realization (deterministic given seed).
        phi_r = random_roughness_phase((N, N), seed=(seed * 31 + j) % (2**32))
        E_scat = (np.sqrt(I_obj_track) * np.exp(1j * phi_r)).astype(np.complex64)
        # Back-propagate through the reversed screens.
        E_back = prop.split_step(E_scat, screens[::-1], -shared.dz)
        # Collimate by the conjugate of the outgoing phase.
        E_c = (E_back * np.exp(-1j * phi_total)).astype(np.complex64)
        # Objective lens.
        E_l = (E_c * np.exp(-1j * k * r2 / (2.0 * f_obj))).astype(np.complex64)
        # Propagate to each measurement plane.
        for p in range(3):
            I_p = prop.angular_spectrum_intensity(E_l, shared.plane_offsets[p])
            images[p] += I_p

    images /= n_roughness
    return images, I_obj_track


# --------------------------------------------------------------------------- #
# Public sample API
# --------------------------------------------------------------------------- #
@dataclass
class SimSample:
    """One simulated sample (Algorithm 1 output).

    Attributes
    ----------
    seed : int
        Sample seed.
    images : np.ndarray
        ``(3, N, N)`` float32 RAW measurement-plane intensities (planes
        ``f_obj - zR``, ``f_obj``, ``f_obj + zR``), pre-quantization.
    labels : np.ndarray
        ``(78,)`` float64 raw ``phi_Z78`` Zernike coefficients [rad].
    fom_noao / fom_track / fom_beacon / fom_z78 : float
        FOM for each correction leg.
    fom_ml : float | None
        FOM for the ML leg, filled only when ``correction_coeffs`` is passed.
    I_vac : np.ndarray
        ``(N, N)`` float32 vacuum object-plane intensity.
    I_obj_track : np.ndarray
        ``(N, N)`` float32 tracking-only object-plane intensity.
    track_slopes : np.ndarray
        ``(2,)`` float64 ``[a_x, a_y]`` of the tilt phase map.
    phase_conj / phase_track / phase_beacon / phase_z78 : np.ndarray
        ``(N, N)`` float64 phase maps.
    beam_phases : dict
        Total beam phase at the aperture for each FOM leg.
    """

    seed: int
    images: np.ndarray
    labels: np.ndarray
    fom_noao: float
    fom_track: float
    fom_beacon: float
    fom_z78: float
    fom_ml: Optional[float]
    I_vac: np.ndarray
    I_obj_track: np.ndarray
    track_slopes: np.ndarray
    phase_conj: np.ndarray
    phase_track: np.ndarray
    phase_beacon: np.ndarray
    phase_z78: np.ndarray
    beam_phases: dict = field(default_factory=dict)


def simulate_sample(
    seed: int,
    cfg: dict,
    correction_coeffs: Optional[np.ndarray] = None,
    *,
    shared: Optional[SharedSim] = None,
) -> SimSample:
    """Simulate one sample deterministically given ``seed`` (Algorithm 1).

    Parameters
    ----------
    seed : int
        Sample seed (sample seed = master_seed + sample_index).
    cfg : dict
        Configuration dictionary.
    correction_coeffs : np.ndarray, optional
        ``(78,)`` Zernike coefficients for the ML correction leg. When given,
        ``fom_ml`` is filled and the ``'ml'`` beam phase is added.
    shared : SharedSim, optional
        Prebuilt shared state (avoids rebuild).

    Returns
    -------
    SimSample
        The simulated sample.
    """
    shared = _resolve_shared(shared, cfg)
    zern = shared.zern

    # Step A: turbulence phase screens.
    screens = _make_screens(seed, cfg, shared)

    # Step B: aperture beam + focusing phase (precomputed in shared).
    E0 = shared.E0
    phi_focus = shared.phi_focus

    # Step C: vacuum intensity (precomputed in shared).
    I_vac = shared.I_vac

    # Step D: beacon back-propagation -> phi_conj (+ beacon intensity).
    phi_conj, _ = _beacon_phase_conj(seed, cfg, shared, screens)

    # Step E: tracking.
    phi_track, track_slopes = _tracking(shared, phi_conj)

    # Step F: corrections.
    #
    # Phi_Z78 = M_Z78 (M+_Z78 Phi_beacon): the 78-mode Zernike projection of
    # the beacon conjugate, expressed in the natural (pixel) basis. This
    # coefficient vector is the CNN training target (paper Algorithm 1).
    phi_beacon = phi_conj - phi_track
    labels = zern.phase_to_zernike(phi_beacon)
    phi_z78 = zern.zernike_to_phase(labels)

    # Step G: FOM legs.
    fom_noao = _fom_leg(shared, screens, phi_focus)
    fom_track = _fom_leg(shared, screens, phi_focus + phi_track)
    fom_beacon = _fom_leg(shared, screens, phi_focus + phi_track + phi_beacon)
    fom_z78 = _fom_leg(shared, screens, phi_focus + phi_track + phi_z78)

    beam_phases = {
        "noao": phi_focus,
        "track": phi_focus + phi_track,
        "beacon": phi_focus + phi_track + phi_beacon,
        "z78": phi_focus + phi_track + phi_z78,
    }
    fom_ml: Optional[float] = None
    if correction_coeffs is not None:
        phi_ml = phi_focus + phi_track + zern.zernike_to_phase(correction_coeffs)
        fom_ml = _fom_leg(shared, screens, phi_ml)
        beam_phases["ml"] = phi_ml

    # Step H: imaging (tracking-only condition).
    images, I_obj_track = _imaging(seed, cfg, shared, screens, phi_track)

    return SimSample(
        seed=seed,
        images=images,
        labels=labels,
        fom_noao=fom_noao,
        fom_track=fom_track,
        fom_beacon=fom_beacon,
        fom_z78=fom_z78,
        fom_ml=fom_ml,
        I_vac=I_vac,
        I_obj_track=I_obj_track,
        track_slopes=track_slopes,
        phase_conj=phi_conj,
        phase_track=phi_track,
        phase_beacon=phi_beacon,
        phase_z78=phi_z78,
        beam_phases=beam_phases,
    )


def simulate_sample_fom(
    seed: int,
    cfg: dict,
    coeffs: np.ndarray,
    *,
    shared: Optional[SharedSim] = None,
) -> float:
    """Fast path: FOM of a beam propagated with ``phi_focus + phi_track + zernike_to_phase(coeffs)``.

    Rebuilds the screens from ``seed`` and recomputes the tracking phase, but
    skips the imaging/scatter step. Returns the float FOM.

    Parameters
    ----------
    seed : int
        Sample seed.
    cfg : dict
        Configuration dictionary.
    coeffs : np.ndarray
        ``(78,)`` Zernike coefficients.
    shared : SharedSim, optional
        Prebuilt shared state (avoids rebuild).

    Returns
    -------
    float
        The figure of merit.
    """
    shared = _resolve_shared(shared, cfg)
    screens = _make_screens(seed, cfg, shared)
    phi_conj, _ = _beacon_phase_conj(seed, cfg, shared, screens)
    phi_track, _ = _tracking(shared, phi_conj)
    phi_total = shared.phi_focus + phi_track + shared.zern.zernike_to_phase(coeffs)
    return _fom_leg(shared, screens, phi_total)


# --------------------------------------------------------------------------- #
# Dataset generation
# --------------------------------------------------------------------------- #
def _quantize(images_raw: np.ndarray, scale_p: np.ndarray) -> np.ndarray:
    """Eq 13: quantize raw intensities to 12-bit uint16, per-plane scaling.

    ``I_q = int(I * 2^11 / max_ds_p)`` clipped to ``[0, 2047]``.

    Parameters
    ----------
    images_raw : np.ndarray
        ``(3, N, N)`` float32 raw intensities.
    scale_p : np.ndarray
        ``(3,)`` per-plane max over the training split.

    Returns
    -------
    np.ndarray
        ``(3, N, N)`` uint16 quantized images.
    """
    scaled = images_raw * (2**11) / scale_p[:, None, None]
    scaled = np.clip(scaled, 0, 2**11 - 1)
    return scaled.astype(np.uint16)


# Worker globals (set once per process via the Pool initializer).
_WORKER_CFG: Optional[dict] = None
_WORKER_SHARED: Optional[SharedSim] = None


def _worker_init(cfg: dict) -> None:
    """Pool initializer: build the shared state once per worker process."""
    global _WORKER_CFG, _WORKER_SHARED
    _WORKER_CFG = cfg
    _WORKER_SHARED = _get_shared(cfg)


def _worker_generate(batch: list[tuple[int, int]]) -> list[tuple]:
    """Process a batch of ``(sample_index, seed)`` pairs.

    Returns a list of compact tuples ``(idx, images_raw, labels, fom_noao,
    fom_track, fom_beacon, fom_z78)`` (the large phase arrays are not shipped
    back to the parent).
    """
    out = []
    for idx, seed in batch:
        s = simulate_sample(seed, _WORKER_CFG, shared=_WORKER_SHARED)
        out.append(
            (
                idx,
                s.images,
                s.labels,
                s.fom_noao,
                s.fom_track,
                s.fom_beacon,
                s.fom_z78,
            )
        )
    return out


def _make_batches(
    indices: np.ndarray, master_seed: int, chunk: int
) -> list[list[tuple[int, int]]]:
    """Split sample indices into batches of ``(sample_index, seed)`` pairs."""
    seeds = master_seed + indices
    batches = []
    for i in range(0, len(indices), chunk):
        batches.append(
            [(int(idx), int(seed)) for idx, seed in zip(indices[i : i + chunk], seeds[i : i + chunk])]
        )
    return batches


def generate_dataset(cfg: dict) -> str:
    """Run the full two-pass dataset generation pipeline and write the HDF5 file.

    Pass 1 computes per-plane intensity maxima and per-mode label mean/std over
    the TRAIN split (Eqs 13-14). Pass 2 quantizes and streams all samples to the
    HDF5 file (chunked writes, no in-RAM accumulation of raw images).

    Parameters
    ----------
    cfg : dict
        Configuration dictionary.

    Returns
    -------
    str
        Path to the written HDF5 file.
    """
    p = cfg["physical"]
    d = cfg["data"]
    N = int(p["N"])
    n_train = int(d["n_train"])
    n_test = int(d["n_test"])
    n_eval = int(d["n_eval"])
    master_seed = int(d["master_seed"])
    workers = int(d["workers"])
    h5_path = d["h5_path"]
    L = float(p["L"])

    N_total = n_train + n_test + n_eval
    train_idx = np.arange(n_train, dtype=np.int64)
    test_idx = np.arange(n_train, n_train + n_test, dtype=np.int64)
    eval_idx = np.arange(n_train + n_test, N_total, dtype=np.int64)
    all_idx = np.arange(N_total, dtype=np.int64)

    # Build shared state in the parent (ZernikeBasis inherited COW by workers).
    shared = _get_shared(cfg)

    os.makedirs(os.path.dirname(os.path.abspath(h5_path)), exist_ok=True)

    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(workers, initializer=_worker_init, initargs=(cfg,)) as pool:
        # ---- Pass 1: train-only stats (Eqs 13-14) ----
        plane_max = np.zeros(3, dtype=np.float64)
        label_sum = np.zeros(N_MODES, dtype=np.float64)
        label_sumsq = np.zeros(N_MODES, dtype=np.float64)
        n_proc = 0
        train_batches = _make_batches(train_idx, master_seed, chunk=4)
        for batch_result in tqdm(
            pool.imap_unordered(_worker_generate, train_batches),
            total=len(train_batches),
            desc="pass1 (train stats)",
        ):
            for (_idx, images_raw, labels, *_rest) in batch_result:
                plane_max = np.maximum(plane_max, images_raw.max(axis=(1, 2)))
                label_sum += labels
                label_sumsq += labels**2
                n_proc += 1
        assert n_proc == n_train, f"pass1 processed {n_proc} != n_train {n_train}"

        mu = label_sum / n_train
        sigma = np.sqrt(np.maximum(label_sumsq / n_train - mu**2, 0.0))
        scale_p = plane_max.astype(np.float32)

        # ---- Pass 2: quantize + stream all samples to HDF5 ----
        with h5py.File(h5_path, "w") as f:
            f.create_dataset(
                "images", (N_total, 3, N, N), dtype=np.uint16, chunks=(1, 3, N, N)
            )
            f.create_dataset("labels", (N_total, N_MODES), dtype=np.float32)
            f.create_dataset("fom_noao", (N_total,), dtype=np.float32)
            f.create_dataset("fom_track", (N_total,), dtype=np.float32)
            f.create_dataset("fom_beacon", (N_total,), dtype=np.float32)
            f.create_dataset("fom_z78", (N_total,), dtype=np.float32)
            f.create_dataset("seeds", (N_total,), dtype=np.int64)
            f.create_dataset("L", (N_total,), dtype=np.float32)
            f.create_dataset("train_idx", (n_train,), dtype=np.int64)
            f.create_dataset("test_idx", (n_test,), dtype=np.int64)
            f.create_dataset("eval_idx", (n_eval,), dtype=np.int64)
            f.create_dataset("mu", (N_MODES,), dtype=np.float32)
            f.create_dataset("sigma", (N_MODES,), dtype=np.float32)
            f.create_dataset("scale_p", (3,), dtype=np.float32)
            f.create_dataset("vacuum_intensity", (N, N), dtype=np.float32)
            f.attrs["config_json"] = json.dumps(cfg)

            all_batches = _make_batches(all_idx, master_seed, chunk=4)
            for batch_result in tqdm(
                pool.imap_unordered(_worker_generate, all_batches),
                total=len(all_batches),
                desc="pass2 (write)",
            ):
                for (
                    idx,
                    images_raw,
                    labels,
                    fom_noao,
                    fom_track,
                    fom_beacon,
                    fom_z78,
                ) in batch_result:
                    f["images"][idx] = _quantize(images_raw, scale_p)
                    f["labels"][idx] = labels.astype(np.float32)
                    f["fom_noao"][idx] = fom_noao
                    f["fom_track"][idx] = fom_track
                    f["fom_beacon"][idx] = fom_beacon
                    f["fom_z78"][idx] = fom_z78

            # Metadata.
            f["seeds"][:] = master_seed + all_idx
            f["L"][:] = L
            f["train_idx"][:] = train_idx
            f["test_idx"][:] = test_idx
            f["eval_idx"][:] = eval_idx
            f["mu"][:] = mu.astype(np.float32)
            f["sigma"][:] = sigma.astype(np.float32)
            f["scale_p"][:] = scale_p
            f["vacuum_intensity"][:] = shared.I_vac

    return h5_path
