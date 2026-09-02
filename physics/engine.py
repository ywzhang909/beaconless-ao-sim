"""Abstract physics-engine and measurement-source interfaces for the
beaconless-AO simulation.

This module defines the two seam interfaces the pipeline is built around:

- :class:`PhysicsEngine` — the (deterministic, seed-driven) physics forward
  model: turbulence phase screens, beacon back-propagation, tilt tracking,
  forward-propagation FOM legs, and Zernike projection. Everything needed to
  label a sample and score every correction branch lives here.
- :class:`MeasurementSource` — supplies the ``(3, N, N)`` measurement-plane
  camera images. Two concrete sources exist:
  :class:`SimulatedMeasurementSource` (rough-surface scatter + imaging optics,
  defined in ``data/simulate.py``) and :class:`HardwareMeasurementSource`
  (real camera frames read from an on-disk array / device).

Decomposition rationale
-----------------------
The measurement/imaging step is the only stage a real hardware experiment
replaces. The FOM legs (``noao``/``track``/``beacon``/``z78``) and the
78-mode Zernike label projection depend **only** on the physics forward model
(screens, beacon phase, tracking) and never on the camera images. Keeping the
two concerns behind separate interfaces means a hardware acquisition path can
reuse the entire physics engine unchanged and only swap in a camera-backed
:class:`MeasurementSource`.

A hardware source cannot provide the simulated object-plane intensity
``I_obj_track`` (a physics quantity, not a camera measurement); it returns
``None`` for that field.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

__all__ = [
    "PhysicsEngine",
    "MeasurementSource",
    "HardwareMeasurementSource",
]


class PhysicsEngine(ABC):
    """Abstract beaconless-AO physics forward model.

    Implementations expose the read-only grid/optics state needed by callers
    (``N``, ``dx``, ``lam``, ``k``, ``pupil``, ``E0``, ``phi_focus``,
    ``I_vac``, ``r2``, ``X``, ``Y``, ``G``, ``zern``, ``f_obj``,
    ``plane_offsets``, ``dz``, ...) plus the seed-driven step methods below.

    Concrete instance is created from a configuration and holds the
    per-sample physics state; it is built once per process and shared across
    samples.
    """

    # -- read-only optical/geometric state (populated in __init__) ----------- #
    N: int                  # grid side length (px)
    dx: float               # pixel spacing (m)
    lam: float              # centre wavelength (m)
    k: float                # wavenumber 2*pi/lam (rad/m)
    dz: float               # screen separation (m)
    pupil: np.ndarray       # (N, N) bool aperture mask
    E0: np.ndarray          # (N, N) complex64 aperture beam amplitude
    phi_focus: np.ndarray   # (N, N) float64 focusing phase (rad)
    I_vac: np.ndarray       # (N, N) float32 vacuum object-plane intensity
    r2: np.ndarray          # (N, N) radial distance squared (m^2)
    X: np.ndarray           # (N, N) x coordinates (m)
    Y: np.ndarray           # (N, N) y coordinates (m)
    G: np.ndarray           # (N, N) tracking Gaussian weighting
    zern: object            # Zernike basis (phase_to_zernike / zernike_to_phase)
    f_obj: float            # objective focal length (m)
    plane_offsets: np.ndarray  # (3,) measurement-plane offsets behind lens (m)
    n_screens: int          # number of turbulence layers / screens
    cn2: float              # Cn2 (m**(-2/3))
    L0: float               # outer scale (m)
    l0_sim: float           # inner scale (m)

    @abstractmethod
    def make_screens(self, seed: int) -> np.ndarray:
        """Return ``(n_screens, N, N)`` float32 turbulence phase screens (rad).

        Deterministic given ``seed``.
        """

    @abstractmethod
    def beacon_phase_conj(
        self, seed: int, screens: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Back-propagate the diffraction-limited beacon to the pupil.

        Returns ``(phi_conj, I_beacon)``, each ``(N, N)`` float64: the
        conjugated beacon phase and the pupil beacon intensity (diagnostic).
        """

    @abstractmethod
    def track(
        self, phi_conj: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fit the tilt from the conjugated beacon phase.

        Returns ``(phi_track, track_slopes)``: the ``(N, N)`` float64 linear
        tracking phase and the ``(2,)`` float64 slopes ``[a_x, a_y]``.
        """

    @abstractmethod
    def forward_fom(
        self, screens: np.ndarray, phi_total: np.ndarray
    ) -> float:
        """Forward-propagate with a total aperture phase and return the FOM.

        ``phi_total`` is the full aperture beam phase (focus + correction).
        """

    @abstractmethod
    def phase_to_zernike(self, phi: np.ndarray) -> np.ndarray:
        """Fit ``(n_modes,)`` Zernike coefficients (rad) to a phase map."""

    @abstractmethod
    def zernike_to_phase(self, coeffs: np.ndarray) -> np.ndarray:
        """Reconstruct a ``(N, N)`` phase map (rad) from Zernike coefficients."""


class MeasurementSource(ABC):
    """Supplies the measurement-plane camera images for one sample.

    The source is intentionally decoupled from :class:`PhysicsEngine`: it is
    the only stage a hardware experiment replaces. A simulated source pays
    attention to ``seed``/``screens``/``phi_track`` (it must forward- and
    back-propagate the field to form the rough-surface image); a hardware
    source ignores those and instead indexes its frame stream by
    ``sample_index``.
    """

    @abstractmethod
    def acquire(
        self,
        *,
        seed: int,
        sample_index: int,
        screens: np.ndarray,
        phi_track: np.ndarray,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Return ``(images, I_obj_track)`` for one sample.

        Parameters
        ----------
        seed : int
            Sample seed (meaningful only for simulated sources that form the
            image from the physics state; a hardware source ignores it).
        sample_index : int
            Global sample index (meaningful for hardware sources that index a
            frame stream; a simulated source may ignore it).
        screens : np.ndarray
            ``(n_screens, N, N)`` phase screens (used by the simulated source
            to propagate the field; ignored by a hardware source).
        phi_track : np.ndarray
            ``(N, N)`` tracking phase (used by the simulated source).

        Returns
        -------
        tuple
            ``(images, I_obj_track)``: ``images`` is ``(3, N, N)`` float32
            measurement-plane intensities; ``I_obj_track`` is the ``(N, N)``
            float32 tracking-only object-plane intensity, or ``None`` for a
            hardware source (which cannot provide the simulated field).
        """


class HardwareMeasurementSource(MeasurementSource):
    """Supply the measurement-plane images from real (on-disk) camera frames.

    Instead of simulating rough-surface scatter and imaging optics, this
    source serves pre-acquired camera data. It supports two input shapes:

    * A single frame repeated for all planes: ``frame.shape == (N, N)``.
    * One frame per plane: ``frame.shape == (n_planes, N, N)``.

    The source spatially resamples any input to the engine's target pupil grid
    ``(target_N, target_N)`` (crop or pad to centre), so hardware cameras with
    a different pixel count can be ingested directly.

    Calibration / pre-processing (flat-field, gain, background) is the
    caller's responsibility before storage; this source only rescales pixel
    values to ``float32`` and optionally clips negatives to zero.

    Parameters
    ----------
    frames : np.ndarray
        Array of camera frames. Shape ``(N, N)`` (single, reused per plane) or
        ``(3, N, N)`` (per-plane fixed, non-repeatable). ``n_planes`` must be
        ``1`` or ``3``; other counts raise ``ValueError``.
    target_N : int
        Target pupil-grid side length. Frames are centre-resampled to
        ``(target_N, target_N)``.
    repeat_single : bool
        If the input is a single ``(N, N)`` frame, repeat it across the three
        measurement planes. Default ``True``.
    clip_negative : bool
        Clip negative pixel values to zero (physical intensities are
        non-negative). Default ``True``.
    """

    def __init__(
        self,
        frames: np.ndarray,
        target_N: int,
        repeat_single: bool = True,
        clip_negative: bool = True,
    ) -> None:
        frames = np.asarray(frames, dtype=np.float32)
        if frames.ndim == 2:
            if not repeat_single:
                raise ValueError(
                    "Single (N, N) frame provided but repeat_single=False; "
                    "cannot build a 3-plane image."
                )
            # (N, N) -> replicated to (3, N, N)
            frames = np.repeat(frames[np.newaxis, ...], 3, axis=0)
        if frames.ndim != 3:
            raise ValueError(
                f"frames must be (N, N) or (3, N, N), got shape {frames.shape}"
            )
        n_planes = frames.shape[0]
        if n_planes not in (1, 3):
            raise ValueError(
                f"frames must have 1 or 3 planes, got {n_planes}"
            )
        if n_planes == 1:
            # repeat single plane across the pipeline's 3 measurement planes
            frames = np.repeat(frames, 3, axis=0)

        self._raw = frames
        self._target_N = int(target_N)
        self._clip_negative = bool(clip_negative)
        # Pre-compute the centre-resampled frames once.
        self._frames = self._resample(frames, self._target_N)

    @staticmethod
    def _resample(frames: np.ndarray, target_N: int) -> np.ndarray:
        """Centre-crop / centre-pad ``(3, H, W)`` frames to ``(3, t, t)``."""
        n, H, W = frames.shape
        t = target_N
        out = np.zeros((n, t, t), dtype=np.float32)
        # Source window kept from the input frame (centre-anchored).
        src_h0 = max(0, (H - t) // 2)
        src_w0 = max(0, (W - t) // 2)
        src_h = slice(src_h0, src_h0 + min(H, t))
        src_w = slice(src_w0, src_w0 + min(W, t))
        # Destination window in the output grid (centre-anchored).
        dst_h0 = max(0, (t - H) // 2)
        dst_w0 = max(0, (t - W) // 2)
        dst_h = slice(dst_h0, dst_h0 + min(H, t))
        dst_w = slice(dst_w0, dst_w0 + min(W, t))
        out[:, dst_h, dst_w] = frames[:, src_h, src_w]
        return out

    def acquire(
        self,
        *,
        seed: int,
        sample_index: int,
        screens: np.ndarray,
        phi_track: np.ndarray,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Return the pre-acquired camera frame(s) for this sample.

        ``seed`` is ignored (hardware frames are not seed-deterministic).
        ``sample_index`` may index a per-sample frame stream in a subclass;
        for a fixed frame set it is ignored too.

        Returns :math:`(images, None)` — the object-plane field is not
        measurable from a hardware camera.
        """
        if self._clip_negative:
            images = np.clip(self._frames, 0.0, None).astype(np.float32)
        else:
            images = np.array(self._frames, dtype=np.float32)
        return images, None
