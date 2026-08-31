"""Performance metrics for the beaconless AO simulation.

Metric definitions follow DiComo et al., "Beaconless adaptive optics using a
convolutional neural network for wavefront sensing," Opt. Express 33(15):31010
(2025), Eqs 6-8 and 15-17:

- ``nPIB`` (Eq 6): normalized power in bucket.
- ``SIB`` (Eq 7): Strehl-like intensity in bucket.
- ``FOM`` (Eq 8): figure of merit, geometric mean of nPIB and SIB.
- ``gain`` (Eq 15): FOM gain factor relative to the tracking-only solution.
- ``eta`` (Eq 16): CNN effectiveness, fraction of the possible FOM gain realized.
- ``mode_pearson`` (Eq 17): per-Noll-mode Pearson correlation between the ML
  prediction and the test-set amplitude.

All functions are numpy-only and guard every division so they never raise on
degenerate (zero) inputs.
"""

from __future__ import annotations

import numpy as np


def bucket_mask(N: int, diameter_px: float) -> np.ndarray:
    """Boolean (N,N) circular mask centered at grid center with the given DIAMETER in pixels.

    Cell included iff ``(x - xc)**2 + (y - yc)**2 <= (diameter_px/2)**2`` (inclusive).

    Args:
        N: Grid side length in pixels.
        diameter_px: Diameter of the bucket in pixels.

    Returns:
        Boolean array of shape ``(N, N)``.
    """
    y, x = np.mgrid[0:N, 0:N]
    xc = (N - 1) / 2.0
    yc = (N - 1) / 2.0
    r2 = (x - xc) ** 2 + (y - yc) ** 2
    return r2 <= (diameter_px / 2.0) ** 2


def nPIB(I: np.ndarray, I_vac: np.ndarray, mask: np.ndarray) -> float:
    """Eq (6): fractional power in bucket, normalized by the vacuum (focused) beam.

    ``nPIB = (power_in_bucket / total_power) / (power_in_bucket_vac / total_power_vac)``
    where ``power_in_bucket = sum(I[mask])`` and ``total = sum(I)``.

    Args:
        I: Intensity array of the system under consideration.
        I_vac: Intensity array of the focused Gaussian beam in vacuum.
        mask: Boolean bucket mask broadcastable to ``I``.

    Returns:
        The normalized power-in-bucket ratio. If either total is 0, returns 0.0.
    """
    I = np.asarray(I)
    I_vac = np.asarray(I_vac)
    mask = np.asarray(mask, dtype=bool)

    total = float(I.sum())
    total_vac = float(I_vac.sum())
    if total == 0.0 or total_vac == 0.0:
        return 0.0

    in_bucket = float(I[mask].sum())
    in_bucket_vac = float(I_vac[mask].sum())

    frac = in_bucket / total
    frac_vac = in_bucket_vac / total_vac
    if frac_vac == 0.0:
        return 0.0
    return frac / frac_vac


def SIB(I: np.ndarray, I_vac: np.ndarray, mask: np.ndarray) -> float:
    """Eq (7): peak intensity in bucket, normalized by the vacuum peak.

    ``SIB = max(I[mask]) / max(I_vac[mask])``.

    Args:
        I: Intensity array of the system under consideration.
        I_vac: Intensity array of the focused Gaussian beam in vacuum.
        mask: Boolean bucket mask broadcastable to ``I``.

    Returns:
        The peak-intensity ratio. If the vacuum peak is 0, returns 0.0.
    """
    I = np.asarray(I)
    I_vac = np.asarray(I_vac)
    mask = np.asarray(mask, dtype=bool)

    peak = float(I[mask].max())
    peak_vac = float(I_vac[mask].max())
    if peak_vac == 0.0:
        return 0.0
    return peak / peak_vac


def FOM(I: np.ndarray, I_vac: np.ndarray, mask: np.ndarray) -> float:
    """Eq (8): figure of merit, geometric mean of nPIB and SIB.

    ``FOM = sqrt(nPIB * SIB)``.

    Args:
        I: Intensity array of the system under consideration.
        I_vac: Intensity array of the focused Gaussian beam in vacuum.
        mask: Boolean bucket mask broadcastable to ``I``.

    Returns:
        The figure of merit.
    """
    return float(np.sqrt(nPIB(I, I_vac, mask) * SIB(I, I_vac, mask)))


def gain(fom_ml: float, fom_track: float) -> float:
    """Eq (15): FOM gain factor relative to the tracking-only solution.

    ``gain = fom_ml / fom_track``. An AO gain of 2 is equivalent to doubling
    system laser energy.

    Args:
        fom_ml: FOM of the ML (CNN) solution.
        fom_track: FOM of the tracking-only solution.

    Returns:
        The gain factor. If ``fom_track <= 0``, returns 0.0.
    """
    if fom_track <= 0.0:
        return 0.0
    return float(fom_ml / fom_track)


def eta(fom_ml: float, fom_track: float, fom_z78: float) -> float:
    """Eq (16): CNN effectiveness, fraction of the possible FOM gain realized.

    ``eta = (fom_ml - fom_track) / (fom_z78 - fom_track)``.

    Args:
        fom_ml: FOM of the ML (CNN) solution.
        fom_track: FOM of the tracking-only solution (lower bound).
        fom_z78: FOM of the phase-conjugate DM filtered to 78 modes (upper bound).

    Returns:
        The effectiveness in ``[0, 1]``. If the denominator is ~0 (or the
        78-mode ceiling is at/below the tracking baseline), returns 0.0
        (degenerate: no gain to realize).
    """
    num = fom_ml - fom_track
    den = fom_z78 - fom_track
    if den <= 1e-12:
        return 0.0
    return float(num / den)


def mode_pearson(ml: np.ndarray, test: np.ndarray) -> np.ndarray:
    """Eq (17): per-mode Pearson correlation between ML prediction and test amplitude.

    Computed PER COLUMN (mode). Inputs may be shape ``(K, n_modes)`` or
    ``(n_modes,)`` vectors (broadcast to 2D). Returns ``(n_modes,)``.

    ``Rj = cov(ml_j, test_j) / (std(ml_j) * std(test_j))`` with
    ``cov = mean((a - mean(a)) * (b - mean(b)))``. Zero-std columns -> 0.0.

    Args:
        ml: ML prediction amplitudes, shape ``(K, n_modes)`` or ``(n_modes,)``.
        test: Test-set amplitudes, same shape as ``ml``.

    Returns:
        Array of shape ``(n_modes,)`` with the per-mode Pearson correlation.
    """
    ml = np.asarray(ml, dtype=float)
    test = np.asarray(test, dtype=float)

    if ml.ndim == 1:
        ml = ml[:, None]
    if test.ndim == 1:
        test = test[:, None]

    n_modes = ml.shape[1]
    out = np.zeros(n_modes, dtype=float)

    for j in range(n_modes):
        a = ml[:, j]
        b = test[:, j]
        std_a = float(a.std())
        std_b = float(b.std())
        if std_a == 0.0 or std_b == 0.0:
            out[j] = 0.0
            continue
        cov = float(np.mean((a - a.mean()) * (b - b.mean())))
        out[j] = cov / (std_a * std_b)

    return out
