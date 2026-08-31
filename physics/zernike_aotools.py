"""Zernike basis built on aotools for phase fitting and reconstruction.

This module exposes :class:`ZernikeBasis`, a small wrapper around
``aotools.zernikeArray`` (Noll-normalized Zernike modes) and
``aotools.circle`` (circular pupil mask).  It provides a linear design
matrix ``M`` that samples each mode on the in-circle mask pixels, its
Moore-Penrose pseudo-inverse ``M_pinv``, and convenience methods to fit
Zernike coefficients from a phase map (``phase_to_zernike``), to build a
phase map from coefficients (``zernike_to_phase``), to reconstruct the
mask-pixel values (``reconstruct``), and to zero out unwanted modes
(``project_coeffs``).

Convention
----------
Noll index ``1..n_modes`` corresponds to aotools rows ``0..n_modes-1``.
Piston is Noll 1 -> index 0; tip/tilt are Noll 2,3 -> indices 1,2.
Coefficients are in radians and refer to the Noll-normalized basis.
"""

from __future__ import annotations

import numpy as np
from aotools import circle, zernikeArray


class ZernikeBasis:
    """Zernike modes sampled on a circular pupil mask.

    Parameters
    ----------
    N : int
        Side length of the square phase grid in pixels.
    n_modes : int, optional
        Number of Zernike modes (Noll 1..n_modes). Default is 78.
    mask_radius_px : int | None, optional
        Radius of the circular pupil mask in pixels. Defaults to ``N // 2``.

    Attributes
    ----------
    N : int
        Grid side length in pixels.
    n_modes : int
        Number of Zernike modes.
    mask_radius_px : int
        Pupil mask radius in pixels.
    mask : numpy.ndarray of bool, shape (N, N)
        In-circle pupil mask.
    Z : numpy.ndarray of float64, shape (n_modes, N, N)
        Noll-normalized Zernike modes (row ``i`` = Noll ``i+1``).
    M : numpy.ndarray of float64, shape (mask_px_count, n_modes)
        Design matrix: each mode sampled on the mask pixels.
    M_pinv : numpy.ndarray of float64, shape (n_modes, mask_px_count)
        Moore-Penrose pseudo-inverse of ``M``.
    """

    def __init__(
        self,
        N: int,
        n_modes: int = 78,
        mask_radius_px: int | None = None,
    ) -> None:
        self.N = int(N)
        self.n_modes = int(n_modes)
        self.mask_radius_px = (
            int(mask_radius_px) if mask_radius_px is not None else self.N // 2
        )

        # Noll-normalized Zernike modes: row i = Noll i+1 (piston at index 0).
        self.Z: np.ndarray = np.asarray(
            zernikeArray(self.n_modes, self.N, norm="noll"), dtype=np.float64
        )

        # In-circle pupil mask (aotools.circle returns float64 0/1).
        self.mask: np.ndarray = (
            circle(self.mask_radius_px, self.N).astype(bool)
        )

        # Design matrix: (mask_px_count, n_modes), each mode on mask pixels.
        self.M: np.ndarray = self.Z[:, self.mask].T
        self.M_pinv: np.ndarray = np.linalg.pinv(self.M)

    def phase_to_zernike(self, phi: np.ndarray) -> np.ndarray:
        """Fit Zernike coefficients (radians) to a phase map.

        Parameters
        ----------
        phi : numpy.ndarray, shape (N, N)
            Phase map in radians.

        Returns
        -------
        numpy.ndarray of float64, shape (n_modes,)
            Zernike coefficients in radians (Noll-normalized basis),
            least-squares fit over the mask pixels via ``M_pinv``.
        """
        return self.M_pinv @ np.asarray(phi, dtype=np.float64)[self.mask]

    def zernike_to_phase(self, coeffs: np.ndarray) -> np.ndarray:
        """Build a phase map (radians) from Zernike coefficients.

        Parameters
        ----------
        coeffs : numpy.ndarray, shape (n_modes,)
            Zernike coefficients in radians.

        Returns
        -------
        numpy.ndarray of float64, shape (N, N)
            Phase map in radians; zero outside the pupil mask.
        """
        phase = np.zeros((self.N, self.N), dtype=np.float64)
        phase[self.mask] = self.M @ np.asarray(coeffs, dtype=np.float64)
        return phase

    def reconstruct(self, coeffs: np.ndarray) -> np.ndarray:
        """Reconstruct mask-pixel phase values from coefficients.

        Parameters
        ----------
        coeffs : numpy.ndarray, shape (n_modes,)
            Zernike coefficients in radians.

        Returns
        -------
        numpy.ndarray of float64, shape (mask_px_count,)
            Phase values on the mask pixels only (``M @ coeffs``),
            identical to ``zernike_to_phase(coeffs)[mask]``.
        """
        return self.M @ np.asarray(coeffs, dtype=np.float64)

    def project_coeffs(
        self, coeffs: np.ndarray, keep_modes: np.ndarray
    ) -> np.ndarray:
        """Zero out all modes except those listed in ``keep_modes``.

        Parameters
        ----------
        coeffs : numpy.ndarray, shape (n_modes,)
            Zernike coefficients in radians.
        keep_modes : numpy.ndarray of int
            0-based mode indices to keep (e.g. ``[0, 1, 2]`` for
            piston + tilt + tip); all other entries are zeroed.

        Returns
        -------
        numpy.ndarray, shape (n_modes,)
            Copy of ``coeffs`` with non-kept modes set to zero.
        """
        out = np.zeros_like(coeffs)
        out[keep_modes] = coeffs[keep_modes]
        return out