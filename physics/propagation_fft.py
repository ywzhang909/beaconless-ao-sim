"""Angular-spectrum propagation using soapy's AOFFT FFT wrapper.

Provides a :class:`Propagator` that performs Fresnel/angular-spectrum
propagation of a complex scalar field, plus a helper for the Rayleigh range.
"""

from __future__ import annotations

from typing import List, Union

import numpy as np

try:
    import pyfftw

    try:
        pyfftw.config.NUM_THREADS = 1
    except Exception:
        pass
except Exception:
    pyfftw = None

from soapy import AOFFT


def rayleigh_range(w0: float, lam: float) -> float:
    """Return the Rayleigh range z_R = pi * w0**2 / lam.

    Parameters
    ----------
    w0 : float
        Beam waist radius (m).
    lam : float
        Wavelength (m).

    Returns
    -------
    float
        Rayleigh range (m).
    """
    return np.pi * w0**2 / lam


class Propagator:
    """Angular-spectrum field propagator built on soapy.AOFFT.FFT.

    Parameters
    ----------
    N : int
        Grid size (square, N x N).
    dx : float
        Grid sample spacing (m).
    lam : float
        Wavelength (m).
    n_threads : int, optional
        Number of FFTW threads.
    dtype : np.dtype, optional
        Working complex dtype (default ``np.complex64``).
    """

    def __init__(
        self,
        N: int,
        dx: float,
        lam: float,
        n_threads: int = 1,
        dtype: np.dtype = np.complex64,
    ) -> None:
        self.N = N
        self.dx = dx
        self.lam = lam
        self.dtype = dtype

        self._fft = AOFFT.FFT(
            inputSize=(N, N),
            axes=(-2, -1),
            mode="pyfftw",
            dtype="complex64",
            direction="FORWARD",
            fftw_FLAGS=("FFTW_MEASURE",),
            THREADS=n_threads,
        )
        self._ifft = AOFFT.FFT(
            inputSize=(N, N),
            axes=(-2, -1),
            mode="pyfftw",
            dtype="complex64",
            direction="BACKWARD",
            fftw_FLAGS=("FFTW_MEASURE",),
            THREADS=n_threads,
        )

        # Angular-spectrum transfer function for z0 = 1.0 m.
        k = 2.0 * np.pi / lam
        fx = np.fft.fftfreq(N, dx)
        fy = np.fft.fftfreq(N, dx)
        fxx, fyy = np.meshgrid(fx, fy)
        arg = np.maximum(0.0, 1.0 - (lam * fxx) ** 2 - (lam * fyy) ** 2)
        self._H1 = np.exp(1j * k * np.sqrt(arg)).astype(np.complex64)

    def _propagate_raw(self, E: np.ndarray, z: float) -> np.ndarray:
        """Propagate by z metres, returning the (possibly aliased) buffer."""
        if z == 0.0:
            return np.array(E, dtype=np.complex64, copy=True)
        H = np.power(self._H1, z).astype(np.complex64)
        F = self._fft(E)
        F *= H
        out = self._ifft(F)
        return out

    def propagate(self, E: np.ndarray, z: float) -> np.ndarray:
        """Propagate a complex field by distance ``z`` (m).

        Parameters
        ----------
        E : np.ndarray
            Complex field, shape (N, N), complex64.
        z : float
            Propagation distance (m).

        Returns
        -------
        np.ndarray
            Propagated field, complex64. A fresh copy is returned so the
            caller may safely mutate it.
        """
        out = self._propagate_raw(E, z)
        return np.array(out, dtype=np.complex64, copy=True)

    def split_step(
        self,
        E_in: np.ndarray,
        screens: Union[List[np.ndarray], np.ndarray],
        dz: float,
    ) -> np.ndarray:
        """Symmetric split-step propagation through phase screens.

        For each screen ``phi``: propagate ``dz/2``, apply ``exp(1j*phi)``,
        then propagate ``dz/2``. Screens of zeros are equivalent to a single
        ``propagate(E, n*dz)``.

        Parameters
        ----------
        E_in : np.ndarray
            Input complex field, shape (N, N), complex64.
        screens : list or ndarray
            Phase screens, shape (n, N, N), float32.
        dz : float
            Propagation distance between screens (m).

        Returns
        -------
        np.ndarray
            Output complex field, complex64.
        """
        E = np.array(E_in, dtype=np.complex64, copy=True)
        for phi in screens:
            E = self.propagate(E, dz / 2.0)
            E = np.array(E, dtype=np.complex64, copy=True)
            E *= np.exp(1j * phi).astype(np.complex64)
            E = self.propagate(E, dz / 2.0)
        return E

    def angular_spectrum_intensity(self, E_in: np.ndarray, z: float) -> np.ndarray:
        """Return the propagated intensity ``|propagate(E_in, z)|**2``.

        Parameters
        ----------
        E_in : np.ndarray
            Input complex field, shape (N, N), complex64.
        z : float
            Propagation distance (m).

        Returns
        -------
        np.ndarray
            Intensity, float32, shape (N, N).
        """
        E = self.propagate(E_in, z)
        return (np.abs(E) ** 2).astype(np.float32)
