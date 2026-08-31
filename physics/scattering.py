"""Rough-surface (Lambertian) scattering for the beaconless AO simulation.

This module models the incoherent scattering of an object field off a rough
surface. Each roughness realization applies a uniformly random phase in
``[0, 2*pi)`` to the (positive) object intensity, then back-propagates the
resulting field through a set of phase screens. Averaging over many
realizations yields the incoherent (speckle-averaged) back-propagated field.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def random_roughness_phase(
    shape: tuple[int, ...], seed: Optional[int] = None
) -> np.ndarray:
    """Generate a uniform random phase screen in ``[0, 2*pi)``.

    Parameters
    ----------
    shape : tuple of int
        Shape of the output phase screen.
    seed : int, optional
        Seed for the local ``numpy.random.Generator``. When ``None``, a fresh
        non-deterministic generator is used.

    Returns
    -------
    np.ndarray
        Float32 array of shape ``shape`` with values uniformly distributed in
        ``[0, 2*pi)``.
    """
    rng = np.random.default_rng(seed)
    return (rng.random(shape, dtype=np.float32) * (2.0 * np.pi)).astype(
        np.float32
    )


class LambertianScatterer:
    """Scatter an object field off a rough surface and back-propagate it.

    The scatterer applies a random phase to the object intensity for each of
    ``n_roughness`` independent realizations, back-propagates each realization
    through the provided phase screens, and returns the incoherent average.
    """

    def __init__(self, n_roughness: int = 10) -> None:
        """Initialize the scatterer.

        Parameters
        ----------
        n_roughness : int, optional
            Number of independent roughness realizations used for incoherent
            averaging. Defaults to 10.
        """
        if n_roughness < 1:
            raise ValueError("n_roughness must be >= 1")
        self.n_roughness = n_roughness

    def scatter_and_backprop(
        self,
        I_obj: np.ndarray,
        propagator,
        screens_back: list,
        dz: float,
    ) -> np.ndarray:
        """Scatter the object intensity and back-propagate each realization.

        For each realization a fresh random phase screen is drawn and applied
        to the (clipped) object intensity::

            E_scat = sqrt(max(I_obj, 0)) * exp(1j * phi_r)

        Each ``E_scat`` is back-propagated via ``propagator.split_step`` and the
        results are averaged over realizations.

        Parameters
        ----------
        I_obj : np.ndarray
            Object intensity (real-valued). Negative values are clipped to zero
            before taking the square root.
        propagator : object
            Object exposing ``split_step(E, screens_back, dz)`` returning the
            back-propagated field.
        screens_back : list
            Phase screens to back-propagate through.
        dz : float
            Propagation step size.

        Returns
        -------
        np.ndarray
            Complex128 array, the mean field over all roughness realizations,
            with the same spatial shape as ``I_obj``.
        """
        I_clip = np.maximum(I_obj, 0.0)
        E_avg = np.zeros(I_obj.shape, dtype=np.complex128)
        for _ in range(self.n_roughness):
            phi_r = random_roughness_phase(I_obj.shape)
            E_scat = np.sqrt(I_clip) * np.exp(1j * phi_r)
            E_back = propagator.split_step(E_scat, screens_back, dz)
            E_avg += E_back
        return E_avg / self.n_roughness

    @staticmethod
    def field_to_image(E: np.ndarray) -> np.ndarray:
        """Convert a complex field to an intensity image.

        Parameters
        ----------
        E : np.ndarray
            Complex field.

        Returns
        -------
        np.ndarray
            Float32 array of ``|E|**2``.
        """
        return (np.abs(E) ** 2).astype(np.float32)

    @staticmethod
    def quantize12(I: np.ndarray, max_val: float) -> np.ndarray:
        """Quantize an intensity image to 12-bit unsigned integers.

        Maps ``[0, max_val]`` linearly onto ``[0, 2**11 - 1]`` and clips any
        out-of-range values.

        Parameters
        ----------
        I : np.ndarray
            Intensity image.
        max_val : float
            Maximum intensity value mapped to the top of the 12-bit range.
            Must be positive.

        Returns
        -------
        np.ndarray
            Uint16 array with values in ``[0, 2047]``.

        Raises
        ------
        ValueError
            If ``max_val <= 0``.
        """
        if max_val <= 0:
            raise ValueError("max_val must be > 0")
        I_scaled = I * (2**11) / max_val
        I_scaled = np.clip(I_scaled, 0, 2**11 - 1)
        return I_scaled.astype(np.uint16)
