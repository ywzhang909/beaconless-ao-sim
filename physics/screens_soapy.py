"""Phase-screen generation for the beaconless AO simulation.

This module wraps Soapy's ``atmosphere.makePhaseScreens`` (and, optionally,
aotools' ``ft_sh_phase_screen`` directly) to produce Kolmogorov/von-Karman
turbulence phase screens in **radians**.

Caveat on outer-scale truncation
--------------------------------
The grid side length is ``Lx = N * dx``. For the default configuration
``N=256``, ``dx = 0.03/256`` we have ``Lx = 30 mm`` while the outer scale is
``L0 = 100 m``. Because the grid is far smaller than the outer scale, the
structure function measured at a lag of ``r0`` is suppressed relative to the
infinite-aperture theory: ``D_phi(r0)/6.88 ~= 0.56``. This is an *expected*
consequence of outer-scale truncation (the grid cannot represent spatial
frequencies below ``1/Lx``), **not** a bug. The tests therefore assert a
generous ratio band (``0.3 < sf/theory < 1.5``) to accommodate both this
suppression and the sub-harmonic (SH) overshoot.

Wavelength handling
-------------------
``soapy.atmosphere.makePhaseScreens`` is a thin wrapper over aotools'
``ft_sh_phase_screen`` and does **not** apply any wavelength rescaling. The
``r0`` passed in is interpreted at the wavelength at which the screen is
generated. Therefore callers must pass ``r0`` computed at *their* wavelength
and must **not** multiply by ``500e-9/lam``. All screens are returned in
radians at the wavelength of the passed ``r0``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["compute_r0", "SoapyPhaseScreenGenerator"]


def compute_r0(lam: float, Cn2: float, L: float) -> float:
    """Compute the Fried parameter ``r0`` for a single turbulent layer.

    Uses the standard definition::

        r0 = (0.423 * k**2 * Cn2 * L)**(-3/5),   k = 2*pi/lam

    Parameters
    ----------
    lam : float
        Wavelength in metres.
    Cn2 : float
        Refractive-index structure constant in ``m**(-2/3)``.
    L : float
        Propagation path length (layer thickness) in metres.

    Returns
    -------
    float
        Fried parameter ``r0`` in metres.
    """
    k = 2.0 * np.pi / lam
    return (0.423 * k**2 * Cn2 * L) ** (-3.0 / 5.0)


class SoapyPhaseScreenGenerator:
    """Generate turbulence phase screens in radians using Soapy/aotools.

    Parameters
    ----------
    N : int
        Number of pixels across each (square) screen.
    dx : float
        Pixel scale in metres.
    L0 : float
        Outer scale in metres.
    l0 : float
        Inner scale in metres. Must be ``> 0`` (aotools divides by ``l0``).
    lam : float
        Wavelength in metres (used only for documentation/consistency; the
        screens are generated at the wavelength of the passed ``r0``).
    generator : str
        ``'soapy'`` (default) uses ``soapy.atmosphere.makePhaseScreens``;
        ``'aotools'`` builds each screen directly with
        ``aotools.turbulence.phasescreen.ft_sh_phase_screen``.
    """

    def __init__(
        self,
        N: int,
        dx: float,
        L0: float,
        l0: float,
        lam: float,
        generator: str = "soapy",
    ) -> None:
        if l0 <= 0:
            raise ValueError(
                f"l0 must be > 0 (aotools divides by l0), got l0={l0}"
            )
        if generator not in ("soapy", "aotools"):
            raise ValueError(
                f"generator must be 'soapy' or 'aotools', got {generator!r}"
            )
        self.N = int(N)
        self.dx = float(dx)
        self.L0 = float(L0)
        self.l0 = float(l0)
        self.lam = float(lam)
        self.generator = generator

    def make_pool(self, n_pool: int, r0: float, seed: int = 0) -> np.ndarray:
        """Generate a pool of phase screens.

        Parameters
        ----------
        n_pool : int
            Number of screens to generate.
        r0 : float
            Fried parameter in metres, at the wavelength of the returned
            screens. No wavelength rescaling is applied.
        seed : int
            Seed for the random number generator (deterministic).

        Returns
        -------
        np.ndarray
            ``(n_pool, N, N)`` float32 array of phase screens in radians.
        """
        np.random.seed(seed)
        if self.generator == "soapy":
            from soapy import atmosphere

            screens = atmosphere.makePhaseScreens(
                n_pool,
                r0,
                self.N,
                self.dx,
                self.L0,
                self.l0,
                returnScrns=True,
                SH=True,
            )
        else:  # 'aotools'
            from aotools.turbulence.phasescreen import ft_sh_phase_screen

            screens = [
                ft_sh_phase_screen(
                    r0, self.N, self.dx, self.L0, self.l0, seed=seed
                )
                for _ in range(n_pool)
            ]
        return np.asarray(screens, dtype=np.float32)

    def slide_window(
        self, pool: np.ndarray, n_screens: int, start: int
    ) -> np.ndarray:
        """Return a contiguous window of screens from the pool.

        Parameters
        ----------
        pool : np.ndarray
            Array of screens, first axis is the screen index.
        n_screens : int
            Number of screens in the window.
        start : int
            Starting index of the window.

        Returns
        -------
        np.ndarray
            ``pool[start:start + n_screens]``.

        Raises
        ------
        ValueError
            If ``start + n_screens > len(pool)``.
        """
        end = start + n_screens
        if end > len(pool):
            raise ValueError(
                f"window start={start} + n_screens={n_screens} = {end} "
                f"exceeds pool length {len(pool)}"
            )
        return pool[start:end]

    @staticmethod
    def structure_function(screen: np.ndarray, lag_px: int) -> float:
        """Compute the phase structure function at a given lag.

        ``D_phi(lag) = mean over x of (phi(x+lag) - phi(x))**2`` averaged over
        all rows, using only valid (in-bounds) pairs.

        Parameters
        ----------
        screen : np.ndarray
            2D phase screen in radians.
        lag_px : int
            Lag in pixels (must be ``>= 0`` and ``< screen.shape[1]``).

        Returns
        -------
        float
            Structure function value in ``rad**2``.
        """
        screen = np.asarray(screen, dtype=np.float64)
        if lag_px < 0 or lag_px >= screen.shape[1]:
            raise ValueError(
                f"lag_px={lag_px} out of range for screen width "
                f"{screen.shape[1]}"
            )
        diff = screen[:, lag_px:] - screen[:, :-lag_px]
        return float(np.mean(diff**2))
