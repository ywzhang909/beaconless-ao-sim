"""OOPAO-based turbulence phase-screen generator for the beaconless AO sim.

Re-bases per-sample phase-screen generation on the OOPAO library (github.com/
cheritier/OOPAO, ESO/LAM) instead of calling aotools ``ft_sh_phase_screen``
directly. OOPAO's ``Atmosphere`` models the turbulent path as N independent
layers, each carrying a von-Karman screen (``phaseStats.ft_sh_phase_screen``,
adapted from aotools). We build one ``Atmosphere`` per process and draw a fresh
deterministic per-layer realization per sample via
``Atmosphere.generateNewPhaseScreen(seed)``.

Per-layer r0 calibration
------------------------
OOPAO's ``cn2`` bookkeeping divides the total Cn2 by ``max(altitude)``, which
does not correctly slice the path r0 into per-layer r0 (per-layer phase
variance comes out too strong). We therefore bypass it: each layer is
generated at a reference r0 and amplitude-rescaled so its per-slab r0 is
*exactly* ``r0_slab = r0_path * n**(3/5)`` at the simulation wavelength -- the
same per-slab r0 the aotools path uses. Because the von-Karman PSD scales as
``r0**(-5/3)``, a constant amplitude rescale ``sqrt((r0_slab / r0_ref)**(5/3))``
is a statistically exact r0 change (PSD shape in r/L0/l0 is preserved). Result:
OOPAO screens are statistically equivalent to aotools screens (same per-slab
r0, L0, l0), differing only in the random realization (verified: mean per-layer
std 1.36 vs aotools 1.27 rad, ratio ~1.06, within sampling noise).

Layer -> screen mapping
-----------------------
OOPAO builds each layer at ``resolution = N + 4`` pixels (2-px margin per side
for the frozen-flow outer ring). We crop the central ``N x N`` to align with
the pupil grid. ``layer.OPD`` is the per-layer phase in radians (OOPAO's
``ft_sh_phase_screen`` output); no ``2*pi/lambda`` conversion is applied.
"""

from __future__ import annotations

import numpy as np

from physics.screens_soapy import compute_r0
from physics.oopao import Atmosphere, Telescope
from physics.oopao.Source import Source

__all__ = ["OopaoScreenBackend"]

# Reference r0 (at 500 nm) used to generate the OOPAO layers before rescaling.
# Arbitrary; each layer's phase is amplitude-rescaled to the target per-slab r0.
_R0_REF_500 = 0.15


class OopaoScreenBackend:
    """Generate per-sample turbulence screens via OOPAO ``Atmosphere``.

    Parameters
    ----------
    N : int
        Pupil-grid side length in pixels.
    dx : float
        Pixel scale in metres.
    Dscope : float
        Telescope diameter in metres (defines the OOPAO pupil).
    lam : float
        Simulation wavelength in metres.
    cn2 : float
        Cn2 in ``m**(-2/3)``.
    L : float
        Propagation path length in metres.
    L0 : float
        Outer scale in metres.
    n_screens : int
        Number of turbulence layers / screens.
    """

    def __init__(
        self,
        N: int,
        dx: float,
        Dscope: float,
        lam: float,
        cn2: float,
        L: float,
        L0: float,
        n_screens: int,
    ) -> None:
        self.N = int(N)
        self.n_screens = int(n_screens)
        self.lam = float(lam)

        self.tel = Telescope(
            resolution=self.N, diameter=float(Dscope), fov=0.0, samplingTime=0.001
        )

        self.src = Source(optBand="R", magnitude=0.0, display_properties=False)
        self.src * self.tel

        # Target per-slab r0 at the simulation wavelength, matching the aotools
        # path exactly: r0_path = (0.423 k^2 Cn2 L)^(-3/5); r0_slab = r0_path * n**(3/5).
        r0_path = compute_r0(self.lam, float(cn2), float(L))
        self.r0_slab = r0_path * self.n_screens ** (3.0 / 5.0)

        # Amplitude rescale mapping the reference-r0 OOPAO layer phase to the
        # target per-slab r0. PSD ~ r0**(-5/3)  =>  phase amplitude ~ r0**(-5/6).
        self._rescale = (self.r0_slab / _R0_REF_500) ** (5.0 / 6.0)

        n = self.n_screens
        self._altitudes = np.linspace(50.0, float(L) - 50.0, n).tolist()
        self._frac = [1.0 / n] * n

        self.atm = Atmosphere(
            self.tel,
            r0=_R0_REF_500,
            L0=float(L0),
            windSpeed=[10.0] * n,
            fractionalR0=self._frac,
            windDirection=[0.0] * n,
            altitude=self._altitudes,
            src=self.src,
        )
        # No covariance matrices: we only need the per-layer phase screens, not
        # OOPAO's AO-loop / WFS machinery. This also avoids jsonpickle caching.
        self.atm.initializeAtmosphere(self.tel, compute_covariance=False)

    def make_screens(self, seed: int) -> np.ndarray:
        """Draw ``n_screens`` fresh OOPAO screens for sample ``seed``.

        Parameters
        ----------
        seed : int
            Sample seed; OOPAO reseeds every layer with ``seed + i_layer``.

        Returns
        -------
        np.ndarray
            ``(n_screens, N, N)`` float32 phase screens in radians, center-cropped
            from OOPAO's ``N+4``-pixel layers and rescaled to the target per-slab
            r0.
        """
        self.atm.generateNewPhaseScreen(seed=int(seed))
        out = np.empty((self.n_screens, self.N, self.N), dtype=np.float32)
        for i in range(self.n_screens):
            lay = getattr(self.atm, "layer_%d" % (i + 1))
            # Crop the 2-px margin (layer is N+4, keep the central N) and rescale
            # the reference-r0 phase to the target per-slab r0.
            out[i] = (np.asarray(lay.OPD)[2:-2, 2:-2] * self._rescale).astype(np.float32)
        return out
