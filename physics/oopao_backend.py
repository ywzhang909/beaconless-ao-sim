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
generated at the reference r0 ``_R0_REF_500 = 0.15`` at ``_LAM_REF_500 =
500 nm`` (``Atmosphere.wavelength``) and amplitude-rescaled so its per-slab r0
is *exactly* ``r0_slab = r0_path * n**(3/5)`` at the simulation wavelength --
the same per-slab r0 the aotools path uses. Because the von-Karman PSD scales
as ``r0**(-5/3)``, phase amplitude ~ ``r0**(-5/6)``; the rescale (``_rescale_for``)
additionally converts the generated phase from 500 nm to the simulation
wavelength ``lam`` and removes the generator's measured normalization
``_CAL_REF`` (empirically, per-layer raw std = ``_CAL_REF * (D/r0_ref)**(5/6)``
at 500 nm over 25 seeds for D=0.30 m, L0=100 m, cropped NxN from N+4). Result:
OOPAO screens are statistically equivalent to aotools screens (same per-slab
r0, L0, l0), differing only in the random realization (verified: applied-phase
std / theory ratio 1.00 across r0_slab, per-seed spread ~10% sampling noise).

Layer -> screen mapping
-----------------------
OOPAO builds each layer at ``resolution = N + 4`` pixels (2-px margin per side
for the frozen-flow outer ring). We crop the central ``N x N`` to align with
the pupil grid. ``layer.OPD`` is the per-layer phase in radians at 500 nm
(OOPAO's ``ft_sh_phase_screen`` output); ``make_screens`` multiplies it by the
wavelength-corrected rescale ``_rescale_for`` (no separate ``2*pi/lambda``
step).
"""

from __future__ import annotations

import numpy as np

from physics.screens_soapy import compute_r0
from physics._oopao_compat import Atmosphere, Source, Telescope

__all__ = ["OopaoScreenBackend"]

# Reference r0 (at 500 nm) used to generate the OOPAO layers before rescaling.
# Arbitrary; each layer's phase is amplitude-rescaled to the target per-slab r0.
_R0_REF_500 = 0.15
_LAM_REF_500 = 5.0e-7
# Measured OOPAO generator calibration (25-seed average, D=0.30 m, L0=100 m,
# cropped NxN from N+4): per-layer raw std = _CAL_REF * (D/r0_ref)**(5/6).
_CAL_REF = 0.6191


def _rescale_for(r0_slab: float, lam: float) -> float:
    """Amplitude rescale mapping OOPAO's 500 nm reference layers to ``r0_slab``.

    ``layer.OPD`` is phase in radians at the OOPAO generation wavelength
    (``_LAM_REF_500``), and the simulation applies it as phase at ``lam``.
    Converting the raw reference screen to ``lam`` multiplies its std by
    ``_LAM_REF_500 / lam``; scaling it to the target per-slab r0 then requires

    M = (lam / _LAM_REF_500) * (r0_ref / r0_slab)**(5/6) * sqrt(1.03) / _CAL_REF

    (PSD ~ r0**(-5/3) => phase amplitude ~ r0**(-5/6); the sqrt(1.03) factor is
    the Kolmogorov single-aperture phase-variance constant).
    """
    return (float(lam) / _LAM_REF_500) * (
        _R0_REF_500 / float(r0_slab)
    ) ** (5.0 / 6.0) / _CAL_REF * 1.03**0.5


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
        # OOPAO generates layers at _LAM_REF_500, but the simulation evaluates
        # them at self.lam, so the phase must first be converted between
        # wavelengths (phi_lam = phi_ref * lam_ref / lam) before the r0 rescale.
        self._rescale = _rescale_for(self.r0_slab, self.lam)

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

    def make_screens(
        self, seed: int, r0_slab: float | None = None
    ) -> np.ndarray:
        """Draw ``n_screens`` fresh OOPAO screens for sample ``seed``.

        Parameters
        ----------
        seed : int
            Sample seed; OOPAO reseeds every layer with ``seed + i_layer``.
        r0_slab : float, optional
            Per-sample per-slab r0 [m] (CNNL: varies with propagation distance
            L). When given, the reference-r0 OOPAO layers are rescaled to this
            target per-slab r0 instead of the constant-L ``self.r0_slab``.
            Because the von-Karman PSD scales as ``r0**(-5/3)``, a constant
            amplitude rescale ``_rescale_for(r0_slab, lam)`` is a statistically
            exact r0 change (PSD shape in r/L0/l0 is preserved). The OOPAO
            layer *realizations* are L-independent random fields; only the
            amplitude rescale changes with per-L r0, so no atmosphere rebuild
            is needed. 中文：每样本每 slab r0 [m]（CNNL：随 L 变化）；给出时
            把参考 r0 的 OOPAO 层重缩放到该目标 r0，而非常数 L 的 self.r0_slab。
            波数谱随 r0**(-5/3) 缩放，故常数振幅重缩放 _rescale_for(r0_slab,lam)
            在统计上精确（PSD 形状不变）。OOPAO 层实现在 L 上无关，仅振幅重
            缩放随逐 L r0 变化，无需重建大气。

        Returns
        -------
        np.ndarray
            ``(n_screens, N, N)`` float32 phase screens in radians, center-cropped
            from OOPAO's ``N+4``-pixel layers and rescaled to the target per-slab
            r0.
        """
        # Per-L rescale override (Option B): when a per-sample r0_slab is given
        # (CNNL), rescale the reference-r0 layers to it instead of the constant
        # L value stored at construction. 中文：逐 L 重缩放（选项 B）——
        # 当给定逐样本 r0_slab（CNNL）时，按它而非构造时的常数 L 值重缩放。
        rescale = self._rescale
        if r0_slab is not None:
            rescale = _rescale_for(float(r0_slab), self.lam)
        self.atm.generateNewPhaseScreen(seed=int(seed))
        out = np.empty((self.n_screens, self.N, self.N), dtype=np.float32)
        for i in range(self.n_screens):
            lay = getattr(self.atm, "layer_%d" % (i + 1))
            # Crop the 2-px margin (layer is N+4, keep the central N) and rescale
            # the reference-r0 phase to the target per-slab r0.
            out[i] = (np.asarray(lay.OPD)[2:-2, 2:-2] * rescale).astype(
                np.float32
            )
        return out
