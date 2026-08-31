"""
OOPAO substrate (vendored core modules from github.com/cheritier/OOPAO).

We vendor the core optics OOPAO owns -- ``Atmosphere`` (per-layer von-Karman
phase screens via ``ft_sh_phase_screen``), ``Telescope`` (pupil geometry),
and ``Zernike`` (Noll-indexed basis) -- so the training-data generator can be
re-based on OOPAO without pulling in the full OOPAO package (whose
``requirements.txt`` pins numpy 1.21-1.23, conflicting with our numpy 2.x env).

Only the modules needed for screen generation, pupil, and Zernike are
vendored. ``Atmosphere``'s import of display/interpolation helpers is stubbed
out in ``_oopao_shim.py``.

Provenance: ESO/LAM "Object Oriented Python Adaptive Optics" (C. T. Heritier
et al.), inspired by OOMAO. Adapted from aotools.
"""
from .Atmosphere import Atmosphere  # noqa: F401
from .Telescope import Telescope  # noqa: F401
from .Zernike import Zernike  # noqa: F401

__all__ = ["Atmosphere", "Telescope", "Zernike"]
