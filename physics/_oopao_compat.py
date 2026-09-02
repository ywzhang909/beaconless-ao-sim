"""Compat shim for upstream OOPAO 0.x.

Upstream ``OOPAO/__init__.py`` is broken in two ways on numpy 2.x + Windows:

1. It searches ``sys.path`` for entries containing ``"OOPAO"`` to locate its
   own install directory and write ``precision_oopao.npy``. When the package
   is installed normally, no ``sys.path`` entry matches, and the lookup
   raises ``ValueError: attempt to get argmin of an empty sequence``.
2. Even after that, it runs ``from OOPAO.tools.tools import warning`` --
   a sub-module that no longer ships with upstream OOPAO (the ``tools/``
   directory was removed when modules were flattened into the top-level
   package). Importing the ``OOPAO`` package therefore always fails.

Workaround: do not import the package ``OOPAO`` at all. Load only the three
modules we use (``Atmosphere``, ``Source``, ``Telescope``) directly from the
installed package directory via ``importlib``, and plant a *shadow* ``OOPAO``
package object (``__path__`` only) in ``sys.modules`` so that relative imports
inside these modules (``phaseStats`` → ``tools.tools``) resolve without ever
executing the broken ``OOPAO/__init__.py``. The upstream bugs above are
sidestepped entirely.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from types import ModuleType

import numpy as np

_PKG_NAME = "OOPAO"
_MODULES = ("Atmosphere", "Source", "Telescope")

_spec = importlib.util.find_spec(_PKG_NAME)
if _spec is None or _spec.origin is None:
    raise ImportError("OOPAO package not importable; install via `uv sync`.")
_pkg_dir = os.path.dirname(_spec.origin)

# =====================================================================
# Upstream workarounds (numpy 2.x / Windows)
# =====================================================================
# 1) Both OOPAO/__init__.py and Telescope.__init__ locate the install dir by
#    scanning ``sys.path`` for entries containing "OOPAO" and then doing
#    ``np.argmin`` on the matching path lengths. When installed normally no
#    entry matches, which raises ``ValueError: attempt to get argmin of an
#    empty sequence``. Fix: make the install dir visible on ``sys.path``.
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

# 2) Telescope.__init__ then loads ``<pkg_dir>/precision_oopao.npy`` to pick
#    float64/float32 precision. Upstream writes that file from the (broken,
#    now-skipped) package __init__, so we create it here when missing.
_precision_file = os.path.join(_pkg_dir, "precision_oopao.npy")
if not os.path.exists(_precision_file):
    try:
        np.save(_precision_file, 64)
    except OSError:
        pass  # read-only site-packages: fall back to float64 default below

# Replace the package entry in ``sys.modules`` with a lightweight *shadow*
# package object (a plain ModuleType carrying only ``__path__``). We never run
# upstream ``OOPAO/__init__.py`` — its broken ``sys.path`` search + missing
# ``tools`` import crash on numpy 2.x/Windows — yet relative imports inside
# the submodules (e.g. ``phaseStats``' ``from .tools.tools import ...``)
# resolve *through* this shadow parent instead of re-triggering the broken
# package init.
if _PKG_NAME in sys.modules:
    sys.modules.pop(_PKG_NAME, None)
_shadow = ModuleType(_PKG_NAME)
_shadow.__path__ = [_pkg_dir]
sys.modules[_PKG_NAME] = _shadow


def _load_submodule(name: str) -> ModuleType:
    path = os.path.join(_pkg_dir, f"{name}.py")
    spec = importlib.util.spec_from_file_location(
        f"{_PKG_NAME}.{name}", path, submodule_search_locations=[_pkg_dir]
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load OOPAO submodule {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


_loaded = {name: _load_submodule(name) for name in _MODULES}

Atmosphere = _loaded["Atmosphere"].Atmosphere
Source = _loaded["Source"].Source
Telescope = _loaded["Telescope"].Telescope

__all__ = ["Atmosphere", "Source", "Telescope"]
