"""Detection-method tests for the scaled-FFT Fresnel propagator.

These encode the findings of the post-de1524e code review:

* Bug (fixed): the post-multiply quadratic phase was evaluated on the input
  grid (``dx``) instead of the scaled-FFT *output* grid
  ``dx2 = lam*z/(N*dx)``.  The returned field phase did not match the
  documented Fresnel convention (the impulse-response tests below fail with
  ``rel_err ~ sqrt(2)`` under the bug, ~1e-8 after the fix).  Intensity
  ``|E|^2`` never depended on the post phase (unit-modulus factor), so all
  imaging metrics were unaffected.
* Non-bugs documented as measurement guidance:
  - Naive power ratio ``sum|E_out|^2 / sum|E_in|^2`` equals
    ``(dx/dx2)^2``, NOT 1 -- power is conserved only *area-weighted*
    (``sum|E|^2 * dA``).  `test_area_weighted_power_conservation` pins this.
  - Fresnel-vs-ASM pixelwise comparison is only valid on matched grids
    (``dx2 ~ dx``); at project distances the grids differ by ~10x, which
    explains apparent "mismatch".  `test_gaussian_beam_analytic` verifies
    the physics independently of any grid convention.
"""

import numpy as np
import pytest

from physics.propagation_fft import Propagator

N = 128
DX = 0.01
LAM = 800e-9


@pytest.fixture(scope="module")
def prop():
    return Propagator(N=N, dx=DX, lam=LAM)


def _fresnel_impulse_response(prop, z, n_pad):
    """Analytic Fresnel impulse response of a unit point source.

    h(x2) = e^{ikz}/(i·λ·z) · dx² · e^{ik x2²/(2z)}, sampled on the scaled-FFT
    output grid dx2 = λz/(N_pad·dx) — the convention fresnel_padded must
    return.
    """
    dx2 = prop.lam * z / (n_pad * prop.dx)
    vals = (np.arange(n_pad) - (n_pad - 1) / 2.0) * dx2
    X, Y = np.meshgrid(vals, vals)
    k = 2.0 * np.pi / prop.lam
    h = np.exp(1j * k * z) / (1j * prop.lam * z)
    return h * (prop.dx**2) * np.exp(1j * k * (X**2 + Y**2) / (2.0 * z))


def _impulse_matches_analytic(prop, E_out, z, n_pad):
    """Compare a propagated field with the analytic impulse response *modulo a
    global phase*.  A point source placed at integer pixel ``N//2`` sits half
    a pixel from the grid origin ``(N-1)/2``; the DFT then carries a constant
    phase that is physically meaningless (absolute phase is unobservable).
    The pre-fix grid bug instead produced a position-dependent error that no
    global phase can remove, so this comparison still detects it."""
    h = _fresnel_impulse_response(prop, z, n_pad)
    center = n_pad // 2
    g = E_out[center, center] / h[center, center]
    return np.linalg.norm(E_out - g * h) / np.linalg.norm(h)


def test_fresnel_padded_impulse_response_matches_analytic(prop):
    """A unit point source must reproduce the analytic impulse response on the
    *output* grid -- catches any post-phase evaluated on the wrong grid."""
    z = 1000.0
    n_pad = 512
    E_in = np.zeros((N, N), dtype=np.complex64)
    E_in[N // 2, N // 2] = 1.0
    E_out = prop.fresnel_padded(E_in, z, n_pad)
    # float32 rounding floor after the fix is ~1e-4 (measured 3.2e-8 at
    # complex64); the pre-fix grid bug sat at ~1.41 (sqrt 2).
    assert _impulse_matches_analytic(prop, E_out, z, n_pad) < 1e-3


def test_fresnel_propagate_impulse_response_matches_analytic(prop):
    z = 1000.0
    E_in = np.zeros((N, N), dtype=np.complex64)
    E_in[N // 2, N // 2] = 1.0
    E_out = prop.fresnel_propagate(E_in, z)
    assert _impulse_matches_analytic(prop, E_out, z, N) < 1e-3  # n_pad == N


def test_area_weighted_power_conservation(prop):
    """Fresnel propagation is unitary: sum|E|^2·dA must be conserved.  Without
    the dx2-vs-dx area weight the naive sum ratio is (dx/dx2)^2, so a raw
    sum comparison falsely reports non-conservation."""
    rng = np.random.default_rng(0)
    E = (rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))).astype(
        np.complex64
    )
    P_in = np.sum(np.abs(E) ** 2) * prop.dx**2
    for z, n_pad in [(642.43, 2048), (1284.86, 4096), (1927.3, 6144)]:
        E2 = prop.fresnel_padded(E, z, n_pad)
        dx2 = prop.lam * z / (n_pad * prop.dx)
        P_out = np.sum(np.abs(E2) ** 2) * dx2**2
        assert abs(P_in - P_out) / P_in < 0.02


def test_gaussian_beam_matches_analytic():
    """Independent physical cross-check (no grid convention): a Gaussian beam
    under Fresnel must follow the analytic w(z) expansion.  The residual
    (~4-6% at z/zR=0.5-1.0) is the scaled-FFT output-grid sampling error
    (dx2/w_z), which shrinks at the shorter distances the pipeline uses.

    NB: the analytic profile must be evaluated on the *output* grid
    (pixel scale dx2 = λz/(N·dx), NOT the input dx) -- evaluating it on the
    input grid is a test bug that falsely reports ~60% error."""
    N2, dx2_in = 256, 0.005
    w0 = 0.06  # m; 12 px per waist: well-sampled input
    prop2 = Propagator(N=N2, dx=dx2_in, lam=LAM)
    xy = (np.arange(N2) - (N2 - 1) / 2.0) * dx2_in
    X, Y = np.meshgrid(xy, xy)
    r2 = X**2 + Y**2
    E0 = np.exp(-(r2 / w0**2)).astype(np.complex64)
    zR = np.pi * w0**2 / LAM
    for frac in [0.5, 1.0]:
        z = frac * zR
        I = prop2.fresnel_intensity(E0, z)
        w_z = w0 * np.sqrt(1.0 + (z / zR) ** 2)
        dx2 = LAM * z / (N2 * dx2_in)  # output pixel scale
        xc = (np.arange(N2) - (N2 - 1) / 2.0) * dx2
        prof = I[N2 // 2, :].astype(np.float64)
        prof /= np.max(prof)
        analytic = np.exp(-2.0 * xc**2 / w_z**2)
        core = np.abs(xc) < 1.5 * w_z
        err = np.max(np.abs(prof[core] - analytic[core]))
        assert err < 0.10, f"z/zR={frac}: max profile err {err:.3f}"


def test_focal_plane_airy_encircled_energy():
    """Flat-top circular aperture + ideal lens focused on the image plane ->
    Airy pattern.  The 86%-encircled-energy radius must be 1.583·λf/D
    (Airy: EE(a)=1-J0(a)²-J1(a)² = 0.86 at a=πDr/(λf) = 4.973).

    Uses the *exact pipeline geometry* (N=512, dx=5.8594e-4, f=1284.86 m,
    D=0.3 m, N_pad=4096) so the test exercises the same numbers as
    data/simulate.py.  Two test-setup pitfalls are encoded here:
    * the lens phase must be centered on the grid origin (N-1)/2 -- the usual
      N//2 center shifts the spot by half an aperture pixel;
    * the quadratic lens phase must be sampled: k·(D/2)·dx/f < π.  On the
      coarse N=128/dx=0.01 grid with f=1284.86 that criterion needs
      f > D·dx/λ = 10000 m, so it is violated by ~8x (aliased lens)."""
    Np, DXp, F, D, N_PAD = 512, 5.8594e-4, 1284.86, 0.30, 4096
    prop = Propagator(N=Np, dx=DXp, lam=LAM)
    cx = cy = (Np - 1) / 2.0  # grid origin, NOT Np//2
    yy, xx = np.mgrid[0:Np, 0:Np]
    mask = ((xx - cx) ** 2 + (yy - cy) ** 2 <= (D / (2 * DXp)) ** 2).astype(
        np.float32
    )
    k = 2.0 * np.pi / LAM
    r2_lens = ((xx - cx) * DXp) ** 2 + ((yy - cy) * DXp) ** 2
    E_l = mask.astype(np.complex64) * np.exp(-1j * k * r2_lens / (2.0 * F))
    E_f = prop.fresnel_padded(E_l, F, N_PAD)
    c = (N_PAD - Np) // 2
    I = np.abs(E_f[c : c + Np, c : c + Np]) ** 2
    dx2 = LAM * F / (N_PAD * DXp)
    pk = np.unravel_index(np.argmax(I), I.shape)  # (256,256) when centered
    yy2, xx2 = np.mgrid[0:Np, 0:Np]
    rr = np.sqrt((xx2 - pk[1]) ** 2 + (yy2 - pk[0]) ** 2) * dx2
    order = np.argsort(rr.ravel())
    cum = np.cumsum(I.ravel()[order])
    r86 = rr.ravel()[order][np.searchsorted(cum, 0.86 * cum[-1])]
    theory = (4.973 / np.pi) * LAM * F / D
    assert abs(r86 - theory) / theory < 0.10, (
        f"r86={r86*1e3:.2f}mm theory={theory*1e3:.2f}mm"
    )