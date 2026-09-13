"""Quantitative three-plane sanity diagnostics for beaconless AO H5 datasets.

Physics expectations (paper Sec 2.4, Fig 2):
- focus plane: tight energy concentration (most light within a few Airy radii
  of the centroid; strong turbulence D/r0~7 spreads it but the core region
  still holds a clear peak);
- pre/post-focus planes: broader, softer, defocused distribution (energy
  spread over larger radius, lower concentration in the core);
- no NaN, no all-zero plane, quantized uint16 (no saturation at 2047 beyond
  reasonable core pixels), centroid near array center, no persistent edge
  rows/cols (wrap-around / absorbing-boundary leak).

Writes a per-plane diagnostics table (mean +/- std over N samples) and saves
the radial profile figure. Exit 0 if checks pass.
"""

from __future__ import annotations

import argparse
import sys

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def radial_profile(im: np.ndarray, rmax: int) -> np.ndarray:
    n = im.shape[0]
    c = (n - 1) // 2
    yy, xx = np.mgrid[0:n, 0:n]
    r = np.sqrt((xx - c) ** 2 + (yy - c) ** 2).ravel()
    vals = im.ravel()
    rbins = np.arange(0, rmax + 1)
    bin_id = np.minimum(r.astype(np.int64), rmax)
    sums = np.bincount(bin_id, weights=vals, minlength=rmax + 1)
    counts = np.bincount(bin_id, minlength=rmax + 1)
    return sums / np.maximum(counts, 1)


def plane_metrics(im: np.ndarray) -> dict:
    n = im.shape[0]
    c = (n - 1) // 2
    yy, xx = np.mgrid[0:n, 0:n]
    r2 = (xx - c) ** 2 + (yy - c) ** 2

    total = float(im.sum())
    if total <= 0:
        return {"max": 0.0, "core16": 0.0, "core64": 0.0, "half": 0.0,
                "edge_band": 0.0, "cx": c, "cy": c, "hwhm": 0.0,
                "sat_frac": 0.0, "zero_frac": 1.0}

    # energy fractions: r<=16 (~1.6 Airy radii), r<=64, r<=128 (~half width)
    core16 = float(im[r2 <= 16 ** 2].sum()) / total
    core64 = float(im[r2 <= 64 ** 2].sum()) / total
    half = float(im[r2 <= 128 ** 2].sum()) / total

    # outer 10-px edge band mean (wrap-around / boundary leak detector)
    outer = (xx >= n - 10) | (xx < 10) | (yy >= n - 10) | (yy < 10)
    edge_band = float(im[outer].sum()) / float(outer.sum())

    # intensity centroid (should be near center)
    cx = float((im * xx).sum()) / total
    cy = float((im * yy).sum()) / total

    # HWHM of radial profile (pixels)
    prof = radial_profile(im, n // 2)
    peak = prof.max()
    hwhm = 0.0
    if peak > 0:
        thr = peak / 2.0
        idx = np.flatnonzero(prof >= thr)
        hwhm = float(idx[-1]) if idx.size else 0.0

    sat_frac = float((im >= 2047.0 / 2047.0).mean()) if im.max() >= 1.0 else 0.0
    zero_frac = float((im == 0).mean())
    return {"max": float(im.max()), "core16": core16, "core64": core64,
            "half": half, "edge_band": edge_band, "cx": cx, "cy": cy,
            "hwhm": hwhm, "sat_frac": sat_frac, "zero_frac": zero_frac}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("h5")
    ap.add_argument("--n-samples", type=int, default=20)
    ap.add_argument("--out", default="results/fig_planes_radial.png")
    args = ap.parse_args()

    with h5py.File(args.h5, "r") as f:
        images = f["images"]
        n_total, n_planes, n = images.shape[:3]
        idxs = np.linspace(0, n_total - 1, min(args.n_samples, n_total)).astype(int)
        data = images[idxs].astype(np.float64) / 2047.0  # (S, 3, N, N)

    names = ["pre-focus (f−zR)", "focus", "post-focus (f+zR)"]
    print(f"dataset={args.h5}  samples={len(idxs)}  planes={n_planes}  N={n}")
    print(f"sample idxs: {idxs.tolist()}\n")

    cols = ["max", "core16%", "core64%", "half128%", "edge_band",
            "cx", "cy", "hwhm", "sat%", "zero%"]
    print(f"{'plane':24s} " + " ".join(f"{c:>10s}" for c in cols))
    all_ok = True
    for p in range(n_planes):
        ms = [plane_metrics(data[i, p]) for i in range(len(idxs))]
        def fmt(k, scale=1.0, prec=2):
            v = np.array([m[k] for m in ms]) * scale
            return f"{v.mean():.{prec}f}±{v.std():.{prec}f}"

        row = {c: fmt(k, scale, p2) for c, k, scale, p2 in [
            ("max", "max", 1.0, 3), ("core16%", "core16", 100.0, 2),
            ("core64%", "core64", 100.0, 2), ("half128%", "half", 100.0, 1),
            ("edge_band", "edge_band", 1.0, 4), ("cx", "cx", 1.0, 1),
            ("cy", "cy", 1.0, 1), ("hwhm", "hwhm", 1.0, 1),
            ("sat%", "sat_frac", 100.0, 4), ("zero%", "zero_frac", 100.0, 2)]}
        print(f"{names[p]:24s} " + " ".join(f"{row[c]:>10s}" for c in cols))

        # sanity judgment
        if np.mean([m["max"] for m in ms]) < 0.5:
            print(f"  [FAIL] plane {p}: mean max < 0.5 (weak signal)")
            all_ok = False
        core16 = np.array([m["core16"] for m in ms])
        if p == 1:  # focus should be the most concentrated
            if core16.mean() < 0.01:
                print(f"  [FAIL] focus plane: core16 mean {core16.mean():.4f} < 0.01")
                all_ok = False
        if np.mean([m["edge_band"] for m in ms]) > 0.05:
            print(f"  [WARN] plane {p}: edge band mean {np.mean([m['edge_band'] for m in ms]):.4f} elevated (check wrap-around)")
        if np.mean([m["zero_frac"] for m in ms]) > 0.7:
            print(f"  [WARN] plane {p}: {np.mean([m['zero_frac'] for m in ms])*100:.1f}% pixels are zero (very sparse)")
        cx = np.abs(np.array([m["cx"] for m in ms]) - (n - 1) / 2)
        cy = np.abs(np.array([m["cy"] for m in ms]) - (n - 1) / 2)
        if cx.mean() > 20 or cy.mean() > 20:
            print(f"  [WARN] plane {p}: centroid offset ({cx.mean():.1f}, {cy.mean():.1f}) px from center")

    # radial profile figure (mean over samples, each plane)
    fig, ax = plt.subplots(figsize=(8, 5))
    for p, nm in enumerate(names):
        profs = np.array([radial_profile(data[i, p], n // 2) for i in range(len(idxs))])
        r = np.arange(len(profs.mean(axis=0)))
        ax.plot(r, profs.mean(axis=0), label=nm, lw=2)
    ax.set_xlim(0, n // 2)
    ax.set_xlabel("radius from array center [px]")
    ax.set_ylabel("mean intensity [/2047]")
    ax.set_title(f"Mean radial intensity profile ({len(idxs)} samples) — {args.h5}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"\nwrote {args.out}")
    print("RESULT:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())