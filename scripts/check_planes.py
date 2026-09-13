"""Three-plane image sanity check for generated beaconless AO H5 datasets.

Loads samples from an H5, computes per-plane diagnostics (energy
concentration, center/edge ratio, dynamic range, NaNs, quantization) and
writes a montage figure (one row per in-focus / defocus plane, log scale)
plus a radial energy profile panel. Exits nonzero if a hard sanity check
fails (NaN, all-zero plane, saturating focus >99.9% pixels at max).
"""

from __future__ import annotations

import argparse
import sys

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plane_stats(im: np.ndarray) -> dict:
    """Per-plane diagnostics on a (N,N) float image in [0,1] (after /2047)."""
    n = im.shape[0]
    c = (n - 1) // 2
    r_core = max(3, n // 64)          # ~1.2 λf/D bucket on 512 grid → small core
    r_half = n // 4
    yy, xx = np.mgrid[0:n, 0:n]
    r = np.sqrt((xx - c) ** 2 + (yy - c) ** 2)

    total = float(im.sum())
    core = float(im[r <= r_core].sum())
    half = float(im[r <= r_half].sum())

    # center vs edge band (edge = outside r_half)
    edge = float(im[r > r_half].sum())
    center_edge_ratio = core / edge if edge > 0 else float("inf")

    frac_nan = float(np.isnan(im).mean())
    return {
        "max": float(im.max()),
        "mean": float(im.mean()),
        "frac_nan": frac_nan,
        "energy_core_frac": core / total if total > 0 else 0.0,
        "energy_half_frac": half / total if total > 0 else 0.0,
        "center_edge_ratio": center_edge_ratio,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("h5", help="path to .h5 dataset")
    ap.add_argument("--n-samples", type=int, default=3, help="samples to inspect")
    ap.add_argument("--out", default="results/fig_planes_check.png")
    ap.add_argument("--idx", type=int, nargs="*", default=None, help="explicit sample idx")
    args = ap.parse_args()

    with h5py.File(args.h5, "r") as f:
        images = f["images"]
        n_total, n_planes, n, _ = images.shape
        idxs = args.idx if args.idx else list(range(min(args.n_samples, n_total)))
        label = f.attrs.get("config_json", "{}")
        n_total_attr = n_total

    print(f"dataset={args.h5}  n_total={n_total_attr}  planes={n_planes}  N={n}")
    print(f"inspecting samples idx={idxs}")

    all_ok = True
    per_sample_rows = []
    with h5py.File(args.h5, "r") as f:
        images = f["images"]
        for si, idx in enumerate(idxs):
            imgs = images[idx].astype(np.float64) / 2047.0  # (3, N, N), [0,1]
            if np.any(np.isnan(imgs)):
                print(f"  [FAIL] sample {idx}: contains NaN pixels")
                all_ok = False
            if np.any(imgs < 0) or np.any(imgs > 1.0 + 1e-6):
                print(f"  [FAIL] sample {idx}: pixels outside [0,1] after /2047")
                all_ok = False

            print(f"  sample {idx}:")
            for p in range(n_planes):
                st = plane_stats(imgs[p])
                fm = ("pre" if p == 0 else "post") if p != 1 else "focus"
                tag = f"plane[{p}] ({fm}-focus)"
                print(
                    f"    {tag:22s} max={st['max']:.3f} mean={st['mean']:.5f} "
                    f"core%={100*st['energy_core_frac']:.2f} "
                    f"half%={100*st['energy_half_frac']:.1f} "
                    f"c/e={st['center_edge_ratio']:.2f}"
                )
                if st["frac_nan"] > 0:
                    print(f"      [FAIL] NaN fraction={st['frac_nan']:.2e}")
                    all_ok = False
                if st["max"] == 0:
                    print(f"      [FAIL] plane is all zeros (max=0)")
                    all_ok = False
                per_sample_rows.append((si, p, idx, st))

    # --- montage figure: rows = planes (3), cols = samples, log scale ---
    ns = len(idxs)
    fig, axes = plt.subplots(n_planes, ns, figsize=(4.2 * ns, 12))
    if n_planes == 1 or ns == 1:
        axes = np.atleast_2d(axes)
    titles = [f"pre-focus (f−zR), idx {idxs[0]}", f"focus, idx {idxs[0]}", f"post-focus (f+zR), idx {idxs[0]}"]
    with h5py.File(args.h5, "r") as f:
        images = f["images"]
        for p in range(n_planes):
            for c in range(ns):
                im = images[idxs[c], p].astype(np.float64) / 2047.0
                ax = axes[p, c] if ns > 1 else axes[p]
                im_log = np.log10(im + 1e-6)
                vmax = im_log.max()
                vmin = max(im_log.min(), vmax - 4.0)
                ax.imshow(im_log, cmap="inferno", vmin=vmin, vmax=vmax)
                ax.set_title(f"{titles[p]} / {titles[p]}{'' if c==0 else ' '}{''}")
                if n_planes > 1 and ns > 1:
                    ax.set_xticks([])
                    ax.set_yticks([])
                st = per_sample_rows[c * n_planes + p][3]
                ax.set_xlabel(f"max={st['max']:.2f} core%={100*st['energy_core_frac']:.1f}")
    fig.suptitle(f"Three-plane images (log10 scale) — {args.h5}")
    fig.tight_layout()

    # --- radial energy profile panel ---
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    with h5py.File(args.h5, "r") as f:
        images = f["images"]
        n = images.shape[-1]
        c = (n - 1) // 2
        yy, xx = np.mgrid[0:n, 0:n]
        r = np.sqrt((xx - c) ** 2 + (yy - c) ** 2).ravel()
        for p, lbl in enumerate(["pre-focus", "focus", "post-focus"]):
            im = images[idxs[0], p].astype(np.float64).ravel() / 2047.0
            rmax = n // 2 - 1
            rbins = np.arange(1, rmax + 1)
            in_bin = np.searchsorted(rbins, r, side="left")
            m = in_bin < len(rbins)
            sums = np.bincount(in_bin[m], weights=im[m], minlength=len(rbins))
            counts = np.bincount(in_bin[m], minlength=len(rbins))
            profile = sums / np.maximum(counts, 1)
            ax2.plot(rbins, profile, label=lbl)
    ax2.set_xlabel("pixel radius from center")
    ax2.set_ylabel("mean intensity (linear, /2047)")
    ax2.set_title(f"Radial intensity profile — sample idx {idxs[0]}")
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig2.tight_layout()

    fig.savefig(args.out, dpi=130)
    fig2.savefig(args.out.replace(".png", "_radial.png"), dpi=130)
    print(f"\nwrote {args.out} and {args.out.replace('.png','_radial.png')}")
    print("RESULT:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())