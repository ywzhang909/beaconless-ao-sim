"""Generate the full paper-scale beaconless AO dataset in parallel chunks.

Why this script
---------------
- ``data.generate_h5`` runs the single-pass Algorithm-1 pipeline in one process.
  At 512-pixel resolution, n_roughness=10 and a 6144-pixel zero-padded Fresnel
  per imaging plane, each sample costs ~83 s of CPU. The full paper-scale
  dataset (81 000 + 9 000 + 1 000 = 91 000 samples) is therefore ~87 days
  single-process; on a 32-core box we want to use many cores at once.
- On Windows the OOPAO Telescope/Source carry C-level state that is not
  picklable, so we cannot reuse ``generate_dataset``'s ``multiprocessing.Pool``
  with ``spawn``. The fix is to launch multiple *independent* Python processes
  (each builds its own OOPAO Atmosphere), partition the 91 000 indices into
  ``--n_chunks`` disjoint ranges, and write per-chunk HDF5 files. A final merge
  step (``--merge``) concatenates the chunks into ``beaconless_full.h5``.

CLI
---
::

    # 1) Launch 8 parallel chunk workers (each runs in its own process)
    python -m data.generate_full --config config_full.yaml --n_chunks 8 \
        --out_dir data/chunks --launch

    # 2) After all workers finish, merge into the final HDF5
    python -m data.generate_full --config config_full.yaml --out_dir data/chunks \
        --merge --final_h5 data/beaconless_full.h5
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

from physics.config import SimConfig, load_config


CHUNK_DONE = "__done__"  # marker file written when a chunk finishes


def _chunk_index_range(chunk_id: int, n_chunks: int, n_total: int) -> tuple[int, int]:
    """Return the half-open [start, end) sample index range for ``chunk_id``."""
    if not (0 <= chunk_id < n_chunks):
        raise ValueError(f"chunk_id {chunk_id} out of [0, {n_chunks})")
    base = n_total // n_chunks
    rem = n_total % n_chunks
    start = chunk_id * base + min(chunk_id, rem)
    end = start + base + (1 if chunk_id < rem else 0)
    return start, end


def _write_chunk(
    cfg: SimConfig,
    start: int,
    end: int,
    chunk_h5: str,
    workers: int = 1,
) -> str:
    """Generate samples ``[start, end)`` into a per-chunk HDF5 file.

    Uses ``generate_dataset`` with the cfg's ``h5_path`` temporarily rewritten
    to ``chunk_h5`` and the global ``n_train/n_test/n_eval`` aligned to the
    chunk's size so that the H5 schema (split indices) stays consistent. The
    chunk's ``train/test/eval`` are recomputed relative to its own start.
    """
    from data.simulate import generate_dataset  # lazy import (heavy)

    n_chunk = end - start
    if n_chunk <= 0:
        # Empty chunk — just write the schema with no rows.
        with h5py.File(chunk_h5, "w") as f:
            N = int(cfg.physical.N)
            f.create_dataset(
                "images",
                (0, 3, N, N),
                dtype=np.uint16,
                chunks=(1, 3, N, N),
                maxshape=(None, 3, N, N),
            )
            f.create_dataset("labels", (0, 78), dtype=np.float32, maxshape=(None, 78))
            for k in ("fom_noao", "fom_track", "fom_beacon", "fom_z78", "seeds", "L"):
                f.create_dataset(
                    k,
                    (0,),
                    dtype=np.float32 if k == "L" else np.float32,
                    maxshape=(None,),
                )
            f.create_dataset("train_idx", (0,), dtype=np.int64, maxshape=(None,))
            f.create_dataset("test_idx", (0,), dtype=np.int64, maxshape=(None,))
            f.create_dataset("eval_idx", (0,), dtype=np.int64, maxshape=(None,))
            f.create_dataset("mu", (78,), dtype=np.float32)
            f.create_dataset("sigma", (78,), dtype=np.float32)
            f.create_dataset("scale_p", (3,), dtype=np.float32)
            f.create_dataset("vacuum_intensity", (N, N), dtype=np.float32)
            f.attrs["config_json"] = json.dumps(cfg.to_dict())
        return chunk_h5

    # Build a copy of cfg scoped to this chunk: indices [0, n_chunk) inside the
    # chunk file map to the original seeds [start, end).
    from copy import deepcopy

    cfg_chunk = deepcopy(cfg)
    # All chunk samples go into the TRAIN split of the chunk file (we re-split
    # at merge time using the original n_train/n_test/n_eval boundaries).
    cfg_chunk.data.n_train = n_chunk
    cfg_chunk.data.n_test = 0
    cfg_chunk.data.n_eval = 0
    cfg_chunk.data.h5_path = chunk_h5
    cfg_chunk.data.workers = max(1, workers)
    cfg_chunk.data.master_seed = int(cfg.data.master_seed) + start

    os.makedirs(os.path.dirname(os.path.abspath(chunk_h5)), exist_ok=True)
    generate_dataset(cfg_chunk)
    return chunk_h5


def _launch_chunks(
    cfg_path: str,
    n_chunks: int,
    out_dir: str,
    py_exec: str,
    log_dir: Optional[str] = None,
    workers: int = 1,
) -> list[subprocess.Popen]:
    """Launch ``n_chunks`` parallel ``generate_full`` processes and return them."""
    cfg = load_config(cfg_path)
    n_total = int(cfg.data.n_train) + int(cfg.data.n_test) + int(cfg.data.n_eval)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
    procs: list[subprocess.Popen] = []
    project_root = Path(__file__).resolve().parent.parent
    for cid in range(n_chunks):
        start, end = _chunk_index_range(cid, n_chunks, n_total)
        chunk_h5 = str(out / f"chunk_{cid:04d}.h5")
        log_file = None
        if log_dir:
            log_file = open(Path(log_dir) / f"chunk_{cid:04d}.log", "w")
        cmd = [
            py_exec,
            "-m",
            "data.generate_full",
            "--config",
            cfg_path,
            "--chunk",
            str(cid),
            "--n_chunks",
            str(n_chunks),
            "--out_h5",
            chunk_h5,
            "--workers",
            str(workers),
        ]
        print(
            f"[launch] chunk {cid:02d}/{n_chunks} indices [{start}, {end})  -> {chunk_h5}"
        )
        procs.append(
            subprocess.Popen(
                cmd,
                cwd=str(project_root),
                stdout=log_file or subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )
        )
    return procs


def _wait_chunks(procs: list[subprocess.Popen]) -> None:
    for p in procs:
        p.wait()
        if p.returncode != 0:
            raise RuntimeError(f"chunk worker exited with code {p.returncode}")


def _merge_chunks(
    cfg: SimConfig,
    out_dir: str,
    final_h5: str,
    n_chunks: int,
) -> str:
    """Concatenate per-chunk HDF5 files into ``final_h5`` (full schema)."""
    p = cfg.physical
    N = int(p.N)
    n_train = int(cfg.data.n_train)
    n_test = int(cfg.data.n_test)
    n_eval = int(cfg.data.n_eval)
    n_total = n_train + n_test + n_eval
    out = Path(out_dir)

    train_idx = np.arange(n_train, dtype=np.int64)
    test_idx = np.arange(n_train, n_train + n_test, dtype=np.int64)
    eval_idx = np.arange(n_train + n_test, n_total, dtype=np.int64)

    os.makedirs(os.path.dirname(os.path.abspath(final_h5)), exist_ok=True)
    if os.path.exists(final_h5):
        os.remove(final_h5)

    # We re-compute mu/sigma from the train split (Eq 14); scale_p is the
    # per-plane max over the train split.
    from data.simulate import _get_shared

    shared = _get_shared(cfg)
    mu_acc = np.zeros(78, dtype=np.float64)
    sq_acc = np.zeros(78, dtype=np.float64)
    plane_max = np.zeros(3, dtype=np.float64)
    n_train_proc = 0

    with h5py.File(final_h5, "w") as fout:
        fout.create_dataset(
            "images", (n_total, 3, N, N), dtype=np.uint16, chunks=(1, 3, N, N)
        )
        fout.create_dataset("labels", (n_total, 78), dtype=np.float32)
        for name in ("fom_noao", "fom_track", "fom_beacon", "fom_z78"):
            fout.create_dataset(name, (n_total,), dtype=np.float32)
        fout.create_dataset("seeds", (n_total,), dtype=np.int64)
        fout.create_dataset("L", (n_total,), dtype=np.float32)
        fout.create_dataset("train_idx", (n_train,), dtype=np.int64)
        fout.create_dataset("test_idx", (n_test,), dtype=np.int64)
        fout.create_dataset("eval_idx", (n_eval,), dtype=np.int64)
        fout.create_dataset("mu", (78,), dtype=np.float32)
        fout.create_dataset("sigma", (78,), dtype=np.float32)
        fout.create_dataset("scale_p", (3,), dtype=np.float32)
        fout.create_dataset("vacuum_intensity", (N, N), dtype=np.float32)
        fout.attrs["config_json"] = json.dumps(cfg.to_dict())

        # Stream per-chunk in order; chunk 0 is indices [0, n0), chunk 1 is
        # [n0, n0+n1), etc. Each chunk's local index j maps to global
        # g = chunk_start + j.
        for cid in range(n_chunks):
            start, end = _chunk_index_range(cid, n_chunks, n_total)
            chunk_h5 = str(out / f"chunk_{cid:04d}.h5")
            if not os.path.exists(chunk_h5):
                raise FileNotFoundError(f"missing chunk file {chunk_h5}")
            with h5py.File(chunk_h5, "r") as fc:
                n = fc["images"].shape[0]
                if n == 0:
                    continue
                # Block-copy (chunks of 64 images to keep memory bounded).
                COPY_CHUNK = 64
                for j in range(0, n, COPY_CHUNK):
                    j2 = min(j + COPY_CHUNK, n)
                    sl = slice(j, j2)
                    fout["images"][start + j : start + j2] = fc["images"][sl]
                    fout["labels"][start + j : start + j2] = fc["labels"][sl]
                    fout["fom_noao"][start + j : start + j2] = fc["fom_noao"][sl]
                    fout["fom_track"][start + j : start + j2] = fc["fom_track"][sl]
                    fout["fom_beacon"][start + j : start + j2] = fc["fom_beacon"][sl]
                    fout["fom_z78"][start + j : start + j2] = fc["fom_z78"][sl]
                    fout["seeds"][start + j : start + j2] = fc["seeds"][sl]
                    fout["L"][start + j : start + j2] = fc["L"][sl]
                # Train stats: only the train split (indices 0..n_train) feeds
                # mu/sigma. The first n_train global samples come from chunks
                # in order, so stop when we cross the train boundary.
                train_end_global = min(start + n, n_train)
                if start < n_train:
                    j_lo = 0
                    j_hi = train_end_global - start
                    sub = fc["labels"][j_lo:j_hi]
                    mu_acc += sub.sum(axis=0)
                    sq_acc += (sub.astype(np.float64) ** 2).sum(axis=0)
                    n_train_proc += j_hi - j_lo
                # Per-plane max over the train portion of this chunk.
                if start < n_train:
                    j_hi = min(n, n_train - start)
                    # images are uint16 (already per-image normalized to 2047),
                    # so per-plane max in the *quantized* domain is 2047; for
                    # the raw max we need the pre-quantization images, which
                    # we don't ship. Use 2047 as the per-plane max (schema
                    # compatibility only).
                    plane_max = np.maximum(plane_max, 2047.0)
                # vacuum_intensity (same in every chunk, take from first).
                if cid == 0:
                    fout["vacuum_intensity"][:] = fc["vacuum_intensity"][:]

        if n_train_proc != n_train:
            raise RuntimeError(
                f"train stats covered {n_train_proc} != n_train {n_train}"
            )
        mu = mu_acc / n_train
        sigma = np.sqrt(np.maximum(sq_acc / n_train - mu**2, 0.0))
        fout["mu"][:] = mu.astype(np.float32)
        fout["sigma"][:] = sigma.astype(np.float32)
        fout["scale_p"][:] = plane_max.astype(np.float32)
        fout["train_idx"][:] = train_idx
        fout["test_idx"][:] = test_idx
        fout["eval_idx"][:] = eval_idx
    return final_h5


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full paper-scale chunked dataset generation."
    )
    parser.add_argument("--config", required=True, help="YAML config path.")
    parser.add_argument(
        "--n_chunks", type=int, default=8, help="Number of parallel chunks."
    )
    parser.add_argument(
        "--out_dir", default="data/chunks", help="Per-chunk h5 output dir."
    )
    parser.add_argument(
        "--log_dir", default="data/chunks/logs", help="Per-chunk log dir."
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Workers per chunk (forced 1 on Win)."
    )
    parser.add_argument(
        "--py_exec", default=sys.executable, help="Python executable for workers."
    )
    # Mutually exclusive worker actions:
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--launch", action="store_true", help="Launch n_chunks parallel workers."
    )
    action.add_argument(
        "--chunk", type=int, help="Worker mode: process this chunk id only."
    )
    action.add_argument(
        "--merge", action="store_true", help="Merge chunks into final h5."
    )
    parser.add_argument(
        "--out_h5", default=None, help="Per-chunk h5 path (worker mode)."
    )
    parser.add_argument(
        "--final_h5", default=None, help="Final merged h5 path (merge mode)."
    )
    args = parser.parse_args()

    if args.chunk is not None:
        # Worker mode: generate samples for chunk id, write to args.out_h5.
        cfg = load_config(args.config)
        n_total = int(cfg.data.n_train) + int(cfg.data.n_test) + int(cfg.data.n_eval)
        start, end = _chunk_index_range(args.chunk, args.n_chunks, n_total)
        out_h5 = args.out_h5 or os.path.join(args.out_dir, f"chunk_{args.chunk:04d}.h5")
        print(
            f"[worker] chunk {args.chunk}/{args.n_chunks} indices [{start}, {end})  -> {out_h5}"
        )
        t0 = time.time()
        _write_chunk(cfg, start, end, out_h5, workers=args.workers)
        print(f"[worker] done in {time.time() - t0:.1f} s")
        return

    if args.launch:
        procs = _launch_chunks(
            args.config,
            args.n_chunks,
            args.out_dir,
            args.py_exec,
            args.log_dir,
            args.workers,
        )
        _wait_chunks(procs)
        return

    if args.merge:
        cfg = load_config(args.config)
        final = args.final_h5 or os.path.join(args.out_dir, "..", "beaconless_full.h5")
        out = _merge_chunks(cfg, args.out_dir, final, args.n_chunks)
        print(f"[merge] wrote {out}")


if __name__ == "__main__":
    main()
