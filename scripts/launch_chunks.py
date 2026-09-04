"""Launch n_chunks parallel Python workers for chunked full-dataset generation.

Run via:  python scripts/launch_chunks.py --config config_demo_full.yaml --n_chunks 4
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--n_chunks", type=int, required=True)
    p.add_argument("--out_dir", default="data/chunks")
    p.add_argument("--log_dir", default=None)
    args = p.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    log_dir = args.log_dir or os.path.join(args.out_dir, "logs")
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    py_exec = sys.executable
    procs = []
    for cid in range(args.n_chunks):
        chunk_h5 = os.path.join(args.out_dir, f"chunk_{cid:04d}.h5")
        log_path = os.path.join(log_dir, f"chunk_{cid:04d}.log")
        log = open(log_path, "w")
        cmd = [
            py_exec,
            "-m",
            "data.generate_full",
            "--config",
            args.config,
            "--chunk",
            str(cid),
            "--n_chunks",
            str(args.n_chunks),
            "--out_h5",
            chunk_h5,
            "--workers",
            "1",
        ]
        print(
            f"[launch] chunk {cid:02d}/{args.n_chunks} -> {chunk_h5}  (log: {log_path})"
        )
        procs.append(
            subprocess.Popen(cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT)
        )
    print(f"[launch] all {args.n_chunks} workers started; waiting...")
    rc = 0
    for proc in procs:
        proc.wait()
        if proc.returncode != 0:
            print(f"[launch] worker exited with code {proc.returncode}")
            rc = proc.returncode
    sys.exit(rc)


if __name__ == "__main__":
    main()
