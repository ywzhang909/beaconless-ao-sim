"""Print summary of a beaconless AO HDF5 dataset."""

import argparse
import h5py
import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("h5", help="Path to HDF5 dataset")
    args = p.parse_args()

    with h5py.File(args.h5, "r") as f:
        n_total = f["images"].shape[0]
        n_train = f["train_idx"].shape[0]
        n_test = f["test_idx"].shape[0]
        n_eval = f["eval_idx"].shape[0]
        N = f["images"].shape[2]
        print("=" * 60)
        print(f"HDF5 dataset: {args.h5}")
        print(f"  N_total = {n_total}  (train={n_train}, test={n_test}, eval={n_eval})")
        print(f"  images shape = (N_total, 3, {N}, {N}) uint16")
        print(f"  labels shape = (N_total, 78) float32")
        if n_total == 0:
            print("  (empty dataset)")
            return
        images = f["images"][:1]  # one image is enough to check max
        print("-" * 60)
        print(f"  scale_p = {f['scale_p'][:]}")
        print("-" * 60)
        print("Label mu/std (first 8 modes):")
        mu = f["mu"][:]
        sigma = f["sigma"][:]
        for j in range(min(8, 78)):
            print(f"  mode {j}: mu={mu[j]:+.4f}  sigma={sigma[j]:.4f}")
        print("-" * 60)
        print("Median FOMs (full dataset):")
        for leg in ("noao", "track", "beacon", "z78"):
            arr = f[f"fom_{leg}"][:]
            print(
                f"  {leg:6s}: median={np.median(arr):.4f}  mean={arr.mean():.4f}  n={arr.size}"
            )


if __name__ == "__main__":
    main()
