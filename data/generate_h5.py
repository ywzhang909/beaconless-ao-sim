"""CLI for generating the beaconless AO HDF5 dataset.

Usage::

    python -m data.generate_h5 --config config.yaml [-n | --dry]

Reads the YAML configuration, runs :func:`data.simulate.generate_dataset`, and
prints a summary (splits, per-plane max, label mu/std, median FOMs).

中文：生成 beaconless AO 的 HDF5 数据集的命令行入口（CLI）。
用法：python -m data.generate_h5 --config config.yaml [-n | --dry]
读取 YAML 配置，调用 data.simulate.generate_dataset（算法 1 两趟生成），
并打印摘要（数据划分、逐平面最大值、标签 mu/sigma、各分支中位 FOM）。
"""

from __future__ import annotations

import argparse
import json
import time

import h5py
import numpy as np
import yaml

from data.simulate import generate_dataset


def _load_cfg(path: str) -> dict:
    """Load and return the YAML configuration dictionary.

    中文：读取并返回 YAML 配置字典。
    参数 path: 配置文件路径（如 config.yaml）。
    返回: 完整配置字典（含 physical / data / model / training 等节）。
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _print_summary(h5_path: str) -> None:
    """Print a summary of the generated HDF5 dataset.

    中文：打印已生成 HDF5 数据集的摘要。
    参数 h5_path: HDF5 文件路径。
    摘要内容：样本总数与各划分大小、图像/标签形状、逐平面最大值与 scale_p、
    前 8 个 Zernike 模式的 mu/sigma（公式 14）、四个分支的中位 FOM。
    """
    with h5py.File(h5_path, "r") as f:
        # 读取形状与索引划分
        n_total = f["images"].shape[0]    # 总样本数
        n_train = f["train_idx"].shape[0]  # 训练集样本数
        n_test = f["test_idx"].shape[0]    # 测试集样本数
        n_eval = f["eval_idx"].shape[0]    # 评估集样本数
        N = f["images"].shape[2]           # 网格分辨率 (512)

        # 全量读入（演示规模 ~2.5k 样本，可承受）
        images = f["images"][:]            # (N_total, 3, N, N) uint16
        labels = f["labels"][:]            # (N_total, 78) float32
        scale_p = f["scale_p"][:]          # (3,) 逐平面 raw 最大值
        mu = f["mu"][:]                    # (78,) 标签均值
        sigma = f["sigma"][:]              # (78,) 标签标准差

        # 四个分支的逐样本 FOM
        fom_noao = f["fom_noao"][:]        # 无 AO
        fom_track = f["fom_track"][:]      # 仅跟踪
        fom_beacon = f["fom_beacon"][:]    # 信标相位共轭
        fom_z78 = f["fom_z78"][:]          # 78 阶上界

    print("=" * 60)
    print(f"HDF5 dataset: {h5_path}")
    print(f"  N_total = {n_total}  (train={n_train}, test={n_test}, eval={n_eval})")
    print(f"  images shape = (N_total, 3, {N}, {N}) uint16")
    print(f"  labels shape = (N_total, 78) float32")
    print("-" * 60)
    print("Per-plane max (quantized, /2047):")
    for p in range(3):
        print(f"  plane {p}: max={images[:, p].max()}, scale_p={scale_p[p]:.4g}")
    print("-" * 60)
    print("Label mu/std (first 8 modes):")
    for j in range(min(8, 78)):
        print(f"  mode {j}: mu={mu[j]:+.4f}  sigma={sigma[j]:.4f}")
    print("-" * 60)
    print("Median FOMs:")
    print(f"  noao   : {np.median(fom_noao):.4f}")
    print(f"  track  : {np.median(fom_track):.4f}")
    print(f"  beacon : {np.median(fom_beacon):.4f}")
    print(f"  z78    : {np.median(fom_z78):.4f}")
    print("=" * 60)


def main() -> None:
    """Entry point for the ``python -m data.generate_h5`` CLI.

    中文：``python -m data.generate_h5`` 命令行入口。
    参数（argparse）：
      --config (必填): YAML 配置路径（如 config.yaml）。
      -n / --dry: 干跑——只打印配置与计划的数据集，不实际生成。
    流程：加载配置 -> (干跑则打印后返回) -> 调用 generate_dataset 生成
    HDF5 -> 计时 -> 打印生成耗时与摘要。
    """
    parser = argparse.ArgumentParser(
        description="Generate the beaconless AO HDF5 dataset (Algorithm 1)."
    )
    # --config：必填，YAML 配置路径
    parser.add_argument("--config", required=True, help="Path to the YAML config.")
    # -n/--dry：干跑标志（store_true），只打印不生成
    parser.add_argument(
        "-n",
        "--dry",
        action="store_true",
        help="Dry run: print the configuration and planned dataset, do not generate.",
    )
    args = parser.parse_args()

    # 加载配置（_load_cfg 读取 YAML -> 字典）
    cfg = _load_cfg(args.config)

    if args.dry:
        # 干跑：打印物理参数、数据划分与输出路径，不触发生成
        d = cfg["data"]
        p = cfg["physical"]
        n_total = d["n_train"] + d["n_test"] + d["n_eval"]  # 计划总样本数
        print("DRY RUN (no generation):")
        print(f"  config: {args.config}")
        print(f"  N={p['N']}, L={p['L']} m, cn2={p['cn2']:.3e}")
        print(
            f"  splits: train={d['n_train']}, test={d['n_test']}, "
            f"eval={d['n_eval']}  (N_total={n_total})"
        )
        print(f"  workers={d['workers']}, master_seed={d['master_seed']}")
        print(f"  h5_path={d['h5_path']}")
        print(f"  config_json={json.dumps(cfg)[:120]}...")
        return

    # 实际生成：计时并调用两趟流水线，随后打印摘要
    t0 = time.time()
    h5_path = generate_dataset(cfg)  # 返回写出的 HDF5 路径
    elapsed = time.time() - t0
    print(f"\nGeneration completed in {elapsed:.1f} s -> {h5_path}")
    _print_summary(h5_path)


if __name__ == "__main__":
    main()
