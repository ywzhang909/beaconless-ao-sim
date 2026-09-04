"""Typed configuration for beaconless-ao-sim.

将 ``config.yaml`` 解析为嵌套 dataclass 族：全部配置项以属性访问
（``cfg.physical.N``）取代 dict 下标访问（``cfg["physical"]["N"]``），
每个字段附带物理含义与单位的注释。用法::

    from physics.config import load_config
    cfg = load_config("config.yaml")          # -> SimConfig
    print(cfg.physical.N, cfg.bucket.diameter_frac)
    cfg_dict = cfg.to_dict()                  # -> 供 json.dumps / wandb 使用

参数取值与 ``config.yaml`` 完全对齐（论文 Table 1 逐字复现 + 演示规模偏差
在 yaml 中记录）。运行时由 ``attach_run`` / ``attach_eval`` 注入的字段
（``run`` 段、``eval.ckpt_path`` 等）也定义为带默认值的字段，保证
dataclass 与旧 dict 的读写语义一致、可被测试 monkeypatch。
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import yaml


# =============================================================================
# physical —— 大气 + 光学 + 仿真网格参数（论文 Table 1）
# =============================================================================
@dataclass
class PhysicalConfig:
    # --- 大气参数 (Table 1) ---
    cn2: float = 8.13e-15
    # 大气折射率结构常数 Cn^2 [m^-2/3]：衡量湍流强度，决定 r0 与 Rytov 方差。
    l0: float = 0.0
    # 湍流内尺度 [m]。论文取 0（与仿真保护无关——真正的除法保护在 l0_sim）。
    l0_sim: float = 0.01
    # [SIM GUARD] 仿真用内尺度 [m]：aotools/soapy 相位屏除以 l0，0 会崩溃，
    #   故仿真采用 0.01 m（近 Kolmogorov 谱行为）。
    L0: float = 100.0
    # 湍流外尺度 [m]：von-Karman 谱的低频截断长度。
    L: float = 1000.0
    # 传播距离 [m]：光源 -> 目标的总路径（CNN1 固定 1 km；CNNL 为 0.5-2.6 km 随机）。
    rytov_sigma2: float = 0.35
    # Rytov 方差（对数强度起伏方差）[无量纲]，CNN1 固定；CNNL 在 0.1-2.0 随机。

    # --- 光学参数 (Table 1) ---
    Dscope: float = 0.30
    # 望远镜（入瞳）直径 [m]（30 cm），决定衍射分辨率与孔径网格。
    rspot: float = 0.075
    # 初始高斯光束 1/e 半径 [m]（7.5 cm），决定入瞳照度分布。
    wavelength: float = 800e-9
    # 中心波长 [m]（800 nm），决定波数 k=2π/λ 与衍射尺度。
    focal: float = 1000.0
    # 聚焦焦距 f = L [m]（Table 1：把能量聚焦到目标距离）。

    # --- 仿真网格参数 (Table 1) ---
    N: int = 512
    # 网格分辨率 [px]：(N, N) 采样数，决定最高空间频率。
    box_size: float = 0.30
    # 网格物理尺寸 [m]（30 cm，与 Dscope 等宽），像素间距 dx = box_size/N。
    n_screens: int = 10
    # 湍流相位屏层数 = L / screen_sep（分步传播的屏数）。
    screen_sep: float = 100.0
    # 相邻相位屏间距 [m]（Table 1），每屏内按 dz/2 -> 乘屏相位 -> dz/2 传播。
    n_roughness: int = 10
    # 目标粗糙面 realization 数（Sec 2.4：多 realization 平均压低散斑）。
    roughness_seed: int = 42
    # 粗糙面相位 RNG 流的基础种子（实际用 seed*31+j 派生）。
    beam_source: str = "oopao"
    # 湍流屏生成后端：soapy | aotools | oopao（本项目默认 OOPAO 库屏幕）。
    screen_pool: int = 0
    # 相位屏样本池大小：0 = 每个样本独立生成（符合论文设定）。


# =============================================================================
# imaging —— 多平面成像几何（Eqs. 9-12, Sec 2.4）
# =============================================================================
@dataclass
class ImagingConfig:
    zR_APWS: float | None = None
    # 瑞利距离 z_R_APWS [m]：null 时运行时由闭合解 z_R = r0^2/(π·λ) 计算
    #   （Eq 9：z_R_APWS = π·r_APWS²/λ；r0 ≈ 0.04 m @ 800nm/L=1km -> ~640 m）。
    f_obj: float | None = None
    # 物镜焦距 [m]：null 时取 f_obj = 2·z_R_APWS（Eq 12，~1280 m）。
    plane_offset_frac: list = field(default_factory=lambda: [0.0, 1.0, 2.0])
    # 测量平面位置（相对物镜的焦距比例）：0=f_obj-z_R, 1=焦面, 2=f_obj+z_R ——
    #   ±z_R 离焦平面是 CNN 提取深度信息的物理载体。


# =============================================================================
# bucket —— FOM 桶（Eq. 6）
# =============================================================================
@dataclass
class BucketConfig:
    diameter_frac: float = 2.5
    # 桶口径系数：D_bucket = 2.5·L·λ/D_telescope [m]
    #   -> 6.67e-3 m ≈ 11.4 px 直径（半径 ~5.7 px）@ N=512, box=0.3 m。


# =============================================================================
# data —— 数据集规模与生成
# =============================================================================
@dataclass
class DataConfig:
    n_train: int = 2000
    # 训练样本数（演示版；论文 CNN1: 81,000, CNNL: 100,000）。
    n_test: int = 400
    # 测试样本数（演示版；论文 CNN1: 9,000）。
    n_eval: int = 100
    # 评估样本数（演示版；论文 CNN1: 1,000）。
    master_seed: int = 20250830
    # 种子调度基准：样本种子 = master_seed + sample_index（确定性复现）。
    workers: int = 96
    # multiprocessing 工作进程数（演示机 384 核）。
    h5_path: str = "data/beaconless_demo.h5"
    # 生成的 HDF5 数据集路径。


# =============================================================================
# model —— CNN 架构（Sec 2.6 / Table 1）
# =============================================================================
@dataclass
class ModelConfig:
    name: str = "CNN1"
    # 模型变体：CNN1 | CNNL | CNN1Freq | CNN1Star。
    n_modes: int = 78
    # Zernike 截断阶数 J = 78：CNN 回归标签 Φ_Z78 的维数。
    channels: list | None = None
    # 3 级 CNN 各阶段通道数（None 时用 models/cnn.py 内置默认）。
    kernel: int = 3
    # 卷积核尺寸 [px]（Table 1: 3x3）。
    stride: int = 1
    # 卷积步长（Table 1: 1）。
    padding: int = 0
    # 卷积填充（Table 1: 0）。
    mlp_width: int = 512
    # MLP 头隐藏层宽度 [神经元]（Sec 2.6: 512）。
    mlp_depth: int = 4
    # MLP 头隐藏层数（Sec 2.6: 4 层 ReLU）。
    pool_size: int = 18
    # AdaptiveAvgPool2d 输出的网格尺寸：(18,18)，展平 128·18·18 = 41472。
    length_head: bool = False
    # True => CNNL：增加 512 神经元的长度传感器头。
    dropout: float = 0.0
    # 丢包率（无正则化）。
    # ---- CNN1Freq 频谱分支（仅 name == "CNN1Freq" 时生效）----
    freq_pool: int = 8
    # 2D-FFT 对数幅度网格的自适应池化尺寸。
    freq_refine_ch: int = 16
    # 重组频谱带的 1x1 卷积宽度。
    # ---- CNN1Star 注意力主干（仅 name == "CNN1Star" 时生效）----
    base_dim: int = 32
    # 第一级 StarBlock 阶段的通道宽度。
    depths: list = field(default_factory=lambda: [1, 1, 2])
    # 各阶段的 StarBlock 数量。
    mlp_ratio: int = 4
    # 每个 StarBlock 内部的扩张比例。
    use_se: bool = False
    # 是否附加 squeeze-and-excitation 注意力层。
    se_reduction: int = 16
    # SE 瓶颈降维比。


# =============================================================================
# train —— 训练超参数（Table 1: Adam, lr 1e-4, beta 0.9/0.999, batch 32, MSE）
# =============================================================================
@dataclass
class TrainConfig:
    optimizer: str = "adam"
    # 优化器（Table 1: Adam）。
    lr: float = 1.0e-4
    # 学习率。
    beta1: float = 0.9
    # Adam 一阶矩衰减系数。
    beta2: float = 0.999
    # Adam 二阶矩衰减系数。
    batch_size: int = 32
    # 有效批大小（论文 Table 1）= micro_batch x grad_accum。
    micro_batch_size: int = 8
    # 显存受限的每步微批大小；梯度累积恢复 32。
    n_steps: int = 3000
    # 训练步数（演示版；论文 CNN1: 11,500, CNNL: 35,000）。
    loss: str = "mse"
    # 损失函数：缩放 Zernike 模式上的 MSE。
    seed: int = 0
    # 训练随机种子。
    amp: bool = True
    # 启用 FP16 autocast + 梯度缩放（RTX 4090 级 GPU）。
    mixed_precision: bool = True
    # 混合精度总开关（与 amp 一致）。
    sim_eval_every: int = 500
    # 每隔 N 步把当前批在仿真中做一次 FOM 评估（论文为 100）。
    sim_eval_n: int = 8
    # 每次训练中 FOM 评估用的样本数。
    sim_eval_workers: int = 4
    # 仿真 FOM 评估的 multiprocessing 进程数（运行时默认）。
    sim_eval_context: str = "spawn"
    # 仿真 FOM 评估的 multiprocessing 上下文（spawn/fork）。
    sim_eval_z78: bool = False
    # 仿真 FOM 评估是否同时报告 Z78 上界。
    ckpt_dir: str = "checkpoints"
    # 检查点输出目录。
    log_every: int = 10
    # wandb 标量记录间隔（步）。
    grad_clip: float | None = None
    # 梯度裁剪阈值（None = 不裁剪）。
    num_workers: int = 4
    # DataLoader 工作进程数（运行时默认 4）。
    persistent_workers: bool = False
    # DataLoader 是否跨 epoch 保持 worker 进程（减少 spawn 开销；num_workers=0 时无效）。
    prefetch_factor: int = 4
    # DataLoader 每个 worker 预取的批数（仅 num_workers>0 时生效）。
    preload_to_ram: bool = False
    # 将训练 split 的 images/labels 一次性读入 CPU 内存（消除 h5py I/O；
    # 适合 smoke test / demo 规模数据集；大 91000 sample 不建议）。
    channels_last: bool = False
    # 是否使用 channels_last 内存布局（运行时默认）。
    compile: bool = False
    # 是否 torch.compile 模型（运行时默认关闭）。


# =============================================================================
# eval —— 评估协议（Sec 2.7）
# =============================================================================
@dataclass
class EvalConfig:
    bucket_mask_px: int | None = None
    # 桶掩膜像素直径：null 时由 bucket.diameter_frac 换算（~11.4 px）。
    out_dir: str = "results"
    # 评估输出目录。
    plot_every: int = 20
    # 预测-vs-真值图的样本索引步长。
    ckpt_path: str | None = None
    # 评估用的检查点路径（运行时由 CLI 注入）。
    batch_size: int = 32
    # 评估批大小（运行时默认 32）。


# =============================================================================
# wandb —— 实验跟踪
# =============================================================================
@dataclass
class WandbConfig:
    project: str = "beaconless-ao-sim"
    # wandb 项目名。
    entity: str | None = None
    # wandb 团队/账号（None 时不设置 WANDB_ENTITY）。
    run_name: str | None = None
    # 运行名（None = 自动生成）。
    tags: list = field(default_factory=lambda: ["cnn1", "demo"])
    # 运行标签。
    notes: str | None = None
    # 运行备注。


# =============================================================================
# run —— 运行时注入段（由 train.attach_run 填充；不存在于 config.yaml）
# =============================================================================
@dataclass
class RunConfig:
    ckpt_dir: str = "checkpoints"
    # 运行时检查点目录（绝对路径，由 attach_run 注入）。
    out_dir: str = "results"
    # 运行时结果目录（绝对路径，注入）。
    h5_path: str | None = None
    # 数据集路径（注入；None 时回退到 data.h5_path）。
    device: str = "cpu"
    # 训练设备（注入：cuda/cpu）。
    rank: int = 0
    # 分布式进程 rank（注入）。
    world_size: int = 1
    # 分布式总进程数（注入）。
    is_distributed: bool = False
    # 是否分布式训练（注入）。
    amp: bool = False
    # 本进程是否启用 AMP（注入）。
    no_wandb: bool = False
    # 是否禁用 wandb（注入，--no-wandb）。
    resume: Optional[str] = None
    # 恢复的检查点绝对路径（注入，--resume）。


# =============================================================================
# SimConfig —— 顶层配置
# =============================================================================
@dataclass
class SimConfig:
    """完整仿真/训练配置（config.yaml 全部节）。"""

    physical: PhysicalConfig = field(default_factory=PhysicalConfig)
    imaging: ImagingConfig = field(default_factory=ImagingConfig)
    bucket: BucketConfig = field(default_factory=BucketConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    run: RunConfig = field(default_factory=RunConfig)

    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, d: dict | None) -> SimConfig:
        """从 ``yaml.safe_load`` 得到的 dict 构建 SimConfig。

        容错：未知顶层节/未知字段被忽略（各节缺省时回落默认值），
        因此手构的部分 dict（如测试 fixture）也能直接使用。
        """
        if not d:
            return cls()
        known = {
            "physical": PhysicalConfig,
            "imaging": ImagingConfig,
            "bucket": BucketConfig,
            "data": DataConfig,
            "model": ModelConfig,
            "train": TrainConfig,
            "eval": EvalConfig,
            "wandb": WandbConfig,
        }
        kwargs: dict[str, Any] = {}
        for sec, sec_cls in known.items():
            raw = d.get(sec)
            if isinstance(raw, dict):
                # 数值字段防御性转换：PyYAML 会把 "800e-9" 解析成字符串，
                # 旧代码依赖 numpy 隐式转换；dataclass 在此显式纠正为 float。
                inner: dict[str, Any] = {}
                for k, v in raw.items():
                    if k not in sec_cls.__dataclass_fields__:
                        continue  # 未知字段：忽略
                    ty = str(sec_cls.__dataclass_fields__[k].type)
                    if isinstance(v, str) and (
                        "float" in ty or " int" in ty or ty == "int"
                    ):
                        try:
                            v = float(v)
                        except ValueError:
                            pass  # run_name 等字符串字段原样保留
                    inner[k] = v
                try:
                    kwargs[sec] = sec_cls(**inner)
                except TypeError:  # 个别字段类型不匹配时回落默认节
                    kwargs[sec] = sec_cls()
            else:
                kwargs[sec] = sec_cls()
        return cls(**kwargs)

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        """转回普通 dict（asdict）：供 json.dumps / yaml.safe_dump / wandb 使用。"""
        return asdict(self)


# =============================================================================
# 模块级加载器
# =============================================================================
def load_config(path: str | os.PathLike) -> SimConfig:
    """读取 YAML 配置文件并解析为 :class:`SimConfig`。

    Parameters
    ----------
    path : str | os.PathLike
        config.yaml 路径（相对当前工作目录或绝对路径）。

    Returns
    -------
    SimConfig
        类型化配置对象。
    """
    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return SimConfig.from_dict(data)
