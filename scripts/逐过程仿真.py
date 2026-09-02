#!/usr/bin/env python3
"""逐过程仿真脚本 —— 光源 → 相位调控(变形镜) → 大气 → 目标（漫反射）→ 返回大气 → 成像

基于 OOPAO 库和 beaconless-ao-sim 项目的物理仿真模块，逐步展示完整的光路仿真。
每一步都可以独立执行并可视化。

OOPAO 的使用
------------
湍流相位屏由 **OOPAO.Atmosphere**（多层 von-Karman）逐层抽取生成
（见 physics/oopao_backend.py，经 physics/_oopao_compat.py 加载
Atmosphere/Source/Telescope 三个类）。变形镜不在 OOPAO 模块范围内，
本脚本将其简化为**纯相位调控**：把任意相位屏 φ 施加到光场上
（E_out = E_in · exp(iφ)），等价于一个理想无空间量化误差的相位型 DM。

中文概览
--------
本脚本实现 DiComo 等 (Opt. Express 33(15):31010, 2025) 中完整的光路仿真：
    §1 光源：生成入瞳高斯光束
    §2 相位调控：把调控相位施加到光束（变形镜的简化模型）
    §3 大气传播（前向）：分步 FFT 通过 OOPAO 湍流屏到目标面
    §4 目标漫反射：粗糙面散射
    §5 返回大气传播：反向传播到望远镜
    §6 成像系统：物镜聚焦 + 多平面成像
    §7 完整光路串联 & FOM 对比

Usage::

    # 直接运行
    uv run python scripts/逐过程仿真.py

    # 作为模块导入
    from scripts.逐过程仿真 import StepByStepSimulation
    sim = StepByStepSimulation("config.yaml")
    sim.run_all(seed=42)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# ── 项目路径设置 ──────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from physics.config import load_config
from physics.propagation_fft import Propagator
from physics.scattering import random_roughness_phase
from physics.screens_soapy import compute_r0
from physics.zernike_aotools import ZernikeBasis
from utils.metrics import FOM, bucket_mask

# OOPAO 可用性探测（与项目现有 fallback 模式一致：oopao -> aotools）
try:
    from physics.oopao_backend import OopaoScreenBackend as _OopaoBackend
    _OOPAO_AVAILABLE = True
except ImportError:
    _OOPAO_AVAILABLE = False
    print("[提示] OOPAO 库不可用（未安装或兼容层失败），将回退到 aotools 湍流屏。")


# ── 中文字体配置 ──────────────────────────────────────────────────────
def _setup_chinese_font():
    """配置 matplotlib 中文字体（兼容 Windows/macOS/Linux）。"""
    # 直接扫描 fontManager.ttflist（比 findfont 逐次探测更可靠）
    _cjk_candidates = [
        "Microsoft YaHei", "Microsoft YaHei UI",
        "SimHei", "SimSun", "NSimSun",
        "Source Han Sans CN", "Source Han Sans SC",
        "Noto Sans CJK SC", "PingFang SC", "Heiti SC",
        "WenQuanYi Micro Hei", "Arial Unicode MS",
    ]
    _available = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
    _picked = [f for f in _cjk_candidates if f in _available]
    if _picked:
        _cjk_name = _picked[0]
        matplotlib.rcParams["font.family"] = _cjk_name
        matplotlib.rcParams["font.sans-serif"] = [_cjk_name] + _picked + ["DejaVu Sans"]
        matplotlib.rcParams["axes.unicode_minus"] = False
        print(f"[字体] CJK 字体: {_cjk_name}")
        return _cjk_name
    # 降级：使用 sans-serif
    matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    print("[字体] 警告: 未找到 CJK 字体，中文可能仍为方块")
    return None


_CN_FONT = _setup_chinese_font()

# ── 语言标签常量 ──────────────────────────────────────────────────────
# 如果中文字体可用则用中文，否则用英文
if _CN_FONT:
    LABELS = {
        "title_source": "§1 光源 —— 入瞳高斯光束",
        "title_amplitude": "光束幅度 |E0|",
        "title_phase": "光束相位 (rad)",
        "title_pupil": "孔径掩膜",
        "title_dm": "§2 相位调控 —— 变形镜简化模型",
        "title_dm_phase": "调控相位 (rad)",
        "title_corrected": "校正后光束",
        "title_corrected_intensity": "校正后强度",
        "title_atm_fwd": "§3 大气传播（前向）",
        "title_screen": "湍流屏",
        "title_target_intensity": "目标面强度",
        "title_vacuum": "真空参考强度",
        "title_scatter": "§4 目标漫反射",
        "title_roughness": "粗糙面相位 (rad)",
        "title_scattered": "散射后强度",
        "title_atm_ret": "§5 返回大气传播",
        "title_return_intensity": "返回面强度",
        "title_return_phase": "返回面相位 (rad)",
        "title_imaging": "§6 成像系统",
        "title_image": "成像平面",
        "title_full": "§7 完整光路 FOM 对比",
        "title_fom": "FOM 对比",
        "xlabel_pixel": "像素",
        "ylabel_pixel": "像素",
        "xlabel_mode": "Noll 模式",
        "ylabel_coeff": "系数 (rad)",
        "no_ao": "无 AO",
        "track_only": "仅跟踪",
        "beacon_conj": "信标共轭",
        "z78_78mode": "78 阶 Zernike",
        "dm_correction": "DM 校正",
        "fom_value": "FOM",
        # 光线传播路径 / 返回演化 / 多距离成像 / 五角星演示
        "title_light_path": "光线传播路径总览 —— 发射 → 目标漫反射 → 返回 → 成像",
        "title_ret_evol": "§5 返回路径光强 / 相位演化",
        "ret_intensity": "光强 I (log₁₀)",
        "ret_phase": "相位 φ (rad)",
        "title_dist_imaging": "§6 不同距离对远端光斑的成像（教程风格）",
        "dist_image": "远端光斑成像",
        "title_star": "§8 五角星均匀光源望远成像",
        "star_no_turb": "无湍流 (真空传播)",
        "star_with_turb": "有湍流 (大气传播)",
        "star_source": "五角星光源 (均匀亮度)",
        "star_diff_limit": "衍射极限 PSF",
        "star_turb_psf": "湍流退化 PSF",
        "xlabel_arcsec": "角坐标 (arcsec)",
        "ylabel_arcsec": "角坐标 (arcsec)",
    }
else:
    LABELS = {
        "title_source": "S1 Light Source - Pupil Gaussian Beam",
        "title_amplitude": "Beam Amplitude |E0|",
        "title_phase": "Beam Phase (rad)",
        "title_pupil": "Pupil Mask",
        "title_dm": "S2 Phase Control (DM simplified)",
        "title_dm_phase": "Control Phase (rad)",
        "title_corrected": "Corrected Beam",
        "title_corrected_intensity": "Corrected Intensity",
        "title_atm_fwd": "S3 Atmosphere (Forward)",
        "title_screen": "Turbulence Screen",
        "title_target_intensity": "Target Intensity",
        "title_vacuum": "Vacuum Reference Intensity",
        "title_scatter": "S4 Target Diffuse Reflection",
        "title_roughness": "Roughness Phase (rad)",
        "title_scattered": "Scattered Intensity",
        "title_atm_ret": "S5 Return Atmosphere",
        "title_return_intensity": "Return Intensity",
        "title_return_phase": "Return Phase (rad)",
        "title_imaging": "S6 Imaging System",
        "title_image": "Image Plane",
        "title_full": "S7 Full Pipeline FOM Comparison",
        "title_fom": "FOM Comparison",
        "xlabel_pixel": "Pixel",
        "ylabel_pixel": "Pixel",
        "xlabel_mode": "Noll Mode",
        "ylabel_coeff": "Coefficient (rad)",
        "no_ao": "No AO",
        "track_only": "Track Only",
        "beacon_conj": "Beacon Conj.",
        "z78_78mode": "Z78 (78-mode)",
        "dm_correction": "DM Correction",
        "fom_value": "FOM",
        # Light path / return evolution / multi-distance / star demo
        "title_light_path": "Light Path Overview - Launch -> Diffuse Return -> Imaging",
        "title_ret_evol": "S5 Return-Path Intensity / Phase Evolution",
        "ret_intensity": "Intensity I (log10)",
        "ret_phase": "Phase phi (rad)",
        "title_dist_imaging": "S6 Imaging a Distant Spot at Various Distances (tutorial style)",
        "dist_image": "Distant-Spot Image",
        "title_star": "S8 Star-Shaped Uniform Source Telescope Imaging",
        "star_no_turb": "No Turbulence (vacuum)",
        "star_with_turb": "With Turbulence (atmosphere)",
        "star_source": "Star Source (uniform)",
        "star_diff_limit": "Diffraction-Limited PSF",
        "star_turb_psf": "Turbulence-Degraded PSF",
        "xlabel_arcsec": "Angle (arcsec)",
        "ylabel_arcsec": "Angle (arcsec)",
    }


# ── 可视化辅助函数 ──────────────────────────────────────────────────
def plot_2d(
    data: np.ndarray,
    title: str,
    xlabel: str | None = None,
    ylabel: str | None = None,
    cmap: str = "viridis",
    colorbar_label: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    ax: plt.Axes | None = None,
) -> None:
    """绘制 2D 强度/相位图（封装常用 matplotlib 调用）。"""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    im = ax.imshow(data, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(xlabel or LABELS["xlabel_pixel"], fontsize=11)
    ax.set_ylabel(ylabel or LABELS["ylabel_pixel"], fontsize=11)
    plt.colorbar(im, ax=ax, label=colorbar_label)


def plot_phase(phi: np.ndarray, title: str, ax: plt.Axes | None = None) -> None:
    """绘制相位图（使用 radar 色图，中心为 0）。"""
    vmax = float(np.percentile(np.abs(phi[phi != 0]), 99)) if np.any(phi != 0) else 1.0
    plot_2d(phi, title, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
            colorbar_label="rad", ax=ax)


def plot_intensity(I: np.ndarray, title: str, ax: plt.Axes | None = None,
                   log_scale: bool = False) -> None:
    """绘制强度图（可选对数标度）。"""
    if log_scale:
        I_plot = np.log10(np.maximum(I, 1e-20))
        plot_2d(I_plot, title, cmap="inferno", ax=ax,
                colorbar_label="log₁₀(I)")
    else:
        plot_2d(I, title, cmap="inferno", ax=ax, colorbar_label="Intensity")


def show_fig(title: str | None = None) -> None:
    """显示当前图形并优化布局。"""
    if title:
        plt.suptitle(title, fontsize=15, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_light_path_schematic() -> None:
    """绘制光线传播路径总览示意图（发射 → 漫反射 → 返回 → 成像）。

    侧视图示意整个双向链路：
      光源/入瞳 → 相位调控 → 前向大气(10层屏) → 目标(漫反射)
                                                    │
      焦面成像 ← 物镜 ← 望远镜入瞳 ← 返回大气(同屏) ←┘

    该图用 annotate/patches 纯手绘，无物理网格，仅作教学示意
    （对应 OOPAO 教程中 Telescope 成像几何的直观化：
    https://github.com/cheritier/OOPAO/blob/master/tutorials/image_formation.py
    与本项目 `data/simulate.py` 的算法 1 链路）。
    """
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.set_xlim(0, 16)
    ax.set_ylim(-2, 3.5)
    ax.axis("off")

    # ── 第 1 行：前向发射（左 → 右） ──
    # 光源 / 入瞳（x≈1）
    ax.plot([1, 1], [-0.5, 0.9], color="#34495e", lw=3)
    ax.annotate("① 光源/入瞳\n(高斯光束 E0 e^{iφ_focus})",
                xy=(1, 0.9), xytext=(0.2, 1.9), fontsize=10,
                arrowprops=dict(arrowstyle="->"))
    # 相位调控（DM 简化，x≈3）
    ax.plot([3, 3], [-0.5, 0.9], color="#8e44ad", lw=3)
    ax.annotate("② 相位调控\nexp(i·φ_ctrl)", xy=(3, 0.9), xytext=(2.3, 1.95), fontsize=10,
                arrowprops=dict(arrowstyle="->"))
    # 前向大气：10 层屏（x≈4.5..9）
    import matplotlib.patches as mpatches
    for i, x_s in enumerate(np.linspace(4.6, 9.0, 10)):
        ax.plot([x_s, x_s], [-0.4, 0.8], color="#7f8c8d", lw=0.8, alpha=0.5)
    ax.annotate("③ 大气前向\n(10 层湍流屏, split-step)",
                xy=(6.8, 0.8), xytext=(5.6, 1.95), fontsize=10,
                arrowprops=dict(arrowstyle="->"))
    # 前向光线（汇聚到目标）
    for y_off in (-0.35, -0.18, 0.0, 0.18, 0.35):
        ax.plot([1.0, 13.0], [y_off, 0.0], color="#e74c3c", lw=1.1, alpha=0.55)
    ax.annotate("发射光 (前向, L=1000 m)", xy=(7.0, 0.5), xytext=(6.1, 0.68), fontsize=9)

    # ── 目标（漫反射，x≈13） ──
    ax.plot([13, 13], [-0.6, 0.7], color="#27ae60", lw=3.5)
    ax.annotate("④ 目标\n(漫反射, 粗糙面 φ_r)",
                xy=(13, 0.7), xytext=(12.4, 1.8), fontsize=10,
                arrowprops=dict(arrowstyle="->"))
    # 漫反射回波（右 → 左，发散）
    for y_off in (-0.45, -0.25, -0.05, 0.15, 0.35):
        ax.plot([13.0, 6.0], [y_off * 0.3, y_off - 0.85], color="#f39c12", lw=1.0, alpha=0.6)

    # ── 第 2 行：返回大气（右 → 左，屏复用） ──
    for i, x_s in enumerate(np.linspace(9.0, 4.6, 10)):
        ax.plot([x_s, x_s], [-1.9, -1.0], color="#7f8c8d", lw=0.8, alpha=0.5)
    ax.annotate("⑤ 返回大气\n(同 10 层屏反向)",
                xy=(5.0, -1.0), xytext=(4.4, -1.85), fontsize=10,
                arrowprops=dict(arrowstyle="->"))
    # 返回光线（右 → 左，汇聚到入瞳）
    for y_off in (-0.35, -0.18, 0.0, 0.18, 0.35):
        ax.plot([13.0, 1.0], [-1.35 - y_off * 0.25, -1.35], color="#16a085", lw=1.0, alpha=0.55)

    # ── 第 3 行：望远镜成像（x≈1 附近，向左聚焦） ──
    ax.plot([1.0, 1.0], [-1.9, -0.8], color="#c0392b", lw=3)
    ax.annotate("⑥ 望远镜入瞳 (D)", xy=(1, -0.8), xytext=(1.6, -1.35), fontsize=10,
                arrowprops=dict(arrowstyle="->"))
    ax.plot([0.2, 0.2], [-1.9, -0.8], color="#2980b9", lw=3)
    ax.annotate("⑦ 物镜 (f_obj)", xy=(0.2, -0.8), xytext=(1.7, -0.45), fontsize=10,
                arrowprops=dict(arrowstyle="->"))
    for y_off in (-0.3, -0.15, 0.0, 0.15, 0.3):
        ax.plot([1.0, -1.8], [-1.4, y_off * 0.45 - 0.7], color="#9b59b6", lw=1.0, alpha=0.55)
    ax.plot([-1.8, -1.8], [-1.1, -0.3], color="#f1c40f", lw=2.5)
    ax.annotate("⑧ 焦面成像\n(f_obj ± z_R 多平面)",
                xy=(-1.8, -0.3), xytext=(-3.2, 0.5), fontsize=10,
                arrowprops=dict(arrowstyle="->"))

    ax.set_title(LABELS["title_light_path"], fontsize=15)
    plt.tight_layout()
    plt.show()


# ── 核心仿真类 ──────────────────────────────────────────────────────
N_MODES = 78  # Zernike 截断阶数 J = 78（论文表 1）


@dataclass
class StepResult:
    """某一步的仿真结果容器。"""
    name: str
    data: dict = field(default_factory=dict)


class StepByStepSimulation:
    """逐过程光路仿真 —— 光源 → 变形镜 → 大气 → 目标 → 返回大气 → 成像

    Parameters
    ----------
    config_path : str
        配置文件路径（相对于项目根目录或绝对路径）。
    """

    def __init__(self, config_path: str = "config.yaml") -> None:
        # 加载配置
        cfg_path = Path(config_path)
        if not cfg_path.is_absolute():
            cfg_path = _PROJECT_ROOT / cfg_path
        self.cfg = load_config(cfg_path)

        p = self.cfg.physical
        img = self.cfg.imaging
        b = self.cfg.bucket

        # ── 基本量 ──
        self.N: int = int(p.N)
        self.dx: float = float(p.box_size) / self.N
        self.lam: float = float(p.wavelength)
        self.k: float = 2.0 * np.pi / self.lam
        self.rspot: float = float(p.rspot)
        self.focal: float = float(p.focal)
        self.L: float = float(p.L)
        self.Dscope: float = float(p.Dscope)

        # ── 坐标网格 ──
        x = (np.arange(self.N) - (self.N - 1) / 2.0) * self.dx
        self.X, self.Y = np.meshgrid(x, x)
        self.r2 = self.X**2 + self.Y**2

        # ── 孔径掩膜 ──
        self.pupil = self.r2 <= (self.Dscope / 2.0) ** 2

        # ── 入瞳光束：高斯幅度，孔径外置零 ──
        self.E0 = np.exp(-(self.r2 / self.rspot**2)).astype(np.complex64)
        self.E0[~self.pupil] = 0.0

        # ── 聚焦相位 phi_focus = -k r²/(2f) ──
        self.phi_focus = (-self.k * self.r2 / (2.0 * self.focal)).astype(np.float64)

        # ── 倾斜跟踪高斯权重 ──
        self.G = np.exp(-(self.r2 / self.rspot**2))

        # ── 传播器（FFTW 分步传播）──
        print("[初始化] 构建 FFT 传播器...")
        self.prop = Propagator(self.N, self.dx, self.lam)

        # ── Zernike 基底 ──
        print("[初始化] 构建 78 阶 Zernike 基底...")
        self.zern = ZernikeBasis(self.N, N_MODES)

        # ── 成像几何（公式 9-12）──
        self.r0 = compute_r0(self.lam, float(p.cn2), self.L)
        self.zR_APWS = img.zR_APWS if img.zR_APWS is not None else self.r0**2 / (np.pi * self.lam)
        self.f_obj = img.f_obj if img.f_obj is not None else 2.0 * self.zR_APWS
        self.plane_offsets = np.array(
            [self.f_obj + (frac - 1.0) * self.zR_APWS for frac in img.plane_offset_frac],
            dtype=np.float64,
        )

        # ── FOM 桶掩膜 ──
        D_bucket = float(b.diameter_frac) * self.L * self.lam / self.Dscope
        diameter_px = D_bucket / self.dx
        self.bucket_mask = bucket_mask(self.N, diameter_px)

        # ── 真空目标面强度 ──
        self.I_vac = self.prop.angular_spectrum_intensity(
            (self.E0 * np.exp(1j * self.phi_focus)).astype(np.complex64), self.L
        )

        # ── OOPAO 大气后端 ──
        # 与现有 simulate.py 一致：beam_source 配置选择 oopao / aotools / soapy，
        # 若 OOPAO 不可用则自动回退到 aotools（确定性 per-seed 屏幕）。
        beam_source = str(p.beam_source).lower()
        if beam_source == "oopao" and _OOPAO_AVAILABLE:
            print("[初始化] 构建 OOPAO 大气后端...")
            self.oopao = _OopaoBackend(
                N=self.N, dx=self.dx, Dscope=self.Dscope, lam=self.lam,
                cn2=float(p.cn2), L=self.L, L0=float(p.L0),
                n_screens=int(p.n_screens),
            )
        else:
            if beam_source == "oopao":
                print("[初始化] beam_source=oopao 但 OOPAO 不可用，回退到 aotools")
            self.oopao = None

        print(f"[初始化完成] N={self.N}, λ={self.lam*1e9:.0f}nm, L={self.L:.0f}m, "
              f"D={self.Dscope:.2f}m, r₀={self.r0*100:.1f}cm, "
              f"z_R={self.zR_APWS:.1f}m, f_obj={self.f_obj:.1f}m")

    # ── 辅助：生成湍流相位屏 ────────────────────────────────────────
    def _make_screens(self, seed: int) -> np.ndarray:
        # OOPAO 参考: https://github.com/cheritier/OOPAO/tree/master/tutorials
        """生成确定性湍流相位屏 (n_screens, N, N)。"""
        if self.oopao is not None:
            return self.oopao.make_screens(seed)
        # aotools 回退路径
        from aotools.turbulence.phasescreen import ft_sh_phase_screen
        p = self.cfg.physical
        n_screens = int(p.n_screens)
        r0_path = compute_r0(self.lam, float(p.cn2), self.L)
        r0_slab = r0_path * n_screens ** (3.0 / 5.0)
        screens = np.stack([
            ft_sh_phase_screen(r0_slab, self.N, self.dx, float(p.L0),
                               float(p.l0_sim), seed=seed + i)
            for i in range(n_screens)
        ]).astype(np.float32)
        return screens

    # ══════════════════════════════════════════════════════════════════
    # §1 光源 —— 生成入瞳高斯光束
    # ══════════════════════════════════════════════════════════════════
    def step1_light_source(self, visualize: bool = True) -> StepResult:
        """§1 光源：生成入瞳高斯光束并可视化。

        Returns
        -------
        StepResult
            包含 E0, phi_focus, pupil 等字段。
        # OOPAO 参考: https://github.com/cheritier/OOPAO/blob/master/tutorials/how_to_configure_a_telescope.py
        """
        print("\n" + "="*60)
        print("§1 光源 —— 生成入瞳高斯光束")
        print("="*60)
        print(f"  波长 λ = {self.lam*1e9:.0f} nm")
        print(f"  光束半径 r_spot = {self.rspot*100:.1f} cm")
        print(f"  孔径直径 D = {self.Dscope*100:.1f} cm")
        print(f"  聚焦焦距 f = {self.focal:.0f} m")
        print(f"  网格分辨率 N = {self.N}×{self.N}")
        print(f"  像素间距 dx = {self.dx*1e3:.3f} mm")

        I0 = (np.abs(self.E0)**2).astype(np.float32)
        print(f"  入瞳峰值强度 = {I0.max():.6f}")
        print(f"  通光面积占比 = {self.pupil.sum() / self.N**2 * 100:.1f}%")

        if visualize:
            fig, axes = plt.subplots(1, 3, figsize=(16, 5))
            plot_intensity(I0, "|E0|² (入瞳强度)",
                          ax=axes[0])
            plot_phase(self.phi_focus, LABELS["title_phase"] + f"\n(f={self.focal:.0f}m)",
                      ax=axes[1])
            axes[2].imshow(self.pupil.astype(float), cmap="gray", origin="lower")
            axes[2].set_title(LABELS["title_pupil"], fontsize=13)
            axes[2].set_xlabel(LABELS["xlabel_pixel"], fontsize=11)
            axes[2].set_ylabel(LABELS["ylabel_pixel"], fontsize=11)
            show_fig(LABELS["title_source"])

        return StepResult("light_source", {
            "E0": self.E0, "phi_focus": self.phi_focus, "pupil": self.pupil,
        })

    # ══════════════════════════════════════════════════════════════════
    # §2 相位调控（变形镜简化模型）
    #     OOPAO 仅提供 Atmosphere/Source/Telescope 三个类（见 _oopao_compat）。
    #     本仿真不建模变形镜的执行器几何，而是将其简化为纯相位调控：
    #     E_out = E_in * exp(1j * phi_ctrl)，phi_ctrl 由上层（如信标共轭、
    #     Zernike 投影）给出或直接传入任意相位屏。
    # ══════════════════════════════════════════════════════════════════
    def step2_phase_control(
        self,
        screens: np.ndarray,
        phi_ctrl: Optional[np.ndarray] = None,
        control_type: str = "beacon_conj",
        visualize: bool = True,
    ) -> StepResult:
        """§2 相位调控：将相位调控量施加到入瞳光束上（变形镜的简化模型）。

        Parameters
        ----------
        screens : np.ndarray
            (n_screens, N, N) 湍流相位屏（由 OOPAO Atmosphere 生成）。
        phi_ctrl : np.ndarray, optional
            外部给定的相位调控量（N, N）rad。给出时忽略 control_type。
        control_type : str
            调控量生成方式（仅在 phi_ctrl 为 None 时使用）：
            - "beacon_conj": 衍射极限信标反向传播 → 相位共轭（含倾斜跟踪）
            - "z78": 信标反向后的 78 阶 Zernike 投影
            - "none": 零调控（不做校正）
        visualize : bool
            是否绘制可视化图。

        Returns
        -------
        StepResult
            包含 phi_ctrl, zernike_coeffs, E_corrected 等。
        # OOPAO 参考: https://github.com/cheritier/OOPAO/blob/master/tutorials/AO_closed_loop_3_corrector_types.py
        """
        print("\n" + "="*60)
        print(f"§2 相位调控 —— 变形镜简化模型 ({control_type})")
        print("="*60)

        if phi_ctrl is None:
            if control_type == "none":
                phi_ctrl = np.zeros((self.N, self.N), dtype=np.float64)
                coeffs = np.zeros(N_MODES, dtype=np.float64)
                print("  [调控量] 零 —— 不做校正")

            elif control_type in ("beacon_conj", "z78"):
                # 衍射极限信标反向传播（OOPAO 湍流屏倒序）：
                # 目标面放一衍射极限小高斯（束腰 w = λL/D），反向穿过湍流屏
                # 到瞳孔，得到湍流引起的相位畸变。
                w = self.lam * self.L / self.Dscope
                E_beacon = (np.exp(-self.r2 / w**2) * self.pupil).astype(np.complex64)
                dz_screen = self.L / len(screens)
                E_back = self.prop.split_step(E_beacon, screens[::-1], -dz_screen)

                # 解析移除会聚球面波相位（-k r²/(2L)），剩残余畸变相位
                spherical = self.k * self.r2 / (2.0 * self.L)
                E_flat = E_back * np.exp(1j * spherical)

                # 强度引导的相位解卷绕（简化 BFS，完整版见 data/simulate.py）
                phi_raw = np.angle(E_flat)
                I_beacon = (np.abs(E_back)**2).astype(np.float64)
                phi_unwrapped = self._simple_unwrap(phi_raw, I_beacon)
                phi_unwrapped -= phi_unwrapped[self.pupil].mean()

                # 倾斜跟踪（去除整体倾斜，等价于快反镜 tip/tilt）
                gx, gy = np.gradient(phi_unwrapped)
                a_x = float(np.sum(self.G * gx) / np.sum(self.G))
                a_y = float(np.sum(self.G * gy) / np.sum(self.G))
                phi_track = a_x * self.X + a_y * self.Y
                print(f"  [跟踪] 倾斜斜率 a_x = {a_x:.4f}, a_y = {a_y:.4f} rad/m")

                phi_beacon = phi_unwrapped - phi_track
                coeffs = self.zern.phase_to_zernike(phi_beacon)

                if control_type == "beacon_conj":
                    # 相位共轭：直接取残余畸变的相反数作为调控量
                    phi_ctrl = -phi_unwrapped
                    print("  [调控量] beacon_conj —— 信标相位共轭")
                else:
                    # 78 阶 Zernike 投影（有限模式 DM 能力）
                    phi_ctrl = -(phi_track + self.zern.zernike_to_phase(coeffs))
                    print("  [调控量] z78 —— 仅 78 阶 Zernike 可实现的调控")
                print(f"  [Zernike] 78 阶系数 RMS = {np.sqrt(np.mean(coeffs**2)):.4f} rad")

            else:
                raise ValueError(f"未知调控类型: {control_type}")
        else:
            phi_ctrl = np.asarray(phi_ctrl, dtype=np.float64)
            coeffs = self.zern.phase_to_zernike(phi_ctrl)
            print(f"  [调控量] 外部给定相位屏，RMS = {np.sqrt(np.mean(phi_ctrl**2)):.4f} rad")

        # 施加相位调控（变形镜作用：E_out = E_in * exp(i φ_ctrl)）
        E_corrected = (self.E0 * np.exp(1j * (self.phi_focus + phi_ctrl))).astype(np.complex64)
        I_corrected = (np.abs(E_corrected)**2).astype(np.float32)

        if visualize:
            fig, axes = plt.subplots(2, 3, figsize=(18, 11))

            # 第一行：调控相位 + 校正后光束
            plot_phase(phi_ctrl, LABELS["title_dm_phase"] + f"\n({control_type})",
                      ax=axes[0, 0])
            plot_intensity(I_corrected, LABELS["title_corrected_intensity"], ax=axes[0, 1])
            plot_phase(self.phi_focus + phi_ctrl, "总孔径相位 (rad)\n(聚焦 + 调控)",
                      ax=axes[0, 2])

            # 第二行：Zernike 系数分布 + 校正效果对比
            if np.any(coeffs):
                axes[1, 0].bar(np.arange(1, N_MODES + 1), coeffs, width=0.8,
                               color="steelblue", alpha=0.8)
                axes[1, 0].set_xlabel(LABELS["xlabel_mode"], fontsize=11)
                axes[1, 0].set_ylabel(LABELS["ylabel_coeff"], fontsize=11)
                axes[1, 0].set_title(f"Zernike 系数 (前 78 阶)\nRMS = {np.sqrt(np.mean(coeffs**2)):.4f} rad",
                                    fontsize=13)
                axes[1, 0].set_xlim(0.5, N_MODES + 0.5)
            else:
                axes[1, 0].text(0.5, 0.5, "零调控\n(No Correction)",
                               ha="center", va="center", transform=axes[1, 0].transAxes,
                               fontsize=14)
                axes[1, 0].set_title("Zernike 系数", fontsize=13)

            # 校正前后目标面强度对比（经 OOPAO 湍流屏前向传播）
            dz_screen = self.L / len(screens)
            E_after = self.prop.split_step(E_corrected, screens, dz_screen)
            I_after = (np.abs(E_after)**2).astype(np.float32)
            axes[1, 1].imshow(np.log10(np.maximum(self.I_vac, 1e-20)), cmap="inferno",
                             origin="lower")
            axes[1, 1].set_title("无湍流真空强度 (log₁₀)", fontsize=13)
            axes[1, 1].set_xlabel(LABELS["xlabel_pixel"])
            axes[1, 1].set_ylabel(LABELS["ylabel_pixel"])

            axes[1, 2].imshow(np.log10(np.maximum(I_after, 1e-20)), cmap="inferno",
                             origin="lower")
            axes[1, 2].set_title(f"{control_type} 调控后目标面 (log₁₀)", fontsize=13)
            axes[1, 2].set_xlabel(LABELS["xlabel_pixel"])
            axes[1, 2].set_ylabel(LABELS["ylabel_pixel"])

            show_fig(LABELS["title_dm"])

        return StepResult("phase_control", {
            "phi_ctrl": phi_ctrl,
            "zernike_coeffs": coeffs,
            "E_corrected": E_corrected,
            "I_corrected": I_corrected,
        })

    # ══════════════════════════════════════════════════════════════════
    # §3 大气传播（前向）
    # ══════════════════════════════════════════════════════════════════
    def step3_atmosphere_forward(
        self,
        screens: np.ndarray,
        E_in: np.ndarray | None = None,
        visualize: bool = True,
    ) -> StepResult:
        """§3 大气传播（前向）：分步 FFT 通过湍流屏到目标面。

        Parameters
        ----------
        screens : np.ndarray
            (n_screens, N, N) 湍流相位屏。
        E_in : np.ndarray, optional
            入射光场。默认使用 E0 * exp(i * phi_focus)。

        Returns
        -------
        StepResult
            包含 E_obj, I_obj, screens 等。
        # OOPAO 参考: https://github.com/cheritier/OOPAO/tree/master/tutorials
        """
        print("\n" + "="*60)
        print("§3 大气传播（前向）—— 分步 FFT 到目标面")
        print("="*60)

        if E_in is None:
            E_in = (self.E0 * np.exp(1j * self.phi_focus)).astype(np.complex64)

        n_screens = screens.shape[0]
        screen_sep = self.L / n_screens
        print(f"  屏层数 = {n_screens}")
        print(f"  屏间距 = {screen_sep:.0f} m")
        print(f"  总传播距离 = {self.L:.0f} m")
        print(f"  r₀ (路径积分) = {self.r0*100:.1f} cm")
        print(f"  D/r₀ = {self.Dscope / self.r0:.1f}")

        # 分步传播
        print("  [传播] 开始分步 FFT 传播...")
        E_obj = self.prop.split_step(E_in, screens, screen_sep)
        I_obj = (np.abs(E_obj)**2).astype(np.float32)
        print(f"  [传播完成] 目标面峰值强度 = {I_obj.max():.6e}")

        if visualize:
            fig, axes = plt.subplots(1, 4, figsize=(22, 5))

            # 显示前两个湍流屏
            if n_screens >= 2:
                plot_phase(screens[0], f"{LABELS['title_screen']} #1\n(Δz={screen_sep:.0f}m)",
                          ax=axes[0])
                plot_phase(screens[1], f"{LABELS['title_screen']} #2", ax=axes[1])
            else:
                plot_phase(screens[0], LABELS["title_screen"], ax=axes[0])
                axes[1].text(0.5, 0.5, "仅 1 屏", ha="center", va="center",
                            transform=axes[1].transAxes)

            plot_intensity(I_obj, LABELS["title_target_intensity"] + f"\n(L={self.L:.0f}m)",
                          log_scale=True, ax=axes[2])
            plot_intensity(self.I_vac, LABELS["title_vacuum"], log_scale=True, ax=axes[3])

            show_fig(LABELS["title_atm_fwd"])

        return StepResult("atmosphere_forward", {
            "E_obj": E_obj, "I_obj": I_obj, "screens": screens,
        })

    # ══════════════════════════════════════════════════════════════════
    # §4 目标漫反射
    # ══════════════════════════════════════════════════════════════════
    def step4_target_scattering(
        self,
        I_obj: np.ndarray,
        seed: int = 42,
        n_roughness: int = 3,
        visualize: bool = True,
    ) -> StepResult:
        """§4 目标漫反射：粗糙面散射。

        Parameters
        ----------
        I_obj : np.ndarray
            目标面强度。
        seed : int
            随机种子。
        n_roughness : int
            粗糙面 realization 数（演示用 3，论文用 10）。

        Returns
        -------
        StepResult
            包含 phi_roughness, E_scat, I_scat 等。
        # OOPAO 参考: https://github.com/cheritier/OOPAO/blob/master/tutorials/image_formation.py
        """
        print("\n" + "="*60)
        print("§4 目标漫反射 —— 粗糙面散射")
        print("="*60)
        print(f"  粗糙面 realization 数 = {n_roughness}")
        print(f"  种子 = {seed}")

        # 生成多个 realization 的散射场
        results = []
        for j in range(n_roughness):
            phi_r = random_roughness_phase((self.N, self.N), seed=(seed * 31 + j) % (2**32))
            E_scat = (np.sqrt(np.maximum(I_obj, 0)) * np.exp(1j * phi_r)).astype(np.complex64)
            I_scat = (np.abs(E_scat)**2).astype(np.float32)
            results.append({"phi": phi_r, "E": E_scat, "I": I_scat})
            print(f"  [Realization {j+1}] 峰值强度 = {I_scat.max():.6e}")

        if visualize:
            fig, axes = plt.subplots(1, min(n_roughness, 3) + 1, figsize=(5*(min(n_roughness,3)+1), 5))
            if n_roughness == 1:
                axes = [axes]
            for j in range(min(n_roughness, 3)):
                plot_phase(results[j]["phi"],
                          f"{LABELS['title_roughness']} (j={j+1})", ax=axes[j])
            plot_intensity(I_obj, "入射强度\n(散射前)", ax=axes[min(n_roughness,3)])
            show_fig(LABELS["title_scatter"])

        return StepResult("target_scattering", {
            "phi_roughness": [r["phi"] for r in results],
            "E_scat": [r["E"] for r in results],
            "I_scat": [r["I"] for r in results],
            "I_obj": I_obj,
            "n_roughness": n_roughness,
        })

    # ══════════════════════════════════════════════════════════════════
    # §5 返回大气传播
    # ══════════════════════════════════════════════════════════════════
    def step5_atmosphere_return(
        self,
        E_scat_list: list[np.ndarray],
        screens: np.ndarray,
        visualize: bool = True,
    ) -> StepResult:
        """§5 返回大气传播：反向传播到望远镜。

        Parameters
        ----------
        E_scat_list : list of np.ndarray
            各 realization 的散射场列表。
        screens : np.ndarray
            (n_screens, N, N) 湍流相位屏。

        Returns
        -------
        StepResult
            包含 E_back_list, I_back_mean 等。
        # OOPAO 参考: https://github.com/cheritier/OOPAO/tree/master/tutorials
        """
        print("\n" + "="*60)
        print("§5 返回大气传播 —— 反向传播到望远镜入瞳")
        print("="*60)

        n_screens = screens.shape[0]
        screen_sep = self.L / n_screens
        print(f"  反向穿过 {n_screens} 层湍流屏 (Δz={screen_sep:.0f}m)")
        print(f"  [吸收边界] 移除入瞳(直径 D={self.Dscope*100:.1f}cm)外的场 → 有限口径望远镜")
        screens_back = screens[::-1]

        E_back_list = []
        I_back_list = []
        # 返回路径采样：在每层屏之间记录中间截面场（用于 §6 不同距离成像 + 演化图）
        # 每个 realization 取第 0/2/4/6/8 屏后（含目标面）共 6 个截面，z 从目标面算起
        E_snap = [[] for _ in range(6)]
        z_snap = None

        for j, E_scat in enumerate(E_scat_list):
            # 反向传播（逐屏 step-by-step，便于采样中间截面）
            E_back = np.array(E_scat, dtype=np.complex64, copy=True)
            E_snap[0].append(E_back.copy())          # z = L（目标面）
            n_screens_b = screens_back.shape[0]
            for i, phi in enumerate(screens_back):
                # split_step 的单屏内部：dz/2 传播 → 加屏相位 → dz/2 传播
                E_back = self.prop.propagate(E_back, -screen_sep / 2.0)
                E_back = np.array(E_back, dtype=np.complex64, copy=True)
                E_back *= np.exp(1j * phi).astype(np.complex64)
                E_back = self.prop.propagate(E_back, -screen_sep / 2.0)
                # 采样点：0 表示还没穿过任何屏，之后每 2 层记一次（[0,2,4,6,8] 屏后 + 最后）
                if z_snap is None:
                    z_snap = np.linspace(0.0, self.L, 6)  # 距目标面的返回距离
                for s_i in range(1, 6):
                    frac = s_i / 5.0
                    if abs((i + 1) / n_screens_b - frac) < 1.0 / (2 * n_screens_b):
                        E_snap[s_i].append(E_back.copy())
            # 吸收边界（有限孔径）——入瞳处
            E_back = (E_back * self.pupil).astype(np.complex64)
            I_back = (np.abs(E_back)**2).astype(np.float32)

            E_back_list.append(E_back)
            I_back_list.append(I_back)
            print(f"  [Realization {j+1}] 返回面峰值强度 = {I_back.max():.6e}")

        I_back_mean = np.mean(I_back_list, axis=0)

        if visualize:
            # ── 图 A：返回面强度（原有） ──
            fig, axes = plt.subplots(1, min(len(E_back_list), 3) + 1,
                                    figsize=(5*(min(len(E_back_list),3)+1), 5))
            if len(E_back_list) == 1:
                axes = [axes]
            for j in range(min(len(E_back_list), 3)):
                plot_intensity(I_back_list[j],
                              f"返回面强度 (j={j+1})", ax=axes[j])
            plot_intensity(I_back_mean, f"返回面平均强度\n({len(E_back_list)} realization)",
                          ax=axes[min(len(E_back_list), 3)])
            show_fig(LABELS["title_atm_ret"])

            # ── 图 B：返回路径上光强 / 相位演化（新增） ──
            # 参考 OOPAO 教程 image_formation.py 中逐屏显示相位屏/PSF 的方式：
            # https://github.com/cheritier/OOPAO/blob/master/tutorials/image_formation.py
            n_snap = len(E_snap)
            fig, axes = plt.subplots(2, n_snap, figsize=(3.2 * n_snap, 6.5))
            for s_i in range(n_snap):
                E_mid = np.mean(E_snap[s_i], axis=0)  # 多 realization 平均
                I_mid = np.abs(E_mid)**2
                phi_mid = np.angle(E_mid)
                z_here = self.L - z_snap[s_i]          # 距光源/入瞳的坐标
                # 上排：光强（log₁₀）
                plot_intensity(I_mid.astype(np.float32),
                              f"z = {z_here:.0f} m\n(距目标 {z_snap[s_i]:.0f} m)",
                              log_scale=True, ax=axes[0, s_i])
                # 下排：相位
                plot_phase(phi_mid,
                          f"返回相位 @ z={z_here:.0f} m",
                          ax=axes[1, s_i])
            axes[0, 0].set_ylabel(LABELS["ret_intensity"], fontsize=11)
            axes[1, 0].set_ylabel(LABELS["ret_phase"], fontsize=11)
            show_fig(LABELS["title_ret_evol"])

        return StepResult("atmosphere_return", {
            "E_back_list": E_back_list,
            "I_back_list": I_back_list,
            "I_back_mean": I_back_mean,
            # 返回路径中间截面场（§6 不同距离成像用）
            "E_snapshots": [np.mean(s, axis=0) for s in E_snap],  # (6, N, N) 平均场
            "z_snapshots": z_snap,                                 # 距目标面的返回距离 (m)
        })

    # ══════════════════════════════════════════════════════════════════
    # §6 望远镜成像系统
    #
    #     返回光已到达望远镜入瞳（§5 输出，经 pupil 孔径吸收边界）。
    #     §6 实现正确的望远成像链：
    #
    #         望远镜入瞳(直径 D, 孔径掩膜)
    #             │  用出射相位的共轭准直:  E_c = E_back · exp(-i φ_total)
    #             ▼
    #         准直后的平行光
    #             │  物镜(焦距 f_obj):      E_l = E_c · exp(-i k r²/(2 f_obj))
    #             ▼
    #         物镜后方三个测量平面:  f_obj - z_R (焦前), f_obj (焦面), f_obj + z_R (焦后)
    #             │  角谱传播 + 非相干(强度)平均
    #             ▼
    #         images (3, N, N) —— 每平面一张图像
    #
    #     焦面 (f_obj) 应形成清晰聚焦光斑（能量集中在桶内 → FOM 高的前提）；
    #     焦前/焦后平面因离焦而展宽，正是多平面 CNN 感知深度(折射率起伏)的
    #     信息载体（论文 Sec 2.4, Eq 12）。
    # ══════════════════════════════════════════════════════════════════
    def step6_imaging(
        self,
        E_back_list: list[np.ndarray],
        phi_total: np.ndarray,
        visualize: bool = True,
        E_snapshots: list[np.ndarray] | None = None,
        z_snapshots: np.ndarray | None = None,
    ) -> StepResult:
        """§6 望远镜成像：入瞳 → 准直 → 物镜聚焦 → 多平面成像。

        Parameters
        ----------
        E_back_list : list of np.ndarray
            各 realization 的返回面（望远镜入瞳处）光场。
        phi_total : np.ndarray
            出射总相位（聚焦 + 调控），其共轭用于把会聚场准直回平行光。
        visualize : bool
            是否绘制可视化图。
        E_snapshots : list of np.ndarray, optional
            返回路径中间截面光场（来自 step5），用于"不同距离远端光斑成像"。
        z_snapshots : np.ndarray, optional
            各截面距目标面的返回距离 (m)，与 E_snapshots 一一对应。

        Returns
        -------
        StepResult
            包含 images (3, N, N) 等。

        Notes
        -----
        "不同距离对远端光斑成像"图参考 OOPAO 教程 image_formation.py 中
        使用角坐标 (arcsec) 显示 PSF 的方式：
        https://github.com/cheritier/OOPAO/blob/master/tutorials/image_formation.py
        """
        print("\n" + "="*60)
        print("§6 望远镜成像系统 —— 入瞳 → 物镜 → 多平面")
        print("="*60)
        print(f"  望远镜口径 D = {self.Dscope*100:.1f} cm (入瞳光阑)")
        print(f"  物镜焦距 f_obj = {self.f_obj:.1f} m (Eq 12: f_obj = 2 z_R)")
        print(f"  瑞利距离 z_R = {self.zR_APWS:.1f} m (Eq 9)")
        print(f"  测量平面: {self.plane_offsets} m 距物镜")
        print(f"  出射相位共轭准直 → 物镜二次相位 → 角谱传播")

        plane_labels = ["焦前 (f_obj - z_R)", "焦面 (f_obj)", "焦后 (f_obj + z_R)"]
        images = np.zeros((3, self.N, self.N), dtype=np.float32)

        for j, E_back in enumerate(E_back_list):
            # ① 入瞳处已施加 pupil 吸收边界（§5）—— 有限口径望远镜接收
            # ② 准直：用出射总相位的共轭，把会聚的返回场转回平行光
            E_c = (E_back * np.exp(-1j * phi_total)).astype(np.complex64)
            # ③ 物镜：焦距 f_obj 的二次聚焦相位（望远镜的成像元件）
            E_l = (E_c * np.exp(-1j * self.k * self.r2 / (2.0 * self.f_obj))).astype(np.complex64)
            # ④ 各平面成像（非相干：对强度取平均，而非场平均）
            for p in range(3):
                I_p = self.prop.angular_spectrum_intensity(E_l, self.plane_offsets[p])
                images[p] += (I_p * self.pupil).astype(np.float32)

        images /= len(E_back_list)

        for p in range(3):
            print(f"  [{plane_labels[p]}] 峰值 = {images[p].max():.6e}, "
                  f"总能量 = {images[p].sum():.6e}")

        if visualize:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            for p in range(3):
                plot_intensity(images[p], f"{LABELS['title_image']} {plane_labels[p]}\n"
                             f"(d={self.plane_offsets[p]:.1f}m)", ax=axes[p])
            show_fig(LABELS["title_imaging"])

            # ═══ 不同距离对远端光斑的成像（教程风格） ═══
            # 参考 OOPAO 教程 image_formation.py：用角坐标 (arcsec) 显示 PSF/成像。
            # 这里把返回路径上不同位置（距望远镜不同距离）的截面光场分别聚焦成像，
            # 展示"目标距离不同 → 焦面成像清晰度不同"的物理（离焦 + 湍流累积）。
            # 教程: https://github.com/cheritier/OOPAO/blob/master/tutorials/image_formation.py
            if E_snapshots is not None and z_snapshots is not None:
                n_snap = len(E_snapshots)
                # 焦面像素对应的角尺度（弧秒/像素）: θ_px = dx / f_obj
                arcsec_per_px = 206265.0 * self.dx / self.f_obj
                extent = [-self.N / 2.0 * arcsec_per_px, self.N / 2.0 * arcsec_per_px,
                          -self.N / 2.0 * arcsec_per_px, self.N / 2.0 * arcsec_per_px]

                fig, axes = plt.subplots(1, n_snap, figsize=(4.0 * n_snap, 4.2))
                if n_snap == 1:
                    axes = [axes]
                for s_i in range(n_snap):
                    E_mid = E_snapshots[s_i].astype(np.complex64)
                    z_tgt = self.L - z_snapshots[s_i]   # 该截面对应的目标距离（距入瞳）
                    # 物镜二次相位 → 角谱传播到焦面
                    E_l = (E_mid * np.exp(-1j * self.k * self.r2 / (2.0 * self.f_obj))).astype(np.complex64)
                    I_img = self.prop.angular_spectrum_intensity(E_l, self.f_obj)
                    I_img = (I_img * self.pupil).astype(np.float32)
                    plot_intensity(I_img, f"{LABELS['dist_image']}\n目标距离 z={z_tgt:.0f} m",
                                  log_scale=True, ax=axes[s_i])
                    axes[s_i].set_xlabel(f"{LABELS['xlabel_arcsec']}\n({arcsec_per_px:.3f} \"/px)")
                    axes[s_i].set_ylabel(LABELS["ylabel_arcsec"])
                    axes[s_i].set_xlim(0, self.N - 1)
                    axes[s_i].set_ylim(0, self.N - 1)
                    # 标注角坐标（低/中/高弧秒刻度）
                    axes[s_i].set_xticks([0, self.N // 2, self.N - 1])
                    axes[s_i].set_xticklabels([f"{-self.N//2*arcsec_per_px:.1f}", "0",
                                               f"{self.N//2*arcsec_per_px:.1f}"])
                    axes[s_i].set_yticks([0, self.N // 2, self.N - 1])
                    axes[s_i].set_yticklabels([f"{-self.N//2*arcsec_per_px:.1f}", "0",
                                               f"{self.N//2*arcsec_per_px:.1f}"])
                show_fig(LABELS["title_dist_imaging"])

            # 光学布局示意图（望远镜成像几何）
            fig, ax = plt.subplots(figsize=(14, 4))
            ax.set_xlim(0, 10)
            ax.set_ylim(-2, 2)
            ax.axis("off")

            # 返回光线（从右往左汇聚到入瞳）
            for y_off in (-1.2, -0.8, -0.4, 0.0, 0.4, 0.8, 1.2):
                ax.plot([0.4, 2.2], [y_off * 0.6, y_off * 0.55], color="#2c3e50",
                        lw=1.2, alpha=0.7)
            ax.text(0.7, 1.7, "返回光\n(经大气, §5)", ha="center", fontsize=11)

            # 望远镜入瞳（光阑）
            ax.plot([2.2, 2.2], [-1.5, 1.5], color="#c0392b", lw=3)
            ax.annotate("入瞳光阑 D=%.0fcm" % (self.Dscope*100),
                        xy=(2.2, 1.5), xytext=(2.4, 1.8), fontsize=11,
                        arrowprops=dict(arrowstyle="->"))

            # 准直后平行光线
            for y_off in (-1.2, -0.8, -0.4, 0.0, 0.4, 0.8, 1.2):
                ax.plot([2.2, 5.2], [y_off * 0.55, y_off * 0.55], color="#16a085",
                        lw=1.2, alpha=0.7)
            ax.text(3.7, 1.7, "准直平行光\n(相位共轭)", ha="center", fontsize=11)

            # 物镜
            ax.plot([5.2, 5.2], [-1.5, 1.5], color="#2980b9", lw=3)
            ax.annotate("物镜 f_obj=%.0fm" % self.f_obj,
                        xy=(5.2, 1.5), xytext=(5.4, 1.8), fontsize=11,
                        arrowprops=dict(arrowstyle="->"))

            # 聚焦到焦平面
            for y_off in (-1.2, -0.8, -0.4, 0.0, 0.4, 0.8, 1.2):
                ax.plot([5.2, 7.6], [y_off * 0.55, y_off * 0.18], color="#e67e22",
                        lw=1.2, alpha=0.7)
            ax.plot([7.6, 7.6], [-0.6, 0.6], color="#8e44ad", lw=2.5)
            ax.annotate("焦面 f_obj\n(±z_R 处还有\n焦前/焦后平面)",
                        xy=(7.6, 0.6), xytext=(8.1, 1.1), fontsize=11,
                        arrowprops=dict(arrowstyle="->"))
            ax.text(0.0, -1.8, "望远镜成像链: 入瞳(ΦD) → 相位共轭准直 → 物镜(二次相位) → 多平面(角谱传播)",
                    fontsize=12, fontweight="bold")
            plt.tight_layout()
            plt.show()

        return StepResult("imaging", {"images": images})

    # ══════════════════════════════════════════════════════════════════
    # §8 五角星均匀光源 —— 望远镜成像（无湍流 vs 有湍流）
    # ══════════════════════════════════════════════════════════════════
    def _star_mask(self, n_pts: int = 5, r_outer: float = 0.5, r_inner: float | None = None,
                   N: int | None = None) -> np.ndarray:
        """生成五角星形状的均匀亮度掩膜（几何, 返回 float 0/1）。

        Parameters
        ----------
        n_pts : int
            角点数（默认五角星 = 5）。
        r_outer : float
            外接圆半径（相对网格半宽，0~1）。
        r_inner : float, optional
            内凹半径；默认按五角星几何比例 r_inner = r_outer * cos(2π/n) 近似。
        N : int, optional
            网格尺寸（默认 self.N）。

        Returns
        -------
        np.ndarray
            (N, N) float32, 星形内部为 1，外部为 0（均匀光源掩膜）。
        """
        N = N or self.N
        y, x = np.mgrid[0:N, 0:N]  # 行=y, 列=x
        cx = cy = (N - 1) / 2.0
        xx = (x - cx) / ((N - 1) / 2.0)
        yy = (y - cy) / ((N - 1) / 2.0)
        r = np.sqrt(xx**2 + yy**2)
        theta = np.arctan2(yy, xx)

        # 五角星顶点在角度 phi_k，角点交替外/内半径
        # 五角星标准比例: 内凹点半径 ≈ 外接圆半径 × cos(π/5)/cos(2π/5) ≈ 0.382
        if r_inner is None:
            r_inner = r_outer * 0.382

        mask = np.zeros((N, N), dtype=np.float32)
        # 对每个角扇区判断是否在星形内（利用星形是"角度上 5 段交替边界"的简单判定）
        # 简化实现：直接采样一组五角星顶点多边形，用射线法/点在多边形内判定。
        angles = np.pi / 2.0 + 2.0 * np.pi * np.arange(0, 2 * n_pts) / (2 * n_pts)
        radii = np.array([r_outer if i % 2 == 0 else r_inner
                          for i in range(2 * n_pts)])
        vx = cx + radii * np.cos(angles) * ((N - 1) / 2.0)
        vy = cy + radii * np.sin(angles) * ((N - 1) / 2.0)

        from matplotlib.path import Path
        star_path = Path(np.column_stack([vx, vy]))
        pts = np.column_stack([x.ravel(), y.ravel()])
        mask = star_path.contains_points(pts).reshape(N, N).astype(np.float32)
        return mask

    def step8_star_imaging(
        self,
        screens: np.ndarray,
        seed: int = 42,
        visualize: bool = True,
    ) -> StepResult:
        """§8 五角星均匀光源望远镜成像 —— 无湍流 vs 有湍流。

        用一幅五角星形状的均匀亮度"景物"作为扩展光源（模拟远端目标/星座形状的
        亮度分布），经望远镜成像：

        * **无湍流**：入瞳相位仅含理想平面波 → 点扩散函数 (PSF) 为衍射极限
          Airy 斑；成像 = 五角星亮度 ⊛ PSF（清晰的五角星像）。
        * **有湍流**：入瞳相位叠加湍流屏 OPD → PSF 展宽/破碎（散斑）；
          成像 = 五角星亮度 ⊛ 湍流退化 PSF（模糊、破碎的五角星像）。

        计算方式：与 OOPAO 教程 image_formation.py 中 `tel.computePSF` 一致的
        思路 —— PSF 是入瞳自动相干的 FFT 功率谱；扩展光源成像 = 源亮度与 PSF
        的卷积（非相干成像，强度线性叠加）。
        参考:
        https://github.com/cheritier/OOPAO/blob/master/tutorials/image_formation.py

        Parameters
        ----------
        screens : np.ndarray
            湍流相位屏 (n, N, N)。取第一屏投影到入瞳作为 OPD。
        seed : int
            随机种子（湍流屏已由 seed 生成，此处仅用于本演示的散斑渲染）。
        visualize : bool
            是否绘制可视化图。

        Returns
        -------
        StepResult
            包含 star_mask, I_noao, I_turb, psf_noao, psf_turb。
        """
        print("\n" + "="*60)
        print("§8 五角星均匀光源望远成像 —— 无湍流 vs 有湍流")
        print("="*60)

        N = self.N
        star = self._star_mask(N=N)

        # ── 入瞳复振幅（望远镜口径均匀照明） ──
        pupil_amp = self.pupil.astype(np.complex64)

        # 无湍流：相位 = 0 → 衍射极限 PSF
        E_pupil_noao = pupil_amp.copy()

        # 有湍流：叠加第一屏湍流相位（rad）
        phi_turb = screens[0].astype(np.float32)
        E_pupil_turb = (pupil_amp * np.exp(1j * phi_turb)).astype(np.complex64)

        # ══ PSF 角采样与星形掩膜对齐 ══
        # 星形掩膜所在角网格: dθ = dx / f_obj (像面像素角)。
        # 入瞳 FFT 生成的 PSF 角采样为 λ/(N_pad·dx)，令二者相等:
        #   N_pad = λ·f_obj / dx²   (这里 ≈ 3000)
        # 然后把 pupil 零填充到 N_pad×N_pad 再做 FFT，取中心 N×N。
        N_pad = int(np.ceil(self.lam * self.f_obj / (self.dx * self.dx)))
        N_pad += (N_pad % 2)  # 偶数
        c0 = N_pad // 2 - N // 2

        def _psf_same_scale(E_pupil: np.ndarray) -> np.ndarray:
            pad = np.zeros((N_pad, N_pad), dtype=np.complex64)
            pad[c0:c0 + N, c0:c0 + N] = E_pupil
            psf = np.abs(np.fft.fftshift(np.fft.fft2(pad))) ** 2
            # 中心 N×N 与 star 掩膜同角采样
            return (psf[N_pad//2 - N//2:N_pad//2 + N//2,
                        N_pad//2 - N//2:N_pad//2 + N//2] / psf.sum()).astype(np.float32)

        psf_noao = _psf_same_scale(E_pupil_noao)
        psf_turb = _psf_same_scale(E_pupil_turb)

        # ── 扩展光源成像 = 源亮度与 PSF 的卷积（非相干成像） ──
        # 用 FFT 快速卷积（循环卷积，掩膜在网格中心，边界效应可忽略）
        # 参考: Goodman "Introduction to Fourier Optics", Sec on incoherent imaging
        from numpy.fft import fft2, ifft2, fftshift, ifftshift

        def _conv(src: np.ndarray, psf: np.ndarray) -> np.ndarray:
            # 循环卷积: 把 PSF 中心 (经 fftshift 居中) 搬回原点做卷积核,
            # ifft2 输出为常规布局(零频在 (0,0)) → 像的峰值位于 src 中心。
            psf0 = ifftshift(psf)
            return np.real(ifft2(fft2(src) * fft2(psf0)))

        I_noao = _conv(star, psf_noao)
        I_turb = _conv(star, psf_turb)

        # 抓取光斑能量中心附近（裁剪到星形区域展示）
        def _crop(I: np.ndarray, frac: float = 0.75) -> np.ndarray:
            c = N // 2
            half = int(N * frac / 2.0)
            return I[c - half:c + half, c - half:c + half]

        I_noao_c = _crop(I_noao)
        I_turb_c = _crop(I_turb)
        star_c = _crop(star)
        psf_noao_c = _crop(psf_noao, 0.35)
        psf_turb_c = _crop(psf_turb, 0.35)

        if visualize:
            # 与教程 image_formation.py 风格一致：log 显示 + 角坐标
            arcsec_per_px = 206265.0 * self.dx / self.f_obj
            fig, axes = plt.subplots(2, 3, figsize=(16, 10))

            # 第 1 行：源 + 两种 PSF
            plot_intensity(star_c, LABELS["star_source"], ax=axes[0, 0])
            plot_intensity(np.log10(np.maximum(psf_noao_c, 1e-12)),
                           LABELS["star_diff_limit"], ax=axes[0, 1])
            plot_intensity(np.log10(np.maximum(psf_turb_c, 1e-12)),
                           LABELS["star_turb_psf"], ax=axes[0, 2])
            for a in (axes[0, 1], axes[0, 2]):
                a.set_xticks([]); a.set_yticks([])

            # 第 2 行：无湍流成像 / 有湍流成像
            plot_intensity(I_noao_c, LABELS["star_no_turb"], ax=axes[1, 0])
            plot_intensity(I_turb_c, LABELS["star_with_turb"], ax=axes[1, 1])

            # 第 3 格：对比说明（星形区域约束能量 / Strehl）
            ax = axes[1, 2]
            # 像落在五角星形状内的能量占比（形状保真度）
            def _star_confinement(Ic: np.ndarray, src_c: np.ndarray) -> float:
                inside = (Ic * src_c).sum()
                return float(inside / max(Ic.sum(), 1e-30))

            r_noao = _star_confinement(I_noao_c, star_c)
            r_turb = _star_confinement(I_turb_c, star_c)
            strehl = float(psf_turb_c.max() / max(psf_noao_c.max(), 1e-30))
            ax.bar([0, 1], [r_noao, r_turb], color=["#2ecc71", "#e74c3c"], alpha=0.8)
            ax.set_xticks([0, 1])
            ax.set_xticklabels([LABELS["star_no_turb"], LABELS["star_with_turb"]], fontsize=10)
            ax.set_ylabel("星形区域能量占比", fontsize=11)
            ax.set_title(f"像质对比\n(形状保真, Strehl={strehl:.2f})", fontsize=12)
            ax.set_ylim(0, 1)
            for i, (r, name) in enumerate(zip([r_noao, r_turb],
                                              [LABELS["star_no_turb"], LABELS["star_with_turb"]])):
                ax.text(i, r + 0.02, f"{r:.3f}", ha="center", fontsize=11)

            show_fig(LABELS["title_star"])

        return StepResult("star_imaging", {
            "star_mask": star,
            "I_noao": I_noao, "I_turb": I_turb,
            "psf_noao": psf_noao, "psf_turb": psf_turb,
        })

    def step7_full_pipeline(
        self,
        seed: int = 42,
        n_roughness: int = 3,
        visualize: bool = True,
    ) -> StepResult:
        """§7 完整光路串联：无AO → 仅跟踪 → 信标共轭 → 78阶Zernike 的 FOM 对比。
        # OOPAO 参考: https://github.com/cheritier/OOPAO/blob/master/tutorials/AO_closed_loop_3_corrector_types.py

        Parameters
        ----------
        seed : int
            样本种子。
        n_roughness : int
            粗糙面 realization 数。

        Returns
        -------
        StepResult
            包含 FOM 值、各分支强度等。
        """
        print("\n" + "="*60)
        print("§7 完整光路串联 —— FOM 对比")
        print("="*60)

        # 生成湍流屏
        screens = self._make_screens(seed)
        n_screens = screens.shape[0]
        screen_sep = self.L / n_screens

        # ── 无 AO ──
        E_noao = (self.E0 * np.exp(1j * self.phi_focus)).astype(np.complex64)
        E_obj_noao = self.prop.split_step(E_noao, screens, screen_sep)
        I_noao = (np.abs(E_obj_noao)**2).astype(np.float32)
        fom_noao = FOM(I_noao, self.I_vac, self.bucket_mask)

        # ── 信标共轭（含跟踪）──
        # 信标反向
        w = self.lam * self.L / self.Dscope
        E_beacon = (np.exp(-self.r2 / w**2) * self.pupil).astype(np.complex64)
        E_back_beacon = self.prop.split_step(E_beacon, screens[::-1], -screen_sep)
        spherical = self.k * self.r2 / (2.0 * self.L)
        E_flat = E_back_beacon * np.exp(1j * spherical)
        phi_raw = np.angle(E_flat)
        I_beacon = (np.abs(E_back_beacon)**2).astype(np.float64)
        phi_unwrapped = self._simple_unwrap(phi_raw, I_beacon)
        phi_unwrapped -= phi_unwrapped[self.pupil].mean()
        phi_conj = -phi_unwrapped

        # 跟踪
        gx, gy = np.gradient(phi_conj)
        a_x = float(np.sum(self.G * gx) / np.sum(self.G))
        a_y = float(np.sum(self.G * gy) / np.sum(self.G))
        phi_track = a_x * self.X + a_y * self.Y
        phi_beacon = phi_conj - phi_track

        # Z78
        coeffs = self.zern.phase_to_zernike(phi_beacon)
        phi_z78 = self.zern.zernike_to_phase(coeffs)

        # 各分支 FOM
        def _fom(phi_total):
            E = self.prop.split_step(
                (self.E0 * np.exp(1j * phi_total)).astype(np.complex64),
                screens, screen_sep
            )
            I = (np.abs(E)**2).astype(np.float32)
            return FOM(I, self.I_vac, self.bucket_mask)

        fom_track = _fom(self.phi_focus + phi_track)
        fom_beacon = _fom(self.phi_focus + phi_track + phi_beacon)
        fom_z78 = _fom(self.phi_focus + phi_track + phi_z78)

        print(f"  FOM (无 AO)   = {fom_noao:.4f}")
        print(f"  FOM (仅跟踪)  = {fom_track:.4f}")
        print(f"  FOM (信标共轭) = {fom_beacon:.4f}")
        print(f"  FOM (78阶Zernike) = {fom_z78:.4f}")

        # 多平面成像（以跟踪条件为例）
        phi_total_track = self.phi_focus + phi_track
        E_obj_track = self.prop.split_step(
            (self.E0 * np.exp(1j * phi_total_track)).astype(np.complex64),
            screens, screen_sep
        )
        I_obj_track = (np.abs(E_obj_track)**2).astype(np.float32)

        images = np.zeros((3, self.N, self.N), dtype=np.float32)
        for _ in range(n_roughness):
            phi_r = random_roughness_phase((self.N, self.N), seed=(seed * 31 + _) % (2**32))
            E_scat = (np.sqrt(np.maximum(I_obj_track, 0)) * np.exp(1j * phi_r)).astype(np.complex64)
            E_back = self.prop.split_step(E_scat, screens[::-1], -screen_sep)
            E_back = (E_back * self.pupil).astype(np.complex64)
            E_c = (E_back * np.exp(-1j * phi_total_track)).astype(np.complex64)
            E_l = (E_c * np.exp(-1j * self.k * self.r2 / (2.0 * self.f_obj))).astype(np.complex64)
            for p in range(3):
                I_p = self.prop.angular_spectrum_intensity(E_l, self.plane_offsets[p])
                images[p] += (I_p * self.pupil).astype(np.float32)
        images /= n_roughness

        if visualize:
            fig, axes = plt.subplots(2, 4, figsize=(22, 11))

            # 第一行：FOM 对比
            foms = [fom_noao, fom_track, fom_beacon, fom_z78]
            names = [LABELS["no_ao"], LABELS["track_only"],
                    LABELS["beacon_conj"], LABELS["z78_78mode"]]
            colors = ["#e74c3c", "#f39c12", "#2ecc71", "#3498db"]

            for i, (fom_val, name, color) in enumerate(zip(foms, names, colors)):
                axes[0, i].barh([0], [fom_val], color=color, height=0.5, alpha=0.8)
                axes[0, i].set_xlim(0, 1)
                axes[0, i].set_yticks([])
                axes[0, i].set_xlabel(LABELS["fom_value"], fontsize=11)
                axes[0, i].set_title(f"{name}\nFOM = {fom_val:.4f}", fontsize=12)
                axes[0, i].axvline(x=1.0, color="gray", linestyle="--", alpha=0.5)

            # 第二行：成像结果
            for p in range(3):
                plot_intensity(images[p],
                              f"测量平面 {p+1}\n(d={self.plane_offsets[p]:.1f}m)",
                              ax=axes[1, p])
            # 桶掩膜可视化
            axes[1, 3].imshow(self.bucket_mask.astype(float), cmap="gray", origin="lower")
            D_bucket = float(self.cfg.bucket.diameter_frac) * self.L * self.lam / self.Dscope
            axes[1, 3].set_title(f"FOM 桶掩膜\nD={D_bucket*1e3:.1f}mm", fontsize=13)
            axes[1, 3].set_xlabel(LABELS["xlabel_pixel"])
            axes[1, 3].set_ylabel(LABELS["ylabel_pixel"])

            show_fig(LABELS["title_full"])

        return StepResult("full_pipeline", {
            "fom_noao": fom_noao, "fom_track": fom_track,
            "fom_beacon": fom_beacon, "fom_z78": fom_z78,
            "images": images, "coeffs": coeffs,
            "screens": screens,
        })

    # ══════════════════════════════════════════════════════════════════
    # 辅助：简化版相位解卷绕
    # ══════════════════════════════════════════════════════════════════
    def _simple_unwrap(self, phi_wrapped: np.ndarray, quality: np.ndarray) -> np.ndarray:
        """简化版 2D 相位解卷绕（从最亮像素出发 BFS）。

        注：完整版本在 data/simulate.py 中使用 numba 加速的 flood fill。
        此处为演示目的使用纯 numpy 实现。
        """
        from collections import deque

        N = phi_wrapped.shape[0]
        unwrapped = np.zeros_like(phi_wrapped)
        done = np.zeros((N, N), dtype=bool)

        # 从最亮像素出发
        i0, j0 = np.unravel_index(np.argmax(quality), quality.shape)
        unwrapped[i0, j0] = phi_wrapped[i0, j0]
        done[i0, j0] = True

        queue = deque([(i0, j0)])
        while queue:
            ci, cj = queue.popleft()
            for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                ni, nj = ci + di, cj + dj
                if 0 <= ni < N and 0 <= nj < N and not done[ni, nj]:
                    diff = phi_wrapped[ni, nj] - unwrapped[ci, cj]
                    diff = diff - 2.0 * np.pi * np.round(diff / (2.0 * np.pi))
                    unwrapped[ni, nj] = unwrapped[ci, cj] + diff
                    done[ni, nj] = True
                    queue.append((ni, nj))

        return unwrapped

    # ══════════════════════════════════════════════════════════════════
    # 运行所有步骤
    # ══════════════════════════════════════════════════════════════════
    def run_all(self, seed: int = 42, n_roughness: int = 3,
                visualize: bool = True) -> dict:
        """运行完整的逐过程仿真。

        Parameters
        ----------
        seed : int
            样本种子。
        n_roughness : int
            粗糙面 realization 数。
        visualize : bool
            是否显示可视化图。

        Returns
        -------
        dict
            所有步骤的结果字典。
        # OOPAO 参考: https://github.com/cheritier/OOPAO
        """
        print("="*60)
        print(f"  逐过程仿真 —— 种子 {seed}")
        print("="*60)

        # 生成湍流屏（全局复用）
        screens = self._make_screens(seed)
        screen_sep = self.L / screens.shape[0]

        # 光线传播路径总览示意图（教学用，非物理网格）
        if visualize:
            plot_light_path_schematic()

        # §1 光源
        r1 = self.step1_light_source(visualize=visualize)

        # §2 相位调控（beacon 相位共轭）
        r2 = self.step2_phase_control(screens, control_type="beacon_conj",
                                      visualize=visualize)

        # §3 大气传播（前向，使用 DM 校正后的光束）
        E_forward = r2.data["E_corrected"]
        r3 = self.step3_atmosphere_forward(screens, E_in=E_forward,
                                           visualize=visualize)

        # §4 目标漫反射
        r4 = self.step4_target_scattering(r3.data["I_obj"], seed=seed,
                                          n_roughness=n_roughness,
                                          visualize=visualize)

        # §5 返回大气传播
        r5 = self.step5_atmosphere_return(r4.data["E_scat"], screens,
                                         visualize=visualize)

        # §6 成像系统（含不同距离远端光斑成像）
        phi_total = self.phi_focus + r2.data["phi_ctrl"]
        r6 = self.step6_imaging(r5.data["E_back_list"], phi_total,
                               visualize=visualize,
                               E_snapshots=r5.data.get("E_snapshots"),
                               z_snapshots=r5.data.get("z_snapshots"))

        # §7 完整 FOM 对比
        r7 = self.step7_full_pipeline(seed=seed, n_roughness=n_roughness,
                                      visualize=visualize)

        # §8 五角星均匀光源望远成像（无湍流 vs 有湍流）
        r8 = self.step8_star_imaging(screens, seed=seed, visualize=visualize)

        return {
            "step1_source": r1, "step2_dm": r2, "step3_atm_fwd": r3,
            "step4_scatter": r4, "step5_atm_ret": r5, "step6_imaging": r6,
            "step7_fom": r7, "step8_star": r8,
        }


# ── 主入口 ────────────────────────────────────────────────────────
def main():
    """主入口：运行逐过程仿真。"""
    import argparse

    parser = argparse.ArgumentParser(description="逐过程光路仿真")
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="配置文件路径 (default: config.yaml)")
    parser.add_argument("--seed", type=int, default=42,
                       help="样本种子 (default: 42)")
    parser.add_argument("--n-roughness", type=int, default=3,
                       help="粗糙面 realization 数 (default: 3, 论文=10)")
    parser.add_argument("--no-plot", action="store_true",
                       help="不显示可视化图")
    args = parser.parse_args()

    sim = StepByStepSimulation(args.config)
    results = sim.run_all(
        seed=args.seed,
        n_roughness=args.n_roughness,
        visualize=not args.no_plot,
    )

    print("\n" + "="*60)
    print("仿真完成！所有步骤结果已返回。")
    print("="*60)


if __name__ == "__main__":
    main()
