"""算法 1 复现 —— DiComo 等, Opt. Express 33(15):31010 (2025)
（Algorithm 1 reproduction from DiComo et al., Opt. Express 33(15):31010 (2025)）

中文概览
--------
本模块实现论文逐样本仿真的完整流水线（算法 1）：
    湍流相位屏 -> 聚焦光束传播 -> 衍射极限信标反向传播 -> 倾斜跟踪
    -> 相位共轭 / Zernike-78 波前校正 -> FOM 评估 -> 多平面粗糙面成像。

本模块实现论文中完整的逐样本仿真流水线（算法 1）：湍流相位屏、聚焦光束
传播、衍射极限信标反向传播、跟踪、相位共轭 / Zernike-78 DM 校正、FOM 评估
以及多平面粗糙面成像。

Equations referenced are from the paper:
    Eq 6-8  : nPIB / SIB / FOM
    Eq 9-12 : imaging geometry (z_R_APWS, r_APWS, r0_eff, f_obj)
    Eq 13   : 12-bit intensity quantization
    Eq 14   : per-mode Zernike normalization (mu / sigma)

Determinism
-----------
Every sample is fully deterministic given its ``seed``. Phase screens are
generated with aotools ``ft_sh_phase_screen(..., seed=seed+i)`` (one seed per
screen) and roughness realizations use ``np.random.default_rng`` with a
seed derived from the sample seed. No global RNG state is consumed, so results
are reproducible regardless of process/worker assignment.

Screen-generation fallback
--------------------------
The task specifies Soapy's ``makePhaseScreens`` (via
``physics.screens_soapy.SoapyPhaseScreenGenerator``), but that path is NOT
deterministic under ``np.random.seed`` (verified empirically). Per the task
instructions we therefore fall back to aotools directly:
``aotools.turbulence.phasescreen.ft_sh_phase_screen(r0, N, box_size/N, L0,
l0_sim, seed=seed+i)`` for each screen ``i``. This is deterministic and
produces independent screens.
"""

from __future__ import annotations

import json
import multiprocessing
import os
from dataclasses import dataclass, field
from typing import Any, Optional

import h5py
import numpy as np
from numba import njit
from aotools.turbulence.phasescreen import ft_sh_phase_screen
from tqdm import tqdm

from physics.oopao_backend import OopaoScreenBackend
from physics.propagation_fft import Propagator
from physics.scattering import random_roughness_phase
from physics.screens_soapy import compute_r0
from physics.zernike_aotools import ZernikeBasis
from utils.metrics import FOM, bucket_mask

__all__ = [
    "SimSample",
    "physics_from_cfg",
    "simulate_sample",
    "simulate_sample_fom",
    "vacuum_intensity",
    "bucket_mask_nd",
    "generate_dataset",
]

N_MODES = 78  # Zernike truncation J = 78 (Table 1)


# --------------------------------------------------------------------------- #
# Shared per-process state
# --------------------------------------------------------------------------- #
@dataclass
class SharedSim:
    """Everything built once per process and reused across samples.

    The Propagator (FFTW_MEASURE init ~1-3 s) and ZernikeBasis (pinv ~1 s) are
    expensive to construct, so they are built once and shared. Grids, the
    aperture beam, focusing phase, vacuum intensity and imaging geometry are
    also precomputed here.
    """

    # 传播器：FFT 分步传播（FFTW_MEASURE 初始化约 1-3 s），跨样本复用
    prop: Propagator
    # Zernike 基底：78 阶 Noll 基 + 伪逆投影（构建约 1 s），跨样本复用
    zern: ZernikeBasis
    # FOM 桶掩膜 (N, N) bool：中心圆形区（公式 6），桶内积分得 nPIB
    bucket_mask: np.ndarray
    # 屏间距离 dz [m]（= screen_sep，论文表 1 = 100 m）
    dz: float
    # 网格分辨率 N（= N×N 像素）
    N: int
    # 像素间距 dx = box_size / N [m]
    dx: float
    # 中心波长 lam [m]
    lam: float
    # 波数 k = 2π/λ [rad/m]
    k: float
    # 初始光束半径 rspot [m]（= r_spot，论文表 1 = 7.5 cm）
    rspot: float
    # 聚焦焦距 focal [m]（= f，论文表 1 取 f = L）
    focal: float
    # (N, N) 以中心为原点的 x 坐标 [m]
    X: np.ndarray  # (N, N) x coordinates [m]
    # (N, N) 以中心为原点的 y 坐标 [m]
    Y: np.ndarray  # (N, N) y coordinates [m]
    # (N, N) 径向距离平方 r^2 [m^2]
    r2: np.ndarray  # (N, N) r^2 [m^2]
    # (N, N) 望远镜孔径掩膜（半径 Dscope/2），bool
    pupil: np.ndarray  # (N, N) bool aperture mask
    # (N, N) 倾斜跟踪的高斯权重（直径与出射光束一致，exp(-r^2/rspot^2)）
    G: np.ndarray  # (N, N) tracking Gaussian weighting
    # (N, N) 入瞳光束幅度 complex64（高斯幅度，孔径外为 0）
    E0: np.ndarray  # (N, N) complex64 aperture beam amplitude
    # (N, N) 聚焦相位 phi_focus = -k r^2/(2 f) [rad]，float64
    phi_focus: np.ndarray  # (N, N) float64 focusing phase
    # (N, N) 真空目标面强度 |E0 e^{i phi_focus} 传播 L|^2，float32（无湍流参考）
    I_vac: np.ndarray  # (N, N) float32 vacuum object-plane intensity
    # 瑞利距离 z_R_APWS [m]（公式 9-12 解析解）
    zR_APWS: float
    # 物镜焦距 f_obj = 2 z_R_APWS [m]（公式 12）
    f_obj: float
    # (3,) 三个测量平面距物镜的距离 [m]（f_obj - zR, f_obj, f_obj + zR）
    plane_offsets: np.ndarray  # (3,) distances behind objective lens [m]
    # OOPAO 屏幕后端；仅当 beam_source == "oopao" 时设置
    oopao: Optional[OopaoScreenBackend] = None  # set when beam_source == "oopao"


_shared_cache: dict[tuple, SharedSim] = {}


def _cfg_key(cfg: dict) -> tuple:
    """Hashable key identifying the physical/imaging/bucket configuration.

    中文：返回一个可哈希的元组，唯一标识物理/成像/桶配置。
    任一字段变化都会改变该键，从而触发共享状态重建（避免复用错误的缓存）。
    """
    p = cfg["physical"]
    img = cfg["imaging"]
    b = cfg["bucket"]
    return (
        p["N"],
        p["box_size"],
        p["wavelength"],
        p["Dscope"],
        p["rspot"],
        p["focal"],
        p["L"],
        p["cn2"],
        p["l0_sim"],
        p["L0"],
        p["screen_sep"],
        str(p.get("beam_source", "soapy")).lower(),
        img["zR_APWS"],
        img["f_obj"],
        tuple(img["plane_offset_frac"]),
        b["diameter_frac"],
    )


def _build_shared(cfg: dict) -> SharedSim:
    """构建每进程共享的仿真状态（跨样本复用，仅构建一次）。"""
    p = cfg["physical"]   # 物理参数节（湍流 / 光学 / 仿真，论文表 1）
    img = cfg["imaging"]  # 成像几何节（z_R_APWS / f_obj / 平面偏移）
    b = cfg["bucket"]     # FOM 桶参数节（桶直径分数，公式 6）

    # --- 从配置解析基本量 ---
    N = int(p["N"])               # 网格分辨率（N×N 像素，表 1 = 512）
    box = float(p["box_size"])    # 计算盒边长 [m]（表 1 = 0.30 m）
    dx = box / N                  # 像素间距 [m]
    lam = float(p["wavelength"])  # 中心波长 [m]（表 1 = 800 nm）
    k = 2.0 * np.pi / lam         # 波数 k = 2π/λ [rad/m]
    rspot = float(p["rspot"])     # 初始光束半径 [m]（表 1 = 7.5 cm）
    focal = float(p["focal"])     # 聚焦焦距 f [m]（表 1 取 f = L = 1000 m）
    L = float(p["L"])             # 传播距离 [m]（表 1 = 1000 m）
    Dscope = float(p["Dscope"])   # 望远镜口径 [m]（表 1 = 0.30 m）

    # 传播器（FFTW 分步）与 78 阶 Zernike 基底 —— 重计算量，跨样本共享
    prop = Propagator(N, dx, lam)
    zern = ZernikeBasis(N, N_MODES)

    # 以中心为原点的坐标网格 [m]
    x = (np.arange(N) - (N - 1) / 2.0) * dx
    X, Y = np.meshgrid(x, x)
    r2 = X**2 + Y**2  # 径向距离平方 [m^2]

    # 孔径掩膜：半径 Dscope/2（因 box == Dscope，恰为 N/2 px）
    pupil = r2 <= (Dscope / 2.0) ** 2

    # 入瞳光束：高斯幅度，孔径外置零
    E0 = np.exp(-(r2 / rspot**2)).astype(np.complex64)
    E0[~pupil] = 0.0

    # 聚焦相位 phi_focus = -k r^2/(2 f)（会聚到焦距 f 的二次相位）
    phi_focus = (-k * r2 / (2.0 * focal)).astype(np.float64)

    # 真空目标面强度（无湍流屏）：|E0 e^{i phi_focus} 传播 L|^2
    I_vac = prop.angular_spectrum_intensity(
        (E0 * np.exp(1j * phi_focus)).astype(np.complex64), L
    )

    # 倾斜跟踪高斯权重（直径与出射光束一致，用于加权梯度平均）
    G = np.exp(-(r2 / rspot**2))

    # 成像几何（公式 9-12）：
    #   r0 = (0.423 k^2 Cn^2 L)^(-3/5)（公式 11，大气相干长度）
    #   z_R_APWS = r0^2/(π λ)（公式 12 耦合方程解析解）
    #   f_obj = 2 z_R_APWS（公式 12）
    r0 = compute_r0(lam, float(p["cn2"]), L)
    zR_APWS = img["zR_APWS"] if img["zR_APWS"] is not None else r0**2 / (np.pi * lam)
    f_obj = img["f_obj"] if img["f_obj"] is not None else 2.0 * zR_APWS
    # 三个测量平面距物镜的距离（plane_offset_frac 给出相对 f_obj 的 zR 倍数）
    plane_offsets = np.array(
        [f_obj + (frac - 1.0) * zR_APWS for frac in img["plane_offset_frac"]],
        dtype=np.float64,
    )

    # FOM 桶掩膜（公式 6）：D_bucket = diameter_frac · L · λ / Dscope
    D_bucket = float(b["diameter_frac"]) * L * lam / Dscope
    diameter_px = D_bucket / dx
    bmask = bucket_mask(N, diameter_px)

    # OOPAO screen backend (beam_source == "oopao"); None otherwise. Built once
    # per process and shared across samples.
    oopao = None
    if str(p.get("beam_source", "soapy")).lower() == "oopao":
        oopao = OopaoScreenBackend(
            N=N,
            dx=dx,
            Dscope=Dscope,
            lam=lam,
            cn2=float(p["cn2"]),
            L=L,
            L0=float(p["L0"]),
            n_screens=int(p["n_screens"]),
        )

    return SharedSim(
        prop=prop,
        zern=zern,
        bucket_mask=bmask,
        dz=float(p["screen_sep"]),
        N=N,
        dx=dx,
        lam=lam,
        k=k,
        rspot=rspot,
        focal=focal,
        X=X,
        Y=Y,
        r2=r2,
        pupil=pupil,
        G=G,
        E0=E0,
        phi_focus=phi_focus,
        I_vac=I_vac,
        zR_APWS=zR_APWS,
        f_obj=f_obj,
        plane_offsets=plane_offsets,
        oopao=oopao,
    )


def _get_shared(cfg: dict) -> SharedSim:
    """Return the cached shared state for ``cfg``, building it if needed.

    中文：返回 ``cfg`` 对应的缓存共享状态，若未缓存则先构建。
    参数 cfg: 配置字典（见 config.yaml）。
    """
    key = _cfg_key(cfg)
    if key not in _shared_cache:
        _shared_cache[key] = _build_shared(cfg)
    return _shared_cache[key]


def _resolve_shared(shared: Any, cfg: dict) -> SharedSim:
    """Resolve the ``shared`` argument to a :class:`SharedSim`.

    Accepts ``None`` (build/cache from ``cfg``), a :class:`SharedSim`, or the
    ``(Propagator, ZernikeBasis, bucket_mask, dz)`` tuple returned by
    :func:`physics_from_cfg` (resolved through the per-cfg cache).

    中文：把 ``shared`` 参数统一解析成 :class:`SharedSim`。
    接受 None（从 cfg 构建/取缓存）、SharedSim 实例、或 physics_from_cfg
    返回的元组（经 cfg 缓存解析）。
    """
    if shared is None or isinstance(shared, SharedSim):
        return shared if isinstance(shared, SharedSim) else _get_shared(cfg)
    # Tuple from physics_from_cfg -> resolve via the cfg cache.
    # 元组形式 -> 通过 cfg 缓存解析（元组本身不携带足够重建信息）
    return _get_shared(cfg)


def physics_from_cfg(cfg: dict) -> tuple:
    """Build (once per process) and return ``(Propagator, ZernikeBasis, bucket_mask_2d, dz)``.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary (see config.yaml).
        中文：配置字典（见 config.yaml，含 physical / imaging / bucket 三节）。

    Returns
    -------
    tuple
        ``(Propagator, ZernikeBasis, bucket_mask_2d, dz)``.
        中文：（传播器, 78 阶 Zernike 基底, FOM 桶掩膜, 屏间距离 dz[m]）。
    """
    shared = _get_shared(cfg)
    return shared.prop, shared.zern, shared.bucket_mask, shared.dz


def bucket_mask_nd(diameter_px: float, N: int) -> np.ndarray:
    """Boolean (N, N) circular bucket mask with the given diameter in pixels.

    Delegates to :func:`utils.metrics.bucket_mask`.

    参数：
        diameter_px : 桶直径 [像素]（公式 6 的桶直径折算到像素）
        N : 网格分辨率
    返回：(N, N) bool 圆形桶掩膜。
    """
    return bucket_mask(N, diameter_px)


def vacuum_intensity(cfg: dict, shared: Any = None) -> np.ndarray:
    """Return the vacuum object-plane intensity ``|propagate(E0 e^{i phi_focus}, L)|^2``.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary.
        中文：配置字典（见 config.yaml）。
    shared : SharedSim or tuple, optional
        Prebuilt shared state (avoids rebuild).
        中文：预构建的共享状态（传入可避免重复构建，None 则从 cfg 取缓存）。

    Returns
    -------
    np.ndarray
        ``(N, N)`` float32 vacuum intensity.
        中文：(N, N) float32 真空目标面强度（无湍流参考）。
    """
    shared = _resolve_shared(shared, cfg)
    return shared.I_vac


# --------------------------------------------------------------------------- #
# Per-sample helpers
# --------------------------------------------------------------------------- #
def _make_screens(seed: int, cfg: dict, shared: Any) -> np.ndarray:
    """Generate ``n_screens`` deterministic turbulence phase screens.

    Uses aotools ``ft_sh_phase_screen`` directly (Soapy's wrapper is not
    deterministic under ``np.random.seed``). Each screen ``i`` uses seed
    ``seed + i`` so the screens are independent and reproducible.

    中文：生成 n_screens 层确定性湍流相位屏。
    直接使用 aotools 的 ft_sh_phase_screen（Soapy 包装器在 np.random.seed 下
    并非确定性）。第 i 层用种子 seed+i，保证各层独立且可复现。

    Parameters
    ----------
    seed : int
        Sample seed.
        中文：样本种子（样本种子 = master_seed + sample_index）。
    cfg : dict
        Configuration dictionary.
        中文：配置字典。
    shared : SharedSim or tuple
        Shared state (provides N, dx, L0, l0_sim, lam).
        中文：共享状态（提供 N, dx, L0, l0_sim, lam 等）。

    Returns
    -------
    np.ndarray
        ``(n_screens, N, N)`` float32 phase screens in radians.
        中文：(n_screens, N, N) float32 相位屏，单位 rad。
    """
    shared = _resolve_shared(shared, cfg)
    p = cfg["physical"]

    if shared.oopao is not None:
        # OOPAO path: per-layer screens drawn from the shared OOPAO Atmosphere,
        # each already rescaled to the target per-slab r0 and cropped to N x N.
        # 中文：OOPAO 路径 —— 从共享 OOPAO 大气中逐层抽取屏幕，每层已缩放到
        # 目标每 slab r0 并裁剪到 N×N。
        return shared.oopao.make_screens(seed)

    n_screens = int(p["n_screens"])  # 屏层数（= L / screen_sep，表 1 = 10）
    # Per-slab coherence length. ``compute_r0`` returns the path-integrated r0
    # for the full L. Each of the ``n_screens`` slabs (thickness L/n_screens)
    # carries r0_slab = r0_path * n_screens**(3/5); using the path r0 for every
    # slab would make the total turbulence ~n_screens**(3/5) times too strong.
    # 中文：每 slab 相干长度。compute_r0 返回整条 L 路径积分的 r0；
    # n_screens 个 slab（厚 L/n_screens）各取 r0_slab = r0_path * n^(3/5)。
    # 若每层都用整条路径的 r0，总湍流强度会偏大 ~n^(3/5) 倍。
    r0_path = compute_r0(shared.lam, float(p["cn2"]), float(p["L"]))  # 整条路径 r0 [m]
    r0_slab = r0_path * n_screens ** (3.0 / 5.0)                       # 每 slab r0 [m]
    l0_sim = float(p["l0_sim"])  # 内尺度 [m]（仿真守护值，表 1 = 0.01 m）
    L0 = float(p["L0"])          # 外尺度 [m]（表 1 = 100 m）
    screens = np.stack(
        [
            # 第 i 层相位屏：用种子 seed+i，保证可复现且各层独立
            ft_sh_phase_screen(r0_slab, shared.N, shared.dx, L0, l0_sim, seed=seed + i)
            for i in range(n_screens)
        ]
    ).astype(np.float32)
    return screens



def _unwrap_flood_fill(phi_w: np.ndarray, quality: np.ndarray) -> np.ndarray:
    """2D phase unwrap by intensity-guided flood fill (numba).

    The sequential row/column unwrap (``np.unwrap`` twice) creates multi-2pi
    branch cuts/ramps through the bright region of the beacon pupil field,
    which corrupt any phase-based Zernike fit (FOM_z78 ~ 0.02). A flood fill
    that grows from the brightest pixel and unwraps each new pixel against its
    already-unwrapped neighbours pushes the 2pi cuts into the weak-field
    regions, where they are invisible to the FOM and down-weighted by the
    Zernike fit. Verified: FOM_z78 0.02 -> ~0.93 (vs FOM_beacon ~0.95).

    中文：基于强度引导的洪水填充 2D 相位解卷绕（numba 加速）。
    逐行/列解卷绕（np.unwrap 两次）会在信标瞳孔场的亮区产生多 2π 分支切线/
    斜坡，破坏任何基于相位的 Zernike 拟合（FOM_z78 ~ 0.02）。从最亮像素出发
    的洪水填充让每个新像素相对其已解卷绕的邻居展开，把 2π 切线推到弱场区
    （对 FOM 不可见，且被 Zernike 拟合降权）。实测 FOM_z78 0.02 -> ~0.93。

    Parameters
    ----------
    phi_w : np.ndarray
        ``(N, N)`` wrapped phase in radians.
        中文：(N, N) 卷绕后的相位 [rad]。
    quality : np.ndarray
        ``(N, N)`` quality map (beacon intensity); the flood fill starts at
        its argmax.
        中文：(N, N) 质量图（信标强度）；洪水填充从其 argmax（最亮像素）出发。

    Returns
    -------
    np.ndarray
        ``(N, N)`` unwrapped phase (absolute 2pi offset is arbitrary; piston
        is removed by the caller).
        中文：(N, N) 解卷绕后的相位（绝对 2π 偏移任意，piston 由调用方移除）。
    """
    N = phi_w.shape[0]
    return _unwrap_flood_fill_nb(np.ascontiguousarray(phi_w), np.ascontiguousarray(quality), N)


@njit(cache=True)
def _wrap_diff(d: float) -> float:
    """Wrap a phase difference into (-pi, pi].

    中文：把相位差 d 卷绕到 (-π, π]。
    """
    return d - 2.0 * np.pi * np.round(d / (2.0 * np.pi))


@njit(cache=True)
def _median4(v: np.ndarray, cnt: int) -> float:
    """Median of the first ``cnt`` (<= 4) elements of ``v`` (insertion sort).

    中文：取 ``v`` 前 ``cnt``（<=4）个元素的中位数（插入排序，O(1) 常数小）。
    """
    for a in range(1, cnt):
        key = v[a]
        b = a - 1
        while b >= 0 and v[b] > key:
            v[b + 1] = v[b]
            b -= 1
        v[b + 1] = key
    if cnt % 2 == 1:
        return v[cnt // 2]
    return 0.5 * (v[cnt // 2 - 1] + v[cnt // 2])


@njit(cache=True)
def _unwrap_flood_fill_nb(phi_w: np.ndarray, quality: np.ndarray, N: int) -> np.ndarray:
    """Numba kernel: BFS flood fill from the brightest pixel (see wrapper).

    中文：Numba 内核 —— 从最亮像素出发的 BFS 洪水填充解卷绕。
    对每个新像素，用其已解卷绕邻居的相位差中位数解卷绕，把 2π 支切推到弱场区。
    """
    out = np.zeros((N, N))                     # 输出：解卷绕后的相位 (N, N)
    done = np.zeros((N, N), dtype=np.bool_)    # 已处理标记 (N, N)
    i0, j0 = 0, 0
    best = -1e30
    # 找到 quality 图最亮像素作为种子点（洪水填充起点）
    for i in range(N):
        for j in range(N):
            if quality[i, j] > best:
                best = quality[i, j]
                i0, j0 = i, j
    out[i0, j0] = phi_w[i0, j0]
    done[i0, j0] = True
    # BFS 队列（一维化）
    qi = np.zeros(N * N, dtype=np.int64)
    qj = np.zeros(N * N, dtype=np.int64)
    head, tail = 0, 0
    qi[tail], qj[tail] = i0, j0
    tail += 1
    while head < tail:
        i, j = qi[head], qj[head]
        head += 1
        # 四个邻居
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = i + di, j + dj
            if 0 <= ni < N and 0 <= nj < N and not done[ni, nj]:
                vals = np.empty(4)
                cnt = 0
                # 收集该像素已解卷绕的邻居，计算相对相位差
                for di2, dj2 in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    mi, mj = ni + di2, nj + dj2
                    if 0 <= mi < N and 0 <= mj < N and done[mi, mj]:
                        vals[cnt] = out[mi, mj] + _wrap_diff(phi_w[ni, nj] - phi_w[mi, mj])
                        cnt += 1
                if cnt > 0:
                    # 用邻居差的中位数解卷绕（对弱场离群更鲁棒）
                    out[ni, nj] = _median4(vals, cnt)
                else:
                    # 无已解卷绕邻居：直接用包裹相位
                    out[ni, nj] = phi_w[ni, nj]
                done[ni, nj] = True
                qi[tail], qj[tail] = ni, nj
                tail += 1
    return out


def _beacon_phase_conj(
    seed: int, cfg: dict, shared: SharedSim, screens: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Step D: diffraction-limited beacon back-propagation -> phi_conj.

    A diffraction-limited beacon (small Gaussian, waist = lambda*L/Dscope) at
    the object plane is back-propagated through the (reversed) screens to the
    pupil. The pupil-field phase is 2D-unwrapped (a simplification of full 2D
    phase unwrapping), piston (mode 0) and defocus (mode 3) are removed, and
    the result is conjugated:

        phi_conj = -(phi_unwrapped - phi_piston_defocus)

    中文：步骤 D —— 衍射极限信标反向传播 -> 共轭信标相位 phi_conj。
    在目标面放置一个衍射极限信标（小高斯，束腰 = λ·L/Dscope），反向穿过
    （倒序的）相位屏到瞳孔。瞳孔场相位做 2D 解卷绕，移除 piston（0 阶）与
    defocus（3 阶）后取共轭，得到无信标 AO 的波前估计 phi_conj。

    Parameters
    ----------
    seed : int
        Sample seed (unused here, kept for signature symmetry).
        中文：样本种子（此处未用，仅保持签名对称）。
    cfg : dict
        Configuration dictionary.
        中文：配置字典。
    shared : SharedSim
        Shared state.
        中文：共享状态。
    screens : np.ndarray
        ``(n_screens, N, N)`` float32 phase screens.
        中文：(n_screens, N, N) float32 相位屏 [rad]。

    Returns
    -------
    tuple
        ``(phi_conj, I_beacon)`` where ``phi_conj`` is the ``(N, N)`` float64
        conjugated beacon phase and ``I_beacon`` is the ``(N, N)`` float64
        beacon intensity at the pupil (diagnostic).
        中文：(phi_conj, I_beacon)。phi_conj 为 (N, N) float64 共轭信标相位；
        I_beacon 为 (N, N) float64 瞳孔处信标强度（仅诊断用）。
    """
    prop = shared.prop
    zern = shared.zern
    N = shared.N
    p = cfg["physical"]

    # Diffraction-limited beacon: a small Gaussian at the object plane whose
    # waist equals the diffraction limit seen from the telescope,
    # w = lambda * L / Dscope (~2.7 mm). A single-pixel delta has a flat
    # (infinite-bandwidth) angular spectrum, which makes the back-propagated
    # phase numerically spurious and uncorrelated with the true turbulence.
    # 中文：衍射极限信标 —— 目标面的小高斯，束腰等于望远镜看到的衍射极限
    # w = λ·L/Dscope（约 2.7 mm）。单像素 delta 的角谱平坦（无限带宽），会使
    # 反向传播相位在数值上失真、与真实湍流不相关，故用有限束腰高斯。
    w = shared.lam * float(p["L"]) / float(p["Dscope"])         # 信标束腰 [m]
    E_pt = (np.exp(-shared.r2 / w**2) * shared.pupil).astype(np.complex64)  # 信标场 (复)

    # Back-propagate through the reversed screens to the pupil.
    # 中文：反向穿过倒序相位屏到瞳孔（逆分步传播，-dz）。
    E_back = prop.split_step(E_pt, screens[::-1], -shared.dz)

    # The back-propagated beacon is a converging spherical wave with phase
    # -k*r^2/(2L). Remove it analytically (paper: "corrected to remove the
    # parabolic defocus term") so the residual turbulence phase unwraps
    # cleanly; a full-pupil low-order Zernike fit of the defocus would be
    # corrupted by weak-field edge outliers.
    # 中文：反向传播的信标是相位为 -k·r^2/(2L) 的会聚球面波。解析移除该
    # 抛物面离焦项（论文：“corrected to remove the parabolic defocus term”），
    # 使残余湍流相位能干净解卷绕；全口径低阶 Zernike 拟合离焦会被弱场边缘
    # 离群点污染。
    k = 2.0 * np.pi / shared.lam
    spherical = k * shared.r2 / (2.0 * float(p["L"]))   # 会聚球面相位 [rad]
    E_flat = E_back * np.exp(1j * spherical)            # 移除离焦后的平坦场

    # 强度引导的 2D 解卷绕（质量图 = 瞳孔处信标强度）
    phi_unwrapped = _unwrap_flood_fill(np.angle(E_flat), (np.abs(E_back) ** 2).astype(np.float64))
    # 移除 piston（孔径内平均相位置零）
    phi_unwrapped = phi_unwrapped - phi_unwrapped[shared.pupil].mean()

    phi_conj = -phi_unwrapped                    # 取共轭 -> 共轭信标相位
    I_beacon = (np.abs(E_back) ** 2).astype(np.float64)  # 瞳孔处信标强度（诊断）
    return phi_conj, I_beacon


def _tracking(shared: SharedSim, phi_conj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Step E: tilt tracking from the conjugated beacon phase.

    The gradient of ``phi_conj`` is weighted by the Gaussian ``G`` (identical
    diameter to the outgoing beam) and averaged to give the tilt slopes
    ``[a_x, a_y]``; the tracking phase is the linear ramp ``a_x x + a_y y``.

    中文：步骤 E —— 从共轭信标相位做倾斜跟踪。
    phi_conj 的梯度经高斯 G（直径与出射光束一致）加权平均，得到倾斜斜率
    [a_x, a_y]；跟踪相位为线性斜坡 a_x·x + a_y·y。

    Parameters
    ----------
    shared : SharedSim
        Shared state.
        中文：共享状态。
    phi_conj : np.ndarray
        ``(N, N)`` conjugated beacon phase.
        中文：(N, N) 共轭信标相位 [rad]。

    Returns
    -------
    tuple
        ``(phi_track, track_slopes)`` where ``phi_track`` is ``(N, N)`` float64
        and ``track_slopes`` is ``(2,)`` float64 ``[a_x, a_y]``.
        中文：(phi_track, track_slopes)。phi_track 为 (N, N) float64 跟踪相位；
        track_slopes 为 (2,) float64 [a_x, a_y]（斜率，rad/m）。
    """
    G = shared.G                              # 倾斜跟踪高斯权重 (N, N)
    gx, gy = np.gradient(phi_conj)            # 共轭信标相位的梯度
    a_x = float(np.sum(G * gx) / np.sum(G))   # x 方向倾斜斜率 [rad/m]
    a_y = float(np.sum(G * gy) / np.sum(G))   # y 方向倾斜斜率 [rad/m]
    phi_track = a_x * shared.X + a_y * shared.Y  # 线性斜坡跟踪相位 [rad]
    track_slopes = np.array([a_x, a_y], dtype=np.float64)
    return phi_track, track_slopes


def _fom_leg(
    shared: SharedSim, screens: np.ndarray, phi_total: np.ndarray
) -> float:
    """Step G: forward-propagate with a total aperture phase and return FOM.

    ``E_obj = split_step(E0 e^{i phi_total}, screens, dz)`` then
    ``FOM(|E_obj|^2, I_vac, bucket_mask)`` (Eqs 6-8).

    中文：步骤 G —— 用给定孔径总相位前向传播并返回 FOM。
    计算 E_obj = split_step(E0 e^{i phi_total}, screens, dz)，
    再 FOM(|E_obj|^2, I_vac, bucket_mask)（公式 6-8：nPIB / SIB / FOM）。

    Parameters
    ----------
    shared : SharedSim
        Shared state.
        中文：共享状态。
    screens : np.ndarray
        ``(n_screens, N, N)`` float32 phase screens.
        中文：(n_screens, N, N) float32 相位屏 [rad]。
    phi_total : np.ndarray
        ``(N, N)`` total beam phase at the aperture.
        中文：(N, N) 孔径处光束总相位 [rad]（= 聚焦相位 + 校正相位）。

    Returns
    -------
    float
        The figure of merit.
        中文：像质因子 FOM（公式 8，nPIB/SIB 之比，0-1，越大越好）。
    """
    E_obj = shared.prop.split_step(
        (shared.E0 * np.exp(1j * phi_total)).astype(np.complex64),  # 孔径场
        screens,                                                     # 湍流屏
        shared.dz,                                                   # 屏间距
    )
    I_obj = (np.abs(E_obj) ** 2).astype(np.float32)  # 目标面强度
    return FOM(I_obj, shared.I_vac, shared.bucket_mask)


def _imaging(
    seed: int, cfg: dict, shared: SharedSim, screens: np.ndarray, phi_track: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Step H: multi-plane rough-surface imaging (tracking-only condition).

    The tracking-only object field is scattered off ``n_roughness`` independent
    rough surfaces, back-propagated through the (reversed) screens, collimated
    by the conjugate of the outgoing phase, focused by the objective lens, and
    propagated to each measurement plane. The per-plane intensities are averaged
    over realizations.

    中文：步骤 H —— 多平面粗糙面成像（仅跟踪条件）。
    仅跟踪的目标面场经 n_roughness 个独立粗糙面散射，反向穿过（倒序）相位屏，
    用出射相位的共轭准直，经物镜聚焦，传播到各测量平面。各平面强度对
    各 realization 取平均。

    Parameters
    ----------
    seed : int
        Sample seed (drives the roughness RNG stream).
        中文：样本种子（驱动粗糙面 RNG 流）。
    cfg : dict
        Configuration dictionary.
        中文：配置字典。
    shared : SharedSim
        Shared state.
        中文：共享状态。
    screens : np.ndarray
        ``(n_screens, N, N)`` float32 phase screens.
        中文：(n_screens, N, N) float32 相位屏 [rad]。
    phi_track : np.ndarray
        ``(N, N)`` tracking phase.
        中文：(N, N) 跟踪相位 [rad]。

    Returns
    -------
    tuple
        ``(images, I_obj_track)`` where ``images`` is ``(3, N, N)`` float32
        (per-plane mean intensity) and ``I_obj_track`` is ``(N, N)`` float32.
        中文：(images, I_obj_track)。images 为 (3, N, N) float32 各平面平均
        强度；I_obj_track 为 (N, N) float32 仅跟踪目标面强度。
    """
    prop = shared.prop
    N = shared.N
    n_roughness = int(cfg["physical"]["n_roughness"])  # 粗糙面 realization 数（表 1 = 10）
    k = shared.k
    f_obj = shared.f_obj
    r2 = shared.r2

    # Tracking-only object-plane intensity.
    # 中文：仅跟踪目标面强度（聚焦 + 跟踪相位，经湍流屏前向传播）。
    phi_total = shared.phi_focus + phi_track
    E_obj_track = prop.split_step(
        (shared.E0 * np.exp(1j * phi_total)).astype(np.complex64),
        screens,
        shared.dz,
    )
    I_obj_track = (np.abs(E_obj_track) ** 2).astype(np.float32)

    pupil = shared.pupil
    images = np.zeros((3, N, N), dtype=np.float32)
    for j in range(n_roughness):
        # Roughness realization (deterministic given seed).
        # 中文：第 j 个粗糙面 realization（由 seed 决定，确定性可复现）。
        phi_r = random_roughness_phase((N, N), seed=(seed * 31 + j) % (2**32))
        E_scat = (np.sqrt(I_obj_track) * np.exp(1j * phi_r)).astype(np.complex64)
        # Back-propagate through the reversed screens.
        # 中文：反向穿过倒序相位屏（逆分步传播，-dz）。
        E_back = prop.split_step(E_scat, screens[::-1], -shared.dz)
        # Absorbing boundary: the telescope has a finite aperture (Dscope).
        # Field scattered outside the pupil must be removed before re-focusing,
        # otherwise it wraps through the FFT and brightens the image border
        # (paper Sec 2.4: absorbing boundary conditions to prevent boundary
        # reflection).
        # 中文：吸收边界 —— 望远镜口径有限（Dscope）。孔径外散射的场必须在
        # 重新聚焦前移除，否则经 FFT 卷绕会点亮图像边缘（论文 2.4 节：吸收
        # 边界条件，防止边界反射）。
        E_back = (E_back * pupil).astype(np.complex64)
        # Collimate by the conjugate of the outgoing phase.
        # 中文：用出射相位的共轭准直（把会聚场转回平行光）。
        E_c = (E_back * np.exp(-1j * phi_total)).astype(np.complex64)
        # Objective lens.
        # 中文：物镜（焦距 f_obj 的二次聚焦相位）。
        E_l = (E_c * np.exp(-1j * k * r2 / (2.0 * f_obj))).astype(np.complex64)
        # Incoherent imaging: average the per-realization INTENSITIES (paper
        # Sec 2.4: "incoherent image"), not the fields. Field averaging
        # retains the coherent speckle; intensity averaging blurs it out.
        # 中文：非相干成像 —— 对各 realization 的强度（而非场）取平均
        # （论文 2.4 节 “incoherent image”）。场平均会保留相干散斑，强度
        # 平均则将其模糊掉。
        for p in range(3):
            I_p = prop.angular_spectrum_intensity(E_l, shared.plane_offsets[p])
            images[p] += (I_p * pupil).astype(np.float32)

    images /= n_roughness
    return images, I_obj_track


# --------------------------------------------------------------------------- #
# Public sample API
# --------------------------------------------------------------------------- #
@dataclass
class SimSample:
    """One simulated sample (Algorithm 1 output).

    中文：一个仿真样本（算法 1 输出），包含图像、标签、各分支 FOM 及中间相位。

    Attributes
    ----------
    seed : int
        Sample seed.
        中文：样本种子（样本种子 = master_seed + sample_index）。
    images : np.ndarray
        ``(3, N, N)`` float32 RAW measurement-plane intensities (planes
        ``f_obj - zR``, ``f_obj``, ``f_obj + zR``), pre-quantization.
        中文：(3, N, N) float32 原始测量平面强度（未量化）；三平面分别为
        焦平面 f_obj 前后 ±z_R。
    labels : np.ndarray
        ``(78,)`` float64 raw ``phi_Z78`` Zernike coefficients [rad].
        中文：(78,) float64 原始 phi_Z78 Zernike 系数 [rad]（CNN 训练目标）。
    fom_noao / fom_track / fom_beacon / fom_z78 : float
        FOM for each correction leg.
        中文：各校正分支的 FOM：无 AO / 仅跟踪 / 信标共轭 / 78 阶 Zernike。
    fom_ml : float | None
        FOM for the ML leg, filled only when ``correction_coeffs`` is passed.
        中文：ML 分支的 FOM，仅在传入 correction_coeffs 时填充，否则 None。
    I_vac : np.ndarray
        ``(N, N)`` float32 vacuum object-plane intensity.
        中文：(N, N) float32 真空目标面强度（无湍流参考）。
    I_obj_track : np.ndarray
        ``(N, N)`` float32 tracking-only object-plane intensity.
        中文：(N, N) float32 仅跟踪目标面强度。
    track_slopes : np.ndarray
        ``(2,)`` float64 ``[a_x, a_y]`` of the tilt phase map.
        中文：(2,) float64 倾斜相位图的斜率 [a_x, a_y]（rad/m）。
    phase_conj / phase_track / phase_beacon / phase_z78 : np.ndarray
        ``(N, N)`` float64 phase maps.
        中文：(N, N) float64 各相位图：共轭信标 / 跟踪 / 残余 / 78 阶重构。
    beam_phases : dict
        Total beam phase at the aperture for each FOM leg.
        中文：dict，各 FOM 分支在孔径处的总光束相位 [rad]。
    """

    seed: int
    images: np.ndarray
    labels: np.ndarray
    fom_noao: float
    fom_track: float
    fom_beacon: float
    fom_z78: float
    fom_ml: Optional[float]
    I_vac: np.ndarray
    I_obj_track: np.ndarray
    track_slopes: np.ndarray
    phase_conj: np.ndarray
    phase_track: np.ndarray
    phase_beacon: np.ndarray
    phase_z78: np.ndarray
    beam_phases: dict = field(default_factory=dict)


def simulate_sample(
    seed: int,
    cfg: dict,
    correction_coeffs: Optional[np.ndarray] = None,
    *,
    shared: Optional[SharedSim] = None,
) -> SimSample:
    """Simulate one sample deterministically given ``seed`` (Algorithm 1).

    中文：给定 seed 确定性仿真一个样本（算法 1 全流程）。
    步骤 A 屏 -> B 入瞳光束/聚焦相位 -> C 真空强度 -> D 信标反向 -> E 跟踪
    -> F Zernike-78 校正 -> G 各分支 FOM -> H 多平面成像。

    Parameters
    ----------
    seed : int
        Sample seed (sample seed = master_seed + sample_index).
        中文：样本种子（样本种子 = master_seed + sample_index）。
    cfg : dict
        Configuration dictionary.
        中文：配置字典（见 config.yaml）。
    correction_coeffs : np.ndarray, optional
        ``(78,)`` Zernike coefficients for the ML correction leg. When given,
        ``fom_ml`` is filled and the ``'ml'`` beam phase is added.
        中文：(78,) Zernike 系数，用于 ML 校正分支。给出时填充 fom_ml 并
        加入 'ml' 分支光束相位；None 时跳过 ML 分支。
    shared : SharedSim, optional
        Prebuilt shared state (avoids rebuild).
        中文：预构建的共享状态（避免重复构建）。

    Returns
    -------
    SimSample
        The simulated sample.
        中文：仿真样本（含图像、标签、各分支 FOM 与中间相位）。
    """
    shared = _resolve_shared(shared, cfg)
    zern = shared.zern

    # Step A: turbulence phase screens.
    # 中文：A. 湍流相位屏（n_screens 层，由 seed 决定）。
    screens = _make_screens(seed, cfg, shared)

    # Step B: aperture beam + focusing phase (precomputed in shared).
    # 中文：B. 入瞳光束 + 聚焦相位（已在 shared 预计算）。
    E0 = shared.E0
    phi_focus = shared.phi_focus

    # Step C: vacuum intensity (precomputed in shared).
    # 中文：C. 真空目标面强度（无湍流参考，已在 shared 预计算）。
    I_vac = shared.I_vac

    # Step D: beacon back-propagation -> phi_conj (+ beacon intensity).
    # 中文：D. 衍射极限信标反向传播 -> 共轭信标相位 phi_conj。
    phi_conj, _ = _beacon_phase_conj(seed, cfg, shared, screens)

    # Step E: tracking.
    # 中文：E. 倾斜跟踪（从 phi_conj 的加权梯度得斜率，构造线性斜坡）。
    phi_track, track_slopes = _tracking(shared, phi_conj)

    # Step F: corrections.
    #
    # Phi_Z78 = M_Z78 (M+_Z78 Phi_beacon): the 78-mode Zernike projection of
    # the beacon conjugate, expressed in the natural (pixel) basis. This
    # coefficient vector is the CNN training target (paper Algorithm 1).
    # 中文：F. 校正。phi_beacon = phi_conj - phi_track（去除倾斜后的残余）。
    # labels 是共轭信标的 78 阶 Zernike 投影（自然/像素基），即 CNN 训练目标
    # （论文算法 1）；phi_z78 是把这些系数重构回相位。
    phi_beacon = phi_conj - phi_track
    labels = zern.phase_to_zernike(phi_beacon)
    phi_z78 = zern.zernike_to_phase(labels)

    # Step G: FOM legs.
    # 中文：G. 各校正分支 FOM（无 AO / 仅跟踪 / 信标共轭 / 78 阶重构）。
    fom_noao = _fom_leg(shared, screens, phi_focus)
    fom_track = _fom_leg(shared, screens, phi_focus + phi_track)
    fom_beacon = _fom_leg(shared, screens, phi_focus + phi_track + phi_beacon)
    fom_z78 = _fom_leg(shared, screens, phi_focus + phi_track + phi_z78)

    # 各分支孔径处总光束相位（聚焦相位 + 各校正相位）
    beam_phases = {
        "noao": phi_focus,
        "track": phi_focus + phi_track,
        "beacon": phi_focus + phi_track + phi_beacon,
        "z78": phi_focus + phi_track + phi_z78,
    }
    # ML 分支（可选）：用预测的 Zernike 系数做校正，计算其 FOM
    fom_ml: Optional[float] = None
    if correction_coeffs is not None:
        phi_ml = phi_focus + phi_track + zern.zernike_to_phase(correction_coeffs)
        fom_ml = _fom_leg(shared, screens, phi_ml)
        beam_phases["ml"] = phi_ml

    # Step H: imaging (tracking-only condition).
    # 中文：H. 多平面成像（仅跟踪条件），生成 3 平面图像。
    images, I_obj_track = _imaging(seed, cfg, shared, screens, phi_track)

    return SimSample(
        seed=seed,
        images=images,
        labels=labels,
        fom_noao=fom_noao,
        fom_track=fom_track,
        fom_beacon=fom_beacon,
        fom_z78=fom_z78,
        fom_ml=fom_ml,
        I_vac=I_vac,
        I_obj_track=I_obj_track,
        track_slopes=track_slopes,
        phase_conj=phi_conj,
        phase_track=phi_track,
        phase_beacon=phi_beacon,
        phase_z78=phi_z78,
        beam_phases=beam_phases,
    )


def simulate_sample_fom(
    seed: int,
    cfg: dict,
    coeffs: np.ndarray,
    *,
    shared: Optional[SharedSim] = None,
) -> float:
    """Fast path: FOM of a beam propagated with ``phi_focus + phi_track + zernike_to_phase(coeffs)``.

    Rebuilds the screens from ``seed`` and recomputes the tracking phase, but
    skips the imaging/scatter step. Returns the float FOM.

    中文：快速路径 —— 计算用 ``phi_focus + phi_track + zernike_to_phase(coeffs)``
    传播的光束的 FOM。从 seed 重建屏幕并重算跟踪相位，但跳过成像/散射步骤
    （比 simulate_sample 快，训练循环中用于评估预测系数）。

    Parameters
    ----------
    seed : int
        Sample seed.
        中文：样本种子。
    cfg : dict
        Configuration dictionary.
        中文：配置字典。
    coeffs : np.ndarray
        ``(78,)`` Zernike coefficients.
        中文：(78,) Zernike 系数（CNN 预测或真实标签）。
    shared : SharedSim, optional
        Prebuilt shared state (avoids rebuild).
        中文：预构建的共享状态。

    Returns
    -------
    float
        The figure of merit.
        中文：像质因子 FOM（公式 8）。
    """
    shared = _resolve_shared(shared, cfg)
    screens = _make_screens(seed, cfg, shared)
    phi_conj, _ = _beacon_phase_conj(seed, cfg, shared, screens)
    phi_track, _ = _tracking(shared, phi_conj)
    # 总相位 = 聚焦 + 跟踪 + 预测系数重构相位
    phi_total = shared.phi_focus + phi_track + shared.zern.zernike_to_phase(coeffs)
    return _fom_leg(shared, screens, phi_total)


# --------------------------------------------------------------------------- #
# Dataset generation
# --------------------------------------------------------------------------- #
def _quantize(images_raw: np.ndarray, scale_p: np.ndarray) -> np.ndarray:
    """Per-image normalize (paper Fig. 2), then quantize to 12-bit uint16.

    Each image is scaled to its own max, so the focal plane (which
    concentrates energy) reaches full 12-bit depth instead of being left dim
    by a dataset-wide per-plane max. The dataset-wide ``scale_p`` is retained
    in the HDF5 for schema compatibility but is no longer applied here.

    中文：逐图像归一化（论文 Fig. 2），再量化为 12-bit uint16。
    每张图按自身最大值缩放，使焦平面（能量集中）达到 12-bit 满深度，
    而不会被数据集级逐平面最大值压暗。scale_p（数据集级逐平面最大值）仅
    保留在 HDF5 中用于 schema 兼容，本函数不再对其做量化。

    Parameters
    ----------
    images_raw : np.ndarray
        ``(3, N, N)`` float32 raw intensities.
        中文：(3, N, N) float32 原始强度（未量化）。
    scale_p : np.ndarray
        ``(3,)`` per-plane raw max (stored for schema compatibility).
        中文：(3,) 逐平面原始最大值（仅用于 schema 兼容，本函数不再使用）。

    Returns
    -------
    np.ndarray
        ``(3, N, N)`` uint16 quantized images (each image max = 2047).
        中文：(3, N, N) uint16 量化图像（每张图 max = 2047）。
    """
    # 每张图按自身最大值归一化（论文 Fig. 2 “图像分别归一化”），使焦平面
    # （能量集中）达到 12-bit 满量程，而不会被数据集级逐平面最大值压暗。
    img_max = images_raw.max(axis=(1, 2), keepdims=True)
    normalized = images_raw / np.maximum(img_max, 1e-12)          # 归一化到 [0, 1]
    scaled = np.clip(normalized * (2**11 - 1), 0, 2**11 - 1)      # 缩放到 12-bit (0-2047)
    return scaled.astype(np.uint16)


# Worker globals (set once per process via the Pool initializer).
# 中文：worker 进程全局变量（每个 worker 进程经 Pool initializer 设置一次）。
_WORKER_CFG: Optional[dict] = None
_WORKER_SHARED: Optional[SharedSim] = None


def _worker_init(cfg: dict) -> None:
    """Pool initializer: build the shared state once per worker process.

    中文：Pool 初始化器 —— 每个 worker 进程构建一次共享状态（避免每个
    样本重复构建传播器/Zernike 基底）。
    参数 cfg: 配置字典（传入 Pool initargs）。
    """
    global _WORKER_CFG, _WORKER_SHARED
    _WORKER_CFG = cfg
    _WORKER_SHARED = _get_shared(cfg)


def _worker_generate(batch: list[tuple[int, int]]) -> list[tuple]:
    """Process a batch of ``(sample_index, seed)`` pairs.

    Returns a list of compact tuples ``(idx, images_raw, labels, fom_noao,
    fom_track, fom_beacon, fom_z78)`` (the large phase arrays are not shipped
    back to the parent).

    中文：处理一批 (sample_index, seed) 对。返回紧凑元组列表
    (idx, images_raw, labels, fom_noao, fom_track, fom_beacon, fom_z78)
    —— 大相位数组不回传父进程（只回传图像/标签/FOM，省 IPC 开销）。
    参数 batch: [(sample_index, seed), ...] 列表。
    """
    out = []
    for idx, seed in batch:
        s = simulate_sample(seed, _WORKER_CFG, shared=_WORKER_SHARED)
        out.append(
            (
                idx,
                s.images,
                s.labels,
                s.fom_noao,
                s.fom_track,
                s.fom_beacon,
                s.fom_z78,
            )
        )
    return out


def _make_batches(
    indices: np.ndarray, master_seed: int, chunk: int
) -> list[list[tuple[int, int]]]:
    """Split sample indices into batches of ``(sample_index, seed)`` pairs.

    中文：把样本索引切分为 (sample_index, seed) 对的批次。
    参数 indices: 样本索引数组；master_seed: 主种子（样本种子 = master_seed + 索引）；
    chunk: 每批样本数。
    """
    seeds = master_seed + indices
    batches = []
    for i in range(0, len(indices), chunk):
        batches.append(
            [(int(idx), int(seed)) for idx, seed in zip(indices[i : i + chunk], seeds[i : i + chunk])]
        )
    return batches


def generate_dataset(cfg: dict) -> str:
    """Run the full two-pass dataset generation pipeline and write the HDF5 file.

    Pass 1 computes per-plane intensity maxima and per-mode label mean/std over
    the TRAIN split (Eqs 13-14). Pass 2 quantizes and streams all samples to the
    HDF5 file (chunked writes, no in-RAM accumulation of raw images).

    中文：运行完整两趟数据集生成流水线并写出 HDF5 文件。
    趟 1：在训练集上计算逐平面强度最大值与逐模式标签均值/标准差（公式 13-14）。
    趟 2：量化并流式写出全部样本到 HDF5（分块写，不在内存累积原始图像）。

    Parameters
    ----------
    cfg : dict
        Configuration dictionary.
        中文：配置字典（含 physical / data 节）。

    Returns
    -------
    str
        Path to the written HDF5 file.
        中文：写出的 HDF5 文件路径。
    """
    p = cfg["physical"]
    d = cfg["data"]
    N = int(p["N"])                # 网格分辨率
    n_train = int(d["n_train"])    # 训练集样本数
    n_test = int(d["n_test"])      # 测试集样本数
    n_eval = int(d["n_eval"])      # 评估集样本数
    master_seed = int(d["master_seed"])  # 主种子（样本种子 = master_seed + 索引）
    workers = int(d["workers"])    # 多进程 worker 数
    h5_path = d["h5_path"]         # HDF5 输出路径
    L = float(p["L"])              # 传播距离 [m]

    N_total = n_train + n_test + n_eval
    train_idx = np.arange(n_train, dtype=np.int64)
    test_idx = np.arange(n_train, n_train + n_test, dtype=np.int64)
    eval_idx = np.arange(n_train + n_test, N_total, dtype=np.int64)
    all_idx = np.arange(N_total, dtype=np.int64)

    # Build shared state in the parent (ZernikeBasis inherited COW by workers).
    # 中文：在父进程构建共享状态（ZernikeBasis 由 worker 进程以 COW 方式继承）。
    shared = _get_shared(cfg)

    os.makedirs(os.path.dirname(os.path.abspath(h5_path)), exist_ok=True)

    # fork 上下文：子进程继承父进程已构建的 shared（COW，零拷贝）
    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(workers, initializer=_worker_init, initargs=(cfg,)) as pool:
        # ---- Pass 1: train-only stats (Eqs 13-14) ----
        # 中文：趟 1 —— 仅训练集统计（公式 13-14）
        plane_max = np.zeros(3, dtype=np.float64)        # 逐平面 raw 强度最大值 (3,)
        label_sum = np.zeros(N_MODES, dtype=np.float64)  # 逐模式标签累加 (78,)
        label_sumsq = np.zeros(N_MODES, dtype=np.float64)  # 逐模式标签平方和 (78,)
        n_proc = 0
        train_batches = _make_batches(train_idx, master_seed, chunk=4)
        for batch_result in tqdm(
            pool.imap_unordered(_worker_generate, train_batches),
            total=len(train_batches),
            desc="pass1 (train stats)",
        ):
            for (_idx, images_raw, labels, *_rest) in batch_result:
                plane_max = np.maximum(plane_max, images_raw.max(axis=(1, 2)))
                label_sum += labels
                label_sumsq += labels**2
                n_proc += 1
        assert n_proc == n_train, f"pass1 processed {n_proc} != n_train {n_train}"

        # 公式 14：逐模式标签的均值 mu 与标准差 sigma（用于训练时 z 标准化）
        mu = label_sum / n_train
        sigma = np.sqrt(np.maximum(label_sumsq / n_train - mu**2, 0.0))
        scale_p = plane_max.astype(np.float32)  # 逐平面 raw 最大值（schema 兼容）

        # ---- Pass 2: quantize + stream all samples to HDF5 ----
        # 中文：趟 2 —— 量化 + 流式写出全部样本到 HDF5
        with h5py.File(h5_path, "w") as f:
            # HDF5 schema（分块写，不在内存累积 raw 图像）：
            #   images (N_total, 3, N, N) uint16 —— 逐样本 3 平面量化图像
            #   labels (N_total, 78) float32      —— 78 阶 Zernike 系数（训练目标）
            #   fom_*  (N_total,) float32         —— 各分支 FOM
            #   seeds / L                          —— 样本种子 / 传播距离
            #   train/test/eval_idx                —— 数据集划分索引
            #   mu / sigma (78,)                   —— 标签均值/标准差（公式 14）
            #   scale_p (3,)                       —— 逐平面 raw 最大值（schema 兼容）
            #   vacuum_intensity (N, N)            —— 真空目标面强度（无湍流参考）
            f.create_dataset(
                "images", (N_total, 3, N, N), dtype=np.uint16, chunks=(1, 3, N, N)
            )
            f.create_dataset("labels", (N_total, N_MODES), dtype=np.float32)
            f.create_dataset("fom_noao", (N_total,), dtype=np.float32)
            f.create_dataset("fom_track", (N_total,), dtype=np.float32)
            f.create_dataset("fom_beacon", (N_total,), dtype=np.float32)
            f.create_dataset("fom_z78", (N_total,), dtype=np.float32)
            f.create_dataset("seeds", (N_total,), dtype=np.int64)
            f.create_dataset("L", (N_total,), dtype=np.float32)
            f.create_dataset("train_idx", (n_train,), dtype=np.int64)
            f.create_dataset("test_idx", (n_test,), dtype=np.int64)
            f.create_dataset("eval_idx", (n_eval,), dtype=np.int64)
            f.create_dataset("mu", (N_MODES,), dtype=np.float32)
            f.create_dataset("sigma", (N_MODES,), dtype=np.float32)
            f.create_dataset("scale_p", (3,), dtype=np.float32)
            f.create_dataset("vacuum_intensity", (N, N), dtype=np.float32)
            # 把完整配置序列化为 attr，便于离线复现
            f.attrs["config_json"] = json.dumps(cfg)

            # 流式写出全部样本（按样本索引随机顺序，chunk=4 减小 IPC 等待）
            all_batches = _make_batches(all_idx, master_seed, chunk=4)
            for batch_result in tqdm(
                pool.imap_unordered(_worker_generate, all_batches),
                total=len(all_batches),
                desc="pass2 (write)",
            ):
                for (
                    idx,
                    images_raw,
                    labels,
                    fom_noao,
                    fom_track,
                    fom_beacon,
                    fom_z78,
                ) in batch_result:
                    f["images"][idx] = _quantize(images_raw, scale_p)
                    f["labels"][idx] = labels.astype(np.float32)
                    f["fom_noao"][idx] = fom_noao
                    f["fom_track"][idx] = fom_track
                    f["fom_beacon"][idx] = fom_beacon
                    f["fom_z78"][idx] = fom_z78

            # Metadata（一次性写满的标量/向量元数据）。
            f["seeds"][:] = master_seed + all_idx
            f["L"][:] = L
            f["train_idx"][:] = train_idx
            f["test_idx"][:] = test_idx
            f["eval_idx"][:] = eval_idx
            f["mu"][:] = mu.astype(np.float32)
            f["sigma"][:] = sigma.astype(np.float32)
            f["scale_p"][:] = scale_p
            f["vacuum_intensity"][:] = shared.I_vac

    return h5_path
