"""Beam propagation demo — configurable shape and propagation method.

Usage::

    uv run python run.py --shape circle --method fresnel
    uv run python run.py --shape triangle --method asm --seed 123
    uv run python run.py --shape square --method split-step --no-plot
"""
from __future__ import annotations

import sys
from pathlib import Path

import click
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from physics.config import load_config
from physics.propagation_fft import Propagator


# ── beam shape builders ──────────────────────────────────────────────────────


def _make_beam_focus(
    N: int, dx: float, r2: np.ndarray, rspot: float, pupil: np.ndarray
) -> np.ndarray:
    """Gaussian beam (default paper shape)."""
    E = np.exp(-(r2 / rspot**2)).astype(np.complex64)
    E[~pupil] = 0.0
    return E


def _make_beam_circle(
    N: int, dx: float, r2: np.ndarray, rspot: float, pupil: np.ndarray
) -> np.ndarray:
    """Uniform circular aperture (flat-top, radius = rspot)."""
    E = np.zeros((N, N), dtype=np.complex64)
    E[r2 <= rspot**2] = 1.0
    E[~pupil] = 0.0
    return E


def _make_beam_square(
    N: int, dx: float, r2: np.ndarray, rspot: float, pupil: np.ndarray
) -> np.ndarray:
    """Uniform square aperture (side = 2 * rspot, centred)."""
    cx = cy = (N - 1) / 2.0
    x = (np.arange(N) - cx) * dx
    X, Y = np.meshgrid(x, x)
    E = np.zeros((N, N), dtype=np.complex64)
    E[(np.abs(X) <= rspot) & (np.abs(Y) <= rspot)] = 1.0
    E[~pupil] = 0.0
    return E


def _make_beam_triangle(
    N: int, dx: float, r2: np.ndarray, rspot: float, pupil: np.ndarray
) -> np.ndarray:
    """Equilateral triangle (circumradius = rspot, flat-top)."""
    cx = cy = (N - 1) / 2.0
    x = (np.arange(N) - cx) * dx
    X, Y = np.meshgrid(x, x)
    # Equilateral triangle: 3 half-planes defined by edges rotated by 120°.
    theta_verts = np.array([np.pi / 2, np.pi / 2 + 2 * np.pi / 3,
                            np.pi / 2 + 4 * np.pi / 3])
    inside = np.ones((N, N), dtype=bool)
    for th in theta_verts:
        # Edge normal direction
        nx, ny = np.cos(th), np.sin(th)
        # Vertex position
        vx, vy = rspot * nx, rspot * ny
        # Half-plane: dot((X-vx, Y-vy), (nx, ny)) <= 0
        inside &= (nx * (X - vx) + ny * (Y - vy)) <= 0.0
    E = np.zeros((N, N), dtype=np.complex64)
    E[inside] = 1.0
    E[~pupil] = 0.0
    return E


SHAPES = {
    "focus": _make_beam_focus,
    "circle": _make_beam_circle,
    "square": _make_beam_square,
    "triangle": _make_beam_triangle,
}

SHAPE_NAMES_CN = {
    "focus": "聚焦高斯",
    "circle": "均匀圆形",
    "square": "正方形",
    "triangle": "三角形",
}


# ── propagation dispatch ──────────────────────────────────────────────────────


def _propagate_intensity(
    prop: Propagator,
    E: np.ndarray,
    z: float,
    method: str,
    *,
    screens: np.ndarray | None = None,
    dz_screen: float = 0.0,
) -> np.ndarray:
    """Propagate *E* to distance *z* and return intensity (float32)."""
    if method == "asm":
        return prop.angular_spectrum_intensity(E, z)
    if method == "fresnel":
        return prop.fresnel_intensity(E, z)
    if method == "split-step":
        if screens is None:
            return prop.angular_spectrum_intensity(E, z)
        return (np.abs(prop.split_step(E, screens, dz_screen)) ** 2).astype(
            np.float32
        )
    raise ValueError(f"Unknown method: {method}")


# ── CLI ───────────────────────────────────────────────────────────────────────


@click.command()
@click.option(
    "-c", "--config",
    default="config.yaml",
    show_default=True,
    help="YAML 配置文件路径。",
)
@click.option(
    "-s", "--shape",
    type=click.Choice(list(SHAPES), case_sensitive=False),
    default="focus",
    show_default=True,
    help="光束整形：focus=聚焦高斯, circle=均匀圆形, square=正方形, triangle=三角形。",
)
@click.option(
    "-m", "--method",
    type=click.Choice(["asm", "fresnel", "split-step"], case_sensitive=False),
    default="fresnel",
    show_default=True,
    help="传播方法：asm=角谱法, fresnel=Fresnel scaled-FFT, split-step=分步传播。",
)
@click.option("--seed", default=42, show_default=True, help="湍流屏随机种子。")
@click.option("--no-plot", is_flag=True, help="不显示可视化图（仅打印数值）。")
@click.option("--save", default=None, help="将结果强度图保存到指定目录。")
def main(
    config: str,
    shape: str,
    method: str,
    seed: int,
    no_plot: bool,
    save: str | None,
) -> None:
    """光束整形 + 传播演示。"""
    cfg = load_config(_PROJECT_ROOT / config)
    p = cfg.physical
    img = cfg.imaging

    N = int(p.N)
    dx = float(p.box_size) / N
    lam = float(p.wavelength)
    rspot = float(p.rspot)
    focal = float(p.focal)
    L = float(p.L)
    Dscope = float(p.Dscope)
    k = 2.0 * np.pi / lam

    # ── 坐标网格与孔径 ──
    x = (np.arange(N) - (N - 1) / 2.0) * dx
    X, Y = np.meshgrid(x, x)
    r2 = X**2 + Y**2
    pupil = r2 <= (Dscope / 2.0) ** 2

    # ── 构建光束 ──
    E0 = SHAPES[shape](N, dx, r2, rspot, pupil)

    # ── 聚焦相位（所有形状共用）──
    phi_focus = (-k * r2 / (2.0 * focal)).astype(np.float64)

    # ── 传播器 ──
    prop = Propagator(N, dx, lam)

    # ── 湍流屏（split-step 或可视化需要时生成）──
    screens = None
    dz_screen = 0.0
    need_screens = method == "split-step"
    if need_screens:
        try:
            from physics.oopao_backend import OopaoScreenBackend

            oopao = OopaoScreenBackend(
                N, dx, Dscope, lam, float(p.cn2), L, float(p.L0), int(p.n_screens)
            )
            screens = oopao.make_screens(seed)
            dz_screen = L / screens.shape[0]
            print(f"[湍流] OOPAO 屏幕 {screens.shape[0]} 层, dz={dz_screen:.1f} m")
        except Exception as exc:
            try:
                from aotools.turbulence.phasescreen import ft_sh_phase_screen

                n_screens = int(p.n_screens)
                from physics.screens_soapy import compute_r0

                r0_path = compute_r0(lam, float(p.cn2), L)
                r0_slab = r0_path * n_screens ** (3.0 / 5.0)
                screens = np.stack(
                    [
                        ft_sh_phase_screen(r0_slab, N, dx, float(p.L0),
                                           float(p.l0_sim), seed=seed + i)
                        for i in range(n_screens)
                    ]
                ).astype(np.float32)
                dz_screen = L / n_screens
                print(f"[湍流] aotools 屏幕 {n_screens} 层, r0={r0_path*100:.1f}cm")
            except ImportError:
                print("[湍流] OOPAO/aotools 不可用，使用真空传播。")
                need_screens = False

    # ── 真空目标面强度（FOM 基准）──
    E_focus = (E0 * np.exp(1j * phi_focus)).astype(np.complex64)
    I_vac = _propagate_intensity(prop, E_focus, L, "asm")

    # ── 主传播 ──
    print(f"\n{'='*50}")
    print(f"  光束整形: {SHAPE_NAMES_CN[shape]}")
    print(f"  传播方法: {method}")
    print(f"  网格: {N}×{N}, dx={dx*1e3:.3f}mm, λ={lam*1e9:.0f}nm")
    print(f"  传播距离: L={L:.0f}m, 焦距 f={focal:.0f}m")
    if screens is not None:
        print(f"  湍流屏: {screens.shape[0]} 层")
    print(f"{'='*50}")

    I_out = _propagate_intensity(
        prop, E_focus, L, method,
        screens=screens, dz_screen=dz_screen,
    )

    # ── 统计 ──
    peak_vac = float(I_vac.max())
    peak_out = float(I_out.max())
    total_vac = float(I_vac.sum())
    total_out = float(I_out.sum())
    print(f"\n  真空峰值强度: {peak_vac:.6f}")
    print(f"  输出峰值强度: {peak_out:.6f}")
    print(f"  峰值比 (out/vac): {peak_out/peak_vac:.4f}")
    print(f"  总能量比 (out/vac): {total_out/total_vac:.4f}")

    if no_plot:
        return

    # ── 可视化 ──
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    I0 = (np.abs(E0) ** 2).astype(np.float32)
    im0 = axes[0].imshow(I0, cmap="inferno", origin="lower")
    axes[0].set_title(f"入瞳强度 |E0|²\n({SHAPE_NAMES_CN[shape]})", fontsize=12)
    fig.colorbar(im0, ax=axes[0], shrink=0.8)

    im1 = axes[1].imshow(I_vac, cmap="inferno", origin="lower")
    axes[1].set_title("真空目标面 |E|²", fontsize=12)
    fig.colorbar(im1, ax=axes[1], shrink=0.8)

    im2 = axes[2].imshow(I_out, cmap="inferno", origin="lower")
    axes[2].set_title(f"输出 |E|² ({method})", fontsize=12)
    fig.colorbar(im2, ax=axes[2], shrink=0.8)

    for a in axes:
        a.set_xlabel("pixel")
        a.set_ylabel("pixel")

    fig.suptitle(
        f"光束整形={SHAPE_NAMES_CN[shape]}, 方法={method}, L={L:.0f}m",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()

    if save:
        out_dir = Path(save)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"beam_{shape}_{method}_L{int(L)}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"\n  图像已保存: {path}")
    else:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    main()
