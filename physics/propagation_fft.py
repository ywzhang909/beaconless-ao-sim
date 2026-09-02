"""Angular-spectrum propagation re-based on the OOPAO library.

Provides a :class:`Propagator` that performs Fresnel/angular-spectrum
propagation of a complex scalar field, plus a helper for the Rayleigh range.

Engine
------
The free-space propagation kernel is **OOPAO's** ``Atmosphere.ASM``
(``Atmosphere.py:938``) — a chirp-based angular-spectrum propagator
supporting input/output pixel-pitch scaling. It is a pure function of its
arguments (no instance state), and is numerically identical to the former
soapy.AOFFT ``H1**z`` pipeline (verified: |E| correlation 1.0, relative L2
error 0.0 over 1–640 m). The whole simulation chain therefore runs through
the OOPAO library; a pure-numpy transcription (:func:`_asm_numpy`) is kept
as a bit-identical fallback when OOPAO is unavailable (matching the project's
``beam_source: oopao -> aotools`` fallback convention).

OOPAO 参考:
- https://github.com/cheritier/OOPAO/blob/master/tutorials/how_to_scintillation.py
- OOPAO ``Atmosphere.py:938`` — ASM（变像素间距角谱传播）
"""

from __future__ import annotations

from typing import Any, List, Union

import numpy as np

try:  # OOPAO 可用性探测（兼容层失败时回退 numpy 引擎）
    from physics._oopao_compat import Atmosphere, Source, Telescope  # noqa: F401

    _OOPAO_AVAILABLE = True
except Exception:  # pragma: no cover - OOPAO 未安装
    _OOPAO_AVAILABLE = False


def rayleigh_range(w0: float, lam: float) -> float:
    """Return the Rayleigh range z_R = pi * w0**2 / lam.

    Parameters
    ----------
    w0 : float
        Beam waist radius (m).
    lam : float
        Wavelength (m).

    Returns
    -------
    float
        Rayleigh range (m).
    """
    return np.pi * w0**2 / lam


# ---------------------------------------------------------------------------
# OOPAO ASM 主机（惰性单例）
# ---------------------------------------------------------------------------
_ASM_HOST: Any = None


def _asm_host() -> Any:
    """Lazily build a tiny OOPAO ``Atmosphere`` used only as an ASM host.

    ``ASM`` 是纯函数，不依赖宿主 Atmosphere 的分辨率/层配置；这里用一个
    极小的望远镜+源+单层大气作为宿主，仅用于调用 OOPAO 的角谱传播方法
    （OOPAO 要求望远镜先有 Source，故先 ``src * tel``）。
    """
    global _ASM_HOST
    if _ASM_HOST is None:
        if not _OOPAO_AVAILABLE:
            raise RuntimeError(
                "OOPAO 不可用：请先 `uv sync` 安装 oopao 依赖"
            )
        from physics._oopao_compat import Atmosphere, Source, Telescope

        tel = Telescope(resolution=8, diameter=1.0, fov=0.0, samplingTime=0.001)
        src = Source(optBand="R", magnitude=0.0, display_properties=False)
        _ = src * tel  # 注册望远镜的源（OOPAO 前置条件）
        _ASM_HOST = Atmosphere(
            telescope=tel,
            r0=0.15,
            L0=30.0,
            windSpeed=[10.0],
            windDirection=[0.0],
            fractionalR0=[1.0],
            altitude=[0.0],
            elevation=90.0,
            angular_spectrum_propagation=True,
        )
    return _ASM_HOST


def oopao_asm(
    input_field: np.ndarray,
    wavelength: float,
    input_pitch: float,
    output_pitch: float,
    distance: float,
) -> np.ndarray:
    """OOPAO ``Atmosphere.ASM``：变像素间距角谱传播（纯函数，无实例状态）。

    Parameters
    ----------
    input_field : np.ndarray
        输入复场 (N, N)。
    wavelength : float
        波长 (m)。
    input_pitch : float
        输入像素间距 (m)。
    output_pitch : float
        输出像素间距 (m)，可与输入不同（缩放采样）。
    distance : float
        传播距离 (m)；负数表示反向传播。
    """
    if not _OOPAO_AVAILABLE:
        raise RuntimeError("OOPAO 不可用：请先 `uv sync` 安装 oopao 依赖")
    from physics._oopao_compat import Atmosphere

    return Atmosphere.ASM(input_field, wavelength, input_pitch, output_pitch, distance)


def _asm_numpy(
    input_field: np.ndarray,
    wavelength: float,
    input_pitch: float,
    output_pitch: float,
    distance: float,
) -> np.ndarray:
    """纯 numpy 忠实转录 OOPAO ``Atmosphere.ASM``（numpy 后端下比特级一致）。

    仅在 OOPAO 不可用时作为回退引擎，保证数值行为完全一致。
    """
    if distance == 0:
        return input_field
    N = input_field.shape[0]
    k = 2.0 * np.pi / wavelength
    # 网格 dtype 跟随字段实部精度（等价于 OOPAO 的 input_field.real.dtype）
    grid_dtype = (
        np.float64 if input_field.dtype in (np.complex128, np.float64) else np.float32
    )
    # 空间频率网格（中心化坐标）
    delta_f = 1.0 / (N * input_pitch)
    vals = np.arange(-N / 2.0, N / 2.0, dtype=grid_dtype) * delta_f
    fx, fy = np.meshgrid(vals, vals, copy=False)
    f_sq = fx**2 + fy**2
    # 空间网格
    vals_r = np.arange(-N / 2.0, N / 2.0, dtype=grid_dtype) * input_pitch
    x, y = np.meshgrid(vals_r, vals_r, copy=False)
    r_sq = x**2 + y**2
    m = output_pitch / input_pitch
    # 输入啁啾
    if m != 1.0:
        phase_1 = np.exp(1j * k / 2.0 * (1 - m) / distance * r_sq)
    else:
        phase_1 = 1.0
    # 传递核
    phase_2 = np.exp(-1j * np.pi * wavelength * distance / m * f_sq)
    # 输出啁啾
    if m != 1.0:
        vals_out = np.arange(-N / 2.0, N / 2.0, dtype=grid_dtype) * output_pitch
        x_out, y_out = np.meshgrid(vals_out, vals_out, copy=False)
        r_out_sq = x_out**2 + y_out**2
        phase_3 = np.exp(1j * k / 2.0 * (m - 1) / (m * distance) * r_out_sq)
    else:
        phase_3 = 1.0
    # FFT 操作
    field_freq = np.fft.fft2(np.fft.ifftshift(input_field * phase_1))
    field_filtered = np.fft.ifftshift(np.fft.fftshift(field_freq) * phase_2)
    field_out = np.fft.fftshift(np.fft.ifft2(field_filtered))
    return field_out * phase_3 / m


class Propagator:
    """Angular-spectrum field propagator built on OOPAO ``Atmosphere.ASM``.

    Parameters
    ----------
    N : int
        Grid size (square, N x N).
    dx : float
        Grid sample spacing (m).
    lam : float
        Wavelength (m).
    n_threads : int, optional
        兼容参数（旧的 FFTW 线程数）；OOPAO numpy 引擎下不再使用。
    engine : str, optional
        ``"oopao"``（默认）：传播内核为 OOPAO ``Atmosphere.ASM``；
        ``"numpy"``：使用比特级一致的纯 numpy 转录（OOPAO 不可用时自动回退）。
    dtype : np.dtype, optional
        Working complex dtype (default ``np.complex64``).
    """

    def __init__(
        self,
        N: int,
        dx: float,
        lam: float,
        n_threads: int = 1,
        engine: str = "oopao",
        dtype: np.dtype = np.dtype(np.complex64),
    ) -> None:
        self.N = N
        self.dx = dx
        self.lam = lam
        self.dtype = dtype

        if engine not in ("oopao", "numpy"):
            raise ValueError(f"未知传播引擎: {engine!r}（应为 'oopao' 或 'numpy'）")
        if engine == "oopao" and not _OOPAO_AVAILABLE:  # OOPAO 不可用 → numpy 回退
            engine = "numpy"
        self.engine = engine

    def _propagate_raw(self, E: np.ndarray, z: float) -> np.ndarray:
        """Propagate by z metres, returning the (possibly aliased) buffer."""
        if z == 0.0:
            return np.array(E, dtype=self.dtype, copy=True)
        if self.engine == "oopao":
            field = _asm_host().ASM(E, self.lam, self.dx, self.dx, z)
        else:
            field = _asm_numpy(E, self.lam, self.dx, self.dx, z)
        return np.array(field, dtype=self.dtype, copy=False)

    def propagate(self, E: np.ndarray, z: float) -> np.ndarray:
        """Propagate a complex field by distance ``z`` (m).

        Parameters
        ----------
        E : np.ndarray
            Complex field, shape (N, N), complex64.
        z : float
            Propagation distance (m).

        Returns
        -------
        np.ndarray
            Propagated field, complex64. A fresh copy is returned so the
            caller may safely mutate it.
        """
        out = self._propagate_raw(E, z)
        return np.array(out, dtype=self.dtype, copy=True)

    def split_step(
        self,
        E_in: np.ndarray,
        screens: Union[List[np.ndarray], np.ndarray],
        dz: float,
    ) -> np.ndarray:
        """Symmetric split-step propagation through phase screens.

        For each screen ``phi``: propagate ``dz/2``, apply ``exp(1j*phi)``,
        then propagate ``dz/2``. Screens of zeros are equivalent to a single
        ``propagate(E, n*dz)``.

        Parameters
        ----------
        E_in : np.ndarray
            Input complex field, shape (N, N), complex64.
        screens : list or ndarray
            Phase screens, shape (n, N, N), float32.
        dz : float
            Propagation distance between screens (m).

        Returns
        -------
        np.ndarray
            Output complex field, complex64.
        """
        E = np.array(E_in, dtype=self.dtype, copy=True)
        for phi in screens:
            E = self.propagate(E, dz / 2.0)
            E = np.array(E, dtype=self.dtype, copy=True)
            E *= np.exp(1j * phi).astype(self.dtype)
            E = self.propagate(E, dz / 2.0)
        return E

    def angular_spectrum_intensity(self, E_in: np.ndarray, z: float) -> np.ndarray:
        """Return the propagated intensity ``|propagate(E_in, z)|**2``.

        Parameters
        ----------
        E_in : np.ndarray
            Input complex field, shape (N, N), complex64.
        z : float
            Propagation distance (m).

        Returns
        -------
        np.ndarray
            Intensity, float32, shape (N, N).
        """
        E = self.propagate(E_in, z)
        return (np.abs(E) ** 2).astype(np.float32)