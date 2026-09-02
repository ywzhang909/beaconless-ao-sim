# 基于 OOPAO 的 Beaconless ML-AO 训练数据仿真生成优化指南

> 本文档在原始论文 Algorithm 1 基础上进行工程优化，并设计**统一物理抽象层**，使底层仿真引擎可在 **纯 NumPy/Soapy** 与 **OOPAO** 之间无缝切换。

---

## 1. 设计哲学：统一物理抽象层

原始论文的仿真流程（Algorithm 1）与物理对象（Telescope、Atmosphere、Source、DM、Detector）天然契合 OOPAO 的面向对象设计。我们定义一个 `PhysicsEngine` 抽象基类，所有上层数据生成逻辑只依赖此接口，不关心底层实现。

```
                    +------------------+
                    |  DataGenerator   |
                    |  (Algorithm 1)   |
                    +--------+---------+
                             |
              +--------------+--------------+
              |                             |
       +------v------+              +-------v-------+
       | NumPyEngine |              |  OOPAOEngine  |
       | (Soapy+FFT) |              |  (OOPAO lib)  |
       +-------------+              +---------------+
```

---

## 2. 环境准备

### 2.1 安装（按需选择）

```bash
# 方案 A：OOPAO 路径（推荐用于快速原型和闭环保真度验证）
git clone https://github.com/cheritier/OOPAO.git
pip install -e OOPAO
pip install torch h5py tqdm matplotlib

# 方案 B：纯 NumPy/Soapy 路径（推荐用于大规模并行数据生成）
pip install soapy aotools pyfftw torch h5py tqdm
```

---

## 3. 统一物理抽象层定义

**`physics/engine_base.py`**

```python
from abc import ABC, abstractmethod
import numpy as np

class PhysicsEngine(ABC):
    """
    物理仿真引擎抽象基类。
    所有上层数据生成逻辑只调用此接口，与底层库解耦。
    """

    @abstractmethod
    def init_geometry(self, N, D_scope, Lx, lam, L, Cn2, L0, l0):
        """初始化望远镜几何与大气参数。"""
        pass

    @abstractmethod
    def generate_atmosphere(self, seed=None):
        """生成/重置一组湍流相位屏，返回屏列表 [rad]。"""
        pass

    @abstractmethod
    def get_pupil_mask(self):
        """返回圆 pupil 布尔掩膜 (N, N)。"""
        pass

    @abstractmethod
    def prepare_beam(self, r_spot, phi_extra=0.0):
        """
        在 pupil 平面准备高斯光束 + 聚焦相位 + 可选附加相位。
        返回复振幅 E_pupil (N, N)。
        """
        pass

    @abstractmethod
    def propagate_to_object(self, E_pupil, screens, dz):
        """正向 Split-Step 传播至目标面。"""
        pass

    @abstractmethod
    def propagate_to_pupil(self, E_obj, screens, dz):
        """反向 Split-Step 传播回 pupil。"""
        pass

    @abstractmethod
    def fresnel_image(self, E_pupil, z):
        """角谱法 Fresnel 传播到距离 z，返回强度图。"""
        pass

    @abstractmethod
    def get_opd(self):
        """获取当前 pupil 平面总光程差 [m]（用于提取相位）。"""
        pass

    @abstractmethod
    def apply_dm(self, coeffs, M2C):
        """将 Zernike/DM 系数施加到望远镜光路。"""
        pass

    @abstractmethod
    def reset_dm(self):
        """重置 DM 为平坦状态。"""
        pass
```

---

## 4. OOPAO 引擎实现

**`physics/engine_oopao.py`**

```python
import numpy as np
from .engine_base import PhysicsEngine

# OOPAO 核心对象
from OOPAO.Telescope import Telescope
from OOPAO.Atmosphere import Atmosphere
from OOPAO.Source import Source
from OOPAO.Zernike import Zernike
from OOPAO.DeformableMirror import DeformableMirror

class OOPAOEngine(PhysicsEngine):
    """
    基于 OOPAO 的物理仿真引擎实现。
    利用 OOPAO 的 * / ** 光传播链式操作和内置 FFT 加速。
    """

    def __init__(self):
        self.tel = None
        self.atm = None
        self.src = None
        self.dm = None
        self.zern = None
        self._N = None
        self._lam = None
        self._L = None
        self._dz = None
        self._screens = []

    def init_geometry(self, N, D_scope, Lx, lam, L, Cn2, L0, l0):
        self._N = N
        self._lam = lam
        self._L = L
        self._dz = Lx / N  # 像素尺度

        # 1. 望远镜（OOPAO 核心，定义 pupil 和像素尺度）
        # OOPAO 的 resolution = N，diameter = D_scope [m]
        self.tel = Telescope(
            resolution=N,
            diameter=D_scope,
            samplingTime=1e-3  # 时间步，数据生成时不用
        )

        # 2. 光源（定义波长和通量）
        # OOPAO 的 Source 用波段字符串，这里直接用波长数值
        # 注意：OOPAO 内部参数通常归一化到 500nm，需确认缩放
        self.src = Source(
            optBand='K',  # 或自定义，OOPAO 支持自定义波长
            magnitude=0   # 亮度不影响相位仿真
        )
        # 手动覆盖波长（OOPAO 内部可能用 band 查表，需测试）
        self.src.wavelength = lam

        # 3. 大气（Von Karman 多层）
        # OOPAO 的 Atmosphere 用 r0 [m] @ 500nm，需转换
        k0 = 2 * np.pi / lam
        r0 = (0.423 * k0**2 * Cn2 * L)**(-3/5)

        # 论文用均匀湍流 -> 单层等效
        self.atm = Atmosphere(
            telescope=self.tel,
            r0=r0,
            L0=L0,
            windSpeed=[0],       # 数据生成用 frozen flow，风速设 0
            fractionalR0=[1.0],  # 全部湍流强度集中在该层
            windDirection=[0],
            altitude=[0]         # 地面层等效
        )
        self.atm.initializeAtmosphere(telescope=self.tel)

        # 4. Zernike 基（用于标签生成和 tip-tilt 提取）
        self.zern = Zernike(self.tel, nModes=78)
        self.zern.computeZernike(self.tel)

        # 5. DM（用于施加相位校正）
        # 使用 Zernike 模式作为 DM influence functions（模态 DM）
        Z_2D = self.zern.modesFullRes.reshape((N**2, 78))
        self.dm = DeformableMirror(
            telescope=self.tel,
            nSubap=1,  # 占位，实际用 modes 参数覆盖
            modes=Z_2D
        )

    def generate_atmosphere(self, seed=None):
        """生成新的随机湍流屏。"""
        if seed is not None:
            self.atm.generateNewPhaseScreen(seed=seed)
        else:
            self.atm.update()
        # OOPAO 的屏存储在 atm.OPD [m] 或各 layer 中
        # 对于单层，直接取 atm.OPD 并转为相位 [rad]
        self._screens = [self.atm.OPD * (2 * np.pi / self._lam)]
        return self._screens

    def get_pupil_mask(self):
        return self.tel.pupil.astype(bool)

    def prepare_beam(self, r_spot, phi_extra=0.0):
        """
        在 pupil 平面准备光束。
        OOPAO 中通过设置 src.phase 和 tel.src 实现。
        """
        N = self._N
        dx = self.tel.D / N  # OOPAO 的像素尺度
        x = np.linspace(-N/2, N/2-1, N) * dx
        X, Y = np.meshgrid(x, x)
        r = np.sqrt(X**2 + Y**2)

        # 高斯振幅
        amp = np.exp(-(r / r_spot)**2) * self.tel.pupil
        # 聚焦相位 + 附加相位
        phi_focus = - (2 * np.pi / self._lam) * (X**2 + Y**2) / (2 * self._L)
        phi_total = phi_focus + phi_extra

        E = amp * np.exp(1j * phi_total)
        # 存入 OOPAO 的 src.phase（OOPAO 内部用 phase [rad]）
        self.src.phase = np.angle(E)
        self.src.amplitude = np.abs(E)
        return E

    def propagate_to_object(self, E_pupil, screens, dz):
        """
        OOPAO 的正向传播：src * tel * atm。
        但 OOPAO 的 atm 是叠加在 tel 上的，需组合后传播。
        """
        # 将相位屏转换为 OPD [m] 并设置到 atm
        if screens:
            self.atm.OPD = screens[0] * (self._lam / (2 * np.pi))

        # OOPAO 传播链：src ** atm * tel
        # ** 表示通过 atm 的物理传播（含闪烁），* 表示几何叠加
        self.src ** self.atm * self.tel

        # 获取目标面复振幅（OOPAO 的 tel.src 存储结果）
        # 注意：OOPAO 默认焦点在无穷远或特定位置，需确认
        # 这里简化：直接返回 tel.src 的复振幅表示
        E_obj = self.tel.src.amplitude * np.exp(1j * self.tel.src.phase)
        return E_obj

    def propagate_to_pupil(self, E_obj, screens, dz):
        """反向传播：OOPAO 支持反向传播通过 lineOfSight。"""
        # OOPAO 的反向传播需手动实现 Split-Step
        # 或利用 OOPAO 的 lineOfSight 模块
        # 这里回退到通用 FFT 实现（见下文 Hybrid 方案）
        from OOPAO.LineOfSight import LineOfSight
        # ... 反向传播实现
        pass

    def fresnel_image(self, E_pupil, z):
        """利用 OOPAO 的 Detector 或手动 FFT 实现。"""
        # OOPAO 的 Detector 可记录焦平面强度
        # 或直接用 FFT 传播到指定距离
        pass

    def get_opd(self):
        """返回当前总 OPD [m]。"""
        return self.tel.OPD

    def apply_dm(self, coeffs, M2C=None):
        """施加 DM 校正。"""
        if M2C is None:
            self.dm.coefs = coeffs
        else:
            self.dm.coefs = M2C @ coeffs
        # OOPAO 的 DM 通过 * 操作符叠加到 tel
        self.src * self.tel * self.dm

    def reset_dm(self):
        self.dm.coefs = np.zeros(78)
```

---

## 5. 关键优化点（相对原始 Algorithm 1）

### 优化 1：延迟初始化 + 对象池复用

原始 Algorithm 1 每样本都重新创建所有物理对象。OOPAO 的 `Telescope`、`Atmosphere`、`DM` 等对象初始化较耗时（尤其是 Zernike 基计算和 FFT 规划）。优化后：**一次初始化，多次重置**。

```python
# 优化前（原始论文伪代码）：每样本都 init
for seed in range(n_samples):
    screens = makePhaseScreens(...)  # 每次重新分配内存
    prop = Propagator(...)           # 每次重新规划 FFT
    ...

# 优化后（OOPAO 引擎）：对象池复用
engine = OOPAOEngine()
engine.init_geometry(...)  # 只执行一次

for seed in range(n_samples):
    engine.generate_atmosphere(seed=seed)  # 仅更新屏，不重建对象
    engine.reset_dm()
    ...
```

### 优化 2：批量粗糙度并行（joblib）

OOPAO 依赖 `joblib` 做并行。Lambertian 散射的 10 次粗糙度实现天然可并行：

```python
from joblib import Parallel, delayed

def _one_roughness(I_obj, seed_r, engine, screens_back):
    np.random.seed(seed_r)
    phi_rand = 2 * np.pi * np.random.rand(*I_obj.shape)
    E_scat = np.sqrt(I_obj) * np.exp(1j * phi_rand)
    return engine.propagate_to_pupil(E_scat, screens_back, dz)

# 10 次粗糙度并行反向传播
E_back_list = Parallel(n_jobs=4)(
    delayed(_one_roughness)(I_obj, seed*1000+r, engine, screens_back)
    for r in range(10)
)
E_avg = sum(E_back_list) / 10
```

### 优化 3：HDF5 增量写入 + 内存映射

原始流程生成 81,000 组 512x512x3 图像，峰值内存约 238 GB。优化为**分块生成、增量写入 HDF5**：

```python
import h5py

with h5py.File('train.h5', 'w') as f:
    # 预创建数据集（不占用内存）
    ds_img = f.create_dataset('images', shape=(81000, 3, 512, 512),
                              dtype='uint16', chunks=(1, 3, 512, 512),
                              compression='gzip')
    ds_coeff = f.create_dataset('coeffs', shape=(81000, 78), dtype='float32')

    chunk_size = 100
    for chunk_start in range(0, 81000, chunk_size):
        chunk_end = min(chunk_start + chunk_size, 81000)
        # 生成 chunk_size 个样本
        for i in range(chunk_start, chunk_end):
            ... # 单样本生成逻辑
            ds_img[i] = I_planes
            ds_coeff[i] = coeffs_dm
```

### 优化 4：三平面成像的 FFT 复用

三平面（Plane 0/1/2）使用同一输入光场，仅传播距离不同。可一次性计算 FFT，分别乘以三个传递函数：

```python
# 优化前：3 次独立 FFT
I0 = fresnel(E, z0)
I1 = fresnel(E, z1)
I2 = fresnel(E, z2)

# 优化后：1 次 FFT + 3 次点乘
E_ft = np.fft.fft2(E)
I0 = np.abs(np.fft.ifft2(E_ft * H(z0)))**2
I1 = np.abs(np.fft.ifft2(E_ft * H(z1)))**2
I2 = np.abs(np.fft.ifft2(E_ft * H(z2)))**2
```

### 优化 5：Zernike 投影的预计算矩阵

原始论文每次样本都重新做最小二乘拟合。预计算伪逆矩阵后，投影变为矩阵-向量乘法：

```python
# 预计算（一次）
M_zern = zbasis.modesFullRes[:, pupil_mask]  # (n_modes, n_pixels)
M_pinv = np.linalg.pinv(M_zern.T)            # (n_modes, n_pixels)

# 每样本（向量化）
coeffs = M_pinv @ phi_flat[pupil_mask]       # O(n_modes * n_pixels)
```

---

## 6. 完整数据生成流程（优化版 Algorithm 1）

**`data/generate_optimized.py`**

```python
import numpy as np
import h5py
from tqdm import tqdm
from joblib import Parallel, delayed

from physics.engine_oopao import OOPAOEngine  # 或 from physics.engine_numpy import NumPyEngine

def generate_optimized(engine, config, out_path, n_samples, seed_start=0, chunk_size=100):
    """
    优化后的训练数据生成流程。
    支持增量写入、并行粗糙度、FFT 复用。
    """
    N = config['N']
    n_rough = config['n_roughness']

    with h5py.File(out_path, 'w') as f:
        ds_img = f.create_dataset('images', shape=(n_samples, 3, N, N),
                                  dtype='uint16', chunks=(1, 3, N, N), compression='gzip')
        ds_len = f.create_dataset('lengths', shape=(n_samples,), dtype='float32')
        ds_coeff = f.create_dataset('coeffs', shape=(n_samples, 78), dtype='float32')

        # 预计算 Zernike 伪逆
        zbasis = engine.zern  # OOPAO Zernike 对象
        pupil = engine.get_pupil_mask()
        M_zern = zbasis.modesFullRes.reshape((78, N**2))[:, pupil.flatten()]
        M_pinv = np.linalg.pinv(M_zern.T)

        # 预计算 Fresnel 传递函数（三平面）
        zR = compute_zR(config)  # 论文 Eq.9-10
        f_obj = 2 * zR
        z_planes = [f_obj - zR, f_obj, f_obj + zR]
        H_planes = [compute_transfer(N, config['dx'], config['lam'], z) for z in z_planes]

        for chunk_start in range(0, n_samples, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_samples)

            for i in tqdm(range(chunk_start, chunk_end), desc=f"Chunk {chunk_start}"):
                seed = seed_start + i

                # === Step 1: 生成湍流（OOPAO 对象池复用）===
                screens = engine.generate_atmosphere(seed=seed)

                # === Step 2: 真空基准（可选，缓存）===
                if i == 0:
                    E_vac = engine.prepare_beam(config['r_spot'])
                    I_vac = engine.propagate_to_object(E_vac, [np.zeros_like(s) for s in screens], config['dz'])

                # === Step 3: 理想信标反向（生成标签）===
                # 利用 OOPAO 的反向传播或手动 Split-Step
                E_beacon = engine.prepare_beam(config['r_spot'], phi_extra=0)
                # 反向传播获取共轭相位
                phi_conj = compute_beacon_phase(engine, screens)

                # Tip-Tilt 提取（前 3 个 Zernike 模式）
                coeffs_conj = M_pinv @ phi_conj[pupil]
                phi_track = (M_zern.T @ coeffs_conj[:3]).reshape(N, N)
                phi_dm = phi_conj - phi_track
                coeffs_dm = M_pinv @ phi_dm[pupil]

                # === Step 4: 正向跟踪光束 ===
                engine.reset_dm()
                E_track = engine.prepare_beam(config['r_spot'], phi_extra=phi_track)
                I_obj = np.abs(engine.propagate_to_object(E_track, screens, config['dz']))**2

                # === Step 5: Lambertian 散射 + 并行反向 ===
                def _backprop_one(r):
                    phi_r = 2 * np.pi * np.random.rand(N, N)
                    E_scat = np.sqrt(I_obj) * np.exp(1j * phi_r)
                    return engine.propagate_to_pupil(E_scat, screens[::-1], config['dz'])

                E_back_list = Parallel(n_jobs=4, backend='threading')(
                    delayed(_backprop_one)(r) for r in range(n_rough)
                )
                E_pupil_back = sum(E_back_list) / n_rough

                # === Step 6: 三平面成像（FFT 复用）===
                E_ft = np.fft.fft2(E_pupil_back)
                I_planes = []
                for H in H_planes:
                    I = np.abs(np.fft.ifft2(E_ft * H))**2
                    I_planes.append(I)

                # === Step 7: 12-bit 量化 + 写入 ===
                # 全局缩放在 chunk 结束后统一计算，或维护运行最大值
                for p, I in enumerate(I_planes):
                    ds_img[i, p] = np.clip(I * (2**11 / I.max()), 0, 2**11-1).astype(np.uint16)

                ds_len[i] = config['L']
                ds_coeff[i] = coeffs_dm.astype(np.float32)

        # 全局 Zernike 归一化（论文 Eq.10）
        mean = ds_coeff[:].mean(axis=0)
        std = ds_coeff[:].std(axis=0)
        ds_coeff[:] = (ds_coeff[:] - mean) / std
        f.attrs['zern_mean'] = mean
        f.attrs['zern_std'] = std
```

---

## 7. OOPAO 特有优势与注意事项

### 优势

| 特性                 | OOPAO 实现                                               | 说明                                         |
| -------------------- | -------------------------------------------------------- | -------------------------------------------- |
| **对象复用**   | `Telescope/Atmosphere/DM` 一次 init，多次 `update()` | 避免每样本重建对象的开销                     |
| **相位屏统计** | `Atmosphere` 内部使用协方差矩阵法生成非定常屏          | 比纯 FFT 更符合理论 Von Karman 结构函数      |
| **光传播链**   | `src ** atm * tel * dm` 链式语法                       | 物理意义清晰，不易出错                       |
| **并行加速**   | 原生依赖`joblib`                                       | Lambertian 多实现并行天然支持                |
| **Zernike 基** | `Zernike(tel, nModes=78)` 内置                         | 自动处理 Noll 索引和 pupil 掩膜              |
| **DM 模态**    | `DeformableMirror(modes=Z_2D)`                         | 可直接用 Zernike 模式作为 influence function |

### 注意事项

1. **波长单位**：OOPAO 内部大量参数默认归一化到 **500 nm**。使用非 500nm 激光时，需显式设置 `src.wavelength` 并检查 `atm.r0` 的波长依赖性。
2. **反向传播**：OOPAO 的 `LineOfSight` 支持 `propagationDirection='down'`，但主要用于 WFS 光路。论文所需的**激光正向传播 + 散射光反向传播**需要手动实现 Split-Step 或扩展 `LineOfSight`。
3. **焦平面位置**：OOPAO 的 `Telescope` 默认焦点在无穷远（天文配置）。论文的成像系统需要物镜焦平面在有限距离，需手动配置 `f_obj` 和 Fresnel 传播。
4. **闪烁模拟**：OOPAO 的 `Atmosphere` 支持 `**` 操作符进行物理传播（含闪烁），但数据生成中若使用几何近似（`*`），则忽略闪烁。论文的 Split-Step 已包含闪烁，需确保 OOPAO 也启用物理传播模式。
5. **内存管理**：OOPAO 的 `Atmosphere` 对象持有大数组。在多进程并行生成数据时，建议使用 `spawn` 而非 `fork`，或每个进程独立创建 `engine` 实例。

---

## 8. 给 Codex 的完整 Prompt（OOPAO 版）

```
请基于 OOPAO (https://github.com/cheritier/OOPAO) 实现一个无信标 ML-AO 训练数据生成系统：

核心要求：
1. 设计 PhysicsEngine 抽象基类，并提供 OOPAOEngine 实现。
2. 使用 OOPAO 的 Telescope、Atmosphere、Source、DeformableMirror、Zernike 对象。
3. 实现 Algorithm 1 的优化版：
   - 对象池复用（一次 init，多次 update）
   - 10 次 Lambertian 粗糙度并行（joblib）
   - 三平面 Fresnel 成像的 FFT 复用
   - HDF5 增量分块写入
4. 大气：单层均匀 Von Karman，r0 由 Cn2/L/lam 计算，L0=100m，l0=0。
5. 光源：800nm，高斯光束，望远镜口径 30mm，网格 512x512。
6. 散射面：Lambertian，保留强度，赋予 [0,2pi) 随机相位，10 次平均。
7. 标签：理想信标反向传播 -> 共轭相位 -> 减去 tip-tilt -> 投影到 78 阶 Zernike。
8. 输出：HDF5 文件，含 images(3,512,512 uint16)、lengths(float32)、coeffs(78 float32)。

OOPAO 特定约束：
- 注意 OOPAO 内部 500nm 归一化，正确设置 src.wavelength。
- Atmosphere 用物理传播模式（** 操作符）以保留闪烁。
- 焦平面成像需手动配置 f_obj = 2*zR_APWS。
- 提供纯 NumPy 回退引擎（NumPyEngine）用于对比验证。
```
