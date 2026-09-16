# beaconless-ao-sim

对 DiComo 等人 *"Beaconless adaptive optics for atmospheric laser propagation
with multi-plane convolutional neural network,"* Opt. Express 33(15):31010
(2025)，DOI 10.1364/OE.561077 的复现与扩展——物理仿真、多平面 CNN 训练
（CNN1/CNNL）及演示规模评估，湍流屏幕基底已重建为 **OOPAO** 库。

> **完整指标、训练曲线及输入预处理冒烟测试：** 见 [`REPORT.md`](REPORT.md)。

## 概述

本流水线实现了论文完整链路（算法 1 + 第 2.4–2.7 节）：

1. **物理仿真**（`data/simulate.py`，`physics/`）— 聚焦激光束通过 10 层
   Kolmogorov 相位屏幕（间隔 100 m，1 km 路径）的分步传播、衍射极限高斯信标
   （束腰 `λL/D`）、带**解析抛物面离焦移除**的信标反向传播至瞳孔、从共轭
   信标相位提取的倾斜/倾斜跟踪，以及 78 阶 Zernike 投影
   `Φ_Z78 = M_Z78(M⁺_Z78 Φ_beacon)` 作为 CNN 目标（公式 1–5，算法 B–C）。
湍流屏幕从 **OOPAO** 库（`physics/oopao_backend.py` 通过 `OOPAO.Atmosphere`
   /`OOPAO.Telescope`/`OOPAO.Source`）中抽取。物理正向模型封装为
   `PhysicsEngine` 抽象接口（`physics/engine.py`，见下文
   [引擎与测量源抽象](#引擎与测量源抽象)）。
2. **多平面成像** — 3 个测量平面（焦平面附近 ±z_R，公式 12）、粗糙表面散射、
   12-bit 量化（公式 13）、标签 z 标准化（公式 14）。成像环节封装为
   `MeasurementSource` 抽象接口，可替换为**相机实测帧**（硬件数据接入）。
3. **CNN 训练**（`train.py`，`models/cnn.py`）— 3 阶段 CNN + 4 层 MLP 头部
   （第 2.6 节），Adam lr 1e-4，缩放 Zernike 模式上的 MSE，周期性仿真 FOM
   评估。
4. **评估**（`evaluate.py`，`utils/metrics.py`）— nPIB（公式 6），SIB（公式 7），
   FOM（公式 8），增益（公式 15），η（公式 16），逐模式 Pearson（公式 17）。
5. **输入预处理冒烟测试**（`smoke_test.py`）— 在 7 种输入预处理方法
   （基线 `/2047`、原始 uint16、逐样本 / 全局 min-max、z-score、单平面、
   原始+单平面）上运行训练好的 CNN1，并报告每种方法的 R_j、FOM_ML 及
   特征图统计量。

## OOPAO 集成

湍流屏幕生成器已重建为 [OOPAO](https://github.com/cheritier/OOPAO)
（通过 `uv` 从 GitHub 安装；`physics/oopao_backend.py` 调用
`OOPAO.Atmosphere`/`Telescope`/`Source`）。`OopaoScreenBackend` 抽取每层 von-Karman 屏幕，
将其中心裁剪至 512×512 的瞳孔网格，并按比例缩放每层振幅至目标每 slab r0
（`r0_slab = r0_path · n^(3/5)`）。通过 `physical.beam_source` 配置开关选择
（`soapy | aotools | oopao`；默认 `oopao`）。自定义 FFT 分步传播器、算法 1
信标反向传播和多平面成像全部保留 — OOPAO 仅提供湍流 + 瞳孔 + Zernike 基底。
OOPAO 屏幕按种子确定性生成，每层幅度重标定到目标每 slab r0（`_rescale_for`
  完成 500 nm → `lam` 波长换算 + `r0^(-5/6)` 幅度缩放 + 生成器标定常数去除），
  使每 slab OPD 标准差匹配 Kolmogorov 理论（ratio ≈ 0.99，逐种子 ~10% 采样噪声）。

## 目录结构

```
data/simulate.py         算法 1 流水线（屏幕、信标、FOM 分支、数据集）
data/generate_h5.py      CLI：单趟 HDF5 数据集写入器
physics/engine.py        PhysicsEngine / MeasurementSource 抽象 + HardwareMeasurementSource
physics/oopao_backend.py OopaoScreenBackend（OOPAO 湍流、r0 缩放，通过 uv 从 GitHub 安装的 OOPAO 库）
docs/oopao/              OOPAO 中文文档 + 可视化 notebook
physics/                 zernike_aotools、screens_soapy、propagation_fft、scattering
models/cnn.py            CNN1 / CNNL 架构
train.py                 训练循环（支持 DDP、梯度累积）
evaluate.py              评估 / 评估重跑 CLI + WandB 图表
smoke_test.py            逐方法输入预处理冒烟测试 + WandB
utils/metrics.py         公式 6–8、15–17 指标
tests/                   100+ 单元测试（物理、指标、模型、数据模式）
config.yaml              论文表 1 逐字复现 + 演示规模数据/模型/训练参数
REPORT.md                全流程报告（数据 → 模型 → 训练 → 评估 → 冒烟测试）
```

## 引擎与测量源抽象

数据生成流水线（`generate_dataset` / `simulate_sample`）通过两个抽象接口解耦，
可在**保持既有 API 完全兼容**的前提下更换物理正向模型或接入硬件采集数据：

- **`PhysicsEngine`**（`physics/engine.py`）— 物理正向模型：湍流屏幕
  （`make_screens`）、信标反向传播（`beacon_phase_conj`）、倾斜跟踪
  （`track`）、FOM 分支（`forward_fom`）及 Zernike 投影/重建。默认实现
  `data.simulate.SimulatedPhysicsEngine` 即原有算法 1 物理链路的一个封装；
- **`MeasurementSource`**（`physics/engine.py`）— 成像测量源：
  `acquire(seed, sample_index, screens, phi_track) -> (images, I_obj_track)`。
  默认 `SimulatedMeasurementSource` 走原粗糙表面散射成像；`HardwareMeasurementSource`
  直接读取**预先采集的相机帧**（`(3,N,N)` 三平面或 `(N,N)` 单帧复用），
  中心裁剪/填充至仿真网格 `N×N`，并返回 `I_obj_track=None`（硬件端不提供
  共轭目标面强度这一物理量）。

接入硬件数据只需注入测量源（标签仍由 `SimulatedPhysicsEngine` 计算）：

```python
from physics.engine import HardwareMeasurementSource
from data.simulate import generate_dataset

frames = load_camera_frames()          # (3, N, N) 或 (N, N) float32
h5 = generate_dataset(cfg, measurement=HardwareMeasurementSource(frames, target_N=N))
```

硬件测量源强制 `workers=1`（单消费者物理设备/文件流），并会自动警告。
`I_obj_track` 仅在 `MeasurementSource.acquire` 返回非 `None` 时写入数据集。

数据集生成已改为**单趟**：量化 + 流式写出全部样本到 HDF5 的同时，仅对训练子集
增量累计逐平面最大值与逐模式标签均值/平方和（公式 13–14），循环结束后一次性
回填 `mu` / `sigma` / `scale_p`。结果与原两趟实现逐位一致。

## 快速开始

```bash
uv sync                      # 根据 pyproject.toml / uv.lock 创建 .venv
uv run python -m pytest tests/ -v

# 1. 生成数据集（算法 1，单趟，OOPAO 屏幕）：
uv run python -m data.generate_h5 --config config.yaml

# 2. 训练（论文批次 32，通过微批次 8×4 累积实现）：
CUDA_VISIBLE_DEVICES=1 uv run python train.py --config config.yaml

# 3. 评估 + WandB 图表：
CUDA_VISIBLE_DEVICES=1 uv run python evaluate.py --config config.yaml --ckpt checkpoints/best.pt

# 4. 逐方法输入预处理冒烟测试 + WandB：
CUDA_VISIBLE_DEVICES=1 uv run python smoke_test.py --config config.yaml --ckpt checkpoints/best.pt
```

> GPU 说明：演示主机在两个 GPU 上运行 VLLM 工作进程（每个约 20.6 GiB，
> GPU 1 上仅剩约 3.4 GiB）。使用 `CUDA_VISIBLE_DEVICES=1` 并减小批次
> （冒烟测试使用 `--batch-size 8`）以适应显存。

## 演示规模 vs 论文

仿真参数（表 1）逐字复现。数据/模型/训练规模已缩减以便快速演示
（`config.yaml` 中记录了每一处偏差）：

| 参数 | 论文 | 演示版 |
|------|------|--------|
| n_train | 81,000 | 2,000 |
| n_test | 9,000 | 400 |
| n_eval | 1,000 | 100 |
| 训练步数 | 11,500 | 3,000 |
| 批次 | 32 | 32（微批次 8） |

## 核心结果（OOPAO 数据集）

### FOM 含义、公式与意义

**FOM**（Figure of Merit，公式 8）是本仿真的核心性能指标，它将激光在光瓶内的集中程度衡量为一个标量：

```
FOM = sqrt(nPIB × SIB)
```

其中两个分量独立衡量不同维度的对焦质量：

- **nPIB**（normalized Power-In-Bucket，公式 6）：光瓶内功率占总功率的比值，
  相比真空对焦光斑归一化：

  ```
  nPIB = (Σ I[mask] / Σ I) / (Σ I_vac[mask] / Σ I_vac)
  ```

  `mask` 是光瓶区域（半径 ~ 1.2 λf/D）。nPIB ≈ 1 表示所有能量都落到光瓶中
  （完美对焦）；nPIB < 1 表示大量能量散出光瓶边缘（湍流失焦）。

- **SIB**（Strehl-like Intensity in Bucket，公式 7）：光瓶内峰值强度相比真空
  峰值的比值：

  ```
  SIB = max(I[mask]) / max(I_vac[mask])
  ```

  SIB ≈ 1 表示光瓶中心亮度接近真空 diffraction-limited 理论值（Strehl ≈ 1）；
  SIB < 1 表示信标中心被湍流斑点拉低。

FOM ∈ [0, 1]。**1 = 完美**（达到无湍流真空对焦水平），**0 = 全能量丢失光瓶**。

### 各分支含义

| 分支 | 说明 | 物理意义 |
|------|------|----------|
| **noao** | 仅聚焦相位 `phi_focus`，无任何 AO 校正 | 湍流失焦基线；修正后 fixedL 中位 ≈ 0.024 |
| **track** | 加上倾斜移除 `phi_track` | 比 noao 略好，但仍被高阶 aberration 限制；≈ 0.0245 |
| **beacon** | 信标共轭 `phi_beacon = phi_conj - phi_track` | 完整自适应光学校正的理论上界；修正后 fixedL ≈ 0.416 |
| **z78** | 78 阶 Zernike 重构 `phi_z78` | beacon 相位被 78 阶 Zernike 投影/截断后的近似；≈ 0.365 |
| **ML** | CNN 预测的 Zernike 系数重构 | 端到端 ML 替代信标 → 修正后 CNN1 FOM_ML ≈ 0.0469 |

> 上表为**物理修正后**（`e644af0` + `caec5c3`）`fixedL.h5` 数据集的**中位 FOM**
> （n_roughness=3、300 样本演示规模）。绝对 FOM 低于论文（beacon 0.93 / z78 0.88）
> 是演示规模 + 少样本所致；**方法间相对排序**（g/η，`REPORT.md` §4.6）才是结论依据。

### 增益与 CNN 有效率

- **增益** `g = FOM_ML / FOM_track`（公式 15）：ML 分支相比仅跟踪基线的 FOM 改进倍数。
  修正后 CNN1：g ≈ 1.91（ML 使光瓶性能近翻倍，能量增加 ~90%）。

- **η** `(FOM_ML - FOM_track) / (FOM_z78 - FOM_track)`（公式 16）：ML 分支相对
  可能实现范围所占的比例。修正后 CNN1：η ≈ 0.066（演示规模少样本，绝对值偏低；
  方法间相对排序见 `REPORT.md` §4.6）。

### 结果（修正后固定 L，n_eval=50，独立 `evaluate.py`）

| 变体 | 中位 FOM_ML | g (Eq 15) | η (Eq 16) |
|------|------------|-----------|-----------|
| **CNN1Freq**（频域分支） | **0.04880** | **1.99** | **0.071** |
| CNN1FreqInput（频域输入） | 0.04743 | 1.94 | 0.067 |
| CNN1（基线） | 0.04692 | 1.91 | 0.066 |
| CNN1Star（StarNet+SE） | 0.04346 | 1.77 | 0.056 |
| CNNL（per-L，length head） | 0.01643 | 1.04 | 0.058 |

逐方法「先冒烟、后完整」消融（smoke 50 步 → full 800 步 + 独立评估）详见
[`REPORT.md` §4.6](REPORT.md)（修正物理后的逐方法消融）：
**频域分支（CNN1Freq / CNN1FreqInput）是唯一一致优于基线 CNN1 的方法**，
g 与 η 均更高；StarNet 骨干在少数据下未超过基线（参数效率优势待更大数据验证）。

**预处理冒烟测试** — 网络对其训练时输入契约敏感。基线 `/2047`（3 平面）
给出 FOM_ML ≈ 0.57 且 R_j ≈ 0.195；原始 uint16 和 z-score 输入使 CNN 主干饱和
（特征最大值 ~10⁴–10⁵）并将 FOM_ML 崩溃至 ~0.001；单平面输入丢失了深度感知
所需的离焦视差（FOM_ML ≈ 0.31，R_j ≈ 0.02）。逐图像归一化后 min-max ≈ 基线
（FOM_ML ≈ 0.57）。完整表格见 [`REPORT.md` §5.1](REPORT.md)（方法与结果表）。

## PINN 光束整形（`pinn-shaper/` 子项目）

本仓库同时收纳了第二个独立的论文复现：**PINN 光束整形**
（R. de la Fuente Herrezuelo 等，*"Physics-Informed Neural Networks for Optimal
Beam Shaping in Flat Optics"*，[arXiv:2607.18012](https://arxiv.org/abs/2607.18012)）。
它以 `pinn-shaper/` 子目录形式并入，与本 AO 仿真的文件互不冲突，可独立运行。

- **任务**：用物理信息神经网络（PINN）求解相位分布 $\Phi(x,y)$，使高斯入射光传播
  到目标平面后形成指定光斑（六角星、DT 字母，近场 / 远场两种），并与经典
  Gerchberg–Saxton (GS) 算法对比。
- **完整复现报告（含全部指标与图）**：[`pinn-shaper/report.md`](pinn-shaper/report.md)。
  核心结论：PINN 远场 MSE $4.0\times10^6$（论文 $3.59\times10^6$）远优于
  GS $2.08\times10^8$（论文 $2.08\times10^8$，逐位一致），光能利用率
  PINN 99.96% vs GS 95.90% —— PINN 相对 GS 约 **52 倍** MSE 改进。
- **结构**：

  ```
  pinn-shaper/pinn_shaper/     PINN 求解核心（fcnn 网络 + 近/远场 shaper）
  pinn-shaper/*.ipynb          论文 4 个原始 notebook
  pinn-shaper/repro/           从零训练复现脚本（--seed / --no-train）
  pinn-shaper/reproduced_models/  4 个从零训练权重
  pinn-shaper/saved_models/    官方预训练权重（论文数值校准基线）
  pinn-shaper/results/         复现实验图形
  pinn-shaper/report.md        完整复现报告
  ```

- **运行**（用 GPU 剩余显存，以 DT 远场 + GS 为例）：
  `CUDA_VISIBLE_DEVICES=1 /home/ws/pinn-venv/bin/python pinn-shaper/repro/exp4_dt_farfield_gs.py --seed 0`
  依赖 `diffractsim`（CPU 后端，2048×2048）+ CUDA torch，venv 见 `pinn-shaper/README.md`。

## 说明

- 78 阶上界 `FOM_Z78` 在强湍流样本下接近跟踪基线：当 `D/r0≈7.4` 时，
  反向传播的信标具有强度零点，其相位支点点是任何 78 阶相位共轭器都无法指令的。
  这是真实的运行区域限制，而非仿真伪影；它限制了演示集上可实现的 CNN 增益
  （`η`，公式 16）。
- **Fresnel 传播相位约定**（`physics/propagation_fft.py`）：scaled-FFT 的
  输出/后乘二次相位按**输出网格** `dx2=λz/(N·dx)`（padded 版
  `λz/(N_pad·dx)`）求值，与文档规定的 Fresnel 冲激响应一致；强度 `|E|²`
  不受影响。多平面成像的功率校验须**面积加权**
  （`Σ|E|²·dA`，见 `REPORT.md` §1.6 问题 5/6）——输出/输入像素面积不同，
  直接 `Σ|E|²` 比值恒为 `(dx/dx2)²` 而非 1 是网格约定伪影，非物理缺陷。
  相关检测已固化为测试：`tests/test_fresnel_scaled_fft.py`（冲激响应 /
  面积加权能量守恒 / 高斯束 `w(z)` / 焦点 Airy r86）。

## WandB

- 项目：https://wandb.ai/ywzhang909/beaconless-ao-sim
- 训练：`curious-shape-18`（`krg3hrzn`）
- 评估：`zkm1bhiq`
- 预处理冒烟：`s9st1raa`
