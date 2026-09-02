# OOPAO 原生集成文档

本目录说明本项目如何以**原生安装**的方式使用 OOPAO（Object-Oriented Python
Adaptive Optics）库，替代早期直接把 OOPAO 的**部分模块拷贝进仓库**（vendored）
的方案。

---

## 1. 为什么从「内嵌拷贝」改为「原生安装」

早期版本把 OOPAO 的少数核心模块（`Atmosphere`、`Telescope`、`Source`、
`Zernike`、`phaseStats`、`tools`）直接拷贝到 `physics/oopao/` 下，作为仓库的
一部分长期维护。这样做有几个问题：

1. **只保留了子集**：拷贝的只是本项目用到的少数模块，无法使用 OOPAO 的完整
   功能（Pyramid WFS、Shack-Hartmann、DeformableMirror、Detector 等）。
2. **容易与上游脱节**：内嵌拷贝不会随上游 OOPAO 更新，修复与新特性无法同步。
3. **维护成本高**：`_oopao_shim.py` 之类的补丁需要手工维护。

因此改为遵循官方 README 的推荐方式，将 OOPAO 作为**一个真正的 pip 安装包**
（native install）引入项目，代码通过 `import OOPAO` 使用，并在
`physics/oopao_backend.py` 中迁移导入。

---

## 2. 官方参考

- 源码仓库: https://github.com/cheritier/OOPAO
- 官方文档: https://cheritier.github.io/OOPAO/index.html
- 引用: C. Héritier 等, OOPAO AO4ELT7, https://hal.science/AO4ELT7/hal-04402878v1

官方 README 推荐的安装方式为（在克隆的仓库目录下）：

```bash
git clone https://github.com/cheritier/OOPAO.git
python -m pip install -e OOPAO
```

其中 `OOPAO` 是仓库根目录名，安装后 `import OOPAO` 可用。

---

## 3. 本项目的原生安装

由于本环境无法通过 HTTPS 直接访问 GitHub（被防火墙拦截），且项目使用 `uv`
管理 Python 环境，我们采用如下步骤：

### 3.1 用 SSH 克隆官方 OOPAO

```bash
git clone git@github.com:cheritier/OOPAO.git /tmp/opencode/oopao-official
```

### 3.2 将官方包源码纳入本仓库，作为安装来源

把官方包（`OOPAO/` 目录）及配套打包文件拷贝到仓库的 `third_party/` 下：

```bash
mkdir -p third_party
cp -r <official>/OOPAO third_party/OOPAO
cp <official>/README.md <official>/LICENSE \
   <official>/setup.py <official>/setup.cfg <official>/pyproject.toml third_party/
```

这样仓库自包含（`uv sync` 后可直接重建环境），同时 OOPAO 以「真实安装包」的
形式被引入。

### 3.3 用 uv 安装进项目环境

```bash
uv pip install third_party/
```

> 等价于官方 README 的 `pip install -e OOPAO`，只是通过 `uv` 执行。
> 安装了 `oopao==0.0.0`，并自动补装其依赖（如 `numexpr`）；`setup.cfg` 中
> `install_requires` 只写 `numpy`（未锁版本），因此**不会**把项目 numpy 2.5.x
> 降级（官方 `requirements.txt` 锁定 numpy 1.21–1.23，但那是给单独安装用的，
> 与 `pip/uv install` 无关）。

验证：

```bash
uv run python -c "import OOPAO; print(OOPAO.__file__)"
```

### 3.4 我们对官方 OOPAO 所做的小修补

官方 OOPAO 在多个模块文件的构造逻辑中，通过扫描 `sys.path` 中名字包含
`"OOPAO"` 的路径来定位数据缓存目录（`precision_oopao.npy`）。当包被安装到
`site-packages/OOPAO`（或 `third_party/OOPAO`）这类 sys.path **条目名字不含**
`"OOPAO"` 的位置时，`np.argmin([])` 会抛 `ValueError`，导致 `import OOPAO`
直接崩溃。这是上游的一个真实 bug。

我们在仓库内的 `third_party/OOPAO/` 源码里做了两处**行为保持一致**的修补：

1. `__init__.py`：当没有匹配的 `sys.path` 条目时，跳过 `precision_oopao.npy`
   的保存（该文件只是记录模拟精度常量，去掉不影响功能）。
2. `Telescope.py`、`Source.py`、`Atmosphere.py`、`Pyramid.py`、
   `ShackHartmann.py`、`DeformableMirror.py`、`BioEdge.py`、`SpatialFilter.py`
   —— 将原先「扫描 sys.path + `np.load`」的块直接写定为 `precision = 64`
   （float64，即上游始终保存的值），避免在标准安装位置下崩溃。
3. `setup.cfg`：`packages` 增加 `OOPAO.tools`，使嵌套的 `tools` 子包被正确
   安装（否则 `from OOPAO.tools.tools import warning` 会报
   `ModuleNotFoundError`）。

这些补丁只影响「精度常量读取 / 缓存保存」这类非关键路径，不改变湍流屏生成、
光瞳、Zernike 等核心计算结果。

---

## 4. 代码如何迁移

`physics/oopao_backend.py` 是 OOPAO 后端接口。原先（vendored 版）：

```python
from physics.oopao import Atmosphere, Telescope
from physics.oopao.Source import Source
```

迁移后（原生版，注意官方把每个类放在同名模块里，需要 `from OOPAO.<Mod> import <Cls>`）：

```python
from OOPAO.Atmosphere import Atmosphere
from OOPAO.Telescope import Telescope
from OOPAO.Source import Source
```

> 官方 OOPAO 在 `OOPAO/__init__.py` 只导入模块、不把类提升到顶层，因此
> `from OOPAO import Telescope` 得到的是**模块**而不是**类**；必须写成
> `from OOPAO.Telescope import Telescope`。这与旧 vendored 版的
> `from physics.oopao import Telescope`（直接得到类）不同，是迁移时的重点。

`data/simulate.py` 只依赖 `physics.oopao_backend.OopaoScreenBackend`，无需改动。

---

## 5. `OopaoScreenBackend` 工作原理

`physics/oopao_backend.py` 负责为每个样本生成确定性的多层湍流相位屏。

### 5.1 与 aotools 的关系

OOPAO 的 `Atmosphere` 把湍流路径建模为 N 个独立层，每层携带一张 von-Karman
屏幕（`phaseStats.ft_sh_phase_screen`，源自 aotools）。每个进程构建一个
`Atmosphere`，每个样本通过 `Atmosphere.generateNewPhaseScreen(seed)` 得到全新
的、按种子确定的逐层实现。

### 5.2 逐层 r0 标定

OOPAO 自带的 `cn2` 记账会把总 Cn2 除以 `max(altitude)`，导致分层 r0 切分不
正确（逐层的相位方差过强）。因此我们绕过它：

- 每层在参考 r0（`_R0_REF_500 = 0.15` m @ 500 nm）下生成；
- 再做幅度重缩放，使逐层 r0 **恰好等于** `r0_slab = r0_path · n^(3/5)`
  （与 aotools 路径一致）。

由于 von-Karman 功率谱按 `r0^(-5/3)` 缩放，常量幅度重缩放因子
`sqrt((r0_slab / r0_ref)^(5/3))` 是一次统计上精确的 r0 变换（相位屏形状按
r/L0/l0 保持不变）。结果是：OOPAO 屏幕与 aotools 屏幕**统计等价**（同样的
逐层 r0、L0、l0），仅随机实现不同。

### 5.3 层到屏的映射

OOPAO 以 `resolution = N + 4` 像素构建每层（每侧留 2 像素余量，用于冻结流外圈）。
我们裁剪中央 `N × N` 与瞳孔网格对齐。`layer.OPD` 是逐层相位（弧度，即
`ft_sh_phase_screen` 的输出），不做 `2π/λ` 换算。

### 5.4 Python 片段

```python
from physics.oopao_backend import OopaoScreenBackend

backend = OopaoScreenBackend(
    N=64, dx=1.0, Dscope=1.0, lam=1.06e-6,
    cn2=1e-13, L=1000.0, L0=25.0, n_screens=3,
)
screens = backend.make_screens(seed=42)   # shape (n_screens, N, N), float32, 弧度
```

---

## 6. 原生 OOPAO 的更多能力

原生安装（而非 vendored 子集）让项目可以使用 OOPAO 的完整模块：

- `Atmosphere`：多层、无限 / 非平稳相位屏，可实时更新条件，可仿真闪烁。
- `Telescope`：默认圆形光瞳或自定义，可带/不带 spiders。
- `DeformableMirror`：高斯影响函数（默认）或自定义。
- WFS：`Pyramid`、`ShackHartmann`（衍射与几何）、`BioEdge`。
- `Source`：NGS 或 LGS。
- 控制基：KL 模态基、Zernike 多项式。
- `Detector`、`FieldTransformer`、`GainSensingCamera` 等。

上述能力可参考 `third_party/README.md`（官方 README）与官方文档。

---

## 7. 运行可视化 / 测试

`docs/oopao/` 下提供 Jupyter notebook：
`oopao_explore.ipynb` —— 可视化并测试 OOPAO 的核心函数
（`Telescope`、`Atmosphere`、`Source`、`Zernike`），并验证与
`OopaoScreenBackend` 输出的一致性。见同名 notebook 的说明。

---

## 8. 目录结构

```
third_party/                官方 OOPAO 源码（安装来源，含我们的小修补）
  setup.py / setup.cfg       打包配置（packages 含 OOPAO.tools）
  OOPAO/                     官方包（Telescope、Source、Atmosphere、Zernike……）
physics/oopao_backend.py     OopaoScreenBackend（已改用原生 `from OOPAO...`）
docs/oopao/README.md         本文档
docs/oopao/oopao_explore.ipynb  可视化 / 测试 notebook
```

> `physics/oopao/`（旧 vendored 子集）已从仓库移除。
