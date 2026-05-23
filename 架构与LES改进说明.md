# SOG 代码架构与相对 LES 的主要改进

SOG（Sum-of-Gaussians）是一个面向机器学习势函数的长程静电插件库，包布局与 LES 类似，核心用可训练的高斯核叠加近似 Coulomb 长程相互作用。

---

## 1. 目录与模块结构

```
sog-github/
├── src/sog/
│   ├── sog.py                 # 顶层 Sog 模型，LES 风格 API
│   └── module/
│       ├── gaussian.py        # SOG 高斯核：实空间 / 倒空间 / 多极
│       ├── atomwise.py        # 可选：从描述符预测潜电荷
│       ├── fixedcharges.py    # 可选：固定原子电荷偏移
│       ├── atomicalpha.py     # 可选：原子极化率基线
│       ├── bec.py             # 可选：Born 有效电荷
│       └── blocks.py          # MLP 构建块
├── test/                      # 单元与对齐测试
├── example/                   # 配置与 DeepMD 对照脚本
└── plot_bsa_kspace_vs_coulomb_ft.py  # BSA 倒空间验证作图
```

### 1.1 顶层 `Sog`（`src/sog/sog.py`）

- 解析 `sog_arguments` / `les_arguments`（二者等价，便于从 LES 配置迁移）。
- 组装可选子模块：`Atomwise`、`FixedCharges`、`AtomicAlpha`、`Gaussian`、`BEC`。
- `forward` 统一入口：接收坐标、晶胞、潜多极量，返回 `E_lr` 及更新后的潜变量。

### 1.2 核心 `Gaussian`（`src/sog/module/gaussian.py`）

长程能量由 $M$ 项高斯核叠加：

$$
K(r) = \sum_{\ell=0}^{M-1} \omega_\ell \exp\!\left(-\frac{r^2}{s_\ell^2}\right),
\qquad s_\ell = \sqrt{2}\, b^\ell \sigma.
$$

可训练参数为内部存储的 `amp`$_\ell$ 与 `bandwidth`$_\ell$（$= s_\ell^2/2$）。初值固定为 **BSA 实空间 u-series**（见 `BSA初值设置与验证.md`）。

**计算路径：**

| 体系 | 路径 | 说明 |
|------|------|------|
| 非周期 / 奇异晶胞 | `compute_potential_realspace` | 实空间直接求和 |
| 周期（三斜） | `compute_potential_triclinic` | 倒空间结构因子 + `kfac(k)` |
| 周期 + NUFFT | `_compute_periodic_nufft` | 可选 GPU/CPU NUFFT 加速 |

**多极扩展**（`compute_multipole_bundle`）：

- 输入：电荷 $q$、偶极 $\mathbf{u}$、四极 $Q$、可选响应 $\kappa$（感应电荷）、$\alpha$（感应偶极）。
- 周期：结构因子 $S = S_q + S_u + S_Q$，能量 $\propto \sum_{\mathbf{k}\neq 0} \texttt{kfac}(k)\,|S(\mathbf{k})|^2$。
- 非周期：实空间导数核缩并。
- 感应修正：$\Delta q = -\kappa\phi$，$\Delta \mathbf{u} = \alpha \mathbf{E}$。

### 1.3 辅助模块

- **`BEC`**：Born 有效电荷，支持偶极输入。
- **`Atomwise`**：从局部描述符预测潜电荷（`use_atomwise=True` 时启用）。
- **`FixedCharges` / `AtomicAlpha`**：与 LES 相同的固定电荷、原子极化率基线逻辑。

---

## 2. 相对 LES 的主要改进

### 2.1 长程核：SOG 高斯叠加 vs LES Ewald

| 方面 | LES（典型） | SOG（本库） |
|------|-------------|-------------|
| 长程形式 | Ewald / 固定 Coulomb 核 | 可训练 $M$ 项高斯叠加 |
| 核参数 | 多为固定 | `amp`、`bandwidth` 可训练（`trainable_kernel=True`） |
| 初值 | Ewald 分裂参数 | BSA u-series，长程项逼近 $1/r$ |
| 倒空间 | Ewald $k$ 空间公式 | 单一 `kfac(k)`，由 3D FT 与实空间核自洽 |

训练时仍只优化 **`gaussian.amp` 与 `gaussian.bandwidth`**；周期与非周期路径共用同一组参数。

### 2.2 接口对齐 LES

本库自 q-only 扩展为 LES 风格多极接口：

**输入（与 LES 键名兼容）：**

- `latent_charges`（$q$）
- `latent_dipoles`（$\mathbf{u}$）
- `latent_quads`（$Q$）
- `latent_kappas`、`latent_alphas`（响应参数）
- `atomic_numbers`、`e_ext`
- LES 风格配置键：`use_fixed_atomic_charges`、`use_atomic_alpha` 等

**输出：**

```python
{
    "E_lr": ...,
    "latent_charges": ...,   # 含感应电荷修正
    "latent_dipoles": ...,   # 含感应偶极修正
    "latent_quads": ...,
    "latent_alphas": ...,
    "BEC": ...,              # compute_bec=True 时
}
```

旧用法（仅传 `latent_charges`）保持兼容。

### 2.3 实倒空间自洽

- BSA 初值在**实空间**定义 $\omega_\ell$、$s_\ell$。
- 周期 `kfac(k)` 含完整 3D FT 尺度 $\pi^{3/2} s_\ell^3$，与实空间核傅里叶变换一致。
- 避免「实空间一套参数、倒空间另一套系数」的不一致（详见 BSA 文档）。

### 2.4 工程与验证

- 可选 `pytorch_finufft` / `finufft` 加速周期求和。
- `test/` 覆盖：BSA 初值、$kfac$ 与解析 FT、三斜晶胞、DeepMD 对照、有限差分力等。
- `example/compare_with_deepmd_sog.py` 可与 DeepMD SOG 参考实现同输入对比。

---

## 3. 快速使用

```bash
pip install -e .
```

```python
import torch
from sog import Sog

model = Sog({"use_atomwise": False, "r_cut": 5.5, "b": 2.0, "m": 12})

r = torch.rand(16, 3)
q = torch.rand(16) - 0.5
q = q - q.mean()
cell = torch.eye(3).unsqueeze(0) * 12.0

out = model(
    positions=r,
    cell=cell,
    latent_charges=q,
    compute_energy=True,
)
print(out["E_lr"])
```

与 CACE-LES 联用时，通过 `LesWrapper` 传入 `les_arguments`（`r_cut`、`b`、`m` 等），见 `BSA初值设置与验证.md`。

---

## 4. 当前限制

- `compute_multipole_bundle` 主要面向能量路径；`forces` / `virial` 在 bundle 中仍为 `None`，完整 MD 力学需补充显式导数或 autograd。
- `use_epsilon_r_scaling` 仅解析保留，不单独改变 SOG 核公式。

---

*文档对应仓库：`sog-github`，初值策略见 `BSA初值设置与验证.md`。*
