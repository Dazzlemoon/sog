# SOG 接口升级说明：对齐 LES 顶层返回并支持 `q_les/u_les/Q_les`

本文说明 `/data/home/public/qiuqizhi/sog` 的接口升级内容，以及与修改前的区别。

---

## 1. 升级目标

让 `sog` 的顶层接口与 `les` 更一致，支持：

- 输入：`latent_charges`（`q_les`）
- 可选输入：`latent_dipoles`（`u_les`）、`latent_quads`（`Q_les`）
- 可选响应参数：`latent_kappas`（感应电荷）、`latent_alphas`（感应偶极）
- 兼容 LES 风格参数键（减少配置迁移成本）：
  - `use_fixed_atomic_charges`
  - `fixed_atomic_charges_scaling_factor`
  - `use_atomic_alpha`
  - `use_epsilon_r_scaling`（当前仅解析保留，不单独改变 SOG 核公式）
- 顶层返回字段对齐 LES：
  - `E_lr`
  - `latent_charges`（已并入感应电荷）
  - `latent_dipoles`（已并入感应偶极）
  - `latent_quads`
  - `latent_alphas`
  - `BEC`（可选）

---

## 2. 代码改动

### `src/sog/module/gaussian.py`

新增/扩展：

- `forward(...)` 支持 `u, quad, kappa, alpha, e_ext`
- `compute_multipole_bundle(...)`：多极路径（周期/非周期）
- 非周期：实空间导数核缩并
- 周期：结构因子路径
  $$
  S = S_q + S_u + S_Q
  $$
  并计算
  $$
  U \propto \sum_{\mathbf{k}\neq 0} kfac(\mathbf{k}) |S(\mathbf{k})|^2
  $$
- 支持：
  - 感应电荷：$\Delta q = -\kappa\phi$
  - 感应偶极：$\Delta \mathbf{u} = \alpha \mathbf{E}$

`compute_multipole_bundle(...)` 返回中间量：

- `energy`
- `q_induced`
- `u_induced`
- `phi`

### `src/sog/sog.py`

`Sog.forward(...)` 新增参数：

- `latent_dipoles`
- `latent_quads`
- `latent_kappas`
- `latent_alphas`
- `atomic_numbers`
- `e_ext`

并在顶层返回中对齐 LES 字段（不再把 `q_induced/phi` 作为顶层输出字段）。

此外，`src/sog/module/bec.py` 已对齐 LES 的 `u` 输入逻辑：

- `BEC.forward(...)` 现在支持 `u`；
- `Sog.forward` 在 `compute_bec=True` 时会把 `latent_dipoles` 传给 `BEC`。

---

## 3. 修改前 vs 修改后

### 修改前

- q-only 长程路径
- 仅能直接做 `latent_charges -> E_lr`（+可选 BEC）

### 修改后

- 支持 `q/u/Q` 共同进入长程能量
- 支持 `kappa/alpha` 响应项
- 顶层返回与 LES 对齐（见第 1 节）
- 旧用法保持兼容（只传 `latent_charges` 仍可用）

### 高斯核参数是否可训练

是。`sog` 当前高斯核参数默认可训练：

- 权重：`amp`
- 带宽平方：`bandwidth`（内部保存为 $b_m^2$）

由 `trainable_kernel` 控制：

- `trainable_kernel=True`（默认）：`amp/bandwidth` 是 `nn.Parameter`；
- `trainable_kernel=False`：核参数固定不更新。

---

## 4. 使用示例

```python
import torch
from sog import Sog

model = Sog({"use_atomwise": False})

n = 16
pos = torch.randn(n, 3)
cell = torch.eye(3).unsqueeze(0) * 12.0
q = torch.randn(n)
u = torch.randn(n, 3)
Q = torch.randn(n, 3, 3)
kappa = torch.full((n,), 0.1)
alpha = torch.full((n,), 0.2)

out = model(
    positions=pos,
    cell=cell,
    latent_charges=q,
    latent_dipoles=u,
    latent_quads=Q,
    latent_kappas=kappa,
    latent_alphas=alpha,
    compute_energy=True,
)

print(out["E_lr"])
print(out["latent_charges"].shape, out["latent_dipoles"].shape)
```

---

## 5. 当前注意事项

- `compute_multipole_bundle` 当前主要面向能量路径，返回中 `forces/virial` 仍为 `None`。
- 若要完整 MD 力学链路，可继续补充显式导数或自动求导实现。

---

## 6. 总结

本次升级将 `sog` 从 q-only 扩展到 LES 风格的多极接口，并使顶层输出与 `les` 对齐。现在可直接将 `q_les/u_les/Q_les`（以及 `kappa/alpha`）输入 `sog` 计算长程能量与响应修正。
