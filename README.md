# SOG

SOG（Sum-of-Gaussians）是面向机器学习势函数的长程静电插件库，用可训练高斯核叠加近似 Coulomb 长程相互作用，初值采用 BSA（bilateral series approximation）实空间 u-series。

## 安装

```bash
pip install -e .
```

依赖：`torch`、`finufft`、`pytorch_finufft` 等（见 `setup.py`）。

## 文档

| 文档 | 内容 |
|------|------|
| [SOG方法.md](SOG方法.md) | BSA 逼近 $1/r$ 的方法学与数学推导 |
| [架构与LES改进说明.md](架构与LES改进说明.md) | 代码架构与相对 LES 的主要改进 |
| [BSA初值设置与验证.md](BSA初值设置与验证.md) | BSA 初值如何设置及验证结果 |

## 快速示例

```python
import torch
from sog import Sog

model = Sog({"use_atomwise": False, "r_cut": 5.5, "b": 2.0, "m": 12})

r = torch.rand(16, 3)
q = torch.rand(16) - 0.5
q = q - q.mean()
cell = torch.eye(3).unsqueeze(0) * 12.0

out = model(positions=r, cell=cell, latent_charges=q, compute_energy=True)
print(out["E_lr"])
```
