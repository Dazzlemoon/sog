# 方案 B：阶次分辨的 $\widetilde K^{(\ell)}$（后续核扩展思路）

> **状态**：本文描述的是**尚未实现**、可作为 SOG 多极长程模块下一步改进的数学方案。  
> 当前代码（`src/sog/module/gaussian.py`）仍采用**方案 A**：单一结构因子 $S$ + 单一倒空间乘子 $\widetilde K(k)$。  
> 关联：Kim et al. 2026（[arXiv:2605.05746](https://arxiv.org/abs/2605.05746)）、[架构与LES改进说明.md](架构与LES改进说明.md)。

---

## 1. 当前实现（方案 A）与动机

周期多极长程能量（Kim 式 14–15，SOG 核替换 Ewald 屏蔽因子）：

$$
U^{\mathrm{elec}} = \frac{1}{2V}\sum_{\mathbf{k}\neq 0}
\widetilde K_\theta(k)\,
\bigl|S(\mathbf{k})\bigr|^2,
\qquad
S = S_q + S_u + S_Q.
$$

其中 $\widetilde K_\theta(k)=\texttt{kfac}(k)$ 为 BSA/SOG 高斯和的 3D 傅里叶变换（见 [BSA初值设置与验证.md](BSA初值设置与验证.md)）。

**优点：** 公式简洁，与 Kim/Ewald 多极写法一致；$S(\mathbf{k})$ 已含 $q,\mathbf{u},\mathbf{Q}$ 及全部交叉耦合。

**潜在问题：** 可训练参数 $\{\omega_\ell,s_\ell\}$（代码中的 `amp`/`bandwidth`）在训练中往往由 **monopole 主导的 $|S_q|^2$** 决定梯度方向。若实验变体主要为 `-r` / `-uiu`（无四极），dipole/quadrupole 通道未必得到物理合理的远场标度，尽管它们已进入 $S(\mathbf{k})$。

远场渐近：

| 耦合 | 渐近标度 |
|------|----------|
| $q$–$q$ | $\propto 1/r$ |
| $q$–$\mathbf{u}$ | $\propto 1/r^2$ |
| $\mathbf{u}$–$\mathbf{u}$ | $\propto 1/r^3$ |
| 含 $\mathbf{Q}$ | $\propto 1/r^4$ 及以上 |

Kim/LES 用**同一** Ewald 核 $\propto e^{-\sigma^2k^2/2}/k^2$ 乘在整体 $|S|^2$ 上——对固定 erf 核这是标准做法。SOG 将核换成可学习 $\widetilde K(k)$ 后，**单一 $\widetilde K$ 未必同时对各阶多极远场最优**。

---

## 2. 方案 B 的核心想法

将 $|S|^2$ **显式展开**，对不同阶多极项与交叉项使用**阶次分辨**的倒空间权重 $\widetilde K^{(\ell)}(k)$，而不是共用一个 $\widetilde K$：

$$
|S|^2 = |S_q|^2 + |S_u|^2 + |S_Q|^2
+ 2\mathrm{Re}(S_q S_u^* + S_q S_Q^* + S_u S_Q^*).
$$

提议能量形式：

$$
U = \frac{1}{2V}\sum_{\mathbf{k}\neq 0}
\Bigl[
\widetilde K^{(0)}(k)\,|S_q|^2
+ \widetilde K^{(2)}(k)\,|S_u|^2
+ \widetilde K^{(4)}(k)\,|S_Q|^2
+ 2\widetilde K^{(1)}(k)\,\mathrm{Re}(S_q S_u^*)
+ 2\widetilde K^{(3)}(k)\,\mathrm{Re}(S_q S_Q^*)
+ 2\widetilde K^{(3)}(k)\,\mathrm{Re}(S_u S_Q^*)
+ \cdots
\Bigr].
$$

**关键：** $\widetilde K^{(\ell)}$ 仍由**同一组** SOG 实空间参数 $\{\omega_n,s_n\}$（即 `amp`/`bandwidth`）解析导出，**不增加独立可训练核族**；但 monopole 不再单独「绑架」全部倒空间权重。

---

## 3. $\widetilde K^{(\ell)}$ 如何从 SOG 核导出

实空间 SOG 核（BSA 初值，见 `Gaussian`）：

$$
K(r) = \sum_{n=0}^{M-1} \omega_n \exp\!\left(-\frac{r^2}{s_n^2}\right).
$$

对单项高斯 $G_n(r)=\omega_n e^{-r^2/s_n^2}$，在 3D 傅里叶变换约定

$$
\widetilde f(\mathbf{k}) = \int_\Omega f(\mathbf{r})\, e^{-i\mathbf{k}\cdot\mathbf{r}}\,\mathrm{d}\mathbf{r}
$$

下有（与当前 `kfac` 一致）：

$$
\widetilde G_n^{(0)}(k) = \pi^{3/2}\, s_n^3\, \omega_n\,
\exp\!\left(-\frac{s_n^2 k^2}{4}\right).
$$

对 $K(r)$ 作 $\ell$ 阶空间导数后再做 3D FT，$\widetilde K^{(\ell)}$ 在 $\mathbf{k}$ 空间表现为：

- 同一高斯衰减因子 $\exp(-s_n^2 k^2/4)$（代码中即 $\exp(-\texttt{bandwidth}\, k^2/2)$）；
- 额外多项式因子 $\propto k^\ell$（来自 $\nabla^\ell$ 作用于 $e^{-r^2/s^2}$）。

**与 Kim (14) 的退化关系：** 在 monopole 通道取 $\ell=0$ 时，适当截断与标度下

$$
\widetilde K^{(0)}(k) \;\to\; \frac{4\pi}{k^2}\, e^{-\sigma^2 k^2/2},
$$

即 Kim Ewald 长程核。dipole/quadrupole 通道对应 $\ell=2,4,\ldots$，远场分别逼近 $1/r^3,\,1/r^5$ 等多极标度。

**实现侧已有基础：** 实空间路径 `_build_realspace_derivative_tensors` 已构造 $\phi,T^1,T^2,T^3,T^4$（$K$ 的 0–4 阶导数缩并）；方案 B 的 k 空间版本需为各阶导出对应的 $\widetilde K^{(\ell)}(k)$ 闭式，并在 `_compute_multipole_periodic_one` 中替换单一 `kfac * |S|^2`。

---

## 4. 与方案 A、方案 C 的关系

| | **方案 A（当前）** | **方案 B（本文）** | **方案 C** |
|---|-------------------|-------------------|-----------|
| 改什么 | 训练/接线/监控 | **k 空间能量公式** | 自能、无迹 $Q$、实/倒路径 |
| $|S|^2$ 结构 | 不展开 | 展开，各阶不同 $\widetilde K^{(\ell)}$ | 不展开 |
| 可训练参数 | `amp`/`bandwidth` | 同一组 `amp`/`bandwidth` | 同 A |
| 适用时机 | 默认；先补全 `-uQ`/`-iq` 接线 | monopole 仍主导核学习时 | 数值自洽与 Kim 对齐 |

方案 B 与方案 C **正交**：C 解决「算得对不对」，B 解决「单一 $\widetilde K$ 对多极是否足够」。四极无迹投影（Kim 式 6）已在当前代码 `Gaussian.make_quad_traceless` 中实现，是方案 C 的一部分，与 B 可并行推进。

---

## 5. 建议落地步骤（若未来实现）

1. **解析推导**  
   对 $G(r)=\omega e^{-r^2/s^2}$ 写出 $\widetilde{(\nabla^{(2)}\!G)}$、$\widetilde{(\nabla^{(4)}\!G)}$ 等闭式，与 `_kspace_gaussian_ft_scale` 及 `_build_realspace_derivative_tensors` 交叉验证。

2. **k 空间能量改写**  
   在 `_compute_multipole_periodic_one` 中：
   - 保留现有 $S_q,S_u,S_Q$ 构造；
   - 将 `pot = kfac * |S|^2` 换为各阶 $\widetilde K^{(\ell)}$ 的加权和；
   - 自能 `self_sq` 需与展开形式逐项对齐（不能简单沿用 monopole 近似）。

3. **实空间对偶（可选）**  
   实空间已用 $T^1,T^2,T^4$ 分通道；若 B 在 k 空间落地，应验证与 `_compute_multipole_realspace_one` 在大超胞极限下能量一致。

4. **训练与消融**  
   - 对比 `-uiu` vs `-uQiqiu`：方案 A vs B 的 force RMSE、BEC；
   - 日志分解 $U_q,U_u,U_Q$ 及交叉项，确认 dipole 通道不再仅靠 $|S_q S_u^*|$ 凑数。

5. **与方案 A 的轻量替代**  
   若暂不改公式，可先在方案 A 上加多极加权核正则（见 `SOG_multipole_kspace_实现与扩展思路.md` 方案 A 第 3 点），作为 B 的低成本试探。

---

## 6. 当前仓库状态

| 项目 | 状态 |
|------|------|
| 单一 $\widetilde K(k)\,|S|^2$ | **已实现** |
| $|S|^2$ 展开 + $\widetilde K^{(\ell)}$ | **未实现** |
| BSA 初值 + `kfac` 3D FT | **已实现** |
| 四极无迹 `make_quad_traceless` | **已实现** |
| 实/倒空间能量数值对照 | **未系统验证** |

---

## 7. 参考

- Kim D, King D S, Park Y, et al. *Polarizable atomic multipoles for learning long-range electrostatics*. arXiv:2605.05746, 2026.
- 本仓库讨论稿：`Polarizable_atomic_multipoles/SOG_multipole_kspace_实现与扩展思路.md` 第 7 节。
- 代码入口：`src/sog/module/gaussian.py` → `_compute_multipole_periodic_one`、`_kspace_gaussian_ft_scale`。

---

*文档版本：2026-05，描述后续改进方向，非当前发布功能说明。*
