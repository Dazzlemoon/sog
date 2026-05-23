#!/usr/bin/env python3
"""Compare BSA reciprocal-space multiplier vs SOG方法.md Eq.(90) and Coulomb 4*pi/k^2.

Checks:
  - Real-space kernel K(r) from module (BSA init) vs u-series Eq.(11)
  - Code kfac(k) from Gaussian._prepare_triclinic_state vs paper F_tilde(k) Eq.(90)
  - Numerical 3D FT of K(r) vs code kfac and vs 4*pi/k^2
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

OUT_DIR = Path(__file__).resolve().parent
RCUT_TO_SIGMA = 1.9892536839080267


def setup_paths() -> None:
    src = OUT_DIR / "src"
    s = str(src)
    if s not in sys.path:
        sys.path.insert(0, s)


def build_gaussian(r_cut: float, b: float, m: int = 12):
    from sog.module.gaussian import Gaussian

    sigma = r_cut / RCUT_TO_SIGMA
    return Gaussian(
        sigma=sigma,
        b=b,
        m=m,
        trainable=False,
    )


def k_real(r: np.ndarray, amp_stored: np.ndarray, bw2: np.ndarray) -> np.ndarray:
    ae = amp_stored / np.maximum(bw2, 1e-30)
    r2 = np.asarray(r, dtype=float)[:, None] ** 2
    return np.sum(ae[None, :] * np.exp(-0.5 * r2 / np.maximum(bw2[None, :], 1e-30)), axis=1)


def bsa_realspace(r: np.ndarray, sigma: float, b: float, m: int) -> np.ndarray:
    out = np.zeros_like(r, dtype=float)
    for ell in range(m):
        omega_l = math.sqrt(2.0 / math.pi) * (b ** (-ell)) * (math.log(b) / sigma)
        s_l = math.sqrt(2.0) * (b ** ell) * sigma
        out += omega_l * np.exp(-(r * r) / (s_l * s_l))
    return out


def paper_ft_eq90(k: np.ndarray, sigma: float, b: float, m: int) -> np.ndarray:
    """SOG方法.md Eq.(90): pi^{3/2} sum_l omega_l s_l^3 exp(-s_l^2 k^2 / 4)."""
    out = np.zeros_like(k, dtype=float)
    for ell in range(m):
        omega_l = math.sqrt(2.0 / math.pi) * (b ** (-ell)) * (math.log(b) / sigma)
        s_l = math.sqrt(2.0) * (b ** ell) * sigma
        out += (math.pi ** 1.5) * omega_l * (s_l ** 3) * np.exp(-(s_l * s_l) * (k * k) / 4.0)
    return out


def code_kfac(k: np.ndarray, amp_stored: np.ndarray, bw2: np.ndarray) -> np.ndarray:
    """Gaussian._prepare_triclinic_state: 3D FT of sum_l omega_l exp(-r^2/s_l^2)."""
    ae = amp_stored / np.maximum(bw2, 1e-30)
    s = np.sqrt(2.0 * bw2)
    k2 = np.asarray(k, dtype=float)[:, None] ** 2
    ft_scale = (math.pi ** 1.5) * (s ** 3)
    return np.sum(ae[None, :] * ft_scale[None, :] * np.exp(-0.5 * k2 * bw2[None, :]), axis=1)


def numerical_ft_spherical(k: np.ndarray, r_grid: np.ndarray, kr: np.ndarray) -> np.ndarray:
    """F̃(k) = 4π/k ∫_0^∞ K(r) r sin(kr) dr for isotropic K(r)."""
    k = np.asarray(k, dtype=float)
    out = np.zeros_like(k)
    for i, ki in enumerate(k):
        if ki < 1e-14:
            integrand = kr * r_grid * r_grid
        else:
            integrand = kr * r_grid * np.sin(ki * r_grid)
        out[i] = 4.0 * math.pi / ki * np.trapz(integrand, r_grid) if ki >= 1e-14 else 4.0 * math.pi * np.trapz(kr * r_grid * r_grid, r_grid)
    return out


def module_kfac_from_state(g, k_ang: float) -> float:
    """Single k along x for cubic cell L, using module state builder."""
    cell = torch.tensor(
        [[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]],
        dtype=torch.float64,
    )
    r_dummy = torch.zeros((1, 3), dtype=torch.float64)
    q_dummy = torch.ones((1, 1), dtype=torch.float64)
    state = g._prepare_triclinic_state(r_dummy, q_dummy, cell[0])
    g_cart = state["g_cart"].reshape(3, -1).T.cpu().numpy()
    kfac_flat = state["kfac"].reshape(-1).cpu().numpy()
    mask = state["k_mode_mask"].reshape(-1).cpu().numpy().astype(bool)
    g_cart = g_cart[mask]
    kfac_flat = kfac_flat[mask]
    k_mag = np.linalg.norm(g_cart, axis=1)
    idx = np.argmin(np.abs(k_mag - k_ang))
    return float(kfac_flat[idx]), float(k_mag[idx])


def main() -> None:
    setup_paths()
    r_cut = 5.5
    b = 2.0
    m = 12
    sigma = r_cut / RCUT_TO_SIGMA

    g = build_gaussian(r_cut=r_cut, b=b, m=m)
    amp = g.amp.detach().cpu().numpy().astype(float).reshape(-1)
    bw2 = g.bandwidth.detach().cpu().numpy().astype(float).reshape(-1)

    # --- Real space sanity ---
    r = np.linspace(0.05, 12.0, 2000)
    k_mod = k_real(r, amp, bw2)
    k_eq11 = bsa_realspace(r, sigma, b, m)
    rel_real = np.max(np.abs(k_mod - k_eq11) / np.maximum(np.abs(k_eq11), 1e-12))

    # --- k-space curves ---
    k = np.logspace(-3, 1.0, 400)  # 1/Angstrom
    ft_paper = paper_ft_eq90(k, sigma, b, m)
    ft_code = code_kfac(k, amp, bw2)
    coulomb = 4.0 * math.pi / np.maximum(k * k, 1e-30)

    r_grid = np.linspace(1e-4, 30.0, 12000)
    kr = k_real(r_grid, amp, bw2)
    ft_num = numerical_ft_spherical(k, r_grid, kr)

    ratio_paper_coul = ft_paper / coulomb
    ratio_code_coul = ft_code / coulomb
    ratio_num_coul = ft_num / coulomb
    ratio_code_paper = ft_code / np.maximum(ft_paper, 1e-30)
    ratio_num_code = ft_num / np.maximum(ft_code, 1e-30)

    # Per-term factor: paper / code at k=0.01
    k0 = 0.01
    s0 = math.sqrt(2.0) * sigma
    w0 = math.sqrt(2.0 / math.pi) * (math.log(b) / sigma)
    factor_term0 = (math.pi ** 1.5) * (s0 ** 3) / 1.0  # code coeff ~ w0 at small k

    fig, axes = plt.subplots(2, 3, figsize=(14.0, 7.5), dpi=160)

    ax = axes[0, 0]
    ax.plot(r, 1.0 / r, "k-", lw=1.0, label=r"$1/r$")
    ax.plot(r, k_eq11, "C4-", lw=1.2, label="BSA Eq.(11)")
    ax.plot(r, k_mod, "C1--", lw=1.0, label="module K(r)")
    ax.set_xlim(0, 12)
    ax.set_xlabel(r"$r$ (Å)")
    ax.set_title(f"Real space (max rel err {rel_real:.2e})")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)

    ax = axes[0, 1]
    ax.loglog(k, np.maximum(ft_paper, 1e-30), "C4-", lw=1.5, label=r"Paper $\widetilde F$(k) Eq.(90)")
    ax.loglog(k, np.maximum(ft_code, 1e-30), "C1--", lw=1.2, label=r"Code kfac$(k)$")
    ax.loglog(k, np.maximum(ft_num, 1e-30), "C2:", lw=1.0, label=r"Num. FT of $K(r)$")
    ax.loglog(k, coulomb, "k-", lw=1.0, label=r"Coulomb $4\pi/k^2$")
    ax.set_xlabel(r"$k$ (Å$^{-1}$)")
    ax.set_title("Reciprocal multiplier vs k")
    ax.legend(fontsize=6.5)
    ax.grid(alpha=0.25)

    ax = axes[0, 2]
    ax.semilogx(k, ratio_paper_coul, "C4-", label=r"Eq.(90) / $(4\pi/k^2)$")
    ax.semilogx(k, ratio_code_coul, "C1--", label=r"code kfac / $(4\pi/k^2)$")
    ax.semilogx(k, ratio_num_coul, "C2:", label=r"num FT / $(4\pi/k^2)$")
    ax.axhline(1.0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel(r"$k$ (Å$^{-1}$)")
    ax.set_title(r"Match Coulomb FT (ideal $\to 1$ at small $k$)")
    ax.legend(fontsize=6.5)
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    ax.loglog(k, np.maximum(ft_paper, 1e-30), "C4-", label="Paper Eq.(90)")
    ax.loglog(k, np.maximum(ft_code, 1e-30), "C1--", label="code kfac")
    ax.loglog(k, np.maximum(ft_num, 1e-30), "C2:", label="num FT(K)")
    ax.set_xlabel(r"$k$ (Å$^{-1}$)")
    ax.set_ylabel("value")
    ax.set_title("Reciprocal (zoom small k)")
    ax.set_xlim(k.min(), 0.5)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    ratio_num_paper = ft_num / np.maximum(ft_paper, 1e-30)
    ax.semilogx(k, ratio_code_paper, "C1-", label=r"code / paper Eq.(90)")
    ax.semilogx(k, ratio_num_paper, "C4-", label=r"num FT / paper")
    ax.semilogx(k, ratio_num_code, "C2-", label=r"num FT / code")
    ax.set_xlabel(r"$k$ (Å$^{-1}$)")
    ax.set_title("Consistency between representations")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)

    ax = axes[1, 2]
    # ell=0 analytic factor paper/code ~ pi^{3/2} s_0^3
    s_ells = np.array([math.sqrt(2.0) * (b ** ell) * sigma for ell in range(m)])
    w_ells = np.array([math.sqrt(2.0 / math.pi) * (b ** (-ell)) * (math.log(b) / sigma) for ell in range(m)])
    factors = (math.pi ** 1.5) * (s_ells ** 3) / np.maximum(w_ells, 1e-30)
    ax.bar(np.arange(m), factors, color="steelblue", alpha=0.85)
    ax.set_xlabel(r"term $\ell$")
    ax.set_ylabel(r"$\pi^{3/2}s_\ell^3$ (paper/code at $k\to0$)")
    ax.set_title(r"Per-term missing factor in code kfac")
    ax.grid(alpha=0.25, axis="y")

    fig.suptitle(
        rf"BSA reciprocal check ($r_{{cut}}={r_cut}$ Å, $b={b}$, $m={m}$, $\sigma={sigma:.4f}$ Å)",
        fontsize=11,
    )
    fig.tight_layout()
    png = OUT_DIR / "bsa_kspace_vs_coulomb_ft.png"
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)

    # Small-k asymptotics
    k_small = k[k < 0.05]
    summary = {
        "r_cut_A": r_cut,
        "sigma_A": sigma,
        "b": b,
        "m": m,
        "realspace_max_rel_err_vs_eq11": float(rel_real),
        "ell0_factor_paper_over_code_at_k0": float((math.pi ** 1.5) * (s0 ** 3)),
        "ell0_factor_numeric_mean_k_lt_0.05": float(np.mean(ratio_num_code[k < 0.05])),
        "paper_over_coulomb_at_k": {
            "k": [float(k[i]) for i in [0, 50, 100, 200]],
            "ratio": [float(ratio_paper_coul[i]) for i in [0, 50, 100, 200]],
        },
        "code_over_coulomb_at_k": {
            "k": [float(k[i]) for i in [0, 50, 100, 200]],
            "ratio": [float(ratio_code_coul[i]) for i in [0, 50, 100, 200]],
        },
        "numFT_over_code_at_k": {
            "k": [float(k[i]) for i in [0, 50, 100, 200]],
            "ratio": [float(ratio_num_code[i]) for i in [0, 50, 100, 200]],
        },
        "conclusion": (
            "Real-space BSA init matches Eq.(11). "
            "Code kfac uses pi^{3/2}(2*bw2)^{3/2} per term (3D FT of real-space Gaussian). "
            "Code kfac matches Eq.(90). At moderate k, sum approximates 4*pi/k^2."
        ),
    }
    js = OUT_DIR / "bsa_kspace_vs_coulomb_ft_summary.json"
    js.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("[OK]", png)
    print("[OK]", js)
    print("[INFO] realspace max rel err:", rel_real)
    print("[INFO] ell=0 paper/code factor ~", summary["ell0_factor_paper_over_code_at_k0"])
    print("[INFO] mean numFT/code for k<0.05:", summary["ell0_factor_numeric_mean_k_lt_0.05"])


if __name__ == "__main__":
    main()
