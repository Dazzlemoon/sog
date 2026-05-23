"""Tests that periodic kfac matches 3D FT of the real-space SOG kernel."""
from __future__ import annotations

import math

import numpy as np
import torch

from sog.module.gaussian import Gaussian
from sog.sog import RCUT_TO_SIGMA


def _paper_ft_eq90(k: float, sigma: float, b: float, m: int) -> float:
  out = 0.0
  for ell in range(m):
    omega_l = math.sqrt(2.0 / math.pi) * (b ** (-ell)) * (math.log(b) / sigma)
    s_l = math.sqrt(2.0) * (b ** ell) * sigma
    out += (math.pi ** 1.5) * omega_l * (s_l ** 3) * math.exp(-(s_l * s_l) * (k * k) / 4.0)
  return out


def _kfac_from_state_at_kmag(g: Gaussian, k_target: float) -> float:
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
  kmag = np.linalg.norm(g_cart, axis=1)
  idx = int(np.argmin(np.abs(kmag - k_target)))
  return float(kfac_flat[idx]), float(kmag[idx])


def _kfac_analytic_from_params(k: float, amp_stored: np.ndarray, bw2: np.ndarray) -> float:
  ae = amp_stored / np.maximum(bw2, 1e-30)
  s = np.sqrt(2.0 * bw2)
  return float(
    np.sum(
      ae
      * (math.pi ** 1.5)
      * (s ** 3)
      * np.exp(-(s * s) * (k * k) / 4.0)
    )
  )


def test_bsa_kfac_matches_eq90() -> None:
  r_cut = 5.5
  b = 2.0
  m = 12
  sigma = r_cut / RCUT_TO_SIGMA
  g = Gaussian(sigma=sigma, b=b, m=m, trainable=False)

  for k_tgt in (0.01, 0.05, 0.2):
    kfac, k_used = _kfac_from_state_at_kmag(g, k_tgt)
    expected = _paper_ft_eq90(k_used, sigma, b, m)
    rel = abs(kfac - expected) / max(abs(expected), 1e-12)
    assert rel < 1e-5, f"k={k_used}: kfac={kfac}, eq90={expected}, rel={rel}"


def test_bsa_kfac_matches_analytic_3d_ft_of_realspace_kernel() -> None:
  """kfac from _prepare_triclinic_state vs direct 3D FT of K(r)=sum omega_l exp(-r^2/s_l^2)."""
  r_cut = 5.5
  b = 2.0
  m = 12
  g = Gaussian(sigma=r_cut / RCUT_TO_SIGMA, b=b, m=m, trainable=False)

  amp = g.amp.detach().cpu().numpy().reshape(-1)
  bw2 = g.bandwidth.detach().cpu().numpy().reshape(-1)

  for k_tgt in (0.01, 0.05, 0.2):
    kfac, k_used = _kfac_from_state_at_kmag(g, k_tgt)
    expected = _kfac_analytic_from_params(k_used, amp, bw2)
    rel = abs(kfac - expected) / max(abs(expected), 1e-12)
    assert rel < 1e-8, f"k={k_used}: kfac={kfac}, analytic_ft={expected}, rel={rel}"


def test_ft_scale_per_term() -> None:
  bw2 = torch.tensor([1.0, 4.0], dtype=torch.float64)
  scale = Gaussian._kspace_gaussian_ft_scale(bw2)
  expected = torch.tensor(
    [(math.pi ** 1.5) * (2.0 ** 1.5) * (1.0 ** 1.5), (math.pi ** 1.5) * (2.0 ** 1.5) * (4.0 ** 1.5)],
    dtype=torch.float64,
  )
  assert torch.allclose(scale, expected)


if __name__ == "__main__":
  test_ft_scale_per_term()
  test_bsa_kfac_matches_eq90()
  test_bsa_kfac_matches_analytic_3d_ft_of_realspace_kernel()
  print("ok")
