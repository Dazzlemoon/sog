from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

try:
    import pytorch_finufft

    HAS_PYTORCH_FINUFFT = True
except (ImportError, ModuleNotFoundError):
    HAS_PYTORCH_FINUFFT = False
    pytorch_finufft = None


E2_PER_ANGSTROM_TO_EV = 14.3996454784255
SOG_DEFAULT_B = 2
SOG_DEFAULT_SIGMA = 2.180230445405648
SOG_DEFAULT_M = 12


class Gaussian(nn.Module):
    """Gaussian long-range SOG core.

    Periodic systems are computed in reciprocal space when NUFFT is enabled,
    non-periodic/singular-cell inputs fall back to a direct real-space kernel.
    """

    def __init__(
        self,
        n_dl: float = 1.0,
        amp: Optional[float] = None,
        bandwidth: Optional[torch.Tensor] = None,
        b: float = SOG_DEFAULT_B,
        sigma: float = SOG_DEFAULT_SIGMA,
        m: int = SOG_DEFAULT_M,
        remove_self_interaction: bool = True,
        charge_neutral_lambda: Optional[float] = None,
        use_nufft: bool = False,
        nufft_eps: float = 1e-6,
        norm_factor: float = E2_PER_ANGSTROM_TO_EV,
        trainable: bool = True,
        max_cache_size: int = 8,
        nufft: Optional[bool] = None,
    ):
        super().__init__()

        n_dl_value = float(n_dl)
        if (not math.isfinite(n_dl_value)) or n_dl_value <= 0.0:
            raise ValueError("`n_dl` should be a positive finite number.")
        self.n_dl = n_dl_value

        if bandwidth is None:
            if sigma <= 0.0:
                raise ValueError("`sigma` should be positive when `bandwidth` is not provided.")
            m_value = max(1, int(m))
            bw = sigma * torch.pow(
                torch.tensor(float(b), dtype=torch.get_default_dtype()),
                torch.arange(m_value, dtype=torch.get_default_dtype()),
            )
        else:
            bw = torch.as_tensor(bandwidth, dtype=torch.get_default_dtype()).reshape(-1)

        if bw.numel() == 0:
            raise ValueError("`bandwidth` should not be empty.")
        if not torch.isfinite(bw).all():
            raise ValueError("`bandwidth` should be finite.")
        if torch.any(bw <= 0.0):
            raise ValueError("`bandwidth` values should be positive.")
        bw2 = bw.square()

        if amp is None:
            if b <= 0.0:
                raise ValueError("`b` should be positive when `amp` is not provided.")
            coef1 = float(4.0 * torch.pi * math.log(b))
            amp_tensor = torch.full_like(bw2, fill_value=coef1)
        else:
            amp_tensor = torch.as_tensor(amp, dtype=torch.get_default_dtype()).reshape(-1)
        if amp_tensor.numel() == 0:
            raise ValueError("`amp` should not be empty.")
        if not torch.isfinite(amp_tensor).all():
            raise ValueError("`amp` should be finite.")

        # Allow scalar amp and broadcast it to all Gaussian terms.
        if amp_tensor.numel() == 1 and bw2.numel() > 1:
            amp_tensor = amp_tensor.expand_as(bw2).clone()
        elif amp_tensor.numel() != bw2.numel():
            raise ValueError(
                "`amp` should be scalar or have the same length as `bandwidth`."
            )
        # Fixed branch-aware representation:
        # store as k-space style internal parameters (amp_k * bw2).
        amp_tensor = amp_tensor * bw2

        self.amp = nn.Parameter(amp_tensor, requires_grad=trainable)
        self.bandwidth = nn.Parameter(bw2, requires_grad=trainable)

        self.remove_self_interaction = bool(remove_self_interaction)
        self.charge_neutral_lambda = charge_neutral_lambda
        self.use_nufft = bool(use_nufft if nufft is None else nufft)
        self.nufft_eps = float(nufft_eps)
        self.norm_factor = float(norm_factor)

        self._max_cache_size = max(1, int(max_cache_size))
        self._kgrid_base_cache: Dict[
            Tuple[str, str, int, int, int],
            Tuple[torch.Tensor, torch.Tensor, Tuple[int, int, int]],
        ] = {}

    def _amp_kspace(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Return Fourier-space Gaussian amplitudes from stored parameters."""
        amp_real = self.amp.to(dtype=dtype, device=device)
        bw2 = self.bandwidth.to(dtype=dtype, device=device)
        bw2_safe = torch.clamp(bw2, min=torch.finfo(dtype).tiny)
        return amp_real / bw2_safe

    @staticmethod
    def _device_key(device: torch.device) -> str:
        if device.index is None:
            return device.type
        return f"{device.type}:{device.index}"

    def _trim_cache(self) -> None:
        if len(self._kgrid_base_cache) > self._max_cache_size:
            oldest_key = next(iter(self._kgrid_base_cache.keys()))
            self._kgrid_base_cache.pop(oldest_key, None)

    def _get_cached_kgrid_base(
        self,
        nk: Tuple[int, int, int],
        runtime_device: torch.device,
        real_dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int, int]]:
        cache_key = (
            self._device_key(runtime_device),
            str(real_dtype),
            int(nk[0]),
            int(nk[1]),
            int(nk[2]),
        )
        cached = self._kgrid_base_cache.get(cache_key)
        if cached is not None:
            k_cached, zero_mask_cached, output_shape_cached = cached
            # Be defensive against stale caches carried across save/load or device moves.
            if (
                k_cached.device == runtime_device
                and k_cached.dtype == real_dtype
                and zero_mask_cached.device == runtime_device
            ):
                return cached
            self._kgrid_base_cache.pop(cache_key, None)

        n1 = torch.arange(-nk[0], nk[0] + 1, device=runtime_device, dtype=real_dtype)
        n2 = torch.arange(-nk[1], nk[1] + 1, device=runtime_device, dtype=real_dtype)
        n3 = torch.arange(-nk[2], nk[2] + 1, device=runtime_device, dtype=real_dtype)
        kx_grid, ky_grid, kz_grid = torch.meshgrid(n1, n2, n3, indexing="ij")

        k_grid_int = torch.stack((kx_grid, ky_grid, kz_grid), dim=0)
        zero_mask = (k_grid_int[0] == 0) & (k_grid_int[1] == 0) & (k_grid_int[2] == 0)
        output_shape = tuple(int(x) for x in kx_grid.shape)

        out = (k_grid_int, zero_mask, output_shape)
        self._kgrid_base_cache[cache_key] = out
        self._trim_cache()
        return out

    def forward(
        self,
        q: torch.Tensor,
        r: torch.Tensor,
        cell: Optional[torch.Tensor],
        batch: Optional[torch.Tensor] = None,
        u: Optional[torch.Tensor] = None,
        quad: Optional[torch.Tensor] = None,
        kappa: Optional[torch.Tensor] = None,
        alpha: Optional[torch.Tensor] = None,
        e_ext: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if (u is not None) or (quad is not None) or (kappa is not None) or (alpha is not None):
            return self.compute_multipole_bundle(
                q=q, r=r, cell=cell, batch=batch, u=u, quad=quad, kappa=kappa, alpha=alpha, e_ext=e_ext
            )["energy"]
        return self.compute_bundle(
            q=q,
            r=r,
            cell=cell,
            batch=batch,
            compute_force=False,
            compute_virial=False,
        )["energy"]

    @staticmethod
    def _ensure_multipole_shapes(
        q: torch.Tensor,
        u: Optional[torch.Tensor],
        quad: Optional[torch.Tensor],
        kappa: Optional[torch.Tensor],
        alpha: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if q.dim() == 1:
            q = q.unsqueeze(1)
        n, nq = q.shape
        if u is not None:
            if u.dim() == 2 and u.shape[1] == 3:
                u = u.unsqueeze(1)
            if u.shape != (n, nq, 3):
                raise ValueError(f"u shape mismatch: expected {(n, nq, 3)}, got {tuple(u.shape)}")
        if quad is not None:
            if quad.dim() == 3 and quad.shape[1:] == (3, 3):
                quad = quad.unsqueeze(1)
            if quad.shape != (n, nq, 3, 3):
                raise ValueError(
                    f"quad shape mismatch: expected {(n, nq, 3, 3)}, got {tuple(quad.shape)}"
                )
        if kappa is not None:
            if kappa.dim() == 1:
                kappa = kappa.unsqueeze(1)
            if kappa.shape != (n, nq):
                raise ValueError(
                    f"kappa shape mismatch: expected {(n, nq)}, got {tuple(kappa.shape)}"
                )
        if alpha is not None:
            if alpha.dim() == 1:
                alpha = alpha.unsqueeze(1)
            elif alpha.dim() == 3 and alpha.shape[1:] == (3, 3):
                alpha = alpha.unsqueeze(1)
            if alpha.dim() == 2:
                if alpha.shape != (n, nq):
                    raise ValueError(
                        f"alpha shape mismatch: expected {(n, nq)} for scalar alpha, got {tuple(alpha.shape)}"
                    )
            elif alpha.dim() == 4 and alpha.shape[2:] == (3, 3):
                if alpha.shape != (n, nq, 3, 3):
                    raise ValueError(
                        f"alpha shape mismatch: expected {(n, nq, 3, 3)} for tensor alpha, got {tuple(alpha.shape)}"
                    )
            else:
                raise ValueError("alpha should be [n] / [n,nq] or [n,3,3] / [n,nq,3,3]")
        return q, u, quad, kappa, alpha

    def _build_realspace_derivative_tensors(
        self,
        r_raw: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = r_raw.unsqueeze(0) - r_raw.unsqueeze(1)  # [n,n,3]
        r2 = torch.sum(x * x, dim=-1)
        n = r_raw.shape[0]
        eye = torch.eye(3, dtype=r_raw.dtype, device=r_raw.device)

        amp = self._amp_kspace(dtype=r_raw.dtype, device=r_raw.device).view(1, 1, -1)
        bw2 = self.bandwidth.to(dtype=r_raw.dtype, device=r_raw.device).view(1, 1, -1)
        exp_term = torch.exp(-0.5 * r2.unsqueeze(-1) / bw2)

        phi = torch.sum(amp * exp_term, dim=-1)

        c1 = torch.sum(-(amp / bw2) * exp_term, dim=-1)
        T1 = c1[..., None] * x

        xx = x[..., :, None] * x[..., None, :]
        c2 = torch.sum((amp / (bw2 * bw2)) * exp_term, dim=-1)
        c3 = torch.sum((amp / bw2) * exp_term, dim=-1)
        T2 = c2[..., None, None] * xx - c3[..., None, None] * eye[None, None]

        xxx = torch.einsum("...a,...b,...c->...abc", x, x, x)
        term_delta_x = (
            torch.einsum("ab,...c->...abc", eye, x)
            + torch.einsum("ac,...b->...abc", eye, x)
            + torch.einsum("bc,...a->...abc", eye, x)
        )
        c4 = torch.sum(-(amp / (bw2 * bw2 * bw2)) * exp_term, dim=-1)
        c5 = torch.sum((amp / (bw2 * bw2)) * exp_term, dim=-1)
        T3 = c4[..., None, None, None] * xxx + c5[..., None, None, None] * term_delta_x

        xxxx = torch.einsum("...a,...b,...c,...d->...abcd", x, x, x, x)
        term_delta_rr = (
            torch.einsum("ab,...cd->...abcd", eye, xx)
            + torch.einsum("ac,...bd->...abcd", eye, xx)
            + torch.einsum("ad,...bc->...abcd", eye, xx)
            + torch.einsum("bc,...ad->...abcd", eye, xx)
            + torch.einsum("bd,...ac->...abcd", eye, xx)
            + torch.einsum("cd,...ab->...abcd", eye, xx)
        )
        term_delta_delta = (
            torch.einsum("ab,cd->abcd", eye, eye)
            + torch.einsum("ac,bd->abcd", eye, eye)
            + torch.einsum("ad,bc->abcd", eye, eye)
        ).unsqueeze(0).unsqueeze(0)
        c6 = torch.sum((amp / (bw2 * bw2 * bw2 * bw2)) * exp_term, dim=-1)
        c7 = torch.sum((amp / (bw2 * bw2 * bw2)) * exp_term, dim=-1)
        c8 = torch.sum((amp / (bw2 * bw2)) * exp_term, dim=-1)
        T4 = (
            c6[..., None, None, None, None] * xxxx
            - c7[..., None, None, None, None] * term_delta_rr
            + c8[..., None, None, None, None] * term_delta_delta
        )

        if self.remove_self_interaction:
            diag = torch.arange(n, device=r_raw.device)
            phi[diag, diag] = 0.0
            T1[diag, diag] = 0.0
            T2[diag, diag] = 0.0
            T3[diag, diag] = 0.0
            T4[diag, diag] = 0.0
        return phi, T1, T2, T3, T4

    def _compute_multipole_realspace_one(
        self,
        r_raw: torch.Tensor,
        q: torch.Tensor,
        u: Optional[torch.Tensor],
        quad: Optional[torch.Tensor],
        kappa: Optional[torch.Tensor],
        alpha: Optional[torch.Tensor],
        e_ext: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q, u, quad, kappa, alpha = self._ensure_multipole_shapes(q, u, quad, kappa, alpha)
        phi_ij, T1, T2, T3, T4 = self._build_realspace_derivative_tensors(r_raw)
        _n, _nq = q.shape

        e_phi = torch.einsum("iq,ij->jq", q, phi_ij)
        pot = 0.5 * torch.einsum("iq,iq->q", e_phi, q)
        e_field = torch.einsum("iq,ijc->jqc", q, T1)

        if u is not None:
            e_phi_u = -torch.einsum("iqc,ijc->jq", u, T1)
            e_phi = e_phi + e_phi_u
            pot = pot + torch.einsum("iq,iq->q", e_phi_u, q)
            E_u = torch.einsum("ijcd,iqc->jqd", T2, u)
            pot = pot - 0.5 * torch.einsum("iqc,iqc->q", u, E_u)
            e_field = e_field + E_u

        if quad is not None:
            e_phi_Q = 0.5 * torch.einsum("iqab,ijab->jq", quad, T2)
            e_phi = e_phi + e_phi_Q
            pot = pot + torch.einsum("iq,iq->q", q, e_phi_Q)
            pot = pot + 0.125 * torch.einsum("iqab,ijabcd,jqcd->q", quad, T4, quad)
            E_Q = 0.5 * torch.einsum("iqab,ijabc->jqc", quad, T3)
            e_field = e_field + E_Q
            if u is not None:
                pot = pot - torch.einsum("iqc,iqc->q", u, E_Q)

        q_induced = torch.zeros_like(q)
        if kappa is not None:
            q_induced = -kappa * e_phi
            pot = pot + 0.5 * torch.einsum("iq,iq->q", e_phi, q_induced)

        u_induced = torch.zeros_like(e_field)
        if alpha is not None:
            e_field_eff = e_field
            if e_ext is not None:
                e_field_eff = e_field_eff + e_ext[None, None, :]
            if alpha.dim() == 2:
                u_induced = e_field_eff * alpha.unsqueeze(2)
            else:
                u_induced = torch.einsum("iqc,iqcd->iqd", e_field_eff, alpha)
            pot = pot - 0.5 * torch.einsum("iqc,iqc->q", e_field_eff, u_induced)

        return pot.sum().view(-1), q_induced, e_phi, u_induced

    def _compute_multipole_periodic_one(
        self,
        r_raw: torch.Tensor,
        q: torch.Tensor,
        cell_now: torch.Tensor,
        u: Optional[torch.Tensor],
        quad: Optional[torch.Tensor],
        kappa: Optional[torch.Tensor],
        alpha: Optional[torch.Tensor],
        e_ext: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q, u, quad, kappa, alpha = self._ensure_multipole_shapes(q, u, quad, kappa, alpha)
        n, nq = q.shape
        state = self._prepare_triclinic_state(r_raw, q, cell_now)
        volume = state["volume"]
        kfac = state["kfac"].reshape(-1)

        kvec = state["g_cart"].reshape(3, -1).transpose(0, 1)
        k_mask = state["k_mode_mask"].reshape(-1)
        if not torch.any(k_mask):
            z = torch.zeros((nq,), dtype=r_raw.dtype, device=r_raw.device)
            return z.sum().view(-1), torch.zeros_like(q), torch.zeros_like(q)
        kvec = kvec[k_mask]
        kfac = kfac[k_mask]

        k_dot_r = torch.matmul(r_raw, kvec.transpose(0, 1))
        cos_kr = torch.cos(k_dot_r)
        sin_kr = torch.sin(k_dot_r)

        Sq_real = (q.unsqueeze(2) * cos_kr.unsqueeze(1)).sum(dim=0)
        Sq_imag = (q.unsqueeze(2) * sin_kr.unsqueeze(1)).sum(dim=0)

        uk = torch.zeros((n, nq, kvec.shape[0]), dtype=r_raw.dtype, device=r_raw.device)
        if u is not None:
            uk = torch.einsum("nqc,mc->nqm", u, kvec)
        Su_real = -(uk * sin_kr.unsqueeze(1)).sum(dim=0)
        Su_imag = (uk * cos_kr.unsqueeze(1)).sum(dim=0)

        qk2 = torch.zeros((n, nq, kvec.shape[0]), dtype=r_raw.dtype, device=r_raw.device)
        if quad is not None:
            qk2 = torch.einsum("mi,nqij,mj->nqm", kvec, quad, kvec)
        SQ_real = -0.5 * (qk2 * cos_kr.unsqueeze(1)).sum(dim=0)
        SQ_imag = -0.5 * (qk2 * sin_kr.unsqueeze(1)).sum(dim=0)

        S_real = Sq_real + Su_real + SQ_real
        S_imag = Sq_imag + Su_imag + SQ_imag
        S_sq = S_real.square() + S_imag.square()

        pot = (kfac.unsqueeze(0) * S_sq).sum(dim=1) / (2.0 * volume)

        if self.remove_self_interaction:
            self_sq = (q.unsqueeze(2) - 0.5 * qk2).square() + uk.square()
            self_sum = self_sq.sum(dim=0)
            pot = pot - (kfac.unsqueeze(0) * self_sum).sum(dim=1) / (2.0 * volume)

        prefactor = (2.0 * kfac) / volume
        term_real = S_real.unsqueeze(0) * cos_kr.unsqueeze(1) + S_imag.unsqueeze(0) * sin_kr.unsqueeze(1)
        e_phi = (prefactor.unsqueeze(0) * term_real).sum(dim=2)
        if self.remove_self_interaction:
            s_real_i = q.unsqueeze(2) - 0.5 * qk2
            e_phi = e_phi - (prefactor.unsqueeze(0) * s_real_i).sum(dim=2)

        q_induced = torch.zeros_like(q)
        if kappa is not None:
            q_induced = -kappa * e_phi
            pot = pot + 0.5 * torch.einsum("iq,iq->q", e_phi, q_induced)
        term_imag = S_real.unsqueeze(0) * sin_kr.unsqueeze(1) - S_imag.unsqueeze(0) * cos_kr.unsqueeze(1)
        e_field = (
            prefactor.unsqueeze(0).unsqueeze(0).unsqueeze(3)
            * term_imag.unsqueeze(3)
            * kvec.unsqueeze(0).unsqueeze(0)
        ).sum(dim=2)

        if self.remove_self_interaction and u is not None:
            e_field = e_field - (
                prefactor.unsqueeze(0).unsqueeze(0).unsqueeze(3)
                * uk.unsqueeze(3)
                * kvec.unsqueeze(0).unsqueeze(0)
            ).sum(dim=2)

        u_induced = torch.zeros_like(e_field)
        if alpha is not None:
            e_field_eff = e_field
            if e_ext is not None:
                e_field_eff = e_field_eff + e_ext[None, None, :]
            if alpha.dim() == 2:
                u_induced = e_field_eff * alpha.unsqueeze(2)
            else:
                u_induced = torch.einsum("iqc,iqcd->iqd", e_field_eff, alpha)
            pot = pot - 0.5 * torch.einsum("iqc,iqc->q", e_field_eff, u_induced)

        return pot.sum().view(-1), q_induced, e_phi, u_induced

    def compute_multipole_bundle(
        self,
        q: torch.Tensor,
        r: torch.Tensor,
        cell: Optional[torch.Tensor],
        batch: Optional[torch.Tensor] = None,
        u: Optional[torch.Tensor] = None,
        quad: Optional[torch.Tensor] = None,
        kappa: Optional[torch.Tensor] = None,
        alpha: Optional[torch.Tensor] = None,
        e_ext: Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        if q.dim() == 1:
            q = q.unsqueeze(1)
        n, d = r.shape
        assert d == 3, "r dimension error"
        assert n == q.size(0), "q dimension error"
        if batch is None:
            batch = torch.zeros(n, dtype=torch.int64, device=r.device)
        if cell is not None and (cell.dim() != 3 or cell.shape[-2:] != (3, 3)):
            raise ValueError(f"`cell` should be [nbatch, 3, 3], got {tuple(cell.shape)}")

        energies = []
        q_induced_full = torch.zeros_like(q)
        phi_full = torch.zeros_like(q)
        u_induced_full = torch.zeros((n, q.shape[1], 3), dtype=r.dtype, device=r.device)

        for bid_t in torch.unique(batch):
            bid = int(bid_t.item())
            mask = batch == bid_t
            r_now = r[mask]
            q_now = q[mask]
            u_now = u[mask] if u is not None else None
            quad_now = quad[mask] if quad is not None else None
            kappa_now = kappa[mask] if kappa is not None else None
            alpha_now = alpha[mask] if alpha is not None else None

            periodic = False
            box_now = None
            if cell is not None:
                box_now = cell[bid]
                periodic = torch.abs(torch.det(box_now)) > torch.finfo(box_now.dtype).eps

            if periodic:
                assert box_now is not None
                e_now, dq_now, phi_now, du_now = self._compute_multipole_periodic_one(
                    r_now, q_now, box_now, u_now, quad_now, kappa_now, alpha_now, e_ext
                )
            else:
                e_now, dq_now, phi_now, du_now = self._compute_multipole_realspace_one(
                    r_now, q_now, u_now, quad_now, kappa_now, alpha_now, e_ext
                )

            energies.append(e_now)
            q_induced_full[mask] = dq_now
            phi_full[mask] = phi_now
            u_induced_full[mask] = du_now

        # Match LES Ewald: one scalar per batch index as shape [n_batch], not [n_batch, 1]
        # (torch.stack on shape-[1] tensors would yield [n_batch, 1] and break FeatureAdd vs SR_energy).
        if energies:
            energy_out = torch.cat(energies, dim=0) * self.norm_factor
        else:
            energy_out = torch.zeros(0, dtype=r.dtype, device=r.device)
        return {
            "energy": energy_out,
            "q_induced": q_induced_full,
            "u_induced": u_induced_full,
            "phi": phi_full,
            "forces": None,
            "virial": None,
            "used_explicit_derivatives": False,
        }

    def compute_bundle(
        self,
        q: torch.Tensor,
        r: torch.Tensor,
        cell: Optional[torch.Tensor],
        batch: Optional[torch.Tensor] = None,
        compute_force: bool = False,
        compute_virial: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        if q.dim() == 1:
            q = q.unsqueeze(1)

        n, d = r.shape
        assert d == 3, "r dimension error"
        assert n == q.size(0), "q dimension error"

        if batch is None:
            batch = torch.zeros(n, dtype=torch.int64, device=r.device)

        if cell is not None and (cell.dim() != 3 or cell.shape[-2:] != (3, 3)):
            raise ValueError(f"`cell` should be [nbatch, 3, 3], got {tuple(cell.shape)}")

        need_force = bool(compute_force or compute_virial)
        energies = []
        force_full = (
            torch.zeros((n, 3), dtype=r.dtype, device=r.device) if need_force else None
        )
        virial_list = [] if compute_virial else None

        explicit_all = True

        for bid_t in torch.unique(batch):
            bid = int(bid_t.item())
            mask = batch == bid_t
            r_now = r[mask]
            q_now = q[mask]

            if r_now.shape[0] == 0:
                continue

            periodic = False
            box_now = None
            if cell is not None:
                box_now = cell[bid]
                det_now = torch.det(box_now)
                periodic = torch.abs(det_now) > torch.finfo(box_now.dtype).eps

            if periodic and need_force and self.use_nufft and HAS_PYTORCH_FINUFFT:
                assert box_now is not None
                state = self._prepare_triclinic_state(r_now, q_now, box_now)
                pot_now, force_now, virial_now = self._compute_periodic_nufft_bundle(
                    state,
                    need_force=True,
                    need_virial=compute_virial,
                )

                if self.remove_self_interaction:
                    pot_now = pot_now - torch.sum(q_now * q_now) * state["diag_sum"]

                pot_now = pot_now * self.norm_factor
                force_now = force_now * self.norm_factor
                if virial_now is not None:
                    virial_now = virial_now * self.norm_factor

                if force_full is not None:
                    force_full[mask] = force_now
                if virial_list is not None:
                    assert virial_now is not None
                    virial_list.append(virial_now)
            else:
                if need_force:
                    explicit_all = False

                if periodic:
                    assert box_now is not None
                    pot_now = self.compute_potential_triclinic(r_now, q_now, box_now)
                else:
                    pot_now = self.compute_potential_realspace(r_now, q_now)

                if virial_list is not None:
                    virial_list.append(torch.zeros((3, 3), dtype=r.dtype, device=r.device))

            if self.charge_neutral_lambda is not None:
                pot_now = pot_now + float(self.charge_neutral_lambda) * torch.mean(q_now).square()

            energies.append(pot_now)

        if len(energies) == 0:
            energy_out = torch.zeros(0, dtype=r.dtype, device=r.device)
        else:
            energy_out = torch.stack(energies, dim=0)

        used_explicit = bool(need_force and explicit_all)
        if need_force and not explicit_all:
            force_out = None
            virial_out = None
        else:
            force_out = force_full
            virial_out = torch.stack(virial_list, dim=0) if virial_list is not None else None

        return {
            "energy": energy_out,
            "forces": force_out,
            "virial": virial_out,
            "used_explicit_derivatives": used_explicit,
        }

    def compute_potential_realspace(self, r_raw: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        if q.dim() == 1:
            q = q.unsqueeze(1)

        n = r_raw.shape[0]
        r_ij = r_raw.unsqueeze(0) - r_raw.unsqueeze(1)
        r_sq = torch.sum(r_ij * r_ij, dim=-1, keepdim=True)

        amp = self._amp_kspace(dtype=r_raw.dtype, device=r_raw.device).view(1, 1, -1)
        bw2 = self.bandwidth.to(dtype=r_raw.dtype, device=r_raw.device).view(1, 1, -1)
        kernel = amp * torch.exp(-0.5 * r_sq / bw2)
        kernel = kernel.sum(dim=-1)

        diag = torch.arange(n, device=r_raw.device)
        kernel[diag, diag] = 0.0

        pair_q = q.unsqueeze(0) * q.unsqueeze(1)
        pot = 0.5 * torch.sum(pair_q * kernel.unsqueeze(-1))

        if not self.remove_self_interaction:
            k0 = amp.sum()
            pot = pot + 0.5 * torch.sum(q * q) * k0

        return pot * self.norm_factor

    def compute_potential_triclinic(
        self,
        r_raw: torch.Tensor,
        q: torch.Tensor,
        cell_now: torch.Tensor,
    ) -> torch.Tensor:
        state = self._prepare_triclinic_state(r_raw, q, cell_now)

        if self.use_nufft and HAS_PYTORCH_FINUFFT:
            pot = self._compute_periodic_nufft(
                state["r_in"],
                state["q"],
                state["kfac"],
                state["output_shape"],
                state["volume"],
            )
        else:
            pot = self._compute_periodic_direct(
                state["r_raw"],
                state["q"],
                state["g_cart"],
                state["kfac"],
                state["volume"],
                state["k_mode_mask"],
            )

        if self.remove_self_interaction:
            pot = pot - torch.sum(state["q"] * state["q"]) * state["diag_sum"]

        return pot * self.norm_factor

    def _prepare_triclinic_state(
        self,
        r_raw: torch.Tensor,
        q: torch.Tensor,
        cell_now: torch.Tensor,
    ) -> Dict[str, torch.Tensor | Tuple[int, int, int]]:
        if q.dim() == 1:
            q = q.unsqueeze(1)

        runtime_device = r_raw.device
        real_dtype = r_raw.dtype

        box = cell_now.to(dtype=real_dtype, device=runtime_device)
        volume = torch.det(box)
        if torch.abs(volume) <= torch.finfo(real_dtype).eps:
            raise ValueError("`cell` is singular (near-zero volume).")
        volume = torch.abs(volume)

        cell_inv = torch.linalg.inv(box)
        r_frac = torch.matmul(r_raw, cell_inv)
        r_frac = torch.remainder(r_frac + 0.5, 1.0) - 0.5

        pi_tensor = torch.tensor(torch.pi, dtype=real_dtype, device=runtime_device)
        point_limit = pi_tensor - 32.0 * torch.finfo(real_dtype).eps
        r_in = torch.clamp(
            2.0 * pi_tensor * r_frac,
            min=-point_limit,
            max=point_limit,
        ).contiguous()

        norms = torch.norm(box, dim=1)
        nk = tuple(max(1, int(v.item() / self.n_dl)) for v in norms)

        k_grid_int, zero_mask, output_shape = self._get_cached_kgrid_base(
            nk,
            runtime_device,
            real_dtype,
        )

        two_pi = 2.0 * pi_tensor
        n_dl_tensor = torch.as_tensor(self.n_dl, dtype=real_dtype, device=runtime_device)
        k_sq_max = (two_pi / n_dl_tensor) ** 2

        g_cart = two_pi * torch.einsum("ik,k...->i...", cell_inv, k_grid_int)
        k_sq = torch.sum(g_cart * g_cart, dim=0)
        k_mode_mask = (~zero_mask) & (k_sq <= k_sq_max)

        amp = self._amp_kspace(dtype=real_dtype, device=runtime_device).view(
            1,
            1,
            1,
            -1,
        )
        bw2 = self.bandwidth.to(dtype=real_dtype, device=runtime_device).view(
            1,
            1,
            1,
            -1,
        )
        kfac = amp * torch.exp(-0.5 * bw2 * k_sq.unsqueeze(-1))
        kfac = kfac.sum(dim=-1).masked_fill(~k_mode_mask, 0.0)

        diag_sum = kfac.sum() / (2.0 * volume)

        return {
            "r_raw": r_raw,
            "q": q,
            "r_in": r_in,
            "g_cart": g_cart,
            "kfac": kfac,
            "k_mode_mask": k_mode_mask,
            "output_shape": output_shape,
            "volume": volume,
            "diag_sum": diag_sum,
        }

    def _compute_periodic_nufft(
        self,
        r_in: torch.Tensor,
        q: torch.Tensor,
        kfac: torch.Tensor,
        output_shape: Tuple[int, int, int],
        volume: torch.Tensor,
    ) -> torch.Tensor:
        q_t = q.transpose(0, 1).contiguous()
        complex_dtype = torch.complex128 if q.dtype == torch.float64 else torch.complex64
        charge = torch.complex(q_t, torch.zeros_like(q_t)).to(dtype=complex_dtype).contiguous()

        nufft_points = r_in.transpose(0, 1).contiguous()
        recon = pytorch_finufft.functional.finufft_type1(
            nufft_points,
            charge,
            output_shape=output_shape,
            eps=self.nufft_eps,
            isign=-1,
        )

        if recon.dim() == 3:
            recon = recon.unsqueeze(0)

        recon = torch.fft.fftshift(recon, dim=(1, 2, 3))
        rho_sq = recon.real.square() + recon.imag.square()

        return (kfac.unsqueeze(0) * rho_sq).sum() / (2.0 * volume)

    def _compute_periodic_nufft_bundle(
        self,
        state: Dict[str, torch.Tensor | Tuple[int, int, int]],
        need_force: bool,
        need_virial: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        r_in = state["r_in"]
        q = state["q"]
        kfac = state["kfac"]
        output_shape = state["output_shape"]
        volume = state["volume"]
        g_cart = state["g_cart"]
        r_raw = state["r_raw"]

        assert isinstance(r_in, torch.Tensor)
        assert isinstance(q, torch.Tensor)
        assert isinstance(kfac, torch.Tensor)
        assert isinstance(output_shape, tuple)
        assert isinstance(volume, torch.Tensor)
        assert isinstance(g_cart, torch.Tensor)
        assert isinstance(r_raw, torch.Tensor)

        q_t = q.transpose(0, 1).contiguous()
        real_dtype = q.dtype
        complex_dtype = torch.complex128 if real_dtype == torch.float64 else torch.complex64
        charge = torch.complex(q_t, torch.zeros_like(q_t)).to(dtype=complex_dtype).contiguous()

        nufft_points = r_in.transpose(0, 1).contiguous()
        recon = pytorch_finufft.functional.finufft_type1(
            nufft_points,
            charge,
            output_shape=output_shape,
            eps=self.nufft_eps,
            isign=-1,
        )

        if recon.dim() == 3:
            recon = recon.unsqueeze(0)

        recon = torch.fft.fftshift(recon, dim=(1, 2, 3))
        rho_sq = recon.real.square() + recon.imag.square()
        energy = (kfac.unsqueeze(0) * rho_sq).sum() / (2.0 * volume)

        if not need_force:
            return energy, torch.zeros((q.shape[0], 3), dtype=real_dtype, device=q.device), None

        conv = kfac.unsqueeze(0).to(dtype=complex_dtype) * recon
        grad_conv = (1j * g_cart.unsqueeze(1).to(dtype=complex_dtype)) * conv.unsqueeze(0)
        grad_conv = torch.fft.ifftshift(grad_conv, dim=(2, 3, 4))

        grad_field = pytorch_finufft.functional.finufft_type2(
            nufft_points,
            grad_conv,
            eps=self.nufft_eps,
            isign=1,
        )

        force = (
            -(q_t.unsqueeze(0) * grad_field.real.to(dtype=real_dtype)).sum(dim=1).transpose(0, 1)
            / volume
        )

        virial = None
        if need_virial:
            virial = torch.einsum("ni,nj->ij", force, r_raw)

        return energy, force, virial

    def _compute_periodic_direct(
        self,
        r_raw: torch.Tensor,
        q: torch.Tensor,
        g_cart: torch.Tensor,
        kfac: torch.Tensor,
        volume: torch.Tensor,
        k_mode_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        kvec = g_cart.reshape(3, -1).transpose(0, 1)
        kfac_flat = kfac.reshape(-1)

        if k_mode_mask is None:
            # Fallback for old call sites.
            mask = kfac_flat != 0
        else:
            # Keep semantics consistent with condition=(k==0) and cutoff filtering.
            mask = k_mode_mask.reshape(-1)

        if not torch.any(mask):
            return torch.zeros((), dtype=r_raw.dtype, device=r_raw.device)

        kvec = kvec[mask]
        kfac_flat = kfac_flat[mask]

        k_dot_r = torch.matmul(r_raw, kvec.transpose(0, 1))
        cos_k_dot_r = torch.cos(k_dot_r)
        sin_k_dot_r = torch.sin(k_dot_r)

        s_real = (q.unsqueeze(2) * cos_k_dot_r.unsqueeze(1)).sum(dim=0)
        s_imag = (q.unsqueeze(2) * sin_k_dot_r.unsqueeze(1)).sum(dim=0)
        s_sq = s_real.square() + s_imag.square()

        return (kfac_flat.unsqueeze(0) * s_sq).sum() / (2.0 * volume)

    def __repr__(self) -> str:
        return (
            "Gaussian("
            f"n_dl={self.n_dl}, "
            f"remove_self_interaction={self.remove_self_interaction}, "
            f"use_nufft={self.use_nufft and HAS_PYTORCH_FINUFFT}"
            ")"
        )
