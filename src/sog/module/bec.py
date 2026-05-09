from typing import Optional

import torch
import torch.nn as nn

from ..util import grad


class BEC(nn.Module):
    def __init__(
        self,
        remove_mean: bool = True,
        epsilon_factor: float = 1.0,
    ):
        super().__init__()
        self.remove_mean = remove_mean
        self.epsilon_factor = epsilon_factor
        self.normalization_factor = epsilon_factor**0.5

    def forward(
        self,
        q: torch.Tensor,
        r: torch.Tensor,
        cell: Optional[torch.Tensor],
        u: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        output_index: Optional[int] = None,
    ) -> torch.Tensor:
        if q.dim() == 1:
            q = q.unsqueeze(1)

        n_node, n_q = q.shape
        if u is not None:
            if u.dim() == 2 and u.shape[1] == 3:
                u = u.unsqueeze(1)
            assert u.shape == (n_node, n_q, 3), "u dimension error"

        n, d = r.shape
        assert d == 3, "r dimension error"
        assert n == q.size(0), "q dimension error"

        if batch is None:
            batch = torch.zeros(n, dtype=torch.int64, device=r.device)

        all_p = []
        all_p_u = []
        all_phases = []
        for bid in torch.unique(batch):
            mask = batch == bid
            r_now = r[mask]
            q_now = q[mask]
            if self.remove_mean:
                q_now = q_now - torch.mean(q_now, dim=0, keepdim=True)

            if cell is None or torch.abs(torch.det(cell[int(bid)])) < 1e-6:
                pol = torch.sum(q_now * r_now, dim=0)
                phase = torch.ones_like(r_now, dtype=torch.complex64)
            else:
                pol, phase = self.compute_pol_pbc(r_now, q_now, cell[int(bid)])

            if output_index is not None:
                pol = pol[output_index]
                phase = phase[:, output_index]

            all_p.append(pol * self.normalization_factor)
            all_phases.append(phase)
            if u is not None:
                u_now = u[mask]
                pol_u = u_now.sum((0, 1))
                all_p_u.append(pol_u * self.normalization_factor)

        p = torch.stack(all_p, dim=0)
        phases = torch.cat(all_phases, dim=0)
        bec_complex = grad(y=p, x=r).transpose(1, 2).contiguous()
        result = (bec_complex * phases.unsqueeze(2).conj()).real
        if u is not None:
            p_u = torch.stack(all_p_u, dim=0)
            result_u = grad(y=p_u, x=r).transpose(1, 2).contiguous()
            return torch.stack([result, result_u], dim=1)
        return result

    def compute_pol_pbc(
        self,
        r_now: torch.Tensor,
        q_now: torch.Tensor,
        box_now: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r_frac = torch.matmul(r_now, torch.linalg.inv(box_now))
        phase = torch.exp(1j * 2.0 * torch.pi * r_frac)
        s = torch.sum(q_now * phase, dim=0)
        polarization = torch.matmul(box_now.to(s.dtype), s.unsqueeze(1)) / (1j * 2.0 * torch.pi)
        return polarization.reshape(-1), phase
