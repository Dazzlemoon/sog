from __future__ import annotations

from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

from .module import Atomwise, BEC, Gaussian, FixedCharges, AtomicAlpha


RCUT_TO_SIGMA = 1.9892536839080267


class Sog(nn.Module):
    """SOG plugin model with LES-like API."""

    def __init__(
        self,
        sog_arguments: Union[Dict[str, Any], str, None] = None,
        r_cut: Optional[float] = None,
        les_arguments: Union[Dict[str, Any], str, None] = None,
    ):
        super().__init__()

        if sog_arguments is not None and les_arguments is not None:
            raise ValueError("Pass only one of `sog_arguments` and `les_arguments`.")

        if les_arguments is not None:
            sog_arguments = les_arguments

        if sog_arguments is None:
            sog_arguments = {}
        if isinstance(sog_arguments, str):
            import yaml

            with open(sog_arguments, "r", encoding="utf-8") as f:
                parsed = yaml.safe_load(f)
            sog_arguments = {} if parsed is None else parsed

        if r_cut is not None:
            sog_arguments = dict(sog_arguments)
            sog_arguments["r_cut"] = r_cut

        self._parse_arguments(sog_arguments)

        self.atomwise: nn.Module = (
            Atomwise(
                n_in=self.n_in,
                n_out=self.n_out,
                n_layers=self.n_layers,
                n_hidden=self.n_hidden,
                add_linear_nn=self.add_linear_nn,
                output_scaling_factor=self.output_scaling_factor,
            )
            if self.use_atomwise
            else _DummyAtomwise()
        )
        if self.use_fixed_atomic_charges:
            self.fixed_charges = FixedCharges(
                normalization_factor=self.fixed_atomic_charges_scaling_factor
            )
        if self.use_atomic_alpha:
            self.atomic_alpha = AtomicAlpha()

        self.gaussian = Gaussian(
            n_dl=self.n_dl,
            amp=self.amp,
            bandwidth=self.bandwidth,
            b=self.b,
            sigma=self.sigma,
            m=self.m,
            remove_self_interaction=self.remove_self_interaction,
            charge_neutral_lambda=self.charge_neutral_lambda,
            use_nufft=self.nufft,
            nufft_eps=self.nufft_eps,
            norm_factor=self.norm_factor,
            trainable=self.trainable_kernel,
        )

        self.bec = BEC(
            remove_mean=self.remove_mean,
            epsilon_factor=self.epsilon_factor,
        )

    def _parse_arguments(self, args: Dict[str, Any]) -> None:
        self.use_atomwise = bool(args.get("use_atomwise", True))
        self.n_in = args.get("n_in", None)
        self.n_out = int(args.get("n_out", 1))
        self.n_layers = int(args.get("n_layers", 3))
        self.n_hidden = args.get("n_hidden", [32, 16])
        self.add_linear_nn = bool(args.get("add_linear_nn", True))
        self.output_scaling_factor = float(args.get("output_scaling_factor", 0.1))

        self.n_dl = float(args.get("n_dl", 1.0))
        self.amp = args.get("amp", None)
        self.bandwidth = args.get("bandwidth", None)
        self.b = float(args.get("b", 2.0))
        r_cut_arg = args.get("r_cut", None)
        self.r_cut = float(r_cut_arg) if r_cut_arg is not None else None
        if self.r_cut is not None:
            if self.r_cut <= 0.0:
                raise ValueError("`r_cut` should be positive.")
            self.sigma = self.r_cut / RCUT_TO_SIGMA
        else:
            self.sigma = float(args.get("sigma", 2.180230445405648))
        self.m = int(args.get("m", 12))
        self.remove_self_interaction = bool(args.get("remove_self_interaction", True))
        self.charge_neutral_lambda = args.get("charge_neutral_lambda", None)
        self.nufft = bool(args.get("nufft", args.get("use_nufft", False)))
        self.use_nufft = self.nufft
        self.nufft_eps = float(args.get("nufft_eps", 1e-4))
        self.norm_factor = float(args.get("norm_factor", 14.3996454784255))
        self.trainable_kernel = bool(args.get("trainable_kernel", True))

        self.remove_mean = bool(args.get("remove_mean", True))
        self.epsilon_factor = float(args.get("epsilon_factor", 1.0))
        # LES-style alias compatibility
        self.use_fixed_atomic_charges = bool(args.get("use_fixed_atomic_charges", False))
        self.fixed_atomic_charges_scaling_factor = float(
            args.get("fixed_atomic_charges_scaling_factor", 0.5)
        )
        self.use_atomic_alpha = bool(args.get("use_atomic_alpha", False))
        self.use_epsilon_r_scaling = bool(args.get("use_epsilon_r_scaling", False))

    def forward(
        self,
        positions: torch.Tensor,
        cell: Optional[torch.Tensor],
        desc: Optional[torch.Tensor] = None,
        latent_charges: Optional[torch.Tensor] = None,
        latent_dipoles: Optional[torch.Tensor] = None,
        latent_quads: Optional[torch.Tensor] = None,
        latent_kappas: Optional[torch.Tensor] = None,
        latent_alphas: Optional[torch.Tensor] = None,
        atomic_numbers: Optional[torch.Tensor] = None,
        e_ext: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        compute_energy: bool = True,
        compute_bec: bool = False,
        bec_output_index: Optional[int] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        if batch is None:
            batch = torch.zeros(positions.shape[0], dtype=torch.int64, device=positions.device)

        if latent_charges is not None:
            if latent_charges.dim() == 1:
                latent_charges = latent_charges.unsqueeze(1)
            assert latent_charges.shape[0] == positions.shape[0]
        elif desc is not None:
            if not self.use_atomwise:
                raise ValueError("desc is provided but use_atomwise is False")
            assert desc.shape[0] == positions.shape[0]
            latent_charges = self.atomwise(desc, batch)
        else:
            raise ValueError("Either desc or latent_charges must be provided")

        if (
            atomic_numbers is not None
            and hasattr(self, "use_fixed_atomic_charges")
            and self.use_fixed_atomic_charges
        ):
            latent_charges = latent_charges + self.fixed_charges(atomic_numbers)

        if (
            atomic_numbers is not None
            and hasattr(self, "use_atomic_alpha")
            and self.use_atomic_alpha
            and latent_alphas is not None
        ):
            baseline_alphas = self.atomic_alpha(atomic_numbers)
            if latent_alphas.dim() == 1:
                latent_alphas = latent_alphas + baseline_alphas
            elif latent_alphas.dim() == 3:
                latent_alphas = latent_alphas + baseline_alphas[:, None, None] * torch.eye(
                    3, device=baseline_alphas.device
                ).unsqueeze(0)

        if compute_energy:
            if (
                (latent_dipoles is not None)
                or (latent_quads is not None)
                or (latent_kappas is not None)
                or (latent_alphas is not None)
            ):
                out_m = self.gaussian.compute_multipole_bundle(
                    q=latent_charges,
                    r=positions,
                    cell=cell,
                    batch=batch,
                    u=latent_dipoles,
                    quad=latent_quads,
                    kappa=latent_kappas,
                    alpha=latent_alphas,
                    e_ext=e_ext,
                )
                e_lr = out_m["energy"]
                q_induced = out_m["q_induced"]
                u_induced = out_m["u_induced"]
                if q_induced is not None:
                    latent_charges = latent_charges + q_induced
                if u_induced is not None:
                    if latent_dipoles is not None:
                        if latent_dipoles.dim() == 2 and u_induced.dim() == 3:
                            latent_dipoles = latent_dipoles.unsqueeze(1)
                        latent_dipoles = latent_dipoles + u_induced
                    else:
                        latent_dipoles = u_induced
            else:
                e_lr = self.gaussian(
                    q=latent_charges,
                    r=positions,
                    cell=cell,
                    batch=batch,
                )
        else:
            e_lr = None

        bec = None
        if compute_bec:
            bec = self.bec(
                q=latent_charges,
                r=positions,
                cell=cell,
                u=latent_dipoles,
                batch=batch,
                output_index=bec_output_index,
            )

        return {
            "E_lr": e_lr,
            "latent_charges": latent_charges,
            "latent_dipoles": latent_dipoles,
            "latent_quads": latent_quads,
            "latent_alphas": latent_alphas,
            "BEC": bec,
        }


class _DummyAtomwise(nn.Module):
    def forward(self, desc: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        del desc, batch
        raise ValueError("set use_atomwise=True to use the Atomwise module")
