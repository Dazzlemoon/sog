import torch

from sog.module.gaussian import Gaussian


def test_make_quad_traceless_matches_les_convention() -> None:
    torch.manual_seed(7)
    n = 6
    Qr = torch.rand(n, 3, 3, dtype=torch.float64) * 2.0 - 1.0
    Qr = 0.5 * (Qr + Qr.transpose(-1, -2))
    trace = torch.einsum("iaa->i", Qr)
    eye = torch.eye(3, dtype=torch.float64)
    Q_ref = Qr - trace[:, None, None] * eye[None, :, :] / 3.0

    Q_out = Gaussian.make_quad_traceless(Qr)
    assert torch.allclose(Q_out, Q_ref, rtol=0.0, atol=1e-12)
    assert torch.allclose(
        torch.einsum("iaa->i", Q_out),
        torch.zeros(n, dtype=Q_out.dtype, device=Q_out.device),
        atol=1e-12,
    )


def test_ensure_multipole_shapes_applies_traceless() -> None:
    q = torch.randn(4, 1, dtype=torch.float64)
    quad = torch.randn(4, 1, 3, 3, dtype=torch.float64)
    quad = quad + 2.0 * torch.eye(3, dtype=torch.float64).view(1, 1, 3, 3)

    _, _, quad_out, _, _ = Gaussian._ensure_multipole_shapes(q, None, quad, None, None)
    assert quad_out is not None
    assert torch.allclose(
        torch.einsum("nqaa->nq", quad_out),
        torch.zeros(4, 1, dtype=quad_out.dtype, device=quad_out.device),
        atol=1e-12,
    )


def test_sog_forward_returns_traceless_quads() -> None:
    from sog import Sog

    torch.manual_seed(13)
    model = Sog({"use_atomwise": False, "trainable_kernel": False})
    r = torch.rand(5, 3, dtype=torch.float64)
    q = torch.randn(5, dtype=torch.float64)
    q = q - q.mean()
    Q = torch.randn(5, 3, 3, dtype=torch.float64)
    Q = Q + 2.5 * torch.eye(3, dtype=torch.float64)
    cell = torch.eye(3, dtype=torch.float64).unsqueeze(0) * 10.0

    out = model(
        positions=r,
        cell=cell,
        latent_charges=q,
        latent_quads=Q,
        compute_energy=True,
    )
    Q_out = out["latent_quads"]
    assert Q_out is not None
    assert torch.allclose(
        torch.einsum("iaa->i", Q_out),
        torch.zeros(5, dtype=torch.float64),
        atol=1e-12,
    )


if __name__ == "__main__":
    test_make_quad_traceless_matches_les_convention()
    test_ensure_multipole_shapes_applies_traceless()
    test_sog_forward_returns_traceless_quads()
    print("ok")
