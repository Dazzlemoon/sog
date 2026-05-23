import math

import torch

from sog.module.gaussian import Gaussian
from sog.sog import RCUT_TO_SIGMA


def test_bsa_amp_matches_eq14() -> None:
    r_cut = 5.5
    b = 2.0
    m = 12
    sigma = r_cut / RCUT_TO_SIGMA
    g = Gaussian(sigma=sigma, b=b, m=m, trainable=False)
    amp_eff = (g.amp / g.bandwidth).detach().cpu().numpy()
    assert amp_eff.shape == (m,)
    bw = g.bandwidth.detach().cpu().numpy()
    for ell in range(m):
        expected_bw2 = (sigma * (b ** ell)) ** 2
        assert abs(bw[ell] - expected_bw2) < 1e-4 * expected_bw2
        expected_amp_eff = math.sqrt(2.0 / math.pi) * (math.log(b) / sigma) * (b ** (-ell))
        assert abs(amp_eff[ell] - expected_amp_eff) / abs(expected_amp_eff) < 5e-5


def test_bsa_default_b_in_sog() -> None:
    from sog import Sog

    model = Sog({"use_atomwise": False, "r_cut": 5.5})
    assert model.b == 2.0
    assert abs(model.sigma - 5.5 / RCUT_TO_SIGMA) < 1e-9


if __name__ == "__main__":
    test_bsa_amp_matches_eq14()
    test_bsa_default_b_in_sog()
    print("ok")
