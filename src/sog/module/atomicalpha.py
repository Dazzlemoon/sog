import torch
import torch.nn as nn
from typing import Dict

__all__ = ["AtomicAlpha"]

alpha_dict = {
    1: 4.50, 2: 1.38, 3: 164.11, 4: 37.74, 5: 20.50, 6: 11.30, 7: 7.40, 8: 5.30, 9: 3.74,
    10: 2.67, 11: 162.70, 12: 71.30, 13: 60.00, 14: 37.30, 15: 25.00, 16: 19.40, 17: 14.60,
    18: 11.10, 19: 290.00, 20: 169.00, 21: 120.00, 22: 98.00, 23: 84.00, 24: 78.00, 25: 63.00,
    26: 56.00, 27: 50.00, 28: 49.00, 29: 47.00, 30: 38.70, 31: 50.00, 32: 40.00, 33: 30.00,
    34: 28.90, 35: 21.90, 36: 16.80, 37: 319.00, 38: 197.00, 39: 162.00, 40: 121.00, 41: 106.00,
    42: 86.40, 43: 80.00, 44: 65.00, 45: 58.00, 46: 26.10, 47: 55.00, 48: 49.70, 49: 70.00,
    50: 52.00, 51: 43.00, 52: 37.60, 53: 35.00, 54: 27.30, 55: 401.00, 56: 273.00, 57: 213.00,
    58: 204.00, 59: 196.00, 60: 190.00, 61: 185.00, 62: 180.00, 63: 175.00, 64: 160.00, 65: 159.00,
    66: 157.00, 67: 156.00, 68: 153.00, 69: 151.00, 70: 142.00, 71: 148.00, 72: 109.00, 73: 88.00,
    74: 74.00, 75: 65.00, 76: 57.00, 77: 51.00, 78: 39.70, 79: 36.00, 80: 33.90, 81: 50.00, 82: 47.00,
    83: 48.00, 84: 45.00, 85: 38.00, 86: 33.00,
}


class AtomicAlpha(nn.Module):
    def __init__(
        self,
        alpha_dict: Dict[int, float] = alpha_dict,
        normalization_factor: float = 0.1481847 / 14.3996,
    ):
        super().__init__()
        self.alpha_dict = alpha_dict
        self.normalization_factor = normalization_factor

    def forward(self, atomic_numbers: torch.Tensor) -> torch.Tensor:
        alpha = torch.tensor(
            [self.alpha_dict[int(z.item())] for z in atomic_numbers],
            device=atomic_numbers.device,
            dtype=torch.get_default_dtype(),
        )
        return alpha * self.normalization_factor

