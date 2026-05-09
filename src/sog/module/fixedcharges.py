import torch
import torch.nn as nn
from typing import Dict

__all__ = ["FixedCharges"]

typical_charge = {
    1: +1, 2: 0, 3: +1, 4: +2, 5: +3, 6: +4, 7: -3, 8: -2, 9: -1, 10: 0,
    11: +1, 12: +2, 13: +3, 14: +4, 15: +5, 16: -2, 17: -1, 18: 0,
    19: +1, 20: +2, 21: +3, 22: +4, 23: +5, 24: +3, 25: +2, 26: +2,
    27: +2, 28: +2, 29: +1, 30: +2, 31: +3, 32: +4, 33: +5, 34: -2,
    35: -1, 36: 0, 37: +1, 38: +2, 39: +3, 40: +4, 41: +5, 42: +6,
    43: +7, 44: +3, 45: +3, 46: +2, 47: +1, 48: +2, 49: +3, 50: +2,
    51: +3, 52: -2, 53: -1, 54: 0, 55: +1, 56: +2, 57: +3, 58: +3,
    59: +3, 60: +3, 61: +3, 62: +3, 63: +2, 64: +3, 65: +3, 66: +3,
    67: +3, 68: +3, 69: +3, 70: +2, 71: +3, 72: +4, 73: +5, 74: +6,
    75: +7, 76: +4, 77: +3, 78: +2, 79: +1, 80: +2, 81: +1, 82: +2,
    83: +3, 84: +2, 85: -1, 86: 0, 87: +1, 88: +2, 89: +3, 90: +4,
    91: +5, 92: +6, 93: +5, 94: +4, 95: +3, 96: +3, 97: +3, 98: +3,
    99: +3, 100: +3, 101: +3, 102: +2, 103: +3, 104: +4, 105: +5, 106: +6,
    107: +7, 108: +4, 109: +3, 110: +2, 111: +1, 112: +2, 113: +3, 114: +2,
    115: +3, 116: +2, 117: -1, 118: 0,
}


class FixedCharges(nn.Module):
    def __init__(
        self,
        charge_dict: Dict[int, float] = typical_charge,
        normalization_factor: float = 0.5,
    ):
        super().__init__()
        self.charge_dict = charge_dict
        self.normalization_factor = normalization_factor

    def forward(self, atomic_numbers: torch.Tensor) -> torch.Tensor:
        charge = torch.tensor(
            [self.charge_dict[int(z.item())] for z in atomic_numbers],
            device=atomic_numbers.device,
            dtype=torch.get_default_dtype(),
        )
        return charge * self.normalization_factor

