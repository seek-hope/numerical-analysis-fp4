"""NF4 and MXFP4 quantizers compatible with QAT training interface."""

import math
import torch
from src.quantization.fp4_grids import (
    FP4_E2M1_GRID, NF4_GRID, GridQuantizer, MXFP4Quantizer,
)
from src.quantization.fp_quantizer import FPQuantizer


class GridBasedFPQuantizer:
    """
    Wraps a GridQuantizer to expose the same .quantize(x, stochastic)
    interface as FPQuantizer, making it plug-and-play with QAT training.
    """

    def __init__(self, grid: torch.Tensor, name: str = 'custom'):
        self._grid_quantizer = GridQuantizer(grid)
        self.name = name
        self.fmt = name  # for compatibility

    def quantize(self, x: torch.Tensor, stochastic: bool = False,
                 scale: torch.Tensor | None = None) -> torch.Tensor:
        return self._grid_quantizer.quantize(x)


class StochasticGridQuantizer(GridBasedFPQuantizer):
    """Grid-based quantizer with stochastic rounding support."""

    def quantize(self, x: torch.Tensor, stochastic: bool = False,
                 scale: torch.Tensor | None = None) -> torch.Tensor:
        if not stochastic:
            return self._grid_quantizer.quantize(x)

        # Stochastic rounding version
        grid = self._grid_quantizer.grid.to(x.device)
        amax = x.abs().max()
        if amax == 0:
            return x

        scale_factor = grid[-1] / (amax + 1e-12)
        x_scaled = x * scale_factor
        x_abs = x_scaled.abs()
        x_sign = torch.sign(x_scaled)

        idx = torch.searchsorted(grid, x_abs.clamp(0, grid[-1]))
        idx = idx.clamp(0, len(grid) - 1)

        lower = grid[(idx - 1).clamp(0)]
        upper = grid[idx.clamp(0, len(grid) - 1)]

        gap = upper - lower
        safe_gap = gap.clamp(min=1e-12)
        prob_up = ((x_abs - lower) / safe_gap).clamp(0, 1)

        rand = torch.rand_like(x_abs)
        x_q_abs = torch.where(rand < prob_up, upper, lower)

        return x_sign * x_q_abs / scale_factor


class MXFP4StochasticQuantizer:
    """MXFP4 block-scaling quantizer with optional stochastic rounding."""

    def __init__(self, block_size: int = 32):
        self._mx_quantizer = MXFP4Quantizer(block_size)
        self.name = f'mxfp4_b{block_size}'
        self.fmt = self.name

    def quantize(self, x: torch.Tensor, stochastic: bool = False,
                 scale: torch.Tensor | None = None) -> torch.Tensor:
        # For simplicity, use deterministic MXFP4
        # Stochastic could be added to MXFP4Quantizer in the future
        return self._mx_quantizer.quantize(x)
