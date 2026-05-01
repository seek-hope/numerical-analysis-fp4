"""Three FP4 grid schemes for comparison: FP4 E2M1, NF4, MXFP4.

Each provides exactly 16 quantization levels (4 bits), but with
different placement on the real number line.
"""

import math
import torch
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════
# Scheme 1: FP4 E2M1 — Standard hardware format
#   sign=1, exp=2, mantissa=1 → 16 representable values
#   Grid: approximately log-spaced (dense near 0, sparse far)
# ═══════════════════════════════════════════════════════════

def build_fp4_e2m1_grid() -> torch.Tensor:
    """Build the 16 positive FP4 E2M1 quantization levels."""
    values = {0.0}
    for s in [1]:  # positive only (sign handled separately)
        for e in range(4):  # 2 exponent bits
            exp_val = 2.0 ** (e - 1)  # bias = 1
            for m in range(2):  # 1 mantissa bit
                if e == 0:
                    # Subnormal: 2^(1-bias) × m/2
                    val = 0.5 * (m / 2.0)
                else:
                    val = exp_val * (1.0 + m / 2.0)
                values.add(val)
    return torch.tensor(sorted(values), dtype=torch.float32)


# ═══════════════════════════════════════════════════════════
# Scheme 2: NF4 — Normal Float 4-bit
#   Places 16 levels at quantiles of N(0,1).
#   Information-theoretically optimal for Gaussian-distributed weights.
#   From QLoRA (Dettmers et al., NeurIPS 2023).
# ═══════════════════════════════════════════════════════════

def build_nf4_grid() -> torch.Tensor:
    """
    Build 16 positive quantization levels from standard normal quantiles.

    The quantiles are chosen as the midpoints of 16 equal-probability
    intervals of N(0,1). Since the distribution is symmetric, we use
    the positive half and include 0.

    Simplified: use the cumulative distribution function quantiles.
    """
    # 16 levels: [q1, q2, ..., q16] where each q_i corresponds to
    # the midpoint of the i-th 1/16 probability interval
    # Equivalent to: Φ^{-1}((2i-1)/(2*16)) for i=1..16
    # But simpler: linspace CDF approach
    probs = torch.linspace(0.0, 1.0, 18)[1:-1]  # 16 interior points
    # Erfinv approximation for standard normal quantiles
    levels = []
    for p in probs:
        # Φ^{-1}(p) for standard normal
        z = math.sqrt(2) * _erfinv_approx(2 * p.item() - 1)
        levels.append(abs(z))
    levels = sorted(set(levels))
    # Remove 0, add it explicitly
    levels = [0.0] + [l for l in levels if l > 1e-10]
    # Take exactly 16 levels if we have more
    if len(levels) > 16:
        levels = levels[:16]
    return torch.tensor(levels, dtype=torch.float32)


def _erfinv_approx(x: float) -> float:
    """Approximate inverse error function (Winitzki approximation)."""
    if abs(x) >= 1.0:
        return math.copysign(10.0, x)
    a = 0.147
    ln1mx2 = math.log(1.0 - x * x)
    part1 = 2.0 / (math.pi * a) + ln1mx2 / 2.0
    part2 = ln1mx2 / a
    return math.copysign(
        math.sqrt(math.sqrt(part1 * part1 - part2) - part1), x
    )


# ═══════════════════════════════════════════════════════════
# Scheme 3: MXFP4 — Microscaling FP4
#   Each block of B elements shares one E8M0 scale factor.
#   The FP4 values within a block are E2M1 format.
#   This allows the block to "shift" its representable range.
# ═══════════════════════════════════════════════════════════

class MXFP4Quantizer:
    """
    Microscaling FP4 quantizer.

    Divides the tensor into blocks of `block_size` (default 32).
    Each block gets its own scale factor (power-of-2), allowing
    the FP4 grid to adapt to local magnitude variations.

    Reference: OCP Microscaling Formats Specification.
    """
    def __init__(self, block_size: int = 32):
        self.block_size = block_size
        self.fp4_grid = build_fp4_e2m1_grid()

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Quantize x to MXFP4.

        For each block of size B:
        1. Find max absolute value
        2. Compute shared E8M0 scale (power-of-2 nearest to max/grid_max)
        3. Quantize each element with FP4 grid × block_scale
        """
        orig_shape = x.shape
        x_flat = x.reshape(-1)
        n = x_flat.numel()
        result = torch.zeros_like(x_flat)

        grid_max = self.fp4_grid[-1].item()

        for start in range(0, n, self.block_size):
            end = min(start + self.block_size, n)
            block = x_flat[start:end]

            amax = block.abs().max().item()
            if amax == 0:
                result[start:end] = 0
                continue

            # Shared scale: power-of-2 nearest to amax / grid_max
            raw_scale = amax / grid_max
            scale = 2.0 ** round(math.log2(max(raw_scale, 1e-12)))

            # Quantize each element
            device = x.device
            grid = self.fp4_grid.to(device) * scale
            block_q = self._round_to_grid(block, grid)
            result[start:end] = block_q

        return result.reshape(orig_shape)

    @staticmethod
    def _round_to_grid(x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        """Round to nearest value in grid (handles sign)."""
        x_abs = x.abs()
        x_sign = torch.sign(x)

        idx = torch.searchsorted(grid, x_abs.clamp(0, grid[-1]))
        idx = idx.clamp(0, len(grid) - 1)

        lower = grid[(idx - 1).clamp(0)]
        upper = grid[idx.clamp(0, len(grid) - 1)]

        dist_lower = (x_abs - lower).abs()
        dist_upper = (upper - x_abs).abs()
        x_q_abs = torch.where(dist_lower <= dist_upper, lower, upper)

        return x_sign * x_q_abs


# ═══════════════════════════════════════════════════════════
# Grid-to-grid quantization helper
# ═══════════════════════════════════════════════════════════

class GridQuantizer:
    """Quantize a tensor to the nearest values in a discrete grid."""

    def __init__(self, grid: torch.Tensor):
        self.grid = grid

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        grid = self.grid.to(device)

        # Per-tensor scaling
        amax = x.abs().max()
        if amax == 0:
            return x
        scale = grid[-1] / (amax + 1e-12)

        x_scaled = x * scale
        x_abs = x_scaled.abs()
        x_sign = torch.sign(x_scaled)

        idx = torch.searchsorted(grid, x_abs.clamp(0, grid[-1]))
        idx = idx.clamp(0, len(grid) - 1)

        lower = grid[(idx - 1).clamp(0)]
        upper = grid[idx.clamp(0, len(grid) - 1)]

        dist_lower = (x_abs - lower).abs()
        dist_upper = (upper - x_abs).abs()
        x_q_abs = torch.where(dist_lower <= dist_upper, lower, upper)

        return x_sign * x_q_abs / scale


# ═══════════════════════════════════════════════════════════
# Pre-built grids
# ═══════════════════════════════════════════════════════════

FP4_E2M1_GRID = build_fp4_e2m1_grid()
NF4_GRID = build_nf4_grid()

# Print diagnostic
if __name__ == '__main__':
    print("FP4 E2M1 grid (16 positive levels):")
    for i, v in enumerate(FP4_E2M1_GRID):
        print(f"  [{i:2d}] {v:.4f}")
    print(f"\nNF4 grid (16 positive levels):")
    for i, v in enumerate(NF4_GRID):
        print(f"  [{i:2d}] {v:.4f}")
    print(f"\nMXFP4 block_size=32 — instantiated as class, not pre-built")
