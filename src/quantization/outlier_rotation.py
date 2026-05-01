"""DuQuant++ style outlier-aware fine-grained rotation for FP4 PTQ.

Key insight: activation outliers inflate shared block scales in microscaling
formats, compressing the dynamic range of remaining elements. By detecting
outlier channels and applying per-channel scaling or Hadamard rotation aligned
with the block size, we can reduce quantization error.

Reference: Lin et al., "DuQuant++: Fine-grained Rotation Enhances
Microscaling FP4 Quantization", arXiv:2604.17789, 2026.
"""

import math
import torch
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════
# Outlier detection
# ═══════════════════════════════════════════════════════════

def per_channel_kurtosis(W: torch.Tensor) -> torch.Tensor:
    """
    Compute excess kurtosis per output channel.

    High kurtosis (> 3) indicates heavy tails / outliers.
    Returns tensor of shape (out_features,).
    """
    W_f = W.float()
    mean = W_f.mean(dim=1, keepdim=True)
    centered = W_f - mean
    var = centered.pow(2).mean(dim=1)
    m4 = centered.pow(4).mean(dim=1)
    return m4 / (var.pow(2) + 1e-12) - 3  # Excess kurtosis


def detect_outlier_channels(W: torch.Tensor, kurtosis_threshold: float = 5.0) -> torch.Tensor:
    """
    Return boolean mask of outlier output channels.

    Args:
        W: weight matrix (out_features, in_features)
        kurtosis_threshold: channels with excess kurtosis > threshold are outliers
    """
    kurt = per_channel_kurtosis(W)
    return kurt > kurtosis_threshold


# ═══════════════════════════════════════════════════════════
# Per-channel outlier-aware scaling
# ═══════════════════════════════════════════════════════════

class OutlierAwareScaler:
    """
    Per-channel scaling that compresses outlier channels before quantization.

    For each outlier channel, computes a per-channel scale factor that
    brings the channel's distribution closer to the overall distribution,
    reducing the impact of outliers on quantization error.
    """

    def __init__(self, kurtosis_threshold: float = 5.0):
        self.threshold = kurtosis_threshold

    def compute_scales(self, W: torch.Tensor) -> torch.Tensor:
        """
        Compute per-channel scale factors. Outlier channels get smaller
        scales to compress their range.
        """
        outlier_mask = detect_outlier_channels(W, self.threshold)
        global_std = W.std().item()

        scales = torch.ones(W.size(0), device=W.device, dtype=W.dtype)
        for ch in outlier_mask.nonzero(as_tuple=True)[0]:
            ch_std = W[ch].std().item()
            if ch_std > global_std * 1.5:
                scales[ch] = global_std / ch_std

        return scales

    def apply(self, W: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply per-channel scaling. Returns (scaled_W, scales).
        """
        scales = self.compute_scales(W)
        W_scaled = W * scales.unsqueeze(1)
        return W_scaled, scales


# ═══════════════════════════════════════════════════════════
# Block-aligned Hadamard rotation
# ═══════════════════════════════════════════════════════════

def block_hadamard_transform(x: torch.Tensor, block_size: int = 32,
                              dim: int = -1) -> torch.Tensor:
    """
    Apply Hadamard transform within blocks of size `block_size` along `dim`.

    This aligns the rotation granularity with MXFP4's block size,
    avoiding the cross-block variance issue that plagues global Hadamard.

    Unlike the slow Python loop in hadamard.py, this uses a more
    efficient block-wise implementation.

    Reference: DuQuant++ Section 3.2.
    """
    orig_shape = x.shape
    n = orig_shape[dim]

    # Pad to multiple of block_size
    pad = (block_size - n % block_size) % block_size
    if pad > 0:
        pad_shape = list(orig_shape)
        pad_shape[dim] = pad
        x = torch.cat([x, torch.zeros(pad_shape, device=x.device, dtype=x.dtype)], dim=dim)
        n += pad

    # Reshape to expose blocks
    new_shape = list(orig_shape)
    new_shape[dim] = n // block_size
    new_shape.insert(dim + 1, block_size)
    x_blocks = x.reshape(new_shape)

    # Apply Hadamard within each block
    h = 1
    while h < block_size:
        for i in range(0, block_size, h * 2):
            u = x_blocks[..., i].clone()
            v = x_blocks[..., i + h].clone()
            x_blocks[..., i] = u + v
            x_blocks[..., i + h] = u - v
        h *= 2

    # Reshape back
    x = x_blocks.reshape(x.shape)
    # Trim padding
    if pad > 0:
        x = x.narrow(dim, 0, orig_shape[dim])

    # Scale (per-block normalization)
    x = x / math.sqrt(block_size)
    return x


# ═══════════════════════════════════════════════════════════
# DuQuant++ quantizer
# ═══════════════════════════════════════════════════════════

class DuQuantStyleQuantizer:
    """
    DuQuant++ inspired quantization pipeline:

    1. Detect outlier channels via kurtosis
    2. Apply per-channel scaling to compress outliers
    3. Apply block-aligned Hadamard rotation (B=32)
    4. Quantize to target grid (FP4 E2M1 / NF4)
    5. Inverse rotation + inverse scaling
    """

    def __init__(self, grid: torch.Tensor, block_size: int = 32,
                 kurtosis_threshold: float = 5.0):
        self.grid = grid
        self.block_size = block_size
        self.scaler = OutlierAwareScaler(kurtosis_threshold)
        self.use_rotation = True
        self.use_outlier_scale = True

    def quantize(self, W: torch.Tensor, use_rotation: bool = True,
                 use_outlier_scale: bool = True) -> torch.Tensor:
        """Full DuQuant++ quantization pipeline."""
        W_q = W.clone()
        scales = None

        # Step 1: Per-channel outlier scaling
        if use_outlier_scale and W_q.dim() == 2:
            W_q, scales = self.scaler.apply(W_q)

        # Step 2: Block-aligned Hadamard rotation (with torch.no_grad for stability)
        if use_rotation and W_q.dim() == 2:
            with torch.no_grad():
                W_q = block_hadamard_transform(W_q.clone(), self.block_size, dim=1)

        # Step 3: Quantize to target grid
        with torch.no_grad():
            amax = W_q.abs().max()
            if amax > 0:
                scale = self.grid[-1] / amax
                W_q = self._quantize_to_grid(W_q * scale) / scale

        # Step 4: Inverse rotation
        if use_rotation and W_q.dim() == 2:
            with torch.no_grad():
                W_q = block_hadamard_transform(W_q.clone(), self.block_size, dim=1)

        # Step 5: Inverse scaling
        if use_outlier_scale and scales is not None:
            W_q = W_q / scales.unsqueeze(1)

        return W_q

    def _quantize_to_grid(self, x: torch.Tensor) -> torch.Tensor:
        """Round to nearest grid point."""
        grid = self.grid.to(x.device)
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
# Analysis tools
# ═══════════════════════════════════════════════════════════

def analyze_outlier_channels(model) -> dict:
    """Analyze which layers have the most outlier channels."""
    results = {}
    for name, param in model.named_parameters():
        if param.dim() < 2:
            continue
        kurt = per_channel_kurtosis(param.data)
        outliers = (kurt > 5.0).sum().item()
        total = kurt.numel()
        if total > 0:
            results[name] = {
                'outlier_channels': outliers,
                'total_channels': total,
                'outlier_ratio': outliers / total,
                'max_kurtosis': kurt.max().item(),
            }
    return results
