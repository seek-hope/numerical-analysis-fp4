"""FP8/FP4 quantization core. Supports deterministic and stochastic rounding."""

import torch
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════
# FP Format Definitions
# ═══════════════════════════════════════════════════════════════

FP_FORMAT_SPECS = {
    # name: {exp_bits, mantissa_bits, bias, emax, has_subnormals}
    'fp8_e4m3':  {'exp': 4, 'man': 3, 'bias': 7,  'emax': 8},
    'fp8_e5m2':  {'exp': 5, 'man': 2, 'bias': 15, 'emax': 16},
    'fp4_e2m1':  {'exp': 2, 'man': 1, 'bias': 1,  'emax': 2},
}

def _build_fp_grid(exp_bits, man_bits, bias):
    """Build sorted list of all representable positive values for a given FP format.

    Handles both normal and subnormal (denormal) numbers:
    - Normal (e > 0): 2^(e-bias) × (1 + m/2^man_bits)
    - Subnormal (e = 0): 2^(1-bias) × m/2^man_bits
    """
    num_exp = 1 << exp_bits
    num_man = 1 << man_bits
    values = set()
    values.add(0.0)

    for e in range(num_exp):
        if e == 0:
            # Subnormal: 2^(1-bias) × m/2^man_bits
            scale = 2.0 ** (1 - bias)
            for m in range(num_man):
                if m > 0:  # m=0 already covered by values.add(0.0)
                    values.add(scale * m / num_man)
        else:
            exp_val = 2.0 ** (e - bias)
            for m in range(num_man):
                values.add(exp_val * (1.0 + m / num_man))
    return torch.tensor(sorted(values), dtype=torch.float32)


# ═══════════════════════════════════════════════════════════════
# Quantizer
# ═══════════════════════════════════════════════════════════════

class FPQuantizer:
    """
    Simulated FP8/FP4 quantizer.

    Supports per-tensor and per-channel scaling, deterministic and
    stochastic rounding (unbiased for gradient accumulation).
    """

    def __init__(self, fmt: str = 'fp8_e4m3', per_channel: bool = True):
        if fmt not in FP_FORMAT_SPECS and fmt != 'fp8_e4m3fn':
            raise ValueError(f"Unknown format: {fmt}. Options: {list(FP_FORMAT_SPECS.keys())}")

        self.fmt = fmt
        self.per_channel = per_channel

        if fmt == 'fp8_e4m3fn' and hasattr(torch, 'float8_e4m3fn'):
            self._use_native = True
            self._native_dtype = torch.float8_e4m3fn
        else:
            self._use_native = False
            self._spec = FP_FORMAT_SPECS.get(fmt, FP_FORMAT_SPECS['fp8_e4m3'])
            self._grid = _build_fp_grid(
                self._spec['exp'], self._spec['man'], self._spec['bias']
            )
            self._num_levels = len(self._grid)

    def quantize(self, x: torch.Tensor, stochastic: bool = False,
                 scale: torch.Tensor | None = None) -> torch.Tensor:
        """
        Quantize tensor x to the target FP format.

        Args:
            x: input tensor
            stochastic: if True, use stochastic rounding (unbiased)
            scale: pre-computed scale factor. If None, auto-compute.
        """
        if self._use_native:
            return x.to(self._native_dtype).to(x.dtype)
        return self._simulate_quantize(x, stochastic, scale)

    def _simulate_quantize(self, x: torch.Tensor, stochastic: bool,
                           scale: torch.Tensor | None) -> torch.Tensor:
        device = x.device
        grid = self._grid.to(device)

        # Compute scale
        if scale is None:
            if self.per_channel and x.dim() >= 2:
                # Per-channel: one scale per output channel
                # Weight matrices: (out_features, in_features) → scale (out_features, 1)
                amax = x.detach().abs().amax(dim=-1, keepdim=True)
                amax = amax.clamp(min=1e-12)
                scale = grid[-1] / amax
            else:
                # Per-tensor: single scale for entire tensor
                amax = x.detach().abs().max()
                if amax == 0:
                    return x
                scale = grid[-1] / amax

        # Scale to grid space
        x_scaled = x * scale

        if stochastic:
            x_quant = self._stochastic_round(x_scaled, grid)
        else:
            x_quant = self._round_to_nearest(x_scaled, grid)

        # Dequantize
        return x_quant / scale

    def _round_to_nearest(self, x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        """Round to nearest grid point."""
        x_abs = x.abs()
        # Find nearest grid point by binary search approximation
        # For speed: quantize to nearest integer and clamp
        idx = torch.searchsorted(grid, x_abs.clamp(0, grid[-1]))
        idx = idx.clamp(0, len(grid) - 1)

        # Candidates: grid[idx-1] and grid[idx]
        lower = grid[(idx - 1).clamp(0)].to(x.device)
        upper = grid[idx.clamp(0, len(grid) - 1)].to(x.device)

        # Choose nearer
        dist_lower = (x_abs - lower).abs()
        dist_upper = (upper - x_abs).abs()
        x_quant_abs = torch.where(dist_lower <= dist_upper, lower, upper)

        return torch.sign(x) * x_quant_abs

    def _stochastic_round(self, x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        """
        Stochastic rounding: P(round up) = (x - lower) / (upper - lower).

        This provides an unbiased estimator of the true value.
        """
        x_abs = x.abs()
        x_sign = torch.sign(x)

        idx = torch.searchsorted(grid, x_abs.clamp(0, grid[-1]))
        idx = idx.clamp(0, len(grid) - 1)

        lower = grid[(idx - 1).clamp(0)].to(x.device)
        upper = grid[idx.clamp(0, len(grid) - 1)].to(x.device)

        # Probability of rounding up
        gap = upper - lower
        safe_gap = gap.clamp(min=1e-12)
        prob_up = (x_abs - lower) / safe_gap
        prob_up = prob_up.clamp(0, 1)

        # Stochastic choice
        rand = torch.rand_like(x_abs)
        x_quant_abs = torch.where(rand < prob_up, upper, lower)

        return x_sign * x_quant_abs


# ═══════════════════════════════════════════════════════════════
# QAT Helpers
# ═══════════════════════════════════════════════════════════════

def apply_weight_quantization(model, quantizer: FPQuantizer, stochastic: bool = False):
    """
    Quantize all quantizable weights of the model in-place during forward pass.
    Used for Group A (PTQ evaluation) — quantize once after training.
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.dim() >= 2 and any(k in name for k in ('proj', 'weight')):
                param.data = quantizer.quantize(param.data, stochastic=stochastic)


class QuantizedLinear(torch.autograd.Function):
    """
    Forward: quantize weights to FP8/FP4, then compute linear.
    Backward: Straight-Through Estimator (gradient passes through quantization).
    """

    @staticmethod
    def forward(ctx, weight, bias, x, quantizer, stochastic):
        w_q = quantizer.quantize(weight, stochastic=stochastic)
        ctx.save_for_backward(weight, x)
        ctx.has_bias = bias is not None
        return F.linear(x, w_q, bias)

    @staticmethod
    def backward(ctx, grad_output):
        weight, x = ctx.saved_tensors

        grad_weight = grad_bias = grad_x = None

        if ctx.needs_input_grad[0]:
            grad_out_2d = grad_output.reshape(-1, grad_output.shape[-1])
            x_2d = x.reshape(-1, x.shape[-1])
            grad_weight = grad_out_2d.t() @ x_2d

        if ctx.needs_input_grad[1] and ctx.has_bias:
            grad_bias = grad_output.sum(dim=tuple(range(grad_output.dim() - 1)))

        if ctx.needs_input_grad[2]:
            grad_x = grad_output @ weight

        return grad_weight, grad_bias, grad_x, None, None


def make_qat_forward_hook(quantizer: FPQuantizer, stochastic: bool = False):
    """
    Returns a forward pre-hook that quantizes weights before each linear layer.

    Usage:
        hook = make_qat_forward_hook(FPQuantizer('fp8_e4m3'))
        layer.register_forward_pre_hook(hook)
    """
    def hook(module, input):
        if hasattr(module, 'weight') and module.weight is not None:
            with torch.no_grad():
                module._qat_weight_cache = module.weight.data.clone()
                module.weight.data = quantizer.quantize(
                    module.weight.data, stochastic=stochastic)
    return hook


def make_qat_forward_hook_restore():
    """Returns a forward hook that restores original weights after forward."""
    def hook(module, input, output):
        if hasattr(module, '_qat_weight_cache'):
            module.weight.data = module._qat_weight_cache
            del module._qat_weight_cache
    return hook
