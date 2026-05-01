"""Stochastic rounding utilities with limited-precision random numbers."""

import torch


def stochastic_round(x: torch.Tensor, target_dtype=None) -> torch.Tensor:
    """
    Stochastic rounding of tensor to integer values.

    P(ceil(x)) = x - floor(x)
    P(floor(x)) = ceil(x) - x

    This provides an unbiased estimator: E[SR(x)] = x.

    Reference: El Arar et al., "Limited-Precision Stochastic Rounding", 2026.
    """
    x_floor = x.floor()
    prob_up = (x - x_floor)
    rand = torch.rand_like(x)
    return x_floor + (rand < prob_up).to(x.dtype)


def stochastic_round_fp(x: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    """
    Stochastic round tensor x to the nearest values in `levels`.

    Args:
        x: input tensor
        levels: sorted tensor of allowed quantization levels
    """
    x_flat = x.reshape(-1)
    idx = torch.searchsorted(levels, x_flat.abs())
    idx = idx.clamp(0, len(levels) - 1)

    lower = levels[(idx - 1).clamp(0)].to(x.device)
    upper = levels[idx.clamp(0, len(levels) - 1)].to(x.device)

    gap = upper - lower
    safe_gap = gap.clamp(min=1e-12)
    prob_up = ((x_flat.abs() - lower) / safe_gap).clamp(0, 1)

    rand = torch.rand_like(x_flat)
    result_abs = torch.where(rand < prob_up, upper, lower)
    return torch.sign(x) * result_abs.reshape(x.shape)


def compare_rounding_error(x: torch.Tensor, n_trials: int = 1000) -> dict:
    """
    Empirically compare deterministic (round-to-nearest) vs stochastic rounding
    for a given tensor x.

    Returns statistics on bias and variance.
    """
    x = x.detach()
    rn_result = x.round()

    # Stochastic: average over many trials
    sr_results = []
    for _ in range(n_trials):
        sr_results.append(stochastic_round(x.clone()))
    sr_mean = torch.stack(sr_results).mean(0)
    sr_var = torch.stack(sr_results).var(0)

    return {
        'round_nearest_bias': (rn_result - x).mean().item(),
        'round_nearest_mse': ((rn_result - x) ** 2).mean().item(),
        'stochastic_bias': (sr_mean - x).mean().item(),
        'stochastic_mse': ((sr_mean - x) ** 2).mean().item(),
        'stochastic_var': sr_var.mean().item(),
        'n_trials': n_trials,
    }
