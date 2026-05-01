"""Error propagation analysis via Lipschitz constants.

Theory: for layer ℓ with quantization error ε_ℓ, the error at the output
is amplified by Π_{k>ℓ} L_k, where L_k is the Lipschitz constant of layer k.

L_k ≈ spectral norm of weight matrices in that layer.
"""

import torch
import torch.nn as nn
from src.analysis.condition import power_iteration


def estimate_layer_lipschitz(layer) -> float:
    """
    Estimate the Lipschitz constant of a transformer layer.

    Uses the maximum spectral norm across all linear projections in the layer
    as a proxy for the Lipschitz constant.

    For transformer layers: L ≈ max(σ_max(W_Q), σ_max(W_K), σ_max(W_V),
                                      σ_max(W_O), σ_max(W_gate), σ_max(W_up),
                                      σ_max(W_down))
    """
    max_sigma = 0.0
    for name, param in layer.named_parameters():
        if param.dim() >= 2 and 'weight' in name:
            sigma = power_iteration(param.data, n_iter=3)
            max_sigma = max(max_sigma, sigma)
    return max_sigma


def compute_propagation_factors(model) -> dict:
    """
    For each layer, compute the error propagation factor:
      factor_ℓ = Π_{k>ℓ} L_k

    Layer N (last) has factor = 1.0 (no downstream amplification).
    Layer 0 (first) has the largest factor.

    Returns dict: layer_idx → {lipschitz, propagation_factor}
    """
    layers = model.model.layers
    n_layers = len(layers)
    lipschitz_constants = [estimate_layer_lipschitz(l) for l in layers]

    # Compute from output to input
    prop_factors = [1.0] * n_layers
    accumulated = 1.0
    for i in range(n_layers - 1, -1, -1):
        prop_factors[i] = accumulated
        accumulated *= lipschitz_constants[i]

    return {
        f'layer_{i}': {
            'lipschitz': lipschitz_constants[i],
            'propagation_factor': prop_factors[i],
        }
        for i in range(n_layers)
    }


def predict_quantization_output_error(
    model,
    layer_errors: dict[int, float],  # layer_idx → estimated quantization error
) -> dict[int, float]:
    """
    Predict output error from per-layer quantization errors using Lipschitz
    propagation model.

    Predicted output error at layer ℓ:
      E_out(ℓ) = ε_ℓ × Π_{k>ℓ} L_k
    """
    prop = compute_propagation_factors(model)
    predicted = {}
    for layer_idx, err in layer_errors.items():
        factor = prop[f'layer_{layer_idx}']['propagation_factor']
        predicted[layer_idx] = err * factor
    return predicted


def compare_propagation_strategies(model, fp_errors: dict, q_errors: dict) -> dict:
    """
    Compare different mixed-precision strategies:
    - Uniform: same bitwidth everywhere
    - Greedy: assign fewer bits to layers with small propagation factor
    - Lipschitz-aware: proportional to log(propagation_factor)

    Returns predicted output errors for each strategy.
    """
    prop = compute_propagation_factors(model)
    n_layers = len(model.model.layers)

    strategies = {}

    # Strategy 1: Uniform quantization
    uniform_error = sum(fp_errors.values()) / n_layers
    strategies['uniform'] = {
        'total_output_error': uniform_error * sum(
            prop[f'layer_{i}']['propagation_factor'] for i in range(n_layers)
        )
    }

    # Strategy 2: Lipschitz-aware — assign FP8 to sensitive layers, FP4 to insensitive
    factors = [prop[f'layer_{i}']['propagation_factor'] for i in range(n_layers)]
    median_factor = sorted(factors)[n_layers // 2]

    # Sensitive layers (> median propagation factor) get FP8
    # Insensitive layers get FP4
    fp8_error = min(fp_errors.values())
    fp4_error = max(fp_errors.values())
    total = 0.0
    for i in range(n_layers):
        err = fp8_error if factors[i] > median_factor else fp4_error
        total += err * factors[i]
    strategies['lipschitz_aware'] = {'total_output_error': total}

    return strategies
