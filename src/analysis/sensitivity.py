"""Per-layer quantization sensitivity analysis.

Combines condition number analysis and Lipschitz propagation to identify
which layers are most sensitive to quantization.
"""

import torch
from src.analysis.condition import estimate_condition_number, compute_all_condition_numbers
from src.analysis.lipschitz import compute_propagation_factors


def per_layer_sensitivity_report(model, quantizer) -> dict:
    """
    Generate a comprehensive per-layer sensitivity report.

    For each layer, compute:
    - condition number κ(W) for each weight matrix
    - Lipschitz constant L
    - Error propagation factor Π_{k>ℓ} L_k
    - Quantization MSE (actual)
    - Predicted output impact: κ(W) × MSE × propagation_factor

    Returns sorted list of layers by predicted impact.
    """
    cond_numbers = compute_all_condition_numbers(model)
    prop_factors = compute_propagation_factors(model)

    report = []
    for i, layer in enumerate(model.model.layers):
        layer_name = f'layer_{i}'
        layer_type = model.config.layer_types[i]

        # Average condition number for this layer's weights
        layer_cond = 0.0
        weight_count = 0
        for name, kappa in cond_numbers.items():
            if f'layers.{i}.' in name:
                layer_cond += kappa
                weight_count += 1
        avg_kappa = layer_cond / max(weight_count, 1)

        # Quantize and measure error
        total_mse = 0.0
        for name, param in layer.named_parameters():
            if param.dim() >= 2:
                W_q = quantizer.quantize(param.data)
                mse = ((W_q - param.data) ** 2).mean().item()
                total_mse += mse

        prop = prop_factors[layer_name]
        predicted_impact = avg_kappa * total_mse * prop['propagation_factor']

        report.append({
            'layer_idx': i,
            'layer_type': layer_type,
            'avg_kappa': avg_kappa,
            'lipschitz': prop['lipschitz'],
            'propagation_factor': prop['propagation_factor'],
            'quantization_mse': total_mse,
            'predicted_impact': predicted_impact,
        })

    # Sort by predicted impact (descending)
    report.sort(key=lambda x: x['predicted_impact'], reverse=True)
    return report


def suggest_mixed_precision(report: list[dict], fp8_threshold: float = 0.67) -> dict[int, str]:
    """
    Based on sensitivity report, suggest per-layer precision.

    By default, top 67% most sensitive layers → FP8, remainder → FP4.
    Adjust fp8_threshold to control the fraction assigned to FP8.
    """
    n = len(report)
    cutoff = int(n * fp8_threshold)
    suggestion = {}
    for i, entry in enumerate(report):
        suggestion[entry['layer_idx']] = 'fp8' if i < cutoff else 'fp4'
    return dict(sorted(suggestion.items()))
