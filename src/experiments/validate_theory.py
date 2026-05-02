#!/usr/bin/env python3
"""
Phase 1: Validate numerical analysis predictions of quantization error.

Tests three theoretical predictions:
  P1: κ(W) correlates with per-layer quantization MSE
  P2: Early layers' errors are amplified more (Lipschitz propagation)
  P3: Combined prediction E_ℓ = ε_ℓ × Π_{k>ℓ} L_k correlates with PPL


Usage:
    ./remote_python.sh src/experiments/validate_theory.py \
        --checkpoint checkpoints/scaled_fp16_baseline/model.pt \
        --data_dir data/real_tiers
"""

import json, copy, argparse
import torch
from src.model.config import MicroGemmaFPConfig
from src.model.transformer import MicroGemmaFPForCausalLM
from src.quantization.fp_quantizer import FPQuantizer
from src.analysis.condition import estimate_condition_number
from src.analysis.lipschitz import compute_propagation_factors, estimate_layer_lipschitz
from src.experiments.training_utils import (
    get_dataloader, evaluate_perplexity, load_checkpoint,
)


@torch.no_grad()
def quantize_single_layer(model, layer_idx, quantizer):
    """Quantize only one layer's weights, keep rest FP16."""
    layer = model.model.layers[layer_idx]
    mse_total = 0.0
    n_params = 0
    for name, param in layer.named_parameters():
        if param.dim() >= 2:
            W_fp = param.data.clone()
            W_q = quantizer.quantize(param.data)
            mse_total += ((W_q - W_fp) ** 2).sum().item()
            n_params += param.numel()
            param.data = W_q
    return mse_total / max(n_params, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoints/scaled_fp16_baseline/model.pt')
    parser.add_argument('--data_dir', default=None)
    parser.add_argument('--format', default='fp4_e2m1')
    parser.add_argument('--max_eval_steps', type=int, default=100)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load baseline
    config = MicroGemmaFPConfig()
    model = MicroGemmaFPForCausalLM(config).to(device)
    load_checkpoint(model, None, args.checkpoint, device)
    model.eval()

    # FP16 baseline PPL
    loader = get_dataloader(8, 512, args.max_eval_steps, data_dir=args.data_dir)
    fp16_ppl = evaluate_perplexity(model, loader, device, args.max_eval_steps)
    print(f"FP16 baseline PPL: {fp16_ppl:.2f}")

    n_layers = len(model.model.layers)
    quantizer = FPQuantizer(args.format, per_channel=True)

    # ── P1: Compute κ(W) for all layers ──
    print(f"\n{'='*60}")
    print("P1: Condition numbers per layer")
    kappa_per_layer = {}
    for i, layer in enumerate(model.model.layers):
        kappas = []
        for name, param in layer.named_parameters():
            if param.dim() >= 2:
                kappas.append(estimate_condition_number(param.data, n_iter=3))
        kappa_per_layer[i] = max(kappas) if kappas else 0
        lt = model.config.layer_types[i]
        print(f"  layer {i:2d} ({lt:8s}): max κ = {kappa_per_layer[i]:.1f}")

    # ── P2: Lipschitz propagation factors ──
    print(f"\n{'='*60}")
    print("P2: Lipschitz propagation factors")
    prop = compute_propagation_factors(model)
    for i in range(n_layers):
        pf = prop[f'layer_{i}']
        print(f"  layer {i:2d}: L = {pf['lipschitz']:.2f}, "
              f"propagation = {pf['propagation_factor']:.2e}")

    # ── P3: Per-layer quantization experiment ──
    print(f"\n{'='*60}")
    print("P3: Per-layer quantization → actual PPL degradation")
    print(f"{'='*60}")

    results = []
    for i in range(n_layers):
        # Fresh model copy
        model_i = MicroGemmaFPForCausalLM(config).to(device)
        load_checkpoint(model_i, None, args.checkpoint, device)
        model_i.eval()

        # Quantize only layer i
        mse = quantize_single_layer(model_i, i, quantizer)

        # Measure PPL
        loader_i = get_dataloader(8, 512, args.max_eval_steps, data_dir=args.data_dir)
        ppl_i = evaluate_perplexity(model_i, loader_i, device, args.max_eval_steps)
        degradation = ppl_i - fp16_ppl

        # Predicted error
        kappa = kappa_per_layer[i]
        prop_factor = prop[f'layer_{i}']['propagation_factor']
        predicted = kappa * (mse ** 0.5) * prop_factor

        lt = model.config.layer_types[i]
        print(f"  layer {i:2d} ({lt:8s}): PPL={ppl_i:.2f} Δ={degradation:+.2f} "
              f"κ={kappa:.1f} prop={prop_factor:.2e} pred={predicted:.2e}")

        results.append({
            'layer': i, 'layer_type': lt,
            'kappa': kappa,
            'lipschitz': prop[f'layer_{i}']['lipschitz'],
            'propagation_factor': prop_factor,
            'quantization_mse': mse,
            'predicted_error': predicted,
            'ppl': ppl_i,
            'ppl_degradation': degradation,
        })
        del model_i
        torch.cuda.empty_cache()

    # ── Correlation analysis ──
    degradations = [r['ppl_degradation'] for r in results]
    predictions = [r['predicted_error'] for r in results]
    kappas = [r['kappa'] for r in results]
    prop_factors = [r['propagation_factor'] for r in results]

    # Pearson correlation
    def pearson_r(x, y):
        n = len(x)
        mx, my = sum(x)/n, sum(y)/n
        sx = (sum((v-mx)**2 for v in x)/n) ** 0.5
        sy = (sum((v-my)**2 for v in y)/n) ** 0.5
        if sx == 0 or sy == 0:
            return 0
        return sum((x[i]-mx)*(y[i]-my) for i in range(n)) / (n*sx*sy)

    r_kappa = pearson_r(kappas, degradations)
    r_prop = pearson_r(prop_factors, degradations)
    r_combined = pearson_r(predictions, degradations)

    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"  P1: corr(κ, PPL degradation) = {r_kappa:.3f}")
    print(f"  P2: corr(propagation, PPL degradation) = {r_prop:.3f}")
    print(f"  P3: corr(predicted, actual PPL) = {r_combined:.3f}")
    print(f"\n  Interpretation:")
    for name, r in [('P1: κ', r_kappa), ('P2: Lipschitz', r_prop),
                     ('P3: Combined', r_combined)]:
        if r > 0.7:
            status = "STRONG — theory validated"
        elif r > 0.4:
            status = "MODERATE — partially validated"
        else:
            status = "WEAK — theory needs refinement"
        print(f"    {name}: r={r:.3f} → {status}")

    # Save
    with open('checkpoints/theory_validation.json', 'w') as f:
        json.dump({
            'fp16_ppl': fp16_ppl,
            'correlations': {
                'kappa': r_kappa,
                'propagation': r_prop,
                'combined': r_combined,
            },
            'per_layer': results,
        }, f, indent=2)
    print(f"\nResults saved to checkpoints/theory_validation.json")


if __name__ == '__main__':
    main()
