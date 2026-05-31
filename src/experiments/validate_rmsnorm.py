#!/usr/bin/env python3
"""
Phase 1 (Revised): Validate RMSNorm's error-blocking effect.

Tests the hypothesis that RMSNorm prevents cross-layer error propagation.
If the theory is correct:
  - With RMSNorm: per-layer quantization errors stay local (P3 weak correlation)
  - Without RMSNorm: errors compound exponentially (P3 strong correlation)

Implementation: temporarily replace RMSNorm with Identity for specific layers,
quantize those layers, and measure whether errors propagate further downstream.

Usage:
    ./remote_python.sh src/experiments/validate_rmsnorm.py \
        --checkpoint checkpoints/scaled_fp16_baseline/model.pt \
        --data_dir data/real_tiers
"""

import json, copy, argparse
import torch
import torch.nn as nn
from src.model.config import MicroGemmaFPConfig
from src.model.transformer import MicroGemmaFPForCausalLM
from src.quantization.fp_quantizer import FPQuantizer
from src.analysis.condition import estimate_condition_number
from src.experiments.training_utils import (
    get_dataloader, evaluate_loss, load_checkpoint,
)


class IdentityNorm(nn.Module):
    """Replace RMSNorm: passes through unchanged (no normalization)."""
    def forward(self, x):
        return x


@torch.no_grad()
def replace_norms(model, layers_to_replace: set[int], norm_type='input'):
    """Replace RMSNorm with Identity in specified layers."""
    saved = {}
    for i, layer in enumerate(model.model.layers):
        if i in layers_to_replace:
            attr = 'input_norm' if norm_type == 'input' else 'post_attn_norm'
            saved[f'layer_{i}_{attr}'] = getattr(layer, attr)
            setattr(layer, attr, IdentityNorm())
    return saved


@torch.no_grad()
def restore_norms(model, saved):
    """Restore original RMSNorm layers."""
    for name, original in saved.items():
        parts = name.split('_')
        layer_idx = int(parts[1])
        attr = '_'.join(parts[2:])
        setattr(model.model.layers[layer_idx], attr, original)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoints/scaled_fp16_baseline/model.pt')
    parser.add_argument('--data_dir', default=None)
    parser.add_argument('--max_eval_steps', type=int, default=100)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = MicroGemmaFPConfig()
    quantizer = FPQuantizer('fp4_e2m1', per_channel=True)
    n_layers = len(config.layer_types)

    def load_model():
        m = MicroGemmaFPForCausalLM(config).to(device)
        load_checkpoint(m, None, args.checkpoint, device)
        m.eval()
        return m

    def eval_loss(m):
        loader = get_dataloader(8, 512, args.max_eval_steps, data_dir=args.data_dir)
        return evaluate_loss(m, loader, device, args.max_eval_steps)

    # ── Baseline ──
    model_base = load_model()
    fp16_loss = eval_loss(model_base)
    print(f"FP16 baseline loss: {fp16_loss:.4f}")
    del model_base

    # ── Experiment 1: Quantize layer, WITH vs WITHOUT RMSNorm ──
    print(f"\n{'='*65}")
    print("Experiment: RMSNorm error-blocking effect")
    print(f"{'='*65}")
    print(f"{'Layer':>6s} {'Type':>8s} {'Normal Δ':>8s} {'No Norm Δ':>8s} {'Δ diff':>8s} {'Block ratio':>10s}")
    print("-" * 65)

    results = []
    for i in range(n_layers):
        lt = config.layer_types[i]

        # Normal (with RMSNorm): quantize layer i only
        m_normal = load_model()
        # Quantize layer i
        layer = m_normal.model.layers[i]
        for name, param in layer.named_parameters():
            if param.dim() >= 2:
                param.data = quantizer.quantize(param.data)
        loss_normal = eval_loss(m_normal)
        del m_normal
        torch.cuda.empty_cache()

        # Without RMSNorm: replace norms in layer i, quantize
        m_no_norm = load_model()
        saved = replace_norms(m_no_norm, {i}, 'input')
        saved.update(replace_norms(m_no_norm, {i}, 'post_attn'))
        layer = m_no_norm.model.layers[i]
        for name, param in layer.named_parameters():
            if param.dim() >= 2:
                param.data = quantizer.quantize(param.data)
        loss_no_norm = eval_loss(m_no_norm)
        del m_no_norm
        torch.cuda.empty_cache()

        delta_normal = loss_normal - fp16_loss
        delta_no_norm = loss_no_norm - fp16_loss
        # How much worse is no-RMSNorm vs with-RMSNorm?
        block_ratio = delta_no_norm / max(abs(delta_normal), 1e-8)

        print(f"  {i:4d}  {lt:>8s}  {delta_normal:+7.4f}  {delta_no_norm:+7.4f}  "
              f"{delta_no_norm - delta_normal:+7.4f}  {block_ratio:>9.1f}x")

        results.append({
            'layer': i, 'layer_type': lt,
            'loss_normal': loss_normal,
            'loss_no_norm': loss_no_norm,
            'delta_normal': delta_normal,
            'delta_no_norm': delta_no_norm,
            'block_ratio': block_ratio,
        })

    # ── Experiment 2: Cascade error with/without RMSNorm ──
    print(f"\n{'='*65}")
    print("Experiment: Cascade error across consecutive layers")
    print(f"{'='*65}")

    # Quantize layers 0..k with and without RMSNorm, measure loss
    for k in [2, 5, 8, 11]:
        # With RMSNorm
        m_norm = load_model()
        for i in range(k + 1):
            layer = m_norm.model.layers[i]
            for name, param in layer.named_parameters():
                if param.dim() >= 2:
                    param.data = quantizer.quantize(param.data)
        loss_norm = eval_loss(m_norm)
        del m_norm
        torch.cuda.empty_cache()

        # Without RMSNorm
        m_no = load_model()
        saved_all = {}
        for i in range(k + 1):
            saved_all.update(replace_norms(m_no, {i}, 'input'))
            saved_all.update(replace_norms(m_no, {i}, 'post_attn'))
        for i in range(k + 1):
            layer = m_no.model.layers[i]
            for name, param in layer.named_parameters():
                if param.dim() >= 2:
                    param.data = quantizer.quantize(param.data)
        loss_no = eval_loss(m_no)
        del m_no
        torch.cuda.empty_cache()

        print(f"  Layers 0..{k:2d}:  w/ RMSNorm Δ={loss_norm-fp16_loss:+7.4f}  "
              f"w/o RMSNorm Δ={loss_no-fp16_loss:+7.4f}  "
              f"ratio={(loss_no-fp16_loss)/max(abs(loss_norm-fp16_loss),1):.1f}x")

    # ── Summary ──
    avg_ratio = sum(r['block_ratio'] for r in results) / len(results)
    print(f"\n{'='*65}")
    print("CONCLUSION")
    print(f"{'='*65}")
    print(f"  Average error amplification without RMSNorm: {avg_ratio:.1f}x")
    if avg_ratio > 2:
        print(f"  VERDICT: RMSNorm strongly blocks error propagation (Theorem 2 supported)")
    elif avg_ratio > 1.2:
        print(f"  VERDICT: RMSNorm moderately blocks error propagation")
    else:
        print(f"  VERDICT: RMSNorm effect is weak — theory needs revision")

    with open('checkpoints/rmsnorm_validation.json', 'w') as f:
        json.dump({
            'fp16_loss': fp16_loss,
            'avg_block_ratio': avg_ratio,
            'per_layer': results,
        }, f, indent=2)
    print(f"\nResults saved to checkpoints/rmsnorm_validation.json")


if __name__ == '__main__':
    main()
