#!/usr/bin/env python3
"""
Phase 2: Systematic comparison of all quantization methods.

Runs the complete comparison matrix on both the standard baseline and the
condition-number-regularized model. Produces a ranked summary.

PTQ methods:
  - simple_pt:    round-to-nearest, per-tensor
  - simple_pc:    round-to-nearest, per-channel
  - gptq:         GPTQ weight compensation
  - mixed:        sensitivity-guided mixed precision
  - adaptive:     Lloyd-Max per-layer adaptive grid
  - kappa_adapt:  κ-weighted Lloyd-Max adaptive grid

QAT methods:
  - qat_fp8:      STE QAT with FP8 E4M3
  - qat_fp4:      STE QAT with FP4 E2M1

Usage:
    ./remote_python.sh src/experiments/phase2_comparison.py \
        --data_dir data/real_tiers
"""

import json, argparse, copy
import torch
from src.model.config import MicroGemmaFPConfig
from src.model.transformer import MicroGemmaFPForCausalLM
from src.quantization.fp_quantizer import FPQuantizer
from src.quantization.gptq import GPTQQuantizer
from src.quantization.adaptive_grid import AdaptiveGridQuantizer
from src.analysis.sensitivity import per_layer_sensitivity_report, suggest_mixed_precision
from src.experiments.training_utils import (
    get_dataloader, evaluate_perplexity, load_checkpoint,
)


def eval_ppl(model, device, data_dir, steps=100):
    loader = get_dataloader(8, 512, steps, data_dir=data_dir)
    return evaluate_perplexity(model, loader, device, steps)


def load_model(ckpt, device):
    config = MicroGemmaFPConfig()
    model = MicroGemmaFPForCausalLM(config).to(device)
    load_checkpoint(model, None, ckpt, device)
    model.eval()
    return model


@torch.no_grad()
def apply_ptq_simple(model, quantizer):
    for name, param in model.named_parameters():
        if param.dim() >= 2 and 'embed' not in name and 'lm_head' not in name:
            param.data = quantizer.quantize(param.data)


@torch.no_grad()
def apply_ptq_gptq(model, quantizer, data_dir, device):
    calib_loader = get_dataloader(4, 512, 100, data_dir=data_dir)
    gptq = GPTQQuantizer(quantizer)
    gptq.quantize_model(model, calib_loader, device)


@torch.no_grad()
def apply_ptq_adaptive(model, quantizer, kappa_weight=0.0):
    q = AdaptiveGridQuantizer(kappa_weight=kappa_weight)
    q.calibrate(model)
    q.quantize_model(model)


@torch.no_grad()
def apply_ptq_mixed(model, q8, q4, data_dir, device):
    calib_loader = get_dataloader(4, 512, 50, data_dir=data_dir)
    report = per_layer_sensitivity_report(model, q4)
    suggestion = suggest_mixed_precision(report, fp8_threshold=0.33)
    layer_map = {i: l for i, l in enumerate(model.model.layers)}
    for layer_idx, precision in suggestion.items():
        q = q8 if precision == 'fp8' else q4
        for name, param in layer_map[layer_idx].named_parameters():
            if param.dim() >= 2:
                param.data = q.quantize(param.data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline_ckpt', default='checkpoints/scaled_fp16_baseline/model.pt')
    parser.add_argument('--condreg_ckpt', default='checkpoints/cond_regularized/model.pt')
    parser.add_argument('--data_dir', default=None)
    parser.add_argument('--max_eval_steps', type=int, default=100)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = MicroGemmaFPConfig()

    checkpoints = {
        'baseline': args.baseline_ckpt,
        'cond_regularized': args.condreg_ckpt,
    }

    methods = [
        ('simple_pt', 'FPQuantizer(per_tensor)'),
        ('simple_pc', 'FPQuantizer(per_channel)'),
        ('gptq', 'GPTQ'),
        ('mixed', 'Mixed precision'),
        ('adaptive', 'Lloyd-Max adaptive'),
        ('kappa_adapt', 'κ-weighted Lloyd-Max'),
    ]

    formats = {'FP8': 'fp8_e4m3', 'FP4': 'fp4_e2m1'}

    all_results = {}

    for ckpt_name, ckpt_path in checkpoints.items():
        print(f"\n{'='*60}")
        print(f"Checkpoint: {ckpt_name}")
        print(f"{'='*60}")

        model_ref = load_model(ckpt_path, device)
        fp16_ppl = eval_ppl(model_ref, device, args.data_dir, args.max_eval_steps)
        print(f"  FP16 PPL: {fp16_ppl:.2f}")
        del model_ref; torch.cuda.empty_cache()
        all_results[f'{ckpt_name}_fp16'] = fp16_ppl

        for fmt_name, fmt_str in formats.items():
            for method_name, method_desc in methods:
                key = f"{ckpt_name}/{fmt_name}/{method_name}"
                print(f"  {fmt_name} {method_name} ({method_desc})...", end=' ', flush=True)

                model = load_model(ckpt_path, device)

                try:
                    if method_name == 'simple_pt':
                        q = FPQuantizer(fmt_str, per_channel=False)
                        apply_ptq_simple(model, q)
                    elif method_name == 'simple_pc':
                        q = FPQuantizer(fmt_str, per_channel=True)
                        apply_ptq_simple(model, q)
                    elif method_name == 'gptq':
                        q = FPQuantizer(fmt_str, per_channel=True)
                        apply_ptq_gptq(model, q, args.data_dir, device)
                    elif method_name == 'mixed':
                        q8 = FPQuantizer('fp8_e4m3', per_channel=True)
                        q4 = FPQuantizer(fmt_str, per_channel=True)
                        apply_ptq_mixed(model, q8, q4, args.data_dir, device)
                    elif method_name == 'adaptive':
                        apply_ptq_adaptive(model, None, kappa_weight=0.0)
                    elif method_name == 'kappa_adapt':
                        apply_ptq_adaptive(model, None, kappa_weight=0.5)

                    ppl = eval_ppl(model, device, args.data_dir, args.max_eval_steps)
                    print(f"PPL={ppl:.2f}")
                    all_results[key] = ppl
                except Exception as e:
                    print(f"FAILED: {e}")
                    all_results[key] = None

                del model
                torch.cuda.empty_cache()

    # ── Summary Table ──
    print(f"\n{'='*80}")
    print("PHASE 2: COMPLETE COMPARISON MATRIX")
    print(f"{'='*80}")

    for ckpt_name in checkpoints:
        fp16 = all_results.get(f'{ckpt_name}_fp16', 0)
        print(f"\n  {ckpt_name} (FP16 PPL: {fp16:.2f})")
        print(f"  {'Method':<18s} {'FP8 PPL':>8s} {'FP8 Δ':>7s}  "
              f"{'FP4 PPL':>8s} {'FP4 Δ':>7s}")
        print(f"  {'-'*55}")

        best_fp8, best_fp8_val = '', 1e9
        best_fp4, best_fp4_val = '', 1e9

        for method_name, _ in methods:
            fp8_key = f'{ckpt_name}/FP8/{method_name}'
            fp4_key = f'{ckpt_name}/FP4/{method_name}'
            ppl_fp8 = all_results.get(fp8_key)
            ppl_fp4 = all_results.get(fp4_key)

            d8 = f"{ppl_fp8-fp16:+6.2f}" if ppl_fp8 else "  FAIL"
            d4 = f"{ppl_fp4-fp16:+6.2f}" if ppl_fp4 else "  FAIL"
            p8 = f"{ppl_fp8:>8.2f}" if ppl_fp8 else "   FAIL"
            p4 = f"{ppl_fp4:>8.2f}" if ppl_fp4 else "   FAIL"

            print(f"  {method_name:<18s} {p8} {d8}  {p4} {d4}")

            if ppl_fp8 and ppl_fp8 < best_fp8_val:
                best_fp8_val, best_fp8 = ppl_fp8, method_name
            if ppl_fp4 and ppl_fp4 < best_fp4_val:
                best_fp4_val, best_fp4 = ppl_fp4, method_name

        print(f"  {'─'*55}")
        print(f"  Best FP8: {best_fp8} ({best_fp8_val:.2f})")
        print(f"  Best FP4: {best_fp4} ({best_fp4_val:.2f})")

    with open('checkpoints/phase2_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to checkpoints/phase2_results.json")


if __name__ == '__main__':
    main()
