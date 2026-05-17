#!/usr/bin/env python3
"""
Unified Quantization Evaluation — Industry Standard Pipeline.

Runs the full PTQ pipeline as done in production:
  1. Load FP16-pretrained model
  2. Evaluate FP16 baseline PPL
  3. Collect calibration data for GPTQ Hessian estimation
  4. Run PTQ with multiple methods:
     - simple:    round-to-nearest (per-tensor or per-channel)
     - gptq:      GPTQ weight compensation
     - mixed:     sensitivity-guided mixed precision + GPTQ
  5. Compare PPL degradation across methods

Usage:
    python src/experiments/eval_quantization.py \\
        --checkpoint checkpoints/fp16_baseline/model.pt \\
        --data_dir data/real_tiers \\
        --methods simple gptq mixed \\
        --formats fp8_e4m3 fp4_e2m1

    ./remote_python.sh src/experiments/eval_quantization.py \\
        --checkpoint checkpoints/fp16_baseline/model.pt \\
        --data_dir data/real_tiers
"""

import os, json, argparse, copy
import torch
from src.model.config import MicroGemmaFPConfig
from src.model.transformer import MicroGemmaFPForCausalLM
from src.quantization.fp_quantizer import FPQuantizer
from src.quantization.gptq import GPTQQuantizer
from src.analysis.sensitivity import per_layer_sensitivity_report, suggest_mixed_precision
from src.experiments.training_utils import (
    get_dataloader, evaluate_perplexity, load_checkpoint,
)


def evaluate_model(model, device, data_dir, max_steps=200) -> float:
    loader = get_dataloader(8, 512, max_steps, data_dir=data_dir)
    return evaluate_perplexity(model, loader, device, max_steps)


@torch.no_grad()
def ptq_simple(model, quantizer):
    """Round-to-nearest PTQ: quantize all linear weights in-place."""
    for name, param in model.named_parameters():
        if param.dim() >= 2:
            if 'embed' in name.lower() or 'lm_head' in name.lower():
                continue
            param.data = quantizer.quantize(param.data)
    return model


@torch.no_grad()
def ptq_gptq(model, quantizer, data_dir, device, blocksize=128):
    """GPTQ PTQ with weight compensation."""
    calib_loader = get_dataloader(4, 512, 100, data_dir=data_dir)
    gptq = GPTQQuantizer(quantizer, blocksize=blocksize, stochastic=False)
    stats = gptq.quantize_model(model, calib_loader, device)
    return model, stats


@torch.no_grad()
def ptq_mixed_precision(model, quantizer_fp8, quantizer_fp4, data_dir, device):
    """Sensitivity-guided mixed precision: FP8 for sensitive layers, FP4 for rest."""
    calib_loader = get_dataloader(4, 512, 100, data_dir=data_dir)

    # Run sensitivity analysis
    report = per_layer_sensitivity_report(model, quantizer_fp4)
    suggestion = suggest_mixed_precision(report, fp8_threshold=0.33)

    print(f"  Mixed precision: {len([v for v in suggestion.values() if v=='fp8'])} FP8 layers, "
          f"{len([v for v in suggestion.values() if v=='fp4'])} FP4 layers")

    # Quantize layer-by-layer with appropriate precision
    layer_map = {}
    for i, layer in enumerate(model.model.layers):
        layer_map[i] = layer

    for layer_idx, precision in suggestion.items():
        layer = layer_map[layer_idx]
        q = quantizer_fp8 if precision == 'fp8' else quantizer_fp4

        for name, param in layer.named_parameters():
            if param.dim() < 2:
                continue
            param.data = q.quantize(param.data)

    return model, {'suggestion': suggestion, 'report': [r for r in report[:5]]}


def main():
    parser = argparse.ArgumentParser(
        description="Unified Quantization Evaluation — Industry Standard Pipeline")
    parser.add_argument('--checkpoint', default='checkpoints/fp16_baseline/model.pt')
    parser.add_argument('--data_dir', default=None)
    parser.add_argument('--methods', nargs='+',
                        default=['simple_pc', 'gptq'],
                        choices=['simple_pt', 'simple_pc', 'gptq', 'mixed'])
    parser.add_argument('--formats', nargs='+',
                        default=['fp8_e4m3', 'fp4_e2m1'])
    parser.add_argument('--max_eval_steps', type=int, default=200)
    parser.add_argument('--output_dir', default='checkpoints/eval_quantization')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    config = MicroGemmaFPConfig()
    model_base = MicroGemmaFPForCausalLM(config).to(device)
    metrics, _ = load_checkpoint(model_base, None, args.checkpoint, device)
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"  Params: {model_base.count_parameters()['total']:,}")

    # FP16 baseline PPL
    fp16_ppl = evaluate_model(model_base, device, args.data_dir, args.max_eval_steps)
    print(f"  FP16 PPL: {fp16_ppl:.2f}")
    print()

    all_results = {'fp16_baseline': fp16_ppl}

    for fmt in args.formats:
        print(f"{'='*60}")
        print(f"Format: {fmt}")
        print(f"{'='*60}")

        for method in args.methods:
            # Load fresh model copy
            model = MicroGemmaFPForCausalLM(config).to(device)
            load_checkpoint(model, None, args.checkpoint, device)

            key = f"{fmt}_{method}"
            print(f"\n--- {key} ---")

            if method == 'simple_pt':
                q = FPQuantizer(fmt, per_channel=False)
                model = ptq_simple(model, q)

            elif method == 'simple_pc':
                q = FPQuantizer(fmt, per_channel=True)
                model = ptq_simple(model, q)

            elif method == 'gptq':
                q = FPQuantizer(fmt, per_channel=True)
                model, stats = ptq_gptq(model, q, args.data_dir, device)

            elif method == 'mixed':
                q8 = FPQuantizer('fp8_e4m3', per_channel=True)
                q4 = FPQuantizer(fmt, per_channel=True)
                model, stats = ptq_mixed_precision(model, q8, q4, args.data_dir, device)

            ppl = evaluate_model(model, device, args.data_dir, args.max_eval_steps)
            degradation = ppl - fp16_ppl
            degradation_pct = (ppl / fp16_ppl - 1) * 100

            print(f"  PPL: {ppl:.2f}  (FP16: {fp16_ppl:.2f}, "
                  f"delta: {degradation:+.2f}, {degradation_pct:+.2f}%)")

            all_results[key] = ppl

            # Clean up
            del model
            torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Method':<30s} {'PPL':>8s} {'Delta':>8s} {'Delta%':>8s}")
    print("-" * 56)
    for key, ppl in all_results.items():
        if key == 'fp16_baseline':
            continue
        delta = ppl - fp16_ppl
        delta_pct = (ppl / fp16_ppl - 1) * 100
        print(f"{key:<30s} {ppl:>8.2f} {delta:>+8.2f} {delta_pct:>+7.2f}%")

    # Best per format
    for fmt in args.formats:
        format_methods = {k: v for k, v in all_results.items()
                          if k.startswith(fmt)}
        if len(format_methods) > 1:
            best = min(format_methods, key=format_methods.get)
            print(f"\n  Best {fmt}: {best} (PPL {format_methods[best]:.2f})")

    # Save
    result_path = os.path.join(args.output_dir, 'results.json')
    with open(result_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {result_path}")


if __name__ == '__main__':
    main()
