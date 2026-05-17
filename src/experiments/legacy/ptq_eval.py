#!/usr/bin/env python3
"""
Group A1/A2: Post-Training Quantization evaluation.

Takes an FP16-trained checkpoint and quantizes it to FP8/FP4,
measuring the precision loss without any retraining.

Usage:
    # PTQ to FP8
    python src/experiments/ptq_eval.py --checkpoint checkpoints/fp16_baseline/model.pt \\
        --quant fp8

    # PTQ to FP4 with analysis
    python src/experiments/ptq_eval.py --checkpoint checkpoints/fp16_baseline/model.pt \\
        --quant fp4 --analyze
"""

import argparse
import json
import torch
from src.model.config import MicroGemmaFPConfig
from src.model.transformer import MicroGemmaFPForCausalLM
from src.quantization.fp_quantizer import FPQuantizer
from src.quantization.gptq import GPTQQuantizer
from src.analysis.condition import (
    compute_all_condition_numbers,
    analyze_quantization_sensitivity,
)
from src.analysis.sensitivity import per_layer_sensitivity_report, suggest_mixed_precision
from src.experiments.training_utils import (
    get_dataloader, evaluate_perplexity, load_checkpoint,
)


def ptq_quantize_model(model: MicroGemmaFPForCausalLM,
                       quantizer: FPQuantizer) -> MicroGemmaFPForCausalLM:
    """Quantize all quantizable weights in-place (PTQ)."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.dim() >= 2:
                # Skip embeddings
                if 'embed' in name.lower() or 'lm_head' in name.lower():
                    continue
                param.data = quantizer.quantize(param.data)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--quant', type=str, default='fp8',
                        choices=['fp8', 'fp4'])
    parser.add_argument('--method', type=str, default='simple',
                        choices=['simple', 'gptq', 'mixed'],
                        help='PTQ method: simple (round-to-nearest), '
                             'gptq (weight compensation), '
                             'mixed (sensitivity-guided per-layer)')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Data directory for calibration (gptq/mixed methods)')
    parser.add_argument('--analyze', action='store_true',
                        help='Run condition number & sensitivity analysis')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_eval_steps', type=int, default=200)
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fmt_name = {'fp8': 'fp8_e4m3', 'fp4': 'fp4_e2m1'}[args.quant]

    # Load FP16 baseline
    config = MicroGemmaFPConfig()
    model = MicroGemmaFPForCausalLM(config).to(device)
    metrics, meta = load_checkpoint(model, None, args.checkpoint, device)

    # Evaluate FP16 baseline
    eval_loader = get_dataloader(args.batch_size, max_seq_len=512,
                                  max_steps=args.max_eval_steps,
                                  data_dir=args.data_dir)
    fp16_ppl = evaluate_perplexity(model, eval_loader, device,
                                   max_steps=args.max_eval_steps)
    print(f"FP16 baseline PPL: {fp16_ppl:.2f}")

    quantizer = FPQuantizer(fmt_name, per_channel=True)

    # Pre-quantization analysis
    pre_analysis = None
    if args.analyze:
        print("\nRunning pre-quantization analysis...")
        pre_analysis = analyze_quantization_sensitivity(model, quantizer)

    # Apply PTQ with chosen method
    print(f"\nApplying PTQ to {args.quant} (method={args.method})...")

    if args.method == 'simple':
        model = ptq_quantize_model(model, quantizer)

    elif args.method == 'gptq':
        calib_loader = get_dataloader(4, 512, 50, data_dir=args.data_dir)
        gptq = GPTQQuantizer(quantizer)
        model, gptq_stats = gptq.quantize_model(model, calib_loader, device)

    elif args.method == 'mixed':
        q8 = FPQuantizer('fp8_e4m3', per_channel=True)
        q4 = quantizer
        report = per_layer_sensitivity_report(model, q4)
        suggestion = suggest_mixed_precision(report, fp8_threshold=0.33)
        print(f"  Mixed: {sum(1 for v in suggestion.values() if v=='fp8')} FP8, "
              f"{sum(1 for v in suggestion.values() if v=='fp4')} FP4 layers")
        layer_map = {i: l for i, l in enumerate(model.model.layers)}
        for layer_idx, precision in suggestion.items():
            q = q8 if precision == 'fp8' else q4
            for name, param in layer_map[layer_idx].named_parameters():
                if param.dim() >= 2:
                    param.data = q.quantize(param.data)

    # Evaluate PTQ model
    eval_loader = get_dataloader(args.batch_size, max_seq_len=512,
                                  max_steps=args.max_eval_steps,
                                  data_dir=args.data_dir)
    ptq_ppl = evaluate_perplexity(model, eval_loader, device,
                                   max_steps=args.max_eval_steps)
    print(f"PTQ-{args.quant} ({args.method}) PPL: {ptq_ppl:.2f}")
    print(f"PPL degradation: {ptq_ppl - fp16_ppl:+.2f} "
          f"({(ptq_ppl/fp16_ppl - 1)*100:+.1f}%)")

    # Post-quantization analysis
    if args.analyze:
        print("\n--- Per-layer condition numbers ---")
        cond = compute_all_condition_numbers(model_quant)
        cond_items = sorted(cond.items(), key=lambda x: x[1], reverse=True)
        for name, kappa in cond_items[:10]:
            print(f"  κ({name}) = {kappa:.1f}")

    # Save results
    results = {
        'fp16_ppl': fp16_ppl,
        'ptq_ppl': ptq_ppl,
        'ppl_degradation': ptq_ppl - fp16_ppl,
        'ppl_degradation_pct': (ptq_ppl / fp16_ppl - 1) * 100,
        'quant_format': args.quant,
        'ptq_method': args.method,
        'checkpoint': args.checkpoint,
    }

    if args.output_dir:
        import os
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, 'ptq_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output_dir}/ptq_results.json")


if __name__ == '__main__':
    main()
