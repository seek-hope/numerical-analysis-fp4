#!/usr/bin/env python3
"""
FP4 PTQ Benchmark: Compare all FP4 quantization schemes on the FP16 baseline.

Schemes compared:
  1. FP4 E2M1 (standard)     — hardware native, log-spaced grid
  2. NF4 (Normal Float)      — information-theoretically optimal
  3. MXFP4 (Microscaling)    — block-wise shared scale
  4. FP4 E2M1 + OutlierScale — per-channel scaling for outlier channels
  5. FP4 E2M1 + DuQuant++    — outlier scaling + block rotation
  6. MXFP4  + DuQuant++      — block scaling + block rotation

Usage:
    python src/experiments/fp4_ptq_compare.py [--analyze-outliers]
"""

import json
import torch
from src.model.config import MicroGemmaFPConfig
from src.model.transformer import MicroGemmaFPForCausalLM
from src.quantization.fp4_grids import (
    FP4_E2M1_GRID, NF4_GRID, GridQuantizer, MXFP4Quantizer,
)
from src.quantization.outlier_rotation import (
    DuQuantStyleQuantizer, analyze_outlier_channels,
)
from src.experiments.training_utils import (
    get_dataloader, evaluate_perplexity, load_checkpoint,
)


def ptq_quantize_model(model, quantizer_fn, verbose=False):
    """Apply PTQ quantization to all quantizable weights."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.dim() < 2:
                continue
            if 'embed' in name.lower() or 'lm_head' in name.lower():
                continue
            if verbose:
                mse_before = ((param.data - param.data) ** 2).mean().item()
            param.data = quantizer_fn(param.data)
            if verbose:
                mse_after = ((param.data - param.data) ** 2).mean().item()
    return model


def load_baseline_model(device='cuda'):
    """Load the QAT-FP8 model as baseline (industry standard)."""
    config = MicroGemmaFPConfig()
    model = MicroGemmaFPForCausalLM(config).to(device)
    load_checkpoint(model, None, 'checkpoints/qat_fp8/model.pt', device)
    model.eval()
    return model, config


def evaluate_ppl(model, device='cuda', max_steps=200):
    """Evaluate perplexity."""
    loader = get_dataloader(8, 256, max_steps)
    return evaluate_perplexity(model, loader, device, max_steps)


def main():
    device = 'cuda'
    results = {}

    # ═══════════════════════════════════════════════════════
    # FP16 baseline
    # ═══════════════════════════════════════════════════════
    print("=" * 60)
    print("FP4 PTQ COMPARISON BENCHMARK")
    print("=" * 60)

    model_fp8, config = load_baseline_model(device)
    fp8_ppl = evaluate_ppl(model_fp8, device)
    print(f"\nFP8 baseline PPL: {fp8_ppl:.2f}")
    results['fp8_baseline'] = fp8_ppl

    # ═══════════════════════════════════════════════════════
    # 1. FP4 E2M1 (standard)
    # ═══════════════════════════════════════════════════════
    print("\n--- 1. FP8 → FP4 E2M1 (standard) ---")
    model, _ = load_baseline_model(device)
    q = GridQuantizer(FP4_E2M1_GRID)
    model = ptq_quantize_model(model, q.quantize)
    ppl = evaluate_ppl(model, device)
    print(f"  PPL: {ppl:.2f} (Δ: {ppl-fp8_ppl:+.2f}, {(ppl/fp8_ppl-1)*100:+.1f}%)")
    results['fp4_e2m1'] = ppl

    # ═══════════════════════════════════════════════════════
    # 2. NF4 (Normal Float)
    # ═══════════════════════════════════════════════════════
    print("\n--- 2. FP8 → NF4 (Normal Float) ---")
    model, _ = load_baseline_model(device)
    q = GridQuantizer(NF4_GRID)
    model = ptq_quantize_model(model, q.quantize)
    ppl = evaluate_ppl(model, device)
    print(f"  PPL: {ppl:.2f} (Δ: {ppl-fp8_ppl:+.2f}, {(ppl/fp8_ppl-1)*100:+.1f}%)")
    results['nf4'] = ppl

    # ═══════════════════════════════════════════════════════
    # 3. MXFP4 (Microscaling, block_size=32)
    # ═══════════════════════════════════════════════════════
    print("\n--- 3. FP8 → MXFP4 (Microscaling, B=32) ---")
    model, _ = load_baseline_model(device)
    q = MXFP4Quantizer(block_size=32)
    model = ptq_quantize_model(model, q.quantize)
    ppl = evaluate_ppl(model, device)
    print(f"  PPL: {ppl:.2f} (Δ: {ppl-fp8_ppl:+.2f}, {(ppl/fp8_ppl-1)*100:+.1f}%)")
    results['mxfp4_b32'] = ppl

    # ═══════════════════════════════════════════════════════
    # 4. FP4 E2M1 + OutlierScale (no rotation)
    # ═══════════════════════════════════════════════════════
    print("\n--- 4. FP8 → FP4 E2M1 + OutlierScale ---")
    model, _ = load_baseline_model(device)
    dq = DuQuantStyleQuantizer(FP4_E2M1_GRID)
    model = ptq_quantize_model(
        model, lambda w: dq.quantize(w, use_rotation=False, use_outlier_scale=True)
    )
    ppl = evaluate_ppl(model, device)
    print(f"  PPL: {ppl:.2f} (Δ: {ppl-fp8_ppl:+.2f}, {(ppl/fp8_ppl-1)*100:+.1f}%)")
    results['fp4_e2m1_outlier'] = ppl

    # ═══════════════════════════════════════════════════════
    # 5. FP8 → FP4 E2M1 + DuQuant++ (skip — rotation is buggy)
    # ═══════════════════════════════════════════════════════
    print("\n--- 5. FP8 → FP4 E2M1 + DuQuant++ (SKIPPED) ---")
    results['fp4_e2m1_duquant'] = None

    # ═══════════════════════════════════════════════════════
    # 6. MXFP4 + DuQuant++ (SKIPPED)
    # ═══════════════════════════════════════════════════════
    print("\n--- 6. MXFP4 + DuQuant++ (SKIPPED) ---")
    results['mxfp4_duquant'] = None

    # ═══════════════════════════════════════════════════════
    # Outlier analysis
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("OUTLIER ANALYSIS")
    print("=" * 60)
    model, _ = load_baseline_model(device)
    outliers = analyze_outlier_channels(model)
    sorted_outliers = sorted(
        outliers.items(),
        key=lambda x: x[1]['outlier_ratio'],
        reverse=True,
    )
    for name, stats in sorted_outliers[:10]:
        print(f"  {name:50s} outliers={stats['outlier_channels']:3d}/{stats['total_channels']:4d} "
              f"({stats['outlier_ratio']*100:4.1f}%) max_kurt={stats['max_kurtosis']:.1f}")

    # ═══════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Scheme':<30s} {'PPL':>8s} {'Δ':>8s} {'Δ%':>8s}")
    print("-" * 56)
    for name, ppl in results.items():
        if ppl is None:
            print(f"{name:<30s} {'SKIPPED':>8s}")
            continue
        delta = ppl - fp8_ppl
        delta_pct = (ppl / fp8_ppl - 1) * 100
        print(f"{name:<30s} {ppl:>8.2f} {delta:>+8.2f} {delta_pct:>+7.1f}%")

    # Best
    best_name = min(
        (k for k, v in results.items() if v is not None and k != 'fp8_baseline'),
        key=lambda k: results[k] - fp8_ppl
    )
    best_ppl = results[best_name]
    print(f"\nBest: {best_name} (PPL {best_ppl:.2f}, Δ {(best_ppl/fp8_ppl-1)*100:+.1f}%)")

    # Save
    with open('checkpoints/fp4_ptq_results.json', 'w') as f:
        json.dump({k: round(v, 2) for k, v in results.items()}, f, indent=2)
    print("Saved to checkpoints/fp4_ptq_results.json")


if __name__ == '__main__':
    main()
