#!/usr/bin/env python3
"""
Compare uniform E2M1 vs adaptive (Lloyd-Max) vs κ-weighted adaptive FP4 grids.

Usage:
    ./remote_python.sh src/experiments/compare_adaptive_grid.py \
        --checkpoint checkpoints/scaled_fp16_baseline/model.pt \
        --data_dir data/real_tiers
"""

import argparse, torch
from src.model.config import MicroGemmaFPConfig
from src.model.transformer import MicroGemmaFPForCausalLM
from src.quantization.fp_quantizer import FPQuantizer
from src.quantization.adaptive_grid import AdaptiveGridQuantizer
from src.experiments.training_utils import (
    get_dataloader, evaluate_perplexity, load_checkpoint,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoints/scaled_fp16_baseline/model.pt')
    parser.add_argument('--data_dir', default=None)
    parser.add_argument('--max_eval_steps', type=int, default=100)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = MicroGemmaFPConfig()

    def load_model():
        m = MicroGemmaFPForCausalLM(config).to(device)
        load_checkpoint(m, None, args.checkpoint, device)
        m.eval()
        return m

    def eval_ppl(m):
        loader = get_dataloader(8, 512, args.max_eval_steps, data_dir=args.data_dir)
        return evaluate_perplexity(m, loader, device, args.max_eval_steps)

    # FP16 baseline
    model = load_model()
    fp16_ppl = eval_ppl(model)
    print(f"FP16 baseline PPL: {fp16_ppl:.2f}")
    del model
    torch.cuda.empty_cache()

    # ── 1. Uniform E2M1 (per-channel baseline) ──
    m = load_model()
    q_uniform = FPQuantizer('fp4_e2m1', per_channel=True)
    for name, param in m.named_parameters():
        if param.dim() >= 2 and 'embed' not in name and 'lm_head' not in name:
            param.data = q_uniform.quantize(param.data)
    ppl_uniform = eval_ppl(m)
    print(f"Uniform E2M1 (per-channel): PPL={ppl_uniform:.2f}  Δ={ppl_uniform-fp16_ppl:+.2f}")
    del m; torch.cuda.empty_cache()

    # ── 2. Adaptive (no κ-weighting) ──
    m = load_model()
    q_adapt = AdaptiveGridQuantizer(kappa_weight=0.0)
    print("Calibrating adaptive grids (no κ-weighting)...")
    q_adapt.calibrate(m)
    q_adapt.quantize_model(m)
    ppl_adapt = eval_ppl(m)
    print(f"Adaptive (no κ):             PPL={ppl_adapt:.2f}  Δ={ppl_adapt-fp16_ppl:+.2f}")
    del m; torch.cuda.empty_cache()

    # ── 3. κ-weighted adaptive ──
    m = load_model()
    q_kappa = AdaptiveGridQuantizer(kappa_weight=0.5)
    print("Calibrating κ-weighted adaptive grids...")
    q_kappa.calibrate(m)
    q_kappa.quantize_model(m)
    ppl_kappa = eval_ppl(m)
    print(f"κ-weighted Adaptive:         PPL={ppl_kappa:.2f}  Δ={ppl_kappa-fp16_ppl:+.2f}")
    del m; torch.cuda.empty_cache()

    # ── Summary ──
    print(f"\n{'='*55}")
    print(f"{'Method':<25s} {'PPL':>8s} {'Δ(FP16)':>8s} {'vs Uniform':>10s}")
    print("-" * 55)
    for name, ppl in [("Uniform E2M1 (per-ch)", ppl_uniform),
                       ("Adaptive (no kappa)", ppl_adapt),
                       ("Kappa-weighted Adapt", ppl_kappa)]:
        d = ppl - fp16_ppl
        vs = ppl - ppl_uniform
        print(f"{name:<25s} {ppl:>8.2f} {d:>+8.2f} {vs:>+9.2f}")
    print(f"\nFP16 baseline: {fp16_ppl:.2f}")


if __name__ == '__main__':
    main()
