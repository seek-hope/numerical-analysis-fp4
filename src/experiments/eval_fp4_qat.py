#!/usr/bin/env python3
"""Evaluate FP4 QAT optimization checkpoints."""
import torch
from src.model.config import MicroGemmaFPConfig
from src.model.transformer import MicroGemmaFPForCausalLM
from src.experiments.training_utils import get_dataloader, evaluate_perplexity, load_checkpoint

config = MicroGemmaFPConfig()
device = 'cuda'

checkpoints = {
    'fp8_baseline':     'checkpoints/qat_fp8/model.pt',
    'fp4_vanilla':      'checkpoints/qat_fp4/model.pt',
    'fp4_sr':           'checkpoints/qat_fp4_sr/model.pt',
    'fp4_adaptive':     'checkpoints/qat_fp4_adaptive/model.pt',
    'fp4_combined':     'checkpoints/qat_fp4_combined/model.pt',
}

print(f"{'Method':<20s} {'Train PPL':>10s} {'Eval PPL':>10s} {'Gap':>8s}")
print("-" * 50)

for name, ckpt in checkpoints.items():
    model = MicroGemmaFPForCausalLM(config).to(device)
    metrics, _ = load_checkpoint(model, None, ckpt, device)
    train_ppl = metrics[-1]['perplexity'] if metrics else float('nan')

    loader = get_dataloader(8, 256, 200)
    eval_ppl = evaluate_perplexity(model, loader, device, 200)

    gap = eval_ppl / max(train_ppl, 1e-6)
    print(f"{name:<20s} {train_ppl:>10.2f} {eval_ppl:>10.2f} {gap:>7.1f}x")
