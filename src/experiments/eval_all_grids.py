#!/usr/bin/env python3
"""Evaluate all FP4 QAT variants — grid comparison."""
import torch, json
from src.model.config import MicroGemmaFPConfig
from src.model.transformer import MicroGemmaFPForCausalLM
from src.experiments.training_utils import get_dataloader, evaluate_perplexity, load_checkpoint

config = MicroGemmaFPConfig()
device = 'cuda'
all_results = {}

checkpoints = {
    'fp8_baseline':      'checkpoints/qat_fp8/model.pt',
    'fp4_e2m1_vanilla':  'checkpoints/qat_fp4/model.pt',
    'fp4_e2m1_sr':       'checkpoints/qat_fp4_sr/model.pt',
    'fp4_e2m1_combined': 'checkpoints/qat_fp4_combined/model.pt',
    'fp4_nf4_sr':        'checkpoints/qat_nf4/model.pt',
    'fp4_mxfp4b32_sr':   'checkpoints/qat_mxfp4_b32/model.pt',
    'fp4_mxfp4b16_sr':   'checkpoints/qat_mxfp4_b16/model.pt',
}

print(f"{'Method':<25s} {'Train PPL':>10s} {'Eval PPL':>10s}")
print("-" * 48)

for name, ckpt in checkpoints.items():
    try:
        model = MicroGemmaFPForCausalLM(config).to(device)
        metrics, meta = load_checkpoint(model, None, ckpt, device)
        train_ppl = metrics[-1].get('perplexity', metrics[-1].get('ppl', float('nan'))) if metrics else float('nan')

        loader = get_dataloader(8, 256, 200)
        eval_ppl = evaluate_perplexity(model, loader, device, 200)

        all_results[name] = {'train': round(train_ppl, 2), 'eval': round(eval_ppl, 2)}
        print(f"{name:<25s} {train_ppl:>10.2f} {eval_ppl:>10.2f}")
    except FileNotFoundError:
        print(f"{name:<25s} {'(pending)':>10s} {'(pending)':>10s}")

with open('checkpoints/fp4_qat_grid_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)
print("\nSaved to checkpoints/fp4_qat_grid_results.json")
