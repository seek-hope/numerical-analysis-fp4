#!/usr/bin/env python3
"""Evaluate all trained checkpoints on the same eval set for fair comparison."""

import json
import torch
from src.model.config import MicroGemmaFPConfig
from src.model.transformer import MicroGemmaFPForCausalLM
from src.experiments.training_utils import get_dataloader, evaluate_perplexity, load_checkpoint


CHECKPOINTS = {
    'fp16_baseline': 'checkpoints/fp16_baseline/model.pt',
    'qat_fp8': 'checkpoints/qat_fp8/model.pt',
    'qat_fp4': 'checkpoints/qat_fp4/model.pt',
    'qat_fp8_optimized': 'checkpoints/qat_fp8_optimized/model.pt',
    'qat_fp4_optimized': 'checkpoints/qat_fp4_optimized/model.pt',
}

config = MicroGemmaFPConfig()
device = 'cuda'

results = {}
for name, ckpt_path in CHECKPOINTS.items():
    try:
        model = MicroGemmaFPForCausalLM(config).to(device)
        load_checkpoint(model, None, ckpt_path, device)
        loader = get_dataloader(8, 256, 200)
        ppl = evaluate_perplexity(model, loader, device, 200)
        results[name] = round(ppl, 2)
        print(f'{name:25s}  PPL = {ppl:.2f}')
    except FileNotFoundError:
        print(f'{name:25s}  (checkpoint not found)')

# Save
with open('checkpoints/eval_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nSaved to checkpoints/eval_results.json')
