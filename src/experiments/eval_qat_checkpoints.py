#!/usr/bin/env python3
"""Quick eval of all checkpoints for unified comparison table."""
import torch
from src.model.config import MicroGemmaFPConfig
from src.model.transformer import MicroGemmaFPForCausalLM
from src.experiments.training_utils import get_dataloader, evaluate_perplexity, load_checkpoint

device = 'cuda'
config = MicroGemmaFPConfig()
checkpoints = {
    'FP16 baseline': 'checkpoints/scaled_fp16_baseline/model.pt',
    'FP16 + CondReg': 'checkpoints/cond_regularized/model.pt',
    'QAT-FP8': 'checkpoints/qat_fp8/model.pt',
    'QAT-FP4': 'checkpoints/qat_fp4/model.pt',
    'QAT-FP8 + SR + CondReg': 'checkpoints/qat_fp8_sr_cond/model.pt',
    'QAT-FP4 + SR + CondReg': 'checkpoints/qat_fp4_sr_cond/model.pt',
}

for name, ckpt in checkpoints.items():
    try:
        m = MicroGemmaFPForCausalLM(config).to(device)
        load_checkpoint(m, None, ckpt, device)
        m.eval()
        loader = get_dataloader(8, 512, 100, data_dir='data/real_tiers')
        ppl = evaluate_perplexity(m, loader, device, 100)
        print(f'{name:30s} eval PPL={ppl:.2f}')
        del m; torch.cuda.empty_cache()
    except Exception as e:
        print(f'{name:30s} ERROR: {e}')
