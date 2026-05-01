#!/usr/bin/env python3
"""
FP4 QAT with different grid schemes (Number System Design).

Compares:
  - fp4_e2m1 + SR  (baseline from Week 2)
  - nf4 + SR       (Normal Float grid — theoretically optimal)
  - mxfp4_b8 + SR   (Microscaling with B=8)
  - mxfp4_b16 + SR  (Microscaling with B=16)
  - mxfp4_b32 + SR  (Microscaling with B=32)

All use stochastic rounding (proven essential in Week 2).

Usage:
    python src/experiments/train_fp4_grid_qat.py --grid nf4
    python src/experiments/train_fp4_grid_qat.py --grid mxfp4 --block_size 16
"""

import os, math, json, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.config import MicroGemmaFPConfig
from src.model.transformer import MicroGemmaFPForCausalLM
from src.quantization.grid_qat import (
    StochasticGridQuantizer, MXFP4StochasticQuantizer,
)
from src.quantization.fp4_grids import NF4_GRID, FP4_E2M1_GRID
from src.experiments.training_utils import (
    get_dataloader, save_checkpoint, load_checkpoint,
)


# QAT Linear wrapper (reuse pattern from train_qat_fp4_opt.py)
class QATLinearGrid(nn.Module):
    def __init__(self, base, quantizer):
        super().__init__()
        self.weight = base.weight
        self.bias = base.bias
        self.q = quantizer
        self.in_features = base.in_features
        self.out_features = base.out_features

    def forward(self, x):
        W_q = self.q.quantize(self.weight, stochastic=True)
        return F.linear(x, W_q, self.bias)


def wrap_model(model, quantizer):
    def _wrap(module, prefix=''):
        for name, child in list(module.named_children()):
            full = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear):
                if 'embed' in full.lower() or 'lm_head' in full.lower():
                    continue
                setattr(module, name, QATLinearGrid(child, quantizer))
            else:
                _wrap(child, full)
    _wrap(model)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--grid', default='nf4',
                        choices=['fp4_e2m1', 'nf4', 'mxfp4'])
    parser.add_argument('--block_size', type=int, default=32)
    parser.add_argument('--max_steps', type=int, default=1500)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--vocab_size', type=int, default=32000)
    parser.add_argument('--tier', type=str, default=None)
    args = parser.parse_args()

    device = 'cuda'
    name = f"{args.grid}" if args.grid != 'mxfp4' else f"mxfp4_b{args.block_size}"
    output_dir = args.output_dir or f'checkpoints/qat_{name}'
    os.makedirs(output_dir, exist_ok=True)

    # Quantizer
    if args.grid == 'nf4':
        q = StochasticGridQuantizer(NF4_GRID, 'nf4')
    elif args.grid == 'fp4_e2m1':
        q = StochasticGridQuantizer(FP4_E2M1_GRID, 'fp4_e2m1')
    else:
        q = MXFP4StochasticQuantizer(args.block_size)

    print(f"FP4 QAT grid: {q.name} | SR=True | output={output_dir}")

    # Model
    config = MicroGemmaFPConfig()
    model = MicroGemmaFPForCausalLM(config).to(device)
    wrap_model(model, q)

    # Train
    loader = get_dataloader(8, 512, args.max_steps,
                            args.vocab_size, args.data_dir, args.tier)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    metrics = []

    for step, batch in enumerate(loader):
        if step >= args.max_steps:
            break
        ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        loss = model(ids, labels=labels)['loss']
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 50 == 0:
            metrics.append({
                'step': step, 'loss': loss.item(),
                'ppl': math.exp(loss.item()),
            })

    final = metrics[-1]
    print(f"Final: step={final['step']} ppl={final['ppl']:.2f}")

    save_checkpoint(model, opt, metrics,
                    os.path.join(output_dir, 'model.pt'),
                    {'grid': args.grid, 'block_size': args.block_size})


if __name__ == '__main__':
    main()
