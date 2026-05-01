#!/usr/bin/env python3
"""
Group B: QAT training with FP8/FP4 quantization.

Usage:
    # QAT-FP8
    python src/experiments/train_qat.py --quant fp8 --max_steps 2000

    # QAT-FP4
    python src/experiments/train_qat.py --quant fp4 --max_steps 2000

This trains Micro-Gemma-FP with weight quantization simulated on the forward
pass (Straight-Through Estimator for backward).
"""

import argparse
import torch
import torch.nn as nn
from src.model.config import MicroGemmaFPConfig
from src.model.transformer import MicroGemmaFPForCausalLM
from src.quantization.fp_quantizer import FPQuantizer
from src.experiments.training_utils import (
    get_dataloader, train_epoch, save_checkpoint, load_checkpoint,
)


def make_qat_wrapper(linear: nn.Linear, quantizer: FPQuantizer,
                     stochastic: bool = False):
    """Wrap a linear layer to quantize weights before forward pass."""

    class QATLinear(nn.Module):
        def __init__(self, base: nn.Linear, q: FPQuantizer, sr: bool):
            super().__init__()
            self.weight = base.weight
            self.bias = base.bias
            self.q = q
            self.sr = sr
            self.in_features = base.in_features
            self.out_features = base.out_features

        def forward(self, x):
            w_q = self.q.quantize(self.weight, stochastic=self.sr)
            return nn.functional.linear(x, w_q, self.bias)

    return QATLinear(linear, quantizer, stochastic)


def qat_wrap_model(model: MicroGemmaFPForCausalLM, quantizer: FPQuantizer,
                   stochastic: bool = False):
    """Recursively replace all nn.Linear layers with QAT-wrapped versions."""
    _replace_linears(model, quantizer, stochastic, prefix='')


def _replace_linears(module, quantizer, stochastic, prefix):
    for name, child in list(module.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear):
            # Don't quantize embedding/lm_head
            if 'embed' in full_name.lower() or 'lm_head' in full_name.lower():
                continue
            setattr(module, name, make_qat_wrapper(child, quantizer, stochastic))
        else:
            _replace_linears(child, quantizer, stochastic, full_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quant', type=str, default='fp8',
                        choices=['fp8', 'fp4'])
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_steps', type=int, default=1500)
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--vocab_size', type=int, default=32000)
    parser.add_argument('--tier', type=str, default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fmt_name = {'fp8': 'fp8_e4m3', 'fp4': 'fp4_e2m1'}[args.quant]

    # Config
    config = MicroGemmaFPConfig(
        quantize_weights=args.quant,
        quantize_activations='none',
    )
    output_dir = args.output_dir or f'checkpoints/qat_{args.quant}'
    print(f"Model: {config.model_name}")
    print(f"Output: {output_dir}")

    # Model + QAT wrapping
    model = MicroGemmaFPForCausalLM(config).to(device)
    quantizer = FPQuantizer(fmt_name)
    qat_wrap_model(model, quantizer, stochastic=False)
    print(f"QAT wrapping applied: {args.quant}")

    stats = model.count_parameters()
    print(f"Parameters: {stats['total']:,} total")

    # Data
    train_loader = get_dataloader(args.batch_size, args.max_seq_len, args.max_steps,
                                   args.vocab_size, args.data_dir, args.tier)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    all_metrics = []
    if args.resume:
        all_metrics, _ = load_checkpoint(model, optimizer, args.resume, device)

    # Train
    import os
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nTraining QAT-{args.quant} for {args.max_steps} steps...")
    metrics = train_epoch(model, train_loader, optimizer, device,
                          max_steps=args.max_steps, log_interval=10)
    all_metrics.extend(metrics)

    for m in metrics:
        print(f"  step {m['step']:5d}  loss={m['loss']:.4f}  ppl={m['perplexity']:.2f}")

    ckpt_path = os.path.join(output_dir, 'model.pt')
    save_checkpoint(model, optimizer, all_metrics, ckpt_path, {
        'config': vars(config),
        'args': vars(args),
    })
    print(f"\nCheckpoint saved to {ckpt_path}")


if __name__ == '__main__':
    main()
