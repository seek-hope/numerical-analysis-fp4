#!/usr/bin/env python3
"""
Group C: QAT + Numerical Optimization.

Combines all three optimization dimensions:
  - Condition number regularization (λ·log κ(W))
  - Stochastic rounding (unbiased gradient estimation)
  - Hadamard rotation (outlier reduction via orthogonal transform)

Usage:
    # QAT-FP8 with all optimizations
    python src/experiments/train_qat_optimized.py --quant fp8 \\
        --lambda_cond 1e-4 --stochastic --hadamard

    # QAT-FP4 with optimizations
    python src/experiments/train_qat_optimized.py --quant fp4 \\
        --lambda_cond 1e-3 --stochastic --hadamard

    # Ablation: condition number only
    python src/experiments/train_qat_optimized.py --quant fp8 --lambda_cond 1e-4
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.config import MicroGemmaFPConfig
from src.model.transformer import MicroGemmaFPForCausalLM
from src.quantization.fp_quantizer import FPQuantizer
from src.quantization.hadamard import hadamard_rotate_weight, hadamard_rotate_activation
from src.analysis.condition import condition_number_regularization
from src.experiments.training_utils import (
    get_dataloader, train_epoch, save_checkpoint, load_checkpoint,
)


def make_qat_optimized_linear(base: nn.Linear, quantizer: FPQuantizer,
                               stochastic: bool, hadamard: bool,
                               layer_name: str):
    """Wrap a linear layer with QAT + optional Hadamard rotation.

    When hadamard=True: quantize the Hadamard-rotated weight, then apply
    inverse Hadamard on the output (QuIP/QuaRot pattern).
    """
    class QATOptimizedLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = base.weight
            self.bias = base.bias
            self.q = quantizer
            self.stochastic = stochastic
            self.hadamard = hadamard
            self.in_features = base.in_features
            self.out_features = base.out_features
            self.layer_name = layer_name

        def forward(self, x):
            W = self.weight
            if self.hadamard and self.in_features > 16 and self.out_features > 16:
                with torch.no_grad():
                    W_rot = hadamard_rotate_weight(W)
                    x_rot = hadamard_rotate_activation(x)
                W_q = self.q.quantize(W_rot, stochastic=self.stochastic)
                y = F.linear(x_rot, W_q, self.bias)
                # Inverse Hadamard (self-inverse up to scale)
                with torch.no_grad():
                    y = hadamard_rotate_activation(y)
                return y
            else:
                W_q = self.q.quantize(W, stochastic=self.stochastic)
                return F.linear(x, W_q, self.bias)

    return QATOptimizedLinear()


def qat_optimized_wrap(model: MicroGemmaFPForCausalLM, quantizer: FPQuantizer,
                        stochastic: bool, hadamard: bool):
    """Replace all nn.Linear layers with optimized QAT versions."""
    _replace(module=model, q=quantizer, sr=stochastic, hm=hadamard, prefix='')


def _replace(module, q, sr, hm, prefix):
    for name, child in list(module.named_children()):
        full = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear):
            if 'embed' in full.lower() or 'lm_head' in full.lower():
                continue
            setattr(module, name,
                    make_qat_optimized_linear(child, q, sr, hm, full))
        else:
            _replace(child, q, sr, hm, full)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quant', type=str, default='fp8', choices=['fp8', 'fp4'])
    parser.add_argument('--lambda_cond', type=float, default=0.0)
    parser.add_argument('--stochastic', action='store_true')
    parser.add_argument('--hadamard', action='store_true')
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

    config = MicroGemmaFPConfig(
        quantize_weights=args.quant,
        stochastic_rounding=args.stochastic,
        hadamard_rotation=args.hadamard,
        lambda_cond=args.lambda_cond,
    )

    # Auto output dir
    if args.output_dir is None:
        parts = [f'qat_{args.quant}']
        if args.lambda_cond > 0:
            parts.append(f'cond{args.lambda_cond:.0e}')
        if args.stochastic:
            parts.append('sr')
        if args.hadamard:
            parts.append('hadamard')
        args.output_dir = f'checkpoints/{"_".join(parts)}'

    print(f"Model: {config.model_name}")
    print(f"Output: {args.output_dir}")
    print(f"Config: cond_reg={args.lambda_cond}, sr={args.stochastic}, "
          f"hadamard={args.hadamard}")

    # Model
    model = MicroGemmaFPForCausalLM(config).to(device)
    quantizer = FPQuantizer(fmt_name)
    qat_optimized_wrap(model, quantizer, args.stochastic, args.hadamard)
    print(f"Optimized QAT wrapping applied: {args.quant}")

    # Cond reg function
    cond_reg_fn = None
    if args.lambda_cond > 0:
        def cond_reg_fn(m):
            return condition_number_regularization(m, args.lambda_cond)

    print(f"Parameters: {model.count_parameters()['total']:,} total")

    # Data
    train_loader = get_dataloader(args.batch_size, args.max_seq_len, args.max_steps,
                                   args.vocab_size, args.data_dir, args.tier)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    all_metrics = []
    if args.resume:
        all_metrics, _ = load_checkpoint(model, optimizer, args.resume, device)

    # Train
    import os
    os.makedirs(args.output_dir, exist_ok=True)

    flags = []
    if args.lambda_cond > 0:
        flags.append(f'cond={args.lambda_cond}')
    if args.stochastic:
        flags.append('SR')
    if args.hadamard:
        flags.append('Hadamard')
    print(f"\nTraining QAT-{args.quant} [+{' + '.join(flags)}] "
          f"for {args.max_steps} steps...")

    metrics = train_epoch(model, train_loader, optimizer, device,
                          max_steps=args.max_steps, log_interval=10,
                          cond_reg_fn=cond_reg_fn)
    all_metrics.extend(metrics)

    for m in metrics:
        print(f"  step {m['step']:5d}  loss={m['loss']:.4f}  ppl={m['perplexity']:.2f}")

    ckpt_path = os.path.join(args.output_dir, 'model.pt')
    save_checkpoint(model, optimizer, all_metrics, ckpt_path, {
        'config': vars(config),
        'args': vars(args),
    })
    print(f"\nCheckpoint saved to {ckpt_path}")


if __name__ == '__main__':
    main()
