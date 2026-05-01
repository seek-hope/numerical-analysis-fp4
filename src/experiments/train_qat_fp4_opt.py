#!/usr/bin/env python3
"""
FP4 QAT with numerical optimization techniques.

Compares three strategies vs vanilla QAT-FP4 (which achieved train PPL 1.01
but eval PPL 13.86 — severe overfitting):

  1. Stochastic Rounding (SR)  — unbiased gradient estimation
  2. Adaptive Precision        — switch FP4→FP8 when gradient norm too low
  3. SR + Adaptive             — combined

Usage:
    python src/experiments/train_qat_fp4_opt.py --strategy sr
    python src/experiments/train_qat_fp4_opt.py --strategy adaptive
    python src/experiments/train_qat_fp4_opt.py --strategy combined
"""

import os, math, argparse, json
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.config import MicroGemmaFPConfig
from src.model.transformer import MicroGemmaFPForCausalLM
from src.quantization.fp_quantizer import FPQuantizer
from src.experiments.training_utils import (
    get_dataloader, save_checkpoint, load_checkpoint,
)


# ═══════════════════════════════════════════════════════════
# QAT Linear with stochastic rounding
# ═══════════════════════════════════════════════════════════

class QATLinearSR(nn.Module):
    """QAT Linear with stochastic rounding for FP4 quantization."""

    def __init__(self, base: nn.Linear, quantizer: FPQuantizer,
                 stochastic: bool = True):
        super().__init__()
        self.weight = base.weight
        self.bias = base.bias
        self.q = quantizer
        self.stochastic = stochastic
        self.in_features = base.in_features
        self.out_features = base.out_features

    def forward(self, x):
        W_q = self.q.quantize(self.weight, stochastic=self.stochastic)
        return F.linear(x, W_q, self.bias)


def wrap_model_qat(model, quantizer, stochastic=True):
    """Replace all nn.Linear with QATLinearSR."""
    def _wrap(module, prefix=''):
        for name, child in list(module.named_children()):
            full = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear):
                if 'embed' in full.lower() or 'lm_head' in full.lower():
                    continue
                setattr(module, name, QATLinearSR(child, quantizer, stochastic))
            else:
                _wrap(child, full)
    _wrap(model)


# ═══════════════════════════════════════════════════════════
# Training with adaptive precision and gradient monitoring
# ═══════════════════════════════════════════════════════════

def train_with_metrics(model, dataloader, optimizer, device,
                       quantizer_fp4, quantizer_fp8,
                       max_steps=1500, adaptive=False,
                       grad_norm_threshold=1e-4, log_interval=50):
    """
    Train with gradient norm monitoring and optional adaptive precision.

    When adaptive=True and grad_norm < threshold:
      - Switch to FP8 for that step (more stable)
      - Log the switch event
    """
    model.train()
    metrics = []
    switches = []  # (step, grad_norm) when switching to FP8
    current_precision = 'fp4'

    for step, batch in enumerate(dataloader):
        if step >= max_steps:
            break

        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        # Check gradient norm from previous step
        if adaptive and step > 0 and metrics:
            prev_grad = metrics[-1].get('grad_norm', float('inf'))
            if prev_grad < grad_norm_threshold and current_precision == 'fp4':
                current_precision = 'fp8'
                switches.append((step, prev_grad))
                # Update all QAT linears to use FP8
                wrap_model_qat(model, quantizer_fp8, stochastic=True)

        out = model(input_ids, labels=labels)
        loss = out['loss']

        optimizer.zero_grad()
        loss.backward()

        # Compute gradient norm
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.norm().item() ** 2
        grad_norm = math.sqrt(total_norm)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % log_interval == 0:
            metrics.append({
                'step': step,
                'loss': loss.item(),
                'perplexity': math.exp(loss.item()),
                'grad_norm': grad_norm,
                'precision': current_precision,
            })

    return metrics, switches


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', default='sr',
                        choices=['sr', 'adaptive', 'combined'])
    parser.add_argument('--max_steps', type=int, default=1500)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--vocab_size', type=int, default=32000)
    parser.add_argument('--tier', type=str, default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = args.output_dir or f'checkpoints/qat_fp4_{args.strategy}'
    os.makedirs(output_dir, exist_ok=True)

    use_sr = args.strategy in ('sr', 'combined')
    use_adaptive = args.strategy in ('adaptive', 'combined')

    print(f"FP4 QAT Optimization: strategy={args.strategy}")
    print(f"  Stochastic Rounding: {use_sr}")
    print(f"  Adaptive Precision:  {use_adaptive}")
    print(f"  Output: {output_dir}")

    # Model
    config = MicroGemmaFPConfig()
    model = MicroGemmaFPForCausalLM(config).to(device)
    quantizer_fp4 = FPQuantizer('fp4_e2m1')
    quantizer_fp8 = FPQuantizer('fp8_e4m3')
    wrap_model_qat(model, quantizer_fp4, stochastic=use_sr)
    print(f"  Params: {model.count_parameters()['total']:,}")

    # Data
    train_loader = get_dataloader(8, 512, args.max_steps,
                                   args.vocab_size, args.data_dir, args.tier)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Train
    print(f"\nTraining {args.max_steps} steps...")
    metrics, switches = train_with_metrics(
        model, train_loader, optimizer, device,
        quantizer_fp4, quantizer_fp8,
        max_steps=args.max_steps,
        adaptive=use_adaptive,
        log_interval=10,
    )

    # Print summary
    final = metrics[-1] if metrics else None
    if final:
        print(f"\nFinal step {final['step']}: loss={final['loss']:.4f} "
              f"ppl={final['perplexity']:.2f} grad_norm={final['grad_norm']:.6f}")

    if switches:
        print(f"\nPrecision switches: {len(switches)}")
        for step, gn in switches[:5]:
            print(f"  step {step}: grad_norm={gn:.6f} → switched to FP8")

    # Save
    save_checkpoint(model, optimizer, metrics,
                    os.path.join(output_dir, 'model.pt'), {
        'strategy': args.strategy,
        'use_sr': use_sr,
        'use_adaptive': use_adaptive,
    })

    # Save metrics separately for analysis
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved to {output_dir}/")


if __name__ == '__main__':
    main()
