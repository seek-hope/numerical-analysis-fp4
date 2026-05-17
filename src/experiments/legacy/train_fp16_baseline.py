#!/usr/bin/env python3
"""
Group A0: Standard FP16 baseline training of Micro-Gemma-FP.

Usage:
    python src/experiments/train_fp16_baseline.py [--batch_size 8] [--max_steps 2000]

This establishes the reference point against which all quantized experiments
are compared.
"""

import argparse
import torch
from src.model.config import MicroGemmaFPConfig
from src.model.transformer import MicroGemmaFPForCausalLM
from src.experiments.training_utils import (
    get_dataloader, train_epoch, evaluate_perplexity,
    save_checkpoint, load_checkpoint,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_steps', type=int, default=1500)
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--output_dir', type=str, default='checkpoints/fp16_baseline')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--vocab_size', type=int, default=32000)
    parser.add_argument('--tier', type=str, default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Config
    config = MicroGemmaFPConfig(
        quantize_weights='none',
        quantize_activations='none',
    )
    print(f"Model: {config.model_name}")

    # Model
    model = MicroGemmaFPForCausalLM(config).to(device)
    stats = model.count_parameters()
    print(f"Parameters: {stats['total']:,} total, {stats['trainable']:,} trainable")

    # Data
    train_loader = get_dataloader(args.batch_size, args.max_seq_len, args.max_steps,
                                   args.vocab_size, args.data_dir, args.tier)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    start_step = 0
    all_metrics = []
    if args.resume:
        all_metrics, _ = load_checkpoint(model, optimizer, args.resume, device)
        start_step = all_metrics[-1]['step'] + 1 if all_metrics else 0

    # Train
    import os
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\nTraining for {args.max_steps} steps...")
    metrics = train_epoch(model, train_loader, optimizer, device,
                          max_steps=args.max_steps, log_interval=10)
    all_metrics.extend(metrics)

    for m in metrics:
        print(f"  step {m['step']:5d}  loss={m['loss']:.4f}  ppl={m['perplexity']:.2f}")

    # Save
    ckpt_path = os.path.join(args.output_dir, 'model.pt')
    save_checkpoint(model, optimizer, all_metrics, ckpt_path, {
        'config': vars(config),
        'args': vars(args),
    })
    print(f"\nCheckpoint saved to {ckpt_path}")


if __name__ == '__main__':
    main()
