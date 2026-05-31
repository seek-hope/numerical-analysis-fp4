#!/usr/bin/env python3
"""
Strategy B: Train with condition number regularization, then evaluate PTQ.

Hypothesis: Adding λ·log κ(W) during FP16 training produces weights with
lower condition numbers, which are more quantization-friendly. This should
reduce PTQ degradation compared to the unregularized baseline.

Usage:
    # Train with κ regularization
    ./remote_python.sh src/experiments/train_cond_regularized.py \\
        --data_dir data/real_tiers --lambda_cond 1e-4 --max_steps 2000

    # Compare PTQ degradation
    ./remote_python.sh src/experiments/train_cond_regularized.py \\
        --data_dir data/real_tiers --eval_only --lambda_cond 1e-4
"""

import argparse, os
import torch
from src.model.config import MicroGemmaFPConfig
from src.model.transformer import MicroGemmaFPForCausalLM
from src.analysis.condition import (
    condition_number_regularization, compute_all_condition_numbers,
)
from src.experiments.training_utils import (
    get_dataloader, train_epoch,
    save_checkpoint, load_checkpoint,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_steps', type=int, default=2000)
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--lambda_cond', type=float, default=1e-4,
                        help='Condition number regularization strength')
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str,
                        default='checkpoints/cond_regularized')
    parser.add_argument('--eval_only', action='store_true',
                        help='Skip training, just analyze condition numbers on existing checkpoint')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = MicroGemmaFPConfig(lambda_cond=args.lambda_cond)

    # ── Training ──
    if not args.eval_only:
        model = MicroGemmaFPForCausalLM(config).to(device)
        print(f"Training with λ_cond={args.lambda_cond}")
        print(f"  Params: {model.count_parameters()['total']:,}")

        loader = get_dataloader(args.batch_size, args.max_seq_len,
                                 args.max_steps, data_dir=args.data_dir)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

        def cond_reg_fn(m):
            return condition_number_regularization(m, args.lambda_cond)

        os.makedirs(args.output_dir, exist_ok=True)

        print(f"\nTraining {args.max_steps} steps...")
        metrics = train_epoch(model, loader, optimizer, device,
                              max_steps=args.max_steps, log_interval=25,
                              cond_reg_fn=cond_reg_fn)

        for m in metrics[-5:]:
            print(f"  step {m['step']:5d}  loss={m['loss']:.4f}")

        save_checkpoint(model, optimizer, metrics,
                        os.path.join(args.output_dir, 'model.pt'),
                        {'lambda_cond': args.lambda_cond,
                         'args': vars(args)})
        print(f"Saved to {args.output_dir}/model.pt")
    else:
        model = MicroGemmaFPForCausalLM(config).to(device)
        load_checkpoint(model, None,
                        os.path.join(args.output_dir, 'model.pt'), device)

    # ── Condition number analysis ──
    print(f"\n{'='*55}")
    print("Condition number analysis")
    kappa_dict = compute_all_condition_numbers(model)
    kappas = list(kappa_dict.values())
    avg_kappa = sum(kappas) / len(kappas)
    print(f"  Average κ: {avg_kappa:.1f}")
    print(f"  Min κ: {min(kappas):.1f}, Max κ: {max(kappas):.1f}")
    print(f"  Layers with κ > 10: {sum(1 for k in kappas if k > 10)}")

    # ── PTQ evaluation ──
    # PTQ comparison is now handled by run_full_comparison.py using
    # per-matrix ||dy||/||y|| and total ||ΔWX||/||WX|| metrics.
    # See results/full_comparison.json for the complete comparison.


if __name__ == '__main__':
    main()
