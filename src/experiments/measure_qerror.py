#!/usr/bin/env python3
"""End-to-end per-matrix quantization error measurement for the Micro-Gemma-FP Transformer.

Loads the Micro-Gemma-FP model from an FP16 checkpoint, attaches the
ErrorPropagationTracker, runs a single forward pass on validation data,
computes per-matrix kappa(W) via exact SVD, per-matrix output-space
relative error ||dy||/||y|| and weight-space relative error ||dW||/||W||
via FP4 round-to-nearest quantization, validates the pipeline with a null
measurement, and outputs a structured results table.

Usage:
    python src/experiments/measure_qerror.py --checkpoint checkpoints/scaled_fp16_baseline/model.pt
    python src/experiments/measure_qerror.py --checkpoint <path> --output results/phase02.json
"""

import argparse
import sys

import torch

from src.analysis.error_propagation import ErrorPropagationTracker
from src.analysis.condition import compute_all_condition_numbers
from src.model.config import MicroGemmaFPConfig
from src.model.transformer import MicroGemmaFPForCausalLM
from src.quantization.fp_quantizer import FPQuantizer
from src.experiments.training_utils import get_dataloader, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(
        description="Per-matrix quantization error measurement pipeline."
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to FP16 .pt checkpoint file"
    )
    parser.add_argument(
        "--data_dir", type=str, default="data/real_tiers",
        help="Path to tokenized .bin data directory (default: data/real_tiers)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Optional JSON output path for results"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Compute device (default: cuda)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Device setup ────────────────────────────────────────────
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    # ── Model loading ───────────────────────────────────────────
    print("Loading model from checkpoint...")
    config = MicroGemmaFPConfig()
    model = MicroGemmaFPForCausalLM(config)
    load_checkpoint(model, None, args.checkpoint, device)
    model.to(device).eval()
    print(f"  Model loaded from {args.checkpoint} on {device}")
    stats = model.count_parameters()
    print(f"  Parameters: {stats['total']:,} total, {stats['trainable']:,} trainable")

    # ── Tracker setup and single forward pass ───────────────────
    print("\nSetting up ErrorPropagationTracker...")
    tracker = ErrorPropagationTracker()
    tracker.attach(model)

    dataloader = get_dataloader(
        batch_size=1, max_seq_len=512, split="val", data_dir=args.data_dir
    )
    batch = next(iter(dataloader))
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    print("Running single forward pass...")
    with torch.no_grad():
        model(input_ids, attention_mask=attention_mask)

    tracker.detach()
    tracker.compute_p3_p6()

    print(f"  Captured {len(tracker.activations)} Linear layer activations")
    print(f"  Captured {len(tracker.p_points)} measurement points")

    # ── Kappa computation ───────────────────────────────────────
    print("\nComputing condition numbers...")
    kappas_raw = compute_all_condition_numbers(model)
    # Strip .weight suffix to match error dict key convention
    kappas = {k.replace(".weight", ""): v for k, v in kappas_raw.items()}
    print(f"  Computed kappa for {len(kappas)} matrices")

    # ── Error computation (output-space + weight-space) ─────────
    print("\nComputing quantization errors...")
    quantizer = FPQuantizer(fmt="fp4_e2m1", per_channel=True)

    errors = tracker.compute_output_error(model, quantizer)
    print(f"  Computed ||dy||/||y|| for {len(errors)} matrices")

    dw_norms = {}
    for name, param in model.get_quantizable_weights():
        W_fp = param.data
        W_q = quantizer.quantize(W_fp)
        dw_norm_val = (W_q - W_fp).norm().item() / W_fp.norm().item()
        key = name.replace(".weight", "")
        dw_norms[key] = dw_norm_val
    print(f"  Computed ||dW||/||W|| for {len(dw_norms)} matrices")

    # ── Null measurement validation ─────────────────────────────
    print("\nRunning null measurement validation...")
    try:
        max_null_err = tracker.validate_null_measurement(model)
        print(f"  Null measurement: max error = {max_null_err:.2e} -- PASS")
    except ValueError as e:
        print(f"  Null measurement: FAILED -- {e}")
        sys.exit(1)

    print("\nMeasurement complete.")
    sys.exit(0)


if __name__ == "__main__":
    main()
