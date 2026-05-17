#!/usr/bin/env python3
"""
Phase 5: Extended PTQ comparison — definitive 16-20 config evaluation.

Runs 16-20 effective PTQ configurations across 2 checkpoints, 2 formats,
and up to 6 quantization methods, collecting per-matrix ||dy||/||y|| for every configuration. Synthesizes GPTQ-vs-RTN
and Lloyd-Max-vs-uniform comparisons, merges Phase 3/4 results into a 72-row
per-matrix summary table, and exports all results to JSON.

Configuration matrix:
  Checkpoints:  FP16 baseline, condition-number regularized
  Formats:      FP8 (E4M3), FP4 (E2M1)
  Methods:      RTN, GPTQ, Lloyd-Max (FP4 only),
                Hadamard+RTN (FP8 + optional FP4),
                Outlier rotation+RTN (FP8 + optional FP4),
                MXFP4 (FP4 only)

Usage:
    ./remote_python.sh src/experiments/run_full_comparison.py \
        --data_dir data/real_tiers

    # Partial execution (e.g., first 12 configs):
    ./remote_python.sh src/experiments/run_full_comparison.py \
        --data_dir data/real_tiers --config_start 0 --config_end 12
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.model.config import MicroGemmaFPConfig
from src.model.transformer import MicroGemmaFPForCausalLM
from src.quantization.fp_quantizer import FPQuantizer
from src.quantization.gptq import GPTQQuantizer
from src.quantization.adaptive_grid import AdaptiveGridQuantizer
from src.quantization.fp4_grids import MXFP4Quantizer, FP4_E2M1_GRID
from src.quantization.outlier_rotation import DuQuantStyleQuantizer
from src.quantization.hadamard import hadamard_rotate_weight
from src.analysis.error_propagation import ErrorPropagationTracker
from src.experiments.training_utils import (
    get_dataloader,
    load_checkpoint,
    MultiTierDataset,
    collate_batch,
)


# ═══════════════════════════════════════════════════════════════
# Utility functions (copied from validate_theorem1.py)
# ═══════════════════════════════════════════════════════════════

def _classify_matrix(module_path: str) -> tuple[int, str]:
    """Parse module path into (layer_index, matrix_type).

    Layer index: extracted from 'layers.N' subpath, default -1 for global.
    Matrix type: 'attention' for q_proj/k_proj/v_proj/o_proj,
                 'ffn' for gate_proj/up_proj/down_proj,
                 'global' for embed_tokens/lm_head.
    """
    parts = module_path.split(".")
    layer_idx = -1
    if "layers" in parts:
        idx = parts.index("layers") + 1
        if idx < len(parts):
            layer_idx = int(parts[idx])

    last_seg = parts[-1] if parts else "unknown"
    if last_seg in ("q_proj", "k_proj", "v_proj", "o_proj"):
        matrix_type = "attention"
    elif last_seg in ("gate_proj", "up_proj", "down_proj"):
        matrix_type = "ffn"
    elif last_seg in ("embed_tokens", "lm_head"):
        matrix_type = "global"
    else:
        matrix_type = last_seg

    return layer_idx, matrix_type


def _fmt_val(v: float, fmt: str = ".4f") -> str:
    """Format a float, rendering NaN/Inf as 'N/A'."""
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return "N/A"
    return f"{v:{fmt}}"


# ═══════════════════════════════════════════════════════════════
# FP8 E4M3 grid construction (256 levels)
# ═══════════════════════════════════════════════════════════════

def _build_fp8_e4m3_grid() -> torch.Tensor:
    """Build the full FP8 E4M3 grid (256 values).

    E4M3 has 1 sign bit, 4 exponent bits (bias=7), 3 mantissa bits.
    Finite values range: normal=2^{-6}..448, subnormal=2^{-9}..2^{-6}.
    Returns sorted absolute values (positive half) including zero.
    """
    exp_bias = 7
    values = [0.0]
    # Subnormals: exp=0, mantissa=1..7 => 2^{-6} * m/8
    for m in range(1, 8):
        values.append(2.0 ** (-6) * m / 8)
    # Normals: exp=1..15, mantissa=0..7
    for e in range(1, 15):  # exp=15 reserved for NaN/Inf
        for m in range(8):
            values.append((2.0 ** (e - exp_bias)) * (1 + m / 8))
    grid = torch.tensor(sorted(set(values)))
    return grid


# ═══════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════

def load_model(ckpt_path: str, device: torch.device) -> nn.Module:
    """Load a fresh model from checkpoint, eval mode."""
    config = MicroGemmaFPConfig()
    model = MicroGemmaFPForCausalLM(config).to(device)
    load_checkpoint(model, None, ckpt_path, device)
    model.eval()
    return model


# ═══════════════════════════════════════════════════════════════
# Quantizer application functions
# All use @torch.no_grad() and take (model, fmt_str, ...)
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def apply_rtn(model: nn.Module, fmt_str: str, **kwargs) -> None:
    """Round-to-nearest per-channel quantization on all quantizable weights."""
    q = FPQuantizer(fmt_str, per_channel=True)
    for name, param in model.get_quantizable_weights():
        param.data = q.quantize(param.data)


@torch.no_grad()
def apply_gptq(model: nn.Module, fmt_str: str, data_dir: str,
               device: torch.device, args) -> None:
    """GPTQ weight compensation with activation Hessian calibration.

    Uses split='train' for calibration data (D-05).
    """
    q = FPQuantizer(fmt_str, per_channel=True)
    calib_loader = get_dataloader(
        4, args.max_seq_len, args.calib_samples,
        data_dir=data_dir, split='train'
    )
    gptq = GPTQQuantizer(q)
    gptq.quantize_model(model, calib_loader, device)


@torch.no_grad()
def apply_lloyd_max(model: nn.Module, fmt_str: str, **kwargs) -> None:
    """Per-layer Lloyd-Max adaptive FP4 grid quantization.

    Uses uniform Lloyd-Max (kappa_weight=0.0) for fair comparison
    with RTN (D-06/D-07).
    """
    aq = AdaptiveGridQuantizer(kappa_weight=0.0)
    aq.calibrate(model)
    aq.quantize_model(model)


@torch.no_grad()
def apply_hadamard_rtn(model: nn.Module, fmt_str: str, **kwargs) -> None:
    """Hadamard rotation + RTN quantization.

    Hadamard is self-inverse up to scaling: applying twice = identity.
    """
    q = FPQuantizer(fmt_str, per_channel=True)
    for name, param in model.get_quantizable_weights():
        W = param.data
        W_rot = hadamard_rotate_weight(W)
        W_q = q.quantize(W_rot)
        param.data = hadamard_rotate_weight(W_q)


@torch.no_grad()
def apply_outlier_rtn(model: nn.Module, fmt_str: str, **kwargs) -> None:
    """Outlier-aware rotation + RTN quantization (DuQuant-style)."""
    if fmt_str == 'fp4_e2m1':
        grid = FP4_E2M1_GRID
    else:
        grid = _build_fp8_e4m3_grid()
    duquant = DuQuantStyleQuantizer(grid, block_size=32)
    for name, param in model.named_parameters():
        if param.dim() >= 2:
            param.data = duquant.quantize(param.data)


@torch.no_grad()
def apply_mxfp4(model: nn.Module, fmt_str: str, **kwargs) -> None:
    """MXFP4 block-scaling quantization."""
    mx = MXFP4Quantizer(block_size=32)
    for name, param in model.named_parameters():
        if param.dim() >= 2:
            param.data = mx.quantize(param.data)


# ═══════════════════════════════════════════════════════════════
# Method dispatch registry
# ═══════════════════════════════════════════════════════════════

METHOD_DISPATCH = {
    'rtn':       (apply_rtn,       False),
    'gptq':      (apply_gptq,      True),
    'lloyd_max': (apply_lloyd_max, False),   # calibrate() iterates params directly
    'hadamard':  (apply_hadamard_rtn, False),
    'outlier':   (apply_outlier_rtn, False),
    'mxfp4':     (apply_mxfp4,     False),
}


# ═══════════════════════════════════════════════════════════════
# Activation capture (FP16, once per checkpoint)
# ═══════════════════════════════════════════════════════════════

def capture_fp16_activations(
    model: nn.Module, data_dir: str, args, device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Capture pre-activations from one forward pass with fixed seed.

    Returns (activations_dict, batch) where activations_dict maps
    module_path -> input_tensor (CPU). Uses the same seed across all
    configs for cross-config comparability (Research.md Open Question #3).
    """
    torch.manual_seed(args.seed)

    ds = MultiTierDataset(data_dir, args.max_seq_len, split='val')
    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=0,
        pin_memory=True,
    )
    batch = next(iter(loader))
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch.get('attention_mask')
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    tracker = ErrorPropagationTracker()
    tracker.attach(model)
    with torch.no_grad():
        model(input_ids, attention_mask=attention_mask)
    tracker.detach()

    return tracker.activations, batch


# ═══════════════════════════════════════════════════════════════
# Per-matrix error computation (manual, avoids Pitfall 1)
# ═══════════════════════════════════════════════════════════════

def compute_per_matrix_errors(
    original_weights: dict[str, torch.Tensor],
    quantized_model: nn.Module,
    saved_activations: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, float]:
    """Compute per-matrix ||dy||/||y|| manually.

    Uses y_fp = x @ W_fp.T and y_q = x @ W_q.T for each matrix,
    avoiding compute_output_error's internal quantizer re-application
    (Pitfall 1: GPTQ/Lloyd-Max modify weights in-place and have no
    stateless quantizer interface).

    Args:
        original_weights: dict[name -> FP16 weight tensor]
        quantized_model: model after in-place quantization
        saved_activations: dict[module_path -> input_tensor (CPU)]
        device: compute device

    Returns:
        dict[module_path -> relative_error]
    """
    errors = {}
    q_params = dict(quantized_model.named_parameters())

    for name, W_fp in original_weights.items():
        module_path = name.replace('.weight', '') if name.endswith('.weight') else name

        if module_path not in saved_activations:
            continue

        x = saved_activations[module_path].to(device)

        # Get quantized weight
        if name in q_params:
            W_q = q_params[name].data
        else:
            continue

        y_fp = x @ W_fp.T
        y_q = x @ W_q.to(device).T

        denom = y_fp.norm().clamp(min=1e-12)
        err = ((y_q - y_fp).norm() / denom).item()
        errors[module_path] = err

    return errors


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extended PTQ comparison: evaluate 16-20 quantization configurations "
            "across 2 checkpoints, 2 formats, and up to 6 methods, measuring "
            "per-matrix ||dy||/||y|| for every configuration."
        )
    )

    # Checkpoint and data paths
    parser.add_argument('--fp16_checkpoint',
                        default='checkpoints/scaled_fp16_baseline/model.pt')
    parser.add_argument('--cond_checkpoint',
                        default='checkpoints/cond_regularized/model.pt')
    parser.add_argument('--data_dir', default='data/real_tiers')
    parser.add_argument('--output', default='results/full_comparison.json')

    # Hardware
    parser.add_argument('--device', default='cuda')

    # Evaluation
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument('--eval_steps', type=int, default=100)
    parser.add_argument('--calib_samples', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)

    # Experimental methods
    parser.add_argument('--include_experimental', action='store_true',
                        help='Also run Hadamard and Outlier rotation for FP4 format')

    # Re-entrant partial execution
    parser.add_argument('--config_start', type=int, default=None,
                        help='Starting config index for partial execution')
    parser.add_argument('--config_end', type=int, default=None,
                        help='Ending config index (exclusive) for partial execution')

    args = parser.parse_args()

    # ── Device setup ──────────────────────────────────────────
    device = torch.device(args.device if (
        args.device == 'cuda' and torch.cuda.is_available()
    ) else 'cpu')
    print(f"Device: {device}")

    # ── Checkpoint existence guard (D-13) ─────────────────────
    for label, path in [('FP16 baseline', args.fp16_checkpoint),
                        ('Condition-regularized', args.cond_checkpoint)]:
        if not os.path.exists(path):
            print(f"FileNotFoundError: {label} checkpoint not found: {path}",
                  file=sys.stderr)
            sys.exit(1)
    print(f"Checkpoints found: both exist")

    # ── Configuration registry ────────────────────────────────
    checkpoints = {
        'fp16_baseline': args.fp16_checkpoint,
        'cond_regularized': args.cond_checkpoint,
    }
    formats = {'FP8': 'fp8_e4m3', 'FP4': 'fp4_e2m1'}

    # Method table: (method_key, method_label, applicable_formats, needs_calibration)
    methods = [
        ('rtn',       'Round-to-nearest (per-channel)',          ['FP8', 'FP4'], False),
        ('gptq',      'GPTQ (column compensation)',              ['FP8', 'FP4'], True),
        ('lloyd_max', 'Lloyd-Max adaptive',                      ['FP4'],        True),
        ('hadamard',  'Hadamard rotation + RTN',                 ['FP8'],        False),
        ('outlier',   'Outlier rotation + RTN',                  ['FP8'],        False),
        ('mxfp4',     'MXFP4 block-scaling',                     ['FP4'],        False),
    ]

    if args.include_experimental:
        # Add FP4 to Hadamard and Outlier applicable formats
        for i in range(len(methods)):
            if methods[i][0] in ('hadamard', 'outlier'):
                m = list(methods[i])
                m[2] = m[2] + ['FP4']
                methods[i] = tuple(m)

    # Build flat config list for indexing
    config_list = []
    for ckpt_name, ckpt_path in checkpoints.items():
        for fmt_name, fmt_str in formats.items():
            for method_key, method_label, applicable_formats, needs_calib in methods:
                if fmt_name not in applicable_formats:
                    continue
                config_list.append({
                    'ckpt_name': ckpt_name,
                    'ckpt_path': ckpt_path,
                    'fmt_name': fmt_name,
                    'fmt_str': fmt_str,
                    'method_key': method_key,
                    'method_label': method_label,
                    'needs_calib': needs_calib,
                })

    # Apply config slice for partial execution
    config_start = args.config_start if args.config_start is not None else 0
    config_end = args.config_end if args.config_end is not None else len(config_list)
    config_slice = config_list[config_start:config_end]

    print(f"\n{'='*60}")
    print(f"Configuration matrix: {len(config_list)} total ({len(config_slice)} in slice)")
    print(f"{'='*60}")
    print(f"  Checkpoints: {', '.join(checkpoints.keys())}")
    print(f"  Formats: {', '.join(formats.keys())}")
    print(f"  Methods: {', '.join(m[0] for m in methods)}")
    print(f"  Slice: [{config_start}:{config_end}]")

    # ═══════════════════════════════════════════════════════════
    # Step 1: Per-checkpoint FP16 activation capture
    # ═══════════════════════════════════════════════════════════
    ckpt_activations: dict[str, dict[str, torch.Tensor]] = {}
    ckpt_weights: dict[str, dict[str, torch.Tensor]] = {}
    for ckpt_name, ckpt_path in checkpoints.items():
        print(f"\n--- Capturing FP16 activations: {ckpt_name} ---")
        model_ref = load_model(ckpt_path, device)

        # Capture activations
        activations, _ = capture_fp16_activations(model_ref, args.data_dir, args, device)
        ckpt_activations[ckpt_name] = activations

        # Save original weights
        orig_weights = {
            name: param.data.clone()
            for name, param in model_ref.get_quantizable_weights()
        }
        ckpt_weights[ckpt_name] = orig_weights

        print(f"  Captured {len(activations)} activations, "
              f"{len(orig_weights)} weight matrices")

        del model_ref
        torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════════════
    # Step 2: Main config evaluation loop
    # ═══════════════════════════════════════════════════════════
    all_results: dict = {}

    for cfg in config_slice:
        config_key = f"{cfg['ckpt_name']}/{cfg['fmt_name']}/{cfg['method_key']}"
        print(f"\n{'─'*60}")
        print(f"  Config {config_key}: {cfg['method_label']} ({cfg['fmt_name']})")
        print(f"{'─'*60}")

        model = load_model(cfg['ckpt_path'], device)
        t_start = time.time()

        try:
            # Apply quantization via method dispatch
            apply_fn, _ = METHOD_DISPATCH[cfg['method_key']]
            # Pass extra keyword args as needed
            kwargs = {
                'data_dir': args.data_dir,
                'device': device,
                'args': args,
            }
            apply_fn(model, cfg['fmt_str'], **kwargs)

            # Per-matrix error evaluation
            errors = compute_per_matrix_errors(
                ckpt_weights[cfg['ckpt_name']],
                model,
                ckpt_activations[cfg['ckpt_name']],
                device,
            )

            elapsed = time.time() - t_start
            mean_err = np.mean(list(errors.values())) if errors else float('nan')
            print(f"  mean||dy||/||y||={mean_err:.6f}  "
                  f"({elapsed:.1f}s)")

            all_results[config_key] = {
                'per_matrix_errors': errors,
            }

        except Exception as e:
            elapsed = time.time() - t_start
            print(f"  FAILED after {elapsed:.1f}s: {e}")
            all_results[config_key] = None

        finally:
            del model
            torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════════════
    # Step 3: GPTQ vs RTN comparison (COMP-02)
    # ═══════════════════════════════════════════════════════════
    comparisons = {'gptq_vs_rtn': {}, 'lloyd_max_vs_uniform': {}}

    for ckpt_name in checkpoints:
        for fmt_name in formats:
            rtn_key = f'{ckpt_name}/{fmt_name}/rtn'
            gptq_key = f'{ckpt_name}/{fmt_name}/gptq'
            rtn_data = all_results.get(rtn_key)
            gptq_data = all_results.get(gptq_key)

            if rtn_data is None or gptq_data is None:
                continue

            rtn_errs = rtn_data.get('per_matrix_errors', {})
            gptq_errs = gptq_data.get('per_matrix_errors', {})

            rtn_mean = np.mean(list(rtn_errs.values())) if rtn_errs else float('nan')
            gptq_mean = np.mean(list(gptq_errs.values())) if gptq_errs else float('nan')
            mean_dy_delta = gptq_mean - rtn_mean

            # Per-matrix delta (+ means GPTQ worse)
            all_matrix_keys = sorted(set(list(rtn_errs.keys()) + list(gptq_errs.keys())))
            per_matrix_delta = {}
            for k in all_matrix_keys:
                rv = rtn_errs.get(k, float('nan'))
                gv = gptq_errs.get(k, float('nan'))
                per_matrix_delta[k] = gv - rv

            # Per-layer-type subgroup summary
            attn_deltas = []
            ffn_deltas = []
            for k, delta in per_matrix_delta.items():
                _, mtype = _classify_matrix(k)
                if mtype == 'attention' and not (isinstance(delta, float) and math.isnan(delta)):
                    attn_deltas.append(delta)
                elif mtype == 'ffn' and not (isinstance(delta, float) and math.isnan(delta)):
                    ffn_deltas.append(delta)

            comp = {
                'mean_dy_delta': mean_dy_delta,
                'attn_mean_dy_delta': np.mean(attn_deltas) if attn_deltas else float('nan'),
                'ffn_mean_dy_delta': np.mean(ffn_deltas) if ffn_deltas else float('nan'),
                'rtn_mean_dy': rtn_mean,
                'gptq_mean_dy': gptq_mean,
                'per_matrix_delta': per_matrix_delta,
            }
            comparisons['gptq_vs_rtn'][f'{ckpt_name}/{fmt_name}'] = comp

            print(f"\n  GPTQ vs RTN [{ckpt_name}/{fmt_name}]:")
            print(f"    Mean ||dy||/||y||: {rtn_mean:.6f} -> {gptq_mean:.6f} (Δ={mean_dy_delta:+.6f})")
            print(f"    Attention mean Δ: {comp['attn_mean_dy_delta']:+.6f}")
            print(f"    FFN mean Δ: {comp['ffn_mean_dy_delta']:+.6f}")

    # ═══════════════════════════════════════════════════════════
    # Step 5: Lloyd-Max vs uniform E2M1 comparison (COMP-03)
    # ═══════════════════════════════════════════════════════════
    for ckpt_name in checkpoints:
        rtn_key = f'{ckpt_name}/FP4/rtn'
        lm_key = f'{ckpt_name}/FP4/lloyd_max'
        rtn_data = all_results.get(rtn_key)
        lm_data = all_results.get(lm_key)

        if rtn_data is None or lm_data is None:
            continue

        rtn_errs = rtn_data.get('per_matrix_errors', {})
        lm_errs = lm_data.get('per_matrix_errors', {})

        rtn_mean = np.mean(list(rtn_errs.values())) if rtn_errs else float('nan')
        lm_mean = np.mean(list(lm_errs.values())) if lm_errs else float('nan')
        mean_dy_delta = lm_mean - rtn_mean

        all_matrix_keys = sorted(set(list(rtn_errs.keys()) + list(lm_errs.keys())))
        per_matrix_delta = {}
        for k in all_matrix_keys:
            rv = rtn_errs.get(k, float('nan'))
            lv = lm_errs.get(k, float('nan'))
            per_matrix_delta[k] = lv - rv

        attn_deltas = []
        ffn_deltas = []
        for k, delta in per_matrix_delta.items():
            _, mtype = _classify_matrix(k)
            if mtype == 'attention' and not (isinstance(delta, float) and math.isnan(delta)):
                attn_deltas.append(delta)
            elif mtype == 'ffn' and not (isinstance(delta, float) and math.isnan(delta)):
                ffn_deltas.append(delta)

        comp = {
            'mean_dy_delta': mean_dy_delta,
            'attn_mean_dy_delta': np.mean(attn_deltas) if attn_deltas else float('nan'),
            'ffn_mean_dy_delta': np.mean(ffn_deltas) if ffn_deltas else float('nan'),
            'uniform_mean_dy': rtn_mean,
            'lloyd_max_mean_dy': lm_mean,
            'per_matrix_delta': per_matrix_delta,
        }
        comparisons['lloyd_max_vs_uniform'][ckpt_name] = comp

        print(f"\n  Lloyd-Max vs Uniform E2M1 [{ckpt_name}/FP4]:")
        print(f"    Mean ||dy||/||y||: {rtn_mean:.6f} -> {lm_mean:.6f} (Δ={mean_dy_delta:+.6f})")
        print(f"    Attention mean Δ: {comp['attn_mean_dy_delta']:+.6f}")
        print(f"    FFN mean Δ: {comp['ffn_mean_dy_delta']:+.6f}")

    # ═══════════════════════════════════════════════════════════
    # Step 6: Per-matrix summary table (REPORT-01, D-08)
    # ═══════════════════════════════════════════════════════════
    per_matrix_summary = []

    # Load Phase 3 JSON (theorem1_validation.json)
    phase3_path = 'results/theorem1_validation.json'
    phase3_data = None
    if os.path.exists(phase3_path):
        with open(phase3_path, 'r') as f:
            phase3_data = json.load(f)
        phase3_rows = {r['name']: r for r in phase3_data.get('results', [])}
        print(f"\n  Loaded Phase 3 data: {len(phase3_rows)} matrices from {phase3_path}")
    else:
        phase3_rows = {}
        print(f"\n  [WARN] Phase 3 data not found: {phase3_path}")

    # Load Phase 4 JSON (error_propagation_trace.json)
    phase4_path = 'results/error_propagation_trace.json'
    phase4_norm_attenuation = {}
    if os.path.exists(phase4_path):
        with open(phase4_path, 'r') as f:
            phase4_data = json.load(f)
        # Extract norm_attenuation from per-layer data
        # Structure varies — look for per-layer attenuation ratios
        for key, val in phase4_data.items():
            if isinstance(val, dict) and 'norm_attenuation' in val:
                phase4_norm_attenuation[key] = val['norm_attenuation']
            elif 'attenuation' in key.lower():
                phase4_norm_attenuation[key] = val
        print(f"  Loaded Phase 4 data: {len(phase4_norm_attenuation)} entries from {phase4_path}")
    else:
        print(f"  [WARN] Phase 4 data not found: {phase4_path}")

    # Get Phase 5 errors from FP16_baseline / FP4 / RTN (canonical config)
    canonical_key = 'fp16_baseline/FP4/rtn'
    canonical_errors = {}
    if canonical_key in all_results and all_results[canonical_key] is not None:
        canonical_errors = all_results[canonical_key].get('per_matrix_errors', {})
        print(f"  Canonical errors from {canonical_key}: {len(canonical_errors)} matrices")

    # Construct rows
    all_matrix_names = set(phase3_rows.keys()) | set(canonical_errors.keys())
    for name in sorted(all_matrix_names):
        p3 = phase3_rows.get(name, {})
        layer_idx, matrix_type = _classify_matrix(name)
        kappa = p3.get('kappa', float('nan'))
        dw_norm = p3.get('dw_norm', float('nan'))
        dy_norm = canonical_errors.get(name, float('nan'))
        tightness = p3.get('tightness_ratio', float('nan'))

        # Map norm_attenuation by layer index
        layer_key = f"layer_{layer_idx}" if layer_idx >= 0 else "global"
        norm_atten = phase4_norm_attenuation.get(layer_key, float('nan'))

        per_matrix_summary.append({
            'name': name,
            'layer': layer_idx,
            'type': matrix_type,
            'kappa': kappa,
            'dw_norm': dw_norm,
            'dy_norm': dy_norm,
            'tightness_ratio': tightness,
            'norm_attenuation': norm_atten,
        })

    # Sort: by layer ascending, then type priority (attention=0, ffn=1, global=2)
    type_rank = {'attention': 0, 'ffn': 1, 'global': 2}
    per_matrix_summary.sort(key=lambda r: (
        r['layer'] if r['layer'] >= 0 else 999,
        type_rank.get(r['type'], 99),
    ))

    # Print table
    print(f"\n{'='*140}")
    print("  Per-Matrix Summary Table (merged Phase 3/4/5)")
    print(f"{'='*140}")
    header = (
        f"  {'name':<50s}  {'layer':>5s}  {'type':>10s}  "
        f"{'kappa':>12s}  {'||dW||/||W||':>14s}  "
        f"{'||dy||/||y||':>14s}  {'tightness':>12s}  "
        f"{'norm_atten':>12s}"
    )
    print(header)
    print("  " + "-" * 138)
    for row in per_matrix_summary:
        k_str = _fmt_val(row['kappa'], '.2f')
        dw_str = _fmt_val(row['dw_norm'], '.6f')
        dy_str = _fmt_val(row['dy_norm'], '.6f')
        t_str = _fmt_val(row['tightness_ratio'], '.4f')
        na_str = _fmt_val(row['norm_attenuation'], '.4f')
        print(
            f"  {row['name']:<50s}  {row['layer']:>5d}  {row['type']:>10s}  "
            f"{k_str:>12s}  {dw_str:>14s}  {dy_str:>14s}  {t_str:>12s}  {na_str:>12s}"
        )
    print("  " + "-" * 138)
    print(f"  Total matrices: {len(per_matrix_summary)}")

    # ═══════════════════════════════════════════════════════════
    # Step 7: Summary comparison table (stdout)
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*100}")
    print("  Summary: All Configurations")
    print(f"{'='*100}")

    for ckpt_name in checkpoints:
        print(f"\n  Checkpoint: {ckpt_name}")
        print(f"  {'Config':<45s} {'Mean ||dy||/||y||':>20s}")
        print(f"  {'─'*70}")

        best_by_fmt: dict[str, tuple] = {}

        for cfg in config_list:
            if cfg['ckpt_name'] != ckpt_name:
                continue
            config_key = f"{cfg['ckpt_name']}/{cfg['fmt_name']}/{cfg['method_key']}"
            result = all_results.get(config_key)

            if result is None:
                print(f"  {config_key:<45s} {'FAIL':>20s}")
                continue

            errs = result.get('per_matrix_errors', {})
            mean_err = np.mean(list(errs.values())) if errs else float('nan')
            err_str = _fmt_val(mean_err, '.6f')

            print(f"  {config_key:<45s} {err_str:>20s}")

            # Track best per format (lowest ||dy||/||y||)
            fmt = cfg['fmt_name']
            if fmt not in best_by_fmt or mean_err < best_by_fmt[fmt][0]:
                best_by_fmt[fmt] = (mean_err, config_key)

        print(f"  {'─'*70}")
        for fmt, (err, key) in best_by_fmt.items():
            print(f"  Best {fmt}: {key:<30s} mean||dy||/||y||={err:.6f}")

    # GPTQ vs RTN summary
    print(f"\n{'='*60}")
    print("  GPTQ vs RTN Comparison")
    print(f"{'='*60}")
    for comp_key, comp_val in comparisons['gptq_vs_rtn'].items():
        print(f"  {comp_key}:")
        print(f"    Mean ||dy|| Δ={comp_val['mean_dy_delta']:+.6f}  "
              f"Attn Δ={comp_val['attn_mean_dy_delta']:+.6f}  "
              f"FFN Δ={comp_val['ffn_mean_dy_delta']:+.6f}")

    # Lloyd-Max vs Uniform summary
    print(f"\n{'='*60}")
    print("  Lloyd-Max vs Uniform E2M1 Comparison")
    print(f"{'='*60}")
    for comp_key, comp_val in comparisons['lloyd_max_vs_uniform'].items():
        print(f"  {comp_key}:")
        print(f"    Mean ||dy|| Δ={comp_val['mean_dy_delta']:+.6f}  "
              f"Attn Δ={comp_val['attn_mean_dy_delta']:+.6f}  "
              f"FFN Δ={comp_val['ffn_mean_dy_delta']:+.6f}")

    # ═══════════════════════════════════════════════════════════
    # Step 8: JSON export
    # ═══════════════════════════════════════════════════════════
    output_dict = {
        'configs': all_results,
        'comparisons': comparisons,
        'per_matrix_summary': per_matrix_summary,
        'metadata': {
            'checkpoints': list(checkpoints.keys()),
            'formats': list(formats.keys()),
            'methods': [m[0] for m in methods],
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'args': {
                'config_start': config_start,
                'config_end': config_end,
                'batch_size': args.batch_size,
                'max_seq_len': args.max_seq_len,
                'eval_steps': args.eval_steps,
                'calib_samples': args.calib_samples,
                'seed': args.seed,
                'include_experimental': args.include_experimental,
            },
        },
    }

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output_dict, f, indent=2)
    print(f"\n  Full comparison results saved to {args.output}")

    # Export per-matrix summary separately
    summary_output = 'results/per_matrix_summary.json'
    with open(summary_output, 'w') as f:
        json.dump(per_matrix_summary, f, indent=2)
    print(f"  Per-matrix summary saved to {summary_output}")

    print(f"\n{'='*60}")
    print("  Phase 5 comparison complete.")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
