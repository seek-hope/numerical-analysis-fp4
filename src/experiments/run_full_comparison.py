#!/usr/bin/env python3
"""
Phase 5: Extended PTQ comparison — definitive 16-20 config evaluation.

Runs 16-20 effective PTQ configurations across 2 checkpoints, 2 formats,
and up to 6 quantization methods, collecting both PPL (100 validation steps)
and per-matrix ||dy||/||y|| for every configuration. Synthesizes GPTQ-vs-RTN
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
from src.quantization.fp4_grids import MXFP4Quantizer, FP4_E2M1_GRID, build_fp8_e4m3_grid
from src.quantization.outlier_rotation import DuQuantStyleQuantizer
from src.quantization.hadamard import hadamard_rotate_weight
from src.analysis.error_propagation import ErrorPropagationTracker
from src.experiments.training_utils import (
    get_dataloader,
    evaluate_perplexity,
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


# ── Task 2 placeholder: PPL eval, error measurement, comparison, summary table, main() loop ──
# Task 2 will add:
#   - capture_fp16_activations() — one-pass activation capture with fixed seed
#   - compute_per_matrix_errors() — manual ||dy||/||y|| computation (avoids Pitfall 1)
#   - eval_ppl() — PPL evaluation wrapper (split='val')
#   - main() with argparse, config loop, comparison analyses,
#     per-matrix summary table, and JSON export.
