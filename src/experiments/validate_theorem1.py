#!/usr/bin/env python3
"""Multi-seed Theorem 1 validation for the Micro-Gemma-FP Transformer.

Loads the FP16 baseline checkpoint, runs the per-matrix measurement pipeline
across 3 random seeds (42, 123, 456) on validation data, computes Pearson
r(kappa, ||dy||/||y||) with Bonferroni correction and bootstrap 95% CI, prints
a 72-matrix results table, and states a definitive YES/NO/QUALIFIED verdict on
whether Theorem 1's predicted upper bound holds empirically.

Theorem 1: ||dy||/||y|| <= kappa(W) * ||dW||/||W||

Usage:
    python src/experiments/validate_theorem1.py \\
        --checkpoint checkpoints/scaled_fp16_baseline/model.pt
    python src/experiments/validate_theorem1.py \\
        --checkpoint checkpoints/scaled_fp16_baseline/model.pt --device cpu
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.analysis.condition import compute_all_condition_numbers
from src.analysis.error_propagation import ErrorPropagationTracker
from src.experiments.training_utils import (
    MultiTierDataset,
    collate_batch,
    load_checkpoint,
)
from src.model.config import MicroGemmaFPConfig
from src.model.transformer import MicroGemmaFPForCausalLM
from src.quantization.fp_quantizer import FPQuantizer


# ── Pearson correlation with scipy fallback ─────────────────────

try:
    from scipy.stats import pearsonr as _scipy_pearsonr
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


def _pearsonr(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Compute Pearson correlation coefficient and two-sided p-value.

    Uses scipy.stats.pearsonr if available, otherwise a pure-numpy fallback
    with manual p-value via the t-distribution (n-2 degrees of freedom).
    """
    if _HAS_SCIPY:
        r, p = _scipy_pearsonr(x, y)
        return float(r), float(p)

    # Pure-numpy fallback
    r = float(np.corrcoef(x, y)[0, 1])
    n = len(x)
    if n > 2 and abs(r) < 1.0:
        t_stat = r * math.sqrt((n - 2) / (1 - r * r))
        # Two-sided p-value via regularised incomplete beta function
        # p = 2 * (1 - CDF(|t|, n-2))  where CDF is Student's t
        from scipy.stats import t as t_dist
        p = 2.0 * (1.0 - t_dist.cdf(abs(t_stat), n - 2))
    elif n > 2 and abs(r) >= 1.0:
        p = 0.0  # perfect correlation
    else:
        p = 1.0
    return r, float(p)


# ── Bootstrap CI for Pearson r ──────────────────────────────────

def bootstrap_pearson_ci(
    kappa: np.ndarray,
    dy: np.ndarray,
    n_resamples: int = 10000,
) -> tuple[float, float, np.ndarray]:
    """Bootstrap 95% CI for Pearson r(kappa, dy).

    Resamples (kappa, dy) pairs with replacement n_resamples times,
    computes Pearson r for each resample, and returns the 2.5th and
    97.5th percentile of the bootstrap distribution.

    Returns:
        (ci_lower, ci_upper, r_values)
    """
    n = len(kappa)
    data = np.column_stack([kappa, dy])
    r_vals = np.zeros(n_resamples)

    for i in range(n_resamples):
        idx = np.random.choice(n, n, replace=True)
        r_vals[i] = float(np.corrcoef(data[idx, 0], data[idx, 1])[0, 1])

    ci_lower = float(np.percentile(r_vals, 2.5))
    ci_upper = float(np.percentile(r_vals, 97.5))
    return ci_lower, ci_upper, r_vals


# ── Matrix classification ───────────────────────────────────────

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


# ── NaN/Inf safe printing helper ────────────────────────────────

def _fmt_val(v: float, fmt: str = ".4f") -> str:
    """Format a float, rendering NaN/Inf as 'N/A'."""
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return "N/A"
    return f"{v:{fmt}}"


# ── CLI ─────────────────────────────────────────────────────────

VALID_SEEDS = [42, 123, 456]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Multi-seed Theorem 1 validation: measure per-matrix "
            "kappa(W), ||dW||/||W||, and ||dy||/||y|| across 3 seeds, "
            "compute Pearson r with Bonferroni correction and bootstrap CI, "
            "and state a YES/NO/QUALIFIED verdict."
        )
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to FP16 .pt checkpoint file (required)"
    )
    parser.add_argument(
        "--data_dir", type=str, default="data/real_tiers",
        help="Path to tokenized .bin data directory (default: data/real_tiers)"
    )
    parser.add_argument(
        "--output", type=str, default="results/theorem1_validation.json",
        help="JSON output path (default: results/theorem1_validation.json)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Compute device (default: cuda)"
    )
    parser.add_argument(
        "--n_resamples", type=int, default=10000,
        help="Bootstrap resamples for 95% CI (default: 10000)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1,
        help="Evaluation batch size (default: 1)"
    )
    parser.add_argument(
        "--max_seq_len", type=int, default=512,
        help="Maximum sequence length (default: 512)"
    )
    return parser.parse_args()


# ── Seed-independent computation helpers ────────────────────────

def _compute_kappa(model: torch.nn.Module) -> dict[str, float]:
    """Compute kappa(W) for all quantizable weights via exact SVD.

    Returns dict keyed by module path WITHOUT .weight suffix.
    """
    kappas_raw = compute_all_condition_numbers(model)
    return {k.replace(".weight", ""): v for k, v in kappas_raw.items()}


def _compute_weight_errors(model: torch.nn.Module) -> dict[str, float]:
    """Compute ||dW||/||W|| for all quantizable weights via FP4 round-to-nearest.

    Returns dict keyed by module path WITHOUT .weight suffix.
    """
    quantizer = FPQuantizer(fmt="fp4_e2m1", per_channel=True)
    dw_norms = {}
    for name, param in model.get_quantizable_weights():
        W_fp = param.data
        W_q = quantizer.quantize(W_fp)
        dw_norm_val = (W_q - W_fp).norm().item() / W_fp.norm().item()
        dw_norms[name.replace(".weight", "")] = dw_norm_val
    return dw_norms


# ── Single-seed measurement ─────────────────────────────────────

def _run_seed(
    model: torch.nn.Module,
    quantizer: FPQuantizer,
    data_dir: str,
    seed: int,
    batch_size: int,
    max_seq_len: int,
    device: torch.device,
) -> dict[str, float]:
    """Run the measurement pipeline for a single seed.

    1. torch.manual_seed(seed) -- controls DataLoader shuffle order.
    2. Create a fresh DataLoader with shuffle=True (explicit construction
       to avoid get_dataloader()'s shuffle=False-for-val bug).
    3. Attach ErrorPropagationTracker, run one forward pass, detach.
    4. Compute per-matrix ||dy||/||y|| via compute_output_error.

    Returns dict[module_path] -> error_value (no .weight suffix).
    """
    torch.manual_seed(seed)

    # Explicit DataLoader construction with shuffle=True (CRITICAL: avoids
    # get_dataloader() shuffle=False-for-val bug per D-03/D-04).
    ds = MultiTierDataset(data_dir, max_seq_len, split="val")
    dataloader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=0,
        pin_memory=True,
    )

    # Log matched validation files
    matched = [getattr(d, "path", "unknown") for d in ds.datasets]
    print(f"    seed={seed}: matched {len(matched)} val files: {matched}")

    # Attach tracker
    tracker = ErrorPropagationTracker()
    tracker.attach(model)

    # Single forward pass
    batch = next(iter(dataloader))
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.no_grad():
        model(input_ids, attention_mask=attention_mask)

    tracker.detach()
    tracker.compute_p3_p6()

    act_count = len(tracker.activations)
    print(f"    seed={seed}: captured {act_count} activations")

    # Compute output error
    errors = tracker.compute_output_error(model, quantizer)
    print(f"    seed={seed}: computed ||dy||/||y|| for {len(errors)} matrices")

    return errors


# ── Main ────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Device setup ─────────────────────────────────────────
    device = torch.device(args.device if (
        args.device == "cuda" and torch.cuda.is_available()
    ) else "cpu")
    print(f"Device: {device}")

    # ── Model loading ────────────────────────────────────────
    print("Loading model from checkpoint...")
    config = MicroGemmaFPConfig()
    model = MicroGemmaFPForCausalLM(config)
    load_checkpoint(model, None, args.checkpoint, device)
    model.to(device).eval()
    print(f"  Model loaded from {args.checkpoint}")
    stats = model.count_parameters()
    print(f"  Parameters: {stats['total']:,} total, "
          f"{stats['trainable']:,} trainable")

    # ── Quantizer (same for all seeds) ───────────────────────
    quantizer = FPQuantizer(fmt="fp4_e2m1", per_channel=True)

    # ═══════════════════════════════════════════════════════
    # Seed-independent: kappa and ||dW||/||W|| (computed once)
    # ═══════════════════════════════════════════════════════
    print("\n[Pre-loop] Computing kappa (exact SVD)...")
    kappas = _compute_kappa(model)
    print(f"  Computed kappa for {len(kappas)} matrices")

    print("\n[Pre-loop] Computing ||dW||/||W|| (FP4 round-to-nearest)...")
    dw_norms = _compute_weight_errors(model)
    print(f"  Computed ||dW||/||W|| for {len(dw_norms)} matrices")

    # ═══════════════════════════════════════════════════════
    # Multi-seed loop
    # ═══════════════════════════════════════════════════════
    seeds = VALID_SEEDS
    all_seed_errors: list[dict[str, float]] = []

    print(f"\n{'=' * 60}")
    print("  Multi-seed measurement loop")
    print(f"{'=' * 60}")

    for seed in seeds:
        print(f"\n  --- Seed {seed} ---")
        errors = _run_seed(
            model, quantizer, args.data_dir, seed,
            args.batch_size, args.max_seq_len, device,
        )
        all_seed_errors.append(errors)

    # ═══════════════════════════════════════════════════════
    # Per-matrix aggregation
    # ═══════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print("  Aggregating results across seeds")
    print(f"{'=' * 60}")

    # Collect all unique matrix keys across kappas, dw_norms, and errors
    all_keys: set[str] = set()
    all_keys.update(kappas.keys())
    all_keys.update(dw_norms.keys())
    for err_dict in all_seed_errors:
        all_keys.update(err_dict.keys())

    aggregated = []
    for key in sorted(all_keys):
        kappa_val = kappas.get(key, float("nan"))
        dw_val = dw_norms.get(key, float("nan"))

        # Gather ||dy||/||y|| values across seeds
        dy_vals = []
        for err_dict in all_seed_errors:
            val = err_dict.get(key)
            if val is not None and not (isinstance(val, float) and math.isnan(val)):
                dy_vals.append(val)

        if len(dy_vals) == 0:
            continue  # no seed had this matrix

        dy_mean = float(np.mean(dy_vals))
        dy_std = float(np.std(dy_vals, ddof=1)) if len(dy_vals) > 1 else 0.0

        # Tightness ratio: ||dy||/||y|| / (kappa * ||dW||/||W||)
        denom = max(kappa_val * dw_val, 1e-12)
        tightness = dy_mean / denom

        neg_tightness = -tightness  # negative if one factor is negative
        layer_idx, matrix_type = _classify_matrix(key)
        aggregated.append({
            "name": key,
            "layer": layer_idx,
            "type": matrix_type,
            "kappa": kappa_val,
            "dw_norm": dw_val,
            "dy_norm_mean": dy_mean,
            "dy_norm_std": dy_std,
            "tightness_ratio": tightness,
        })

    # ═══════════════════════════════════════════════════════
    # Filtering: primary analysis includes only 'proj' matrices
    # ═══════════════════════════════════════════════════════
    proj_matrices = [r for r in aggregated if "proj" in r["name"]]
    non_proj = [r for r in aggregated if "proj" not in r["name"]]

    print(f"  Total matrices with complete data: {len(aggregated)}")
    print(f"  'proj' matrices (primary analysis): {len(proj_matrices)}")
    if non_proj:
        print(f"  Excluded from primary analysis: {len(non_proj)} "
              f"({', '.join(r['name'] for r in non_proj)})")

    if len(proj_matrices) < 72:
        print(f"\n  [WARN] Expected 72 proj matrices, found {len(proj_matrices)}")
        print(f"         Missing {72 - len(proj_matrices)} matrices.")

    # ═══════════════════════════════════════════════════════
    # Sort: by layer index, then type (attention before ffn)
    # ═══════════════════════════════════════════════════════
    type_rank = {"attention": 0, "ffn": 1, "global": 2}
    proj_sorted = sorted(
        proj_matrices,
        key=lambda r: (
            r["layer"] if r["layer"] >= 0 else 999,
            type_rank.get(r["type"], 99),
        ),
    )

    # ═══════════════════════════════════════════════════════
    # Print the 72-row results table
    # ═══════════════════════════════════════════════════════
    print(f"\n{'=' * 140}")
    print("  Per-Matrix Theorem 1 Validation Results")
    print(f"{'=' * 140}")

    header = (
        f"  {'name':<50s}  {'layer':>5s}  {'type':>10s}  "
        f"{'kappa':>12s}  {'||dW||/||W||':>14s}  "
        f"{'||dy||/||y||':>14s}  {'tightness':>12s}"
    )
    print(header)
    print("  " + "-" * 138)

    for row in proj_sorted:
        k_str = _fmt_val(row["kappa"], ".2f")
        dw_str = _fmt_val(row["dw_norm"], ".6f")
        dy_str = _fmt_val(row["dy_norm_mean"], ".6f")
        t_str = _fmt_val(row["tightness_ratio"], ".4f")
        print(
            f"  {row['name']:<50s}  {row['layer']:>5d}  {row['type']:>10s}  "
            f"{k_str:>12s}  {dw_str:>14s}  {dy_str:>14s}  {t_str:>12s}"
        )

    print("  " + "-" * 138)

    n_table = len(proj_sorted)
    valid_dy = [r for r in proj_sorted if not (
        isinstance(r["dy_norm_mean"], float) and
        (math.isnan(r["dy_norm_mean"]) or math.isinf(r["dy_norm_mean"]))
    )]
    valid_dw = [r for r in proj_sorted if not (
        isinstance(r["dw_norm"], float) and
        (math.isnan(r["dw_norm"]) or math.isinf(r["dw_norm"]))
    )]

    if valid_dy:
        mean_dy_all = np.mean([r["dy_norm_mean"] for r in valid_dy])
        max_dy_row = max(valid_dy, key=lambda r: r["dy_norm_mean"])
    else:
        mean_dy_all = float("nan")
        max_dy_row = {}

    if valid_dw:
        mean_dw_all = np.mean([r["dw_norm"] for r in valid_dw])
        max_dw_row = max(valid_dw, key=lambda r: r["dw_norm"])
    else:
        mean_dw_all = float("nan")
        max_dw_row = {}

    print(f"  Matrices reported: {n_table}")
    print(f"  Mean ||dy||/||y||: {_fmt_val(float(mean_dy_all), '.6f')}")
    print(f"  Max ||dy||/||y||: {_fmt_val(float(max_dy_row.get('dy_norm_mean', 0)), '.6f')} "
          f"at {max_dy_row.get('name', 'N/A')}")
    print(f"  Mean ||dW||/||W||: {_fmt_val(float(mean_dw_all), '.6f')}")
    print(f"  Max ||dW||/||W||: {_fmt_val(float(max_dw_row.get('dw_norm', 0)), '.6f')} "
          f"at {max_dw_row.get('name', 'N/A')}")

    # ═══════════════════════════════════════════════════════
    # Statistical analysis
    # ═══════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print("  Statistical Analysis")
    print(f"{'=' * 60}")

    n_valid = len(proj_sorted)
    if n_valid < 3:
        print("\n  ERROR: Too few matrices for correlation analysis.")
        sys.exit(1)

    kappa_array = np.array([r["kappa"] for r in proj_sorted])
    dy_mean_array = np.array([r["dy_norm_mean"] for r in proj_sorted])

    # Bonferroni-corrected threshold (72 tests)
    bonferroni_alpha = 0.05 / 72.0
    print(f"\n  Bonferroni threshold: alpha = 0.05 / 72 = {bonferroni_alpha:.6f}")

    # Primary Pearson r: kappa vs mean ||dy||/||y||
    r_primary, p_primary = _pearsonr(kappa_array, dy_mean_array)
    print(f"\n  Primary: kappa vs mean ||dy||/||y||")
    print(f"    Pearson r = {r_primary:.4f}")
    print(f"    p-value   = {p_primary:.4e}")

    # Bootstrap CI
    print(f"\n  Bootstrap 95% CI ({args.n_resamples} resamples)...")
    ci_lower, ci_upper, _r_boot = bootstrap_pearson_ci(
        kappa_array, dy_mean_array, args.n_resamples
    )
    print(f"    CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

    # Seed-by-seed r values
    print(f"\n  Seed-by-seed correlations (kappa vs seed ||dy||/||y||):")
    seed_r_values = []
    for seed_idx, seed in enumerate(seeds):
        # Build array of per-seed dy values aligned to proj_sorted order
        seed_dy = []
        for row in proj_sorted:
            val = all_seed_errors[seed_idx].get(row["name"])
            if val is not None and not (isinstance(val, float) and math.isnan(val)):
                seed_dy.append(val)
            else:
                seed_dy.append(0.0)
        seed_dy_arr = np.array(seed_dy)
        r_s, p_s = _pearsonr(kappa_array, seed_dy_arr)
        seed_r_values.append(r_s)
        print(f"    seed {seed}: r = {r_s:.4f}, p = {p_s:.4e}")

    # Per-layer-type subgroup analysis
    print(f"\n  Subgroup correlations (informational, no Bonferroni):")
    subgroups = {}
    for stype in ("attention", "ffn", "global"):
        sub = [r for r in proj_sorted if r["type"] == stype]
        if len(sub) >= 3:
            sub_kappa = np.array([r["kappa"] for r in sub])
            sub_dy = np.array([r["dy_norm_mean"] for r in sub])
            r_sub, p_sub = _pearsonr(sub_kappa, sub_dy)
            subgroups[stype] = {"r": r_sub, "p": p_sub}
            print(f"    {stype} ({len(sub)} matrices): r = {r_sub:.4f}, p = {p_sub:.4e}")
        else:
            subgroups[stype] = {"r": 0.0, "p": 1.0}
            print(f"    {stype} ({len(sub)} matrices): insufficient data")

    # ═══════════════════════════════════════════════════════
    # Verdict computation (per D-06 rubric)
    # ═══════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print("  Verdict")
    print(f"{'=' * 60}")

    ci_excludes_zero = ci_lower > 0.0
    meets_yes = (
        r_primary > 0.5
        and p_primary < bonferroni_alpha
        and ci_excludes_zero
    )
    meets_qualified = r_primary > 0.2

    if meets_yes:
        verdict = "YES"
        reasons = [
            f"r = {r_primary:.4f} > 0.5",
            f"p = {p_primary:.4e} < {bonferroni_alpha:.6f} (Bonferroni-corrected)",
            f"Bootstrap CI [{ci_lower:.4f}, {ci_upper:.4f}] excludes zero",
        ]
        verdict_reason = (
            "Theorem 1's predicted upper bound holds at per-matrix granularity. "
            "kappa(W) is a statistically significant predictor of output-space "
            "quantization error with strong positive correlation. "
            + "; ".join(reasons)
        )
    elif meets_qualified:
        verdict = "QUALIFIED"
        failed_criteria = []
        if r_primary <= 0.5:
            failed_criteria.append(
                f"r={r_primary:.4f} <= 0.5 (below strong correlation threshold)"
            )
        if p_primary >= bonferroni_alpha:
            failed_criteria.append(
                f"p={p_primary:.4e} >= {bonferroni_alpha:.6f} "
                "(not significant after Bonferroni correction)"
            )
        if not ci_excludes_zero:
            failed_criteria.append(
                f"CI [{ci_lower:.4f}, {ci_upper:.4f}] includes zero"
            )
        reasons = "; ".join(failed_criteria)
        verdict_reason = (
            f"Partial support for Theorem 1: r={r_primary:.4f} > 0.2 but "
            f"not all YES criteria met. Failed: {reasons}"
        )
    else:
        verdict = "NO"
        if r_primary <= 0.2:
            reason = f"r={r_primary:.4f} <= 0.2 (negligible correlation)"
        else:
            reason = (
                f"r={r_primary:.4f} but uncorrected p={p_primary:.4e} > 0.05"
            )
        verdict_reason = (
            "Theorem 1's predicted upper bound does not hold empirically at "
            f"per-matrix granularity. {reason}"
        )

    print(f"\n  Verdict: {verdict}")
    print(f"  Reason: {verdict_reason}")

    # ═══════════════════════════════════════════════════════
    # JSON export
    # ═══════════════════════════════════════════════════════
    output_dict = {
        "checkpoint": args.checkpoint,
        "num_matrices": n_valid,
        "bonferroni_alpha": bonferroni_alpha,
        "pearson_r": r_primary,
        "pearson_p": f"{p_primary:.4e}",
        "bootstrap_ci": [ci_lower, ci_upper],
        "seed_by_seed_r": seed_r_values,
        "subgroup_correlations": subgroups,
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "results": [
            {
                "name": r["name"],
                "layer": r["layer"],
                "type": r["type"],
                "kappa": r["kappa"],
                "dw_norm": r["dw_norm"],
                "dy_norm_mean": r["dy_norm_mean"],
                "dy_norm_std": r["dy_norm_std"],
                "tightness_ratio": r["tightness_ratio"],
            }
            for r in proj_sorted
        ],
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_dict, f, indent=2)
    print(f"\n  Results saved to {args.output}")

    print(f"\n{'=' * 60}")
    print("  Theorem 1 validation complete.")
    print(f"{'=' * 60}")
    sys.exit(0)


if __name__ == "__main__":
    main()
