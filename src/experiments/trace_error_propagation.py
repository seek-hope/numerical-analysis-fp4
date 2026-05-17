#!/usr/bin/env python3
"""Error propagation trace for the Micro-Gemma-FP Transformer.

Measures per-source quantization error through all 6 P-points (P0-P6) for
layers 0, 5, and 11. For each of 21 source matrices (7 matrices x 3 layers),
quantizes only that single weight matrix to FP4 E2M1, runs a forward pass,
and computes relative error at each P-point of the source's own layer.
Also computes RMSNorm attenuation ratios and parallel/orthogonal error
decomposition for all 12 layers from the same quantized passes.

Usage:
    python src/experiments/trace_error_propagation.py \
        --checkpoint checkpoints/scaled_fp16_baseline/model.pt
    python src/experiments/trace_error_propagation.py \
        --checkpoint checkpoints/scaled_fp16_baseline/model.pt --device cpu
"""

import argparse
import json
import math
import os
import sys

import torch
from torch.utils.data import DataLoader

from src.analysis.error_propagation import ErrorPropagationTracker
from src.experiments.training_utils import (
    MultiTierDataset,
    collate_batch,
    load_checkpoint,
)
from src.model.config import MicroGemmaFPConfig
from src.model.transformer import MicroGemmaFPForCausalLM
from src.quantization.fp_quantizer import FPQuantizer


# ── Constants ─────────────────────────────────────────────────────

TRACED_LAYERS = [0, 5, 11]
P_POINTS = ['P0', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6']


# ── Matrix classification ────────────────────────────────────────

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


# ── Module resolution ─────────────────────────────────────────────

def _resolve_module(model: torch.nn.Module, module_path: str):
    """Resolve a module from a dot-separated path.

    Splits module_path by '.' and traverses model attributes.
    Returns the resolved nn.Module or None if not found.
    """
    obj = model
    for part in module_path.split('.'):
        if hasattr(obj, part):
            obj = getattr(obj, part)
        else:
            return None
    return obj


# ── CLI ────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Error propagation trace: measure per-source quantization error "
            "through all 6 P-points for layers 0, 5, and 11, plus RMSNorm "
            "attenuation and decomposition for all 12 layers."
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
        "--output", type=str, default="results/error_propagation_trace.json",
        help="JSON output path (default: results/error_propagation_trace.json)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Compute device (default: cuda)"
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


# ── NaN/Inf safe printing helper ──────────────────────────────────

def _fmt_val(v: float, fmt: str = ".4f") -> str:
    """Format a float, rendering NaN/Inf as 'N/A'."""
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return "N/A"
    return f"{v:{fmt}}"


# ── Main ───────────────────────────────────────────────────────────

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

    # ── Quantizer ────────────────────────────────────────────
    quantizer = FPQuantizer(fmt='fp4_e2m1', per_channel=True)
    print(f"  Quantizer: FP4 E2M1 per-channel round-to-nearest")

    # ── Data ─────────────────────────────────────────────────
    ds = MultiTierDataset(args.data_dir, args.max_seq_len, split='val')
    dataloader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=0,
        pin_memory=True,
    )

    # Log matched validation files
    matched = [getattr(d, "path", "unknown") for d in ds.datasets]
    print(f"  Data: matched {len(matched)} val files: {matched}")

    # Single batch reused for ALL forward passes (per Pitfall 5)
    batch = next(iter(dataloader))
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    print(f"  Batch: input_ids shape {tuple(input_ids.shape)}")

    # ═══════════════════════════════════════════════════════════
    # FP16 reference pass (per D-03)
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print("  FP16 Reference Forward Pass")
    print(f"{'=' * 60}")

    ref_tracker = ErrorPropagationTracker()
    ref_tracker.attach(model)

    with torch.no_grad():
        model(input_ids, attention_mask=attention_mask)

    ref_tracker.detach()
    ref_tracker.compute_p3_p6()

    # Save reference P-points for all layers (per Pitfall 4)
    ref_p_points = dict(ref_tracker._p_points)
    p_point_count = len(ref_p_points)
    print(f"  Captured {p_point_count} P-points for all 12 layers")

    # ═══════════════════════════════════════════════════════════
    # Select trace matrices: layers 0/5/11, exclude embed_tokens
    # ═══════════════════════════════════════════════════════════
    selected_matrices: list[tuple[str, int, str]] = []
    for name, _ in model.get_quantizable_weights():
        layer_idx, matrix_type = _classify_matrix(name)
        if layer_idx in TRACED_LAYERS:
            selected_matrices.append((name, layer_idx, matrix_type))

    print(f"  Selected {len(selected_matrices)} source matrices "
          f"across layers {TRACED_LAYERS}")

    # ═══════════════════════════════════════════════════════════
    # Per-source loop (21 sources: 7 matrices x 3 layers)
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print("  Per-Source Error Propagation Trace")
    print(f"{'=' * 60}")

    trace_results: dict[str, list[dict]] = {}
    all_rmsnorm_data: list[dict] = []

    for idx, (module_path, layer_idx, mtype) in enumerate(selected_matrices):
        print(f"\n  [{idx + 1}/{len(selected_matrices)}] {module_path}")

        module = _resolve_module(model, module_path)
        if module is None or not hasattr(module, 'weight'):
            print(f"    SKIP: module not found or has no weight")
            continue

        # Save original weight for restoration (per Pitfall 1)
        original_weight = module.weight.data.clone()

        try:
            # Quantize single matrix in-place (per D-01, D-02, D-10, D-11)
            module.weight.data = quantizer.quantize(module.weight.data)

            # Fresh tracker per pass (per Pitfall 2)
            q_tracker = ErrorPropagationTracker()
            q_tracker.attach(model)

            with torch.no_grad():
                model(input_ids, attention_mask=attention_mask)

            q_tracker.detach()
            q_tracker.compute_p3_p6()  # per Pitfall 4: call after detach

            # TRACE-01: P-point errors for source's OWN layer only (per D-04)
            p_errors: dict[str, float] = {}
            for pp in P_POINTS:
                ref_key = f"{layer_idx}_{pp}"
                q_key = f"{layer_idx}_{pp}"
                if ref_key in ref_p_points and q_key in q_tracker._p_points:
                    ref_tensor = ref_p_points[ref_key]
                    q_tensor = q_tracker._p_points[q_key]
                    # Ensure both on same device
                    if ref_tensor.device != q_tensor.device:
                        ref_tensor = ref_tensor.to(q_tensor.device)
                    d_norm = (q_tensor - ref_tensor).norm().item()
                    y_norm = ref_tensor.norm().clamp(min=1e-12).item()
                    p_errors[pp] = d_norm / y_norm

            # TRACE-01: Waterfall sequence (per D-05)
            waterfall = [p_errors.get(pp, 0.0) for pp in P_POINTS]

            # Store per-source trace
            layer_key = f"layer_{layer_idx}"
            if layer_key not in trace_results:
                trace_results[layer_key] = []
            trace_results[layer_key].append({
                "source_matrix": module_path,
                "matrix_type": mtype,
                "p_points": p_errors,
                "waterfall": waterfall,
            })

            # Store RMSNorm raw data for Task 2 computation (TRACE-02, TRACE-03)
            all_rmsnorm_data.append({
                "source_matrix": module_path,
                "source_layer": layer_idx,
                "q_p_points": dict(q_tracker._p_points),
            })

            print(f"    P0={_fmt_val(p_errors.get('P0', 0.0), '.2e')}  "
                  f"P6={_fmt_val(p_errors.get('P6', 0.0), '.2e')}  "
                  f"waterfall=[{', '.join(_fmt_val(v, '.2e') for v in waterfall)}]")

        finally:
            # Original weight restoration (per Pitfall 1)
            module.weight.data = original_weight

    # ═══════════════════════════════════════════════════════════
    # After loop: compute RMSNorm metrics, print tables, export
    # ═══════════════════════════════════════════════════════════
    total_sources = sum(len(v) for v in trace_results.values())
    print(f"\n{'=' * 60}")
    print(f"  Trace complete: {total_sources} sources traced")
    print(f"  Total forward passes: {1 + total_sources} (1 FP16 + {total_sources} quantized)")
    print(f"{'=' * 60}")

    # ── RMSNorm metrics ─────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  Computing RMSNorm metrics")
    print(f"{'=' * 60}")
    rmsnorm_metrics = _compute_rmsnorm_metrics(all_rmsnorm_data, ref_p_points)

    # ── Waterfall tables ─────────────────────────────────────
    _print_waterfall_tables(trace_results)

    # ── JSON export ──────────────────────────────────────────
    _export_json(args, trace_results, rmsnorm_metrics)

    print(f"\n{'=' * 60}")
    print("  Error propagation trace complete.")
    print(f"{'=' * 60}")
    sys.exit(0)


# ── RMSNorm Metrics ───────────────────────────────────────────────

def _compute_rmsnorm_metrics(
    all_rmsnorm_data: list[dict],
    ref_p_points: dict[str, torch.Tensor],
) -> dict:
    """Compute RMSNorm attenuation ratios and error decomposition.

    For each per-source entry:
      - input_norm: P0 (pre-norm) -> P1 (post-norm) attenuation
      - post_attn_norm: P3 (pre-norm) -> P4 (post-norm) attenuation
      - Decomposition: parallel = |<d,y>|/||y||, orthogonal = ||d-proj||/||y||

    Returns nested dict:
      {
        "by_source": {
            source_matrix: {
                "source_layer": int,
                "layers": {
                    layer_N: {
                        "input_norm": {
                            "d_pre_norm": float, "d_post_norm": float,
                            "ratio": float | None,
                            "decomposition": {
                                "parallel": float, "orthogonal": float,
                                "total": float, "pythagorean_error": float
                            }
                        },
                        "post_attn_norm": { same structure }
                    }
                }
            }
        }
      }
    """
    metrics: dict = {"by_source": {}}

    for entry in all_rmsnorm_data:
        source = entry["source_matrix"]
        source_layer = entry["source_layer"]
        q_p_points = entry["q_p_points"]

        source_metrics: dict = {
            "source_layer": source_layer,
            "layers": {},
        }

        # Process all 12 layers for RMSNorm attenuation/decomposition
        for layer_idx in range(12):
            layer_key = f"layer_{layer_idx}"

            # ── input_norm (P0 -> P1, per D-06) ─────────────
            ref_p0 = ref_p_points.get(f"{layer_idx}_P0")
            ref_p1 = ref_p_points.get(f"{layer_idx}_P1")
            q_p0 = q_p_points.get(f"{layer_idx}_P0")
            q_p1 = q_p_points.get(f"{layer_idx}_P1")

            input_norm_metrics = _compute_norm_pair_metrics(
                ref_p0, q_p0, ref_p1, q_p1,
            )

            # ── post_attn_norm (P3 -> P4, per D-06) ─────────
            ref_p3 = ref_p_points.get(f"{layer_idx}_P3")
            ref_p4 = ref_p_points.get(f"{layer_idx}_P4")
            q_p3 = q_p_points.get(f"{layer_idx}_P3")
            q_p4 = q_p_points.get(f"{layer_idx}_P4")

            post_attn_norm_metrics = _compute_norm_pair_metrics(
                ref_p3, q_p3, ref_p4, q_p4,
            )

            # ── Decomposition ────────────────────────────────
            # input_norm decomposition (per D-08)
            if ref_p1 is not None and q_p1 is not None:
                input_norm_metrics["decomposition"] = _compute_decomposition(
                    ref_p1, q_p1,
                )
            else:
                input_norm_metrics["decomposition"] = {
                    "parallel": float("nan"), "orthogonal": float("nan"),
                    "total": float("nan"), "pythagorean_error": float("nan"),
                }

            # post_attn_norm decomposition (per D-09)
            if ref_p4 is not None and q_p4 is not None:
                post_attn_norm_metrics["decomposition"] = _compute_decomposition(
                    ref_p4, q_p4,
                )
            else:
                post_attn_norm_metrics["decomposition"] = {
                    "parallel": float("nan"), "orthogonal": float("nan"),
                    "total": float("nan"), "pythagorean_error": float("nan"),
                }

            source_metrics["layers"][layer_key] = {
                "input_norm": input_norm_metrics,
                "post_attn_norm": post_attn_norm_metrics,
            }

        metrics["by_source"][source] = source_metrics

    return metrics


def _compute_norm_pair_metrics(
    ref_pre: torch.Tensor | None,
    q_pre: torch.Tensor | None,
    ref_post: torch.Tensor | None,
    q_post: torch.Tensor | None,
) -> dict:
    """Compute attenuation metrics for a norm transition (pre -> post).

    d_pre = q_pre - ref_pre (norm input error)
    d_post = q_post - ref_post (norm output error)
    ratio = ||d_post|| / ||d_pre|| (per D-07)
    """
    result: dict = {
        "d_pre_norm": float("nan"),
        "d_post_norm": float("nan"),
        "ratio": None,
        "decomposition": {},
    }

    if ref_pre is None or q_pre is None or ref_post is None or q_post is None:
        return result

    d_pre = (q_pre - ref_pre).norm().item()
    d_post = (q_post - ref_post).norm().item()

    result["d_pre_norm"] = d_pre
    result["d_post_norm"] = d_post

    # Ratio is NaN when pre-error is negligible (per D-07)
    if d_pre < 1e-8:
        result["ratio"] = None
    else:
        result["ratio"] = d_post / max(d_pre, 1e-12)

    return result


def _compute_decomposition(
    ref_tensor: torch.Tensor,
    q_tensor: torch.Tensor,
) -> dict:
    """Compute parallel/orthogonal decomposition of error at norm output.

    Using vector projection method (per D-08):
      y = clean norm output (ref)
      d = error vector (q - ref)
      parallel = |<d, y>| / ||y||
      orthogonal = ||d - proj|| / ||y||
      total = ||d|| / ||y||

    Returns dict with parallel, orthogonal, total, pythagorean_error.
    """
    # Ensure same device
    if ref_tensor.device != q_tensor.device:
        q_tensor = q_tensor.to(ref_tensor.device)

    y = ref_tensor.reshape(-1)        # clean norm output, flattened
    d = (q_tensor - ref_tensor).reshape(-1)  # error vector, flattened

    y_norm = y.norm().clamp(min=1e-12)
    dot_product = (d * y).sum()

    # Parallel component: projection onto signal direction
    parallel = dot_product.abs().item() / y_norm.item()

    # Orthogonal component: residual after removing projection
    proj = (dot_product / (y_norm * y_norm)) * y
    orthogonal = (d - proj).norm().item() / y_norm.item()

    # Total error (dimensionless ratio per Pitfall 6)
    total = d.norm().item() / y_norm.item()

    # Pythagorean identity verification (per D-08)
    pythagorean_error = abs(
        total * total - (parallel * parallel + orthogonal * orthogonal)
    )

    return {
        "parallel": parallel,
        "orthogonal": orthogonal,
        "total": total,
        "pythagorean_error": pythagorean_error,
    }


# ── Table Printing ─────────────────────────────────────────────────

def _print_waterfall_tables(trace_results: dict[str, list[dict]]):
    """Print per-layer error waterfall tables."""
    print(f"\n{'=' * 120}")
    print("  Error Propagation Waterfall Tables")
    print(f"{'=' * 120}")

    for layer_key in sorted(trace_results.keys()):
        sources = trace_results[layer_key]
        if not sources:
            continue

        layer_idx = int(layer_key.split("_")[1])
        print(f"\n{'=' * 120}")
        print(f"  Layer {layer_idx} Error Propagation Waterfall")
        print(f"{'=' * 120}")

        header_line = (
            f"  {'Source Matrix':<55s}  {'Type':>10s}"
            + ''.join(f"  {'P'+str(i):>10s}" for i in range(7))
        )
        print(header_line)
        print("  " + "-" * (55 + 12 + 7 * 12))

        for src in sources:
            name = src["source_matrix"]
            mtype = src["matrix_type"]
            waterfall = src["waterfall"]

            row = (
                f"  {name:<55s}  {mtype:>10s}"
                + ''.join(f"  {_fmt_val(v, '.6e'):>10s}" for v in waterfall)
            )
            print(row)

        print("  " + "-" * (55 + 12 + 7 * 12))
        print(f"  Note: P0 error for source layer expected ~0 "
              f"(forward_pre_hook fires before weight is used). "
              f"Non-zero P0 indicates weight restoration failure (Pitfall 1).")


# ── JSON Export ────────────────────────────────────────────────────

def _export_json(
    args: argparse.Namespace,
    trace_results: dict[str, list[dict]],
    rmsnorm_metrics: dict,
):
    """Export structured JSON to args.output.

    JSON structure (per D-14):
      {
        "checkpoint": ...,
        "num_selected_layers": 3,
        "selected_layers": [0, 5, 11],
        "trace": { layer_0: [...], ... },
        "rmsnorm_attenuation": { by_source: {...} },
        "rmsnorm_decomposition": { by_source: {...} }
      }
    """
    output_dict = {
        "checkpoint": args.checkpoint,
        "num_selected_layers": len(TRACED_LAYERS),
        "selected_layers": TRACED_LAYERS,
        "trace": trace_results,
        "rmsnorm_attenuation": rmsnorm_metrics["by_source"],
        "rmsnorm_decomposition": {},
    }

    # Build rmsnorm_decomposition section from rmsnorm_metrics
    decomp_section: dict = {}
    for source, mdata in rmsnorm_metrics["by_source"].items():
        decomp_section[source] = {}
        for layer_key, layer_data in mdata["layers"].items():
            decomp_section[source][layer_key] = {
                "input_norm": layer_data["input_norm"].get("decomposition", {}),
                "post_attn_norm": layer_data["post_attn_norm"].get("decomposition", {}),
            }
    output_dict["rmsnorm_decomposition"] = decomp_section

    # Create output directory
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, "w") as f:
        json.dump(output_dict, f, indent=2)

    print(f"\n  Results saved to {args.output}")

    # Print summary
    total_sources = sum(len(v) for v in trace_results.values())
    print(f"\n  Summary:")
    print(f"    Sources traced: {total_sources}")
    print(f"    Forward passes: {1 + total_sources} (1 FP16 + {total_sources} quantized)")
    print(f"    Checkpoint: {args.checkpoint}")
    print(f"    Output: {args.output}")

    # Check P0 errors for warnings
    p0_nonzero = []
    for layer_key, sources in trace_results.items():
        for src in sources:
            p0_val = src["p_points"].get("P0", 0.0)
            if p0_val > 1e-6:
                p0_nonzero.append((src["source_matrix"], p0_val))
    if p0_nonzero:
        print(f"    [WARN] {len(p0_nonzero)} source(s) have non-zero P0 error "
              f"(potential weight restoration failure):")
        for name, val in p0_nonzero[:5]:
            print(f"      {name}: P0={val:.6e}")
    else:
        print(f"    P0 check: all sources have P0 ~0 (weight restoration OK)")


if __name__ == "__main__":
    main()
