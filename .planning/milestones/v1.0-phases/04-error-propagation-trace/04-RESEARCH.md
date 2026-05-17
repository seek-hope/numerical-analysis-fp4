# Phase 4: Error Propagation Trace - Research

**Researched:** 2026-05-17
**Domain:** Numerical analysis of quantization error propagation through Transformer layer pipeline
**Confidence:** HIGH

## Summary

Phase 4 implements the error propagation trace experiment, measuring how quantization error from a single FP4-quantized weight matrix propagates through the Transformer layer pipeline. The core methodology is: (1) run one FP16 reference forward pass and store all per-layer P-point activations, (2) for each of 21 source matrices (7 matrices x 3 layers: 0, 5, 11), quantize only that single matrix in-place, re-run the forward pass, and compute relative error at each of the 6 P-points in the source's layer, (3) from the same quantized passes, compute RMSNorm attenuation ratios and parallel/orthogonal error decomposition for all 12 layers.

The existing `ErrorPropagationTracker` in `src/analysis/error_propagation.py` provides all necessary hook infrastructure (P0-P6 hooks, `compute_p3_p6()`, activation capture). The Phase 4 script extends usage from per-matrix output error (Phase 3) to per-source P-point error waterfall tracing. No tracker code changes are expected -- all new logic is in the trace script.

**Primary recommendation:** Create a single experiment script `src/experiments/trace_error_propagation.py` following the `validate_theorem1.py` pattern (argparse, model loading, tracker attachment, single forward pass, JSON export). The script performs 22 forward passes total: 1 FP16 reference + 21 per-source quantized passes.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| TRACE-01 | Per-matrix P-point error for layers 0/5/11 | ErrorPropagationTracker captures all P-points. Per-source quantization: save original weight, quantize single matrix, run forward pass, compute P-point errors relative to FP16 reference, restore weight. |
| TRACE-02 | RMSNorm attenuation ratio across all 12 layers | P0->P1 captures input_norm transition. P3->P4 captures post_attn_norm transition. Ratio = ||d_post||/||d_pre|| computed from per-source quantized passes. |
| TRACE-03 | RMSNorm parallel/orthogonal decomposition | Vector projection: parallel = |<d,y>|/||y||, orthogonal = ||d - proj||/||y||. Decomposition at P1 (input_norm output) and P4 (post_attn_norm output). |
</phase_requirements>

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01:** Per-matrix quantization for source attribution. Quantize one weight matrix at a time for layers 0, 5, and 11 (7 matrices x 3 layers = 21 sources), re-running forward pass for each to trace that source's error footprint through P-points.
- **D-02:** All other matrices not being traced stay at FP16. Only the single target matrix is quantized per forward pass.
- **D-03:** Two-pass methodology per source matrix: (1) one FP16 reference forward pass stores clean P-point states for all layers, (2) for each source matrix, a quantized forward pass stores perturbed P-point states. Error at each P-point = ||p_q - p_fp16|| / ||p_fp16||.
- **D-04:** For each source matrix in layers 0/5/11, measure ||d||/||y|| at all 6 P-points of that source's own layer only.
- **D-05:** Error waterfall data: for each source matrix in each traced layer, produce the sequence [P0_err, P1_err, P2_err, P3_err, P4_err, P5_err, P6_err].
- **D-06:** Measure ||d_post||/||d_pre|| for both input_norm and post_attn_norm across ALL 12 layers.
- **D-07:** RMSNorm input error d_pre = y_pre_q - y_pre_fp16, RMSNorm output error d_post = y_post_q - y_post_fp16. Attenuation ratio = ||d_post||/||d_pre||.
- **D-08:** Vector projection method for parallel/orthogonal decomposition using y (clean output) and d (error vector).
- **D-09:** Both parallel and orthogonal components reported for input_norm and post_attn_norm across all 12 layers.
- **D-10:** Use FP4 E2M1 round-to-nearest per-channel quantization (FPQuantizer(fmt='fp4_e2m1', per_channel=True)) for all trace measurements.
- **D-11:** No GPTQ compensation for propagation tracing.
- **D-12:** Single experiment script `src/experiments/trace_error_propagation.py`.
- **D-13:** argparse interface: --checkpoint, --data_dir, --output, --device, --batch_size, --max_seq_len. Uses validation data split (split='val').
- **D-14:** JSON output with three sections: trace, rmsnorm_attenuation, rmsnorm_decomposition.

### Claude's Discretion

- Whether to extend full per-matrix trace to all 12 layers (not just 0/5/11) -- infrastructure supports it.
- Exact print table format for waterfall data (column widths, decimal places, sorting).
- Whether to compute cross-layer error propagation (e.g., does layer N's q_proj error show up at layer N+1's P0?).
- Single-seed execution (seed=42) for efficiency; multi-seed belongs to Phase 5.
- Whether to include a "no-quantization" null control trace alongside quantized traces.

### Deferred Ideas (OUT OF SCOPE)

None -- all auto-selected decisions align with prior phase decisions and roadmap requirements.
</user_constraints>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| P-point activation capture | Analysis (error_propagation.py) | Model | ErrorPropagationTracker uses forward hooks on model; model is passive data source |
| Per-source weight quantization | Analysis (error_propagation.py) | Quantization | Script quantizes individual weights in-place; FPQuantizer provides quantize() method |
| P-point error computation | Analysis (error_propagation.py) | -- | compute_output_error() already exists; this phase extends to per-P-point computation |
| RMSNorm attenuation/decomposition | Analysis (error_propagation.py) | -- | Pure vector math on captured tensors; no model changes |
| Results export | Experiment (trace script) | -- | JSON serialization of computed dicts |
| Model loading / data pipeline | Experiment (training_utils.py) | -- | load_checkpoint(), MultiTierDataset already handle this |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | >= 2.3 | All tensor operations, hook registration, SVD | Project standard; required by model |
| FPQuantizer | src/quantization/fp_quantizer.py | FP4 round-to-nearest quantization | Project standard; used in all phases |
| ErrorPropagationTracker | src/analysis/error_propagation.py | P-point activation capture hook architecture | Built for this project in Phase 2 |

### Supporting
| Component | Purpose | When to Use |
|-----------|---------|-------------|
| load_checkpoint() | Load FP16 baseline checkpoint | Model loading at script start |
| MultiTierDataset(data_dir, split='val') | Validation data for forward pass | Providing input_ids for measurement |
| MicroGemmaFPConfig | Architecture dimensions | Model instantiation |
| compute_all_condition_numbers() | Condition number context | Optional -- for cross-referencing kappa with P-point errors |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Per-source single-weight quantization | Full-model quantization + per-layer error | Per-source isolates error attribution but requires 21 forward passes; full-model is faster but can't attribute to specific matrices |

**Installation:**
```bash
pip install -r requirements.txt
```

No new packages needed for this phase. All dependencies (torch, numpy, json, argparse) are already in requirements.

## Package Legitimacy Audit

> No external packages are installed in this phase. The script uses only standard library modules (json, os, sys, math, argparse) and existing project modules (torch, numpy, src.*). No Package Legitimacy Gate needed -- skip.

## Architecture Patterns

### System Architecture Diagram

```
FP16 baseline checkpoint
        |
        v
  Load model (MicroGemmaFPForCausalLM)
        |
        v
  FP16 Reference Forward Pass (tracker.attach)
        |
        +---> Store P-points for all 12 layers as REFERENCE
        |
        v
  Per-Source Loop (21 iterations: 7 matrices x 3 layers)
        |
  +-----+-----+
  |           |
  v           v
Save orig  Quantize
weight      weight
(W_fp16)   (W_q = quantizer.quantize(W_fp16))
  |           |
  +-----+-----+
        |
        v
  Quantized Forward Pass (tracker.attach)
        |
        +---> Store P-points for all 12 layers (QUANTIZED)
        |
        v
  Compute Errors (relative to FP16 reference)
        |
  +-----+-----+-----+
  |           |           |
  v           v           v
TRACE-01:   TRACE-02:   TRACE-03:
P-point     RMSNorm     RMSNorm
errors at   attenuation decomposition
source's    ratios for  for all 12
layer       all 12      layers
(0/5/11)    layers     
  |           |           |
  +-----+-----+-----------+
        |
        v
  Restore original weight (W_fp16)
        |
        v
  Print per-source waterfall tables
        |
        v
  Export results/error_propagation_trace.json
```

### Recommended Project Structure

No new files beyond the script itself:
```
src/
└── experiments/
    └── trace_error_propagation.py   # NEW -- main trace script

results/                              # CREATED at runtime (os.makedirs)
└── error_propagation_trace.json      # CREATED by script
```

### Pattern 1: Per-Source Weight Quantization and Restoration

**What:** Quantize a single weight matrix in-place, run a forward pass, then restore the original weight. This isolates error attribution to exactly one source matrix.

**When to use:** In the per-source loop, before each quantized forward pass.

```python
# Source: dictated by D-01/D-02 per-source isolation principle
module = tracker._resolve_module(model, module_path)
original_weight = module.weight.data.clone()
module.weight.data = quantizer.quantize(module.weight.data)

# Run quantized forward pass
with torch.no_grad():
    model(input_ids, attention_mask=attention_mask)

# Restore original weight
module.weight.data = original_weight
```

### Pattern 2: Per-Point Relative Error Computation

**What:** For each P-point, compute ||p_q - p_fp16|| / ||p_fp16||.

**When to use:** After each quantized forward pass, for the source's own layer.

```python
# Source: D-03 defines error computation formula
p_points = ['P0', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6']
errors = {}
for pp in p_points:
    key = f"{layer_idx}_{pp}"
    p_ref = ref_p_points[key]
    p_q = quantized_p_points[key]
    denom = p_ref.norm().clamp(min=1e-12)
    err = (p_q - p_ref).norm() / denom
    errors[pp] = err.item()
waterfall = [errors[pp] for pp in p_points]  # ordered sequence
```

### Pattern 3: RMSNorm Error Decomposition

**What:** Decompose error vector into parallel (along signal direction) and orthogonal components.

**When to use:** For each RMSNorm (input_norm at P1, post_attn_norm at P4) across all layers.

```python
# Source: D-08 vector projection formulas
y = ref_p_points[f"{layer_idx}_P1"]  # clean norm output, flattened: (B*S, H)
d = (quantized_p_points[f"{layer_idx}_P1"] - ref_p_points[f"{layer_idx}_P1"])
y_flat = y.reshape(-1)
d_flat = d.reshape(-1)

y_norm = y_flat.norm().clamp(min=1e-12)
dot_product = (d_flat * y_flat).sum()

# Parallel component (error along signal direction)
parallel = dot_product.abs() / y_norm

# Orthogonal component (error perpendicular to signal)
proj = (dot_product / (y_norm * y_norm)) * y_flat
orthogonal = (d_flat - proj).norm() / y_norm

# Verify Pythagorean identity: total_error^2 ≈ parallel^2 + orthogonal^2
total_error = d_flat.norm() / y_norm
# total_error^2 ≈ parallel^2 + orthogonal^2 (within floating point tolerance)
```

### Anti-Patterns to Avoid

- **Stale tracker state between passes:** Always call `tracker.detach()` before re-attaching for a new pass. The tracker stores tensors in internal dicts; forgetting to detach accumulates stale activations.
- **Forgetting to restore weights:** If weight restoration is skipped due to an exception, the model is permanently corrupted. Use try/finally blocks around the quantize/forward/restore sequence.
- **Using compute_output_error() for P-point errors:** `compute_output_error()` computes `||(W_q - W)x||/||Wx||` at the matrix output (== P1 level for attention, P4 level for FFN). Phase 4 needs per-P-point errors, which requires manual error computation from captured P-point tensors.
- **Implicit device mismatches:** P-point tensors are on CPU (moved by tracker hooks). Reference P-points are already on CPU. But quantized P-points from the live forward pass are also on CPU. Computation happens on CPU, which is fine.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| P-point hook registration | Custom hook logic | ErrorPropagationTracker.attach() | Already implemented with proper hook factories, CPU offloading, and detach cleanup |
| P3/P6 residual computation | Manual tensor addition | ErrorPropagationTracker.compute_p3_p6() | Handles layer iteration, key construction, idempotency |
| Per-matrix output error | Custom error computation | ErrorPropagationTracker.compute_output_error() | Project standard; used across all phases |
| Model loading from checkpoint | Manual state_dict restore | load_checkpoint(model, optimizer, path, device) | Handles map_location and partial loading |
| Validation data loading | Manual .bin file iteration | MultiTierDataset(data_dir, split='val') | Handles multi-tier concatenation and file discovery |

**Key insight:** The entire measurement infrastructure (P-point hooks, activation capture, residual computation) is already built in Phase 2. Phase 4 is purely a script-writing exercise that composes existing components in a new pattern (per-source quantization + per-P-point error + RMSNorm decomposition).

## Runtime State Inventory

> This phase is a greenfield experiment script with no rename/refactor/migration. Omit.

## Common Pitfalls

### Pitfall 1: Weight Restoration Failure After Quantization
**What goes wrong:** A quantized weight is not restored to FP16, contaminating the next per-source measurement and all subsequent passes.
**Why it happens:** Exception during the forward pass or error computation skips the restoration line.
**How to avoid:** Wrap quantize/measure/restore in try/finally:
```python
original = module.weight.data.clone()
try:
    module.weight.data = quantizer.quantize(module.weight.data)
    # ... forward pass and error computation ...
finally:
    module.weight.data = original
```
**Warning signs:** P0 error is non-zero for the source's own layer (P0 should be identical since quantization hasn't happened yet at the P0 capture point); or after the first source, all subsequent errors become anomalously large.

### Pitfall 2: Tracker State Accumulation Across Passes
**What goes wrong:** _p_points dict carries tensors from the previous quantized pass, so error computation mixes old and new data.
**Why it happens:** tracker.detach() removes hooks but does not clear _p_points or _activations dicts.
**How to avoid:** Create a NEW tracker instance for each pass instead of reusing:
```python
ref_tracker = ErrorPropagationTracker().attach(model)
# ... FP16 pass ...
ref_tracker.detach()
ref_p_points = dict(ref_tracker._p_points)  # save reference

for source in sources:
    # ... quantize weight ...
    q_tracker = ErrorPropagationTracker().attach(model)  # FRESH tracker
    # ... forward pass ...
    q_tracker.detach()
    q_tracker.compute_p3_p6()
    # ... compute errors using ref_p_points vs q_tracker._p_points ...
```

### Pitfall 3: P0 Error Should Be ~Zero for Source Layer
**What goes wrong:** Expecting P0 to have measurable error at the source's own layer.
**Why it happens:** P0 is captured by a forward_pre_hook on the layer, which fires BEFORE any computation inside the layer. Since only the source layer's weight is modified, P0 is identical to the FP16 reference (no quantization has occurred yet at that point in the forward pass).
**How to avoid:** Either skip P0 in the waterfall (only report P1-P6 for per-source error), or expect P0 error << 1e-6.
**Warning signs:** If P0 error is significant, the model state is corrupted (see Pitfall 1).

### Pitfall 4: P3 and P6 Computation Timing
**What goes wrong:** Computing P3/P6 from the wrong tracker instance or before the quantized pass is complete.
**Why it happens:** compute_p3_p6() modifies the internal _p_points dict in-place. Calling it before all forward passes are done, or on the reference tracker after the reference pass, can cause confusion.
**How to avoid:** Always call compute_p3_p6() immediately after detach() on the quantized tracker. Save the reference P-points in a separate dict before any quantized pass begins.

### Pitfall 5: Sequence Length Mismatch in Error Computation
**What goes wrong:** P-point tensors from different passes have different sequence lengths.
**Why it happens:** If the DataLoader uses variable-length sequences with padding (collate_batch pads to max_len within batch), a different batch could produce different tensor shapes, making subtraction impossible.
**How to avoid:** Use the same batch (same input_ids) for ALL passes -- the FP16 reference and all 21 quantized passes. Fetch one batch upfront and re-use it:
```python
batch = next(iter(dataloader))
input_ids = batch["input_ids"].to(device)
# Use this same input_ids for ALL forward passes
```

### Pitfall 6: RKHS / Projection Norm Invariance
**What goes wrong:** When computing the parallel/orthogonal decomposition, the parallel component has units of ||y|| due to the normalization, and the orthogonal also has units of ||y||. The Pythagorean identity holds but the values are dimensionless ratios.
**Why it happens:** This is correct behavior -- both components are normalized by ||y||, making them dimensionless relative errors. The verification |total^2 - (parallel^2 + orthogonal^2)| should be < 1e-6.

## Code Examples

### Pattern A: Full Script Structure

```python
#!/usr/bin/env python3
"""Error propagation trace for the Micro-Gemma-FP Transformer.

Measures per-source quantization error through all 6 P-points for layers
0, 5, and 11. Also computes RMSNorm attenuation and decomposition for
all 12 layers.

Usage:
    python src/experiments/trace_error_propagation.py \\
        --checkpoint checkpoints/scaled_fp16_baseline/model.pt
"""

import argparse, json, math, os, torch
from torch.utils.data import DataLoader
from src.experiments.training_utils import (
    MultiTierDataset, collate_batch, load_checkpoint,
)
from src.analysis.error_propagation import ErrorPropagationTracker
from src.model.config import MicroGemmaFPConfig
from src.model.transformer import MicroGemmaFPForCausalLM
from src.quantization.fp_quantizer import FPQuantizer

TRACED_LAYERS = [0, 5, 11]

def parse_args():
    parser = argparse.ArgumentParser(...)
    # --checkpoint, --data_dir, --output, --device, --batch_size, --max_seq_len
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device if ... else "cpu")

    # Load model
    config = MicroGemmaFPConfig()
    model = MicroGemmaFPForCausalLM(config)
    load_checkpoint(model, None, args.checkpoint, device)
    model.to(device).eval()

    quantizer = FPQuantizer(fmt="fp4_e2m1", per_channel=True)

    # Get one batch
    ds = MultiTierDataset(args.data_dir, args.max_seq_len, split="val")
    dataloader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
    batch = next(iter(dataloader))
    input_ids = batch["input_ids"].to(device)

    # FP16 reference pass
    ref_tracker = ErrorPropagationTracker().attach(model)
    with torch.no_grad():
        model(input_ids)
    ref_tracker.detach()
    ref_tracker.compute_p3_p6()
    ref_p_points = dict(ref_tracker._p_points)  # save reference for all layers

    # Per-source loop
    selected_matrices = []
    for name, _ in model.get_quantizable_weights():
        layer_idx, matrix_type = _classify_matrix(name)
        if layer_idx in TRACED_LAYERS:
            selected_matrices.append((name, layer_idx, matrix_type))

    trace_results = {}
    for module_path, layer_idx, mtype in selected_matrices:
        module = _resolve_module(model, module_path)
        original_weight = module.weight.data.clone()
        try:
            module.weight.data = quantizer.quantize(module.weight.data)
            q_tracker = ErrorPropagationTracker().attach(model)
            with torch.no_grad():
                model(input_ids)
            q_tracker.detach()
            q_tracker.compute_p3_p6()

            # TRACE-01: P-point errors for source's own layer
            p_errors = {}
            for pp in ['P0','P1','P2','P3','P4','P5','P6']:
                ref_key = f"{layer_idx}_{pp}"
                q_key = f"{layer_idx}_{pp}"
                if ref_key in ref_p_points and q_key in q_tracker._p_points:
                    d = q_tracker._p_points[q_key] - ref_p_points[ref_key].to(q_tracker._p_points[q_key].device)
                    y = ref_p_points[ref_key].to(q_tracker._p_points[q_key].device)
                    err = d.norm().item() / y.norm().clamp(min=1e-12).item()
                    p_errors[pp] = err
            waterfall = [p_errors[pp] for pp in ['P0','P1','P2','P3','P4','P5','P6']]

            # TRACE-02: RMSNorm attenuation for all 12 layers
            # TRACE-03: RMSNorm decomposition for all 12 layers
            # ... (computed from q_tracker._p_points vs ref_p_points)

            key = f"layer_{layer_idx}"
            if key not in trace_results:
                trace_results[key] = []
            trace_results[key].append({
                "source_matrix": module_path,
                "matrix_type": mtype,
                "p_points": p_errors,
                "waterfall": waterfall,
            })
        finally:
            module.weight.data = original_weight

    # Print tables
    # Export JSON
    # ...

if __name__ == "__main__":
    main()
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Phase 3: compute_output_error() for per-matrix ||dy||/||y|| at matrix output | Phase 4: per-P-point error through full layer pipeline | Phase 4 | Enables error waterfall tracing through attention, FFN, RMSNorm, and residual connections |
| Phase 3: multi-seed (42, 123, 456) with bootstrap CI | Phase 4: single-seed (42) for efficiency | Phase 4 | Faster execution; multi-seed deferred to Phase 5 |
| Phase 3: 72-matrix correlation analysis | Phase 4: 21-source trace in 3 representative layers | Phase 4 | Deeper per-source analysis at higher measurement resolution (7 P-points per source) |

**Deprecated/outdated:**
- `compute_output_error()` is not suitable for Phase 4 -- it computes error at the matrix output only. Phase 4 needs per-P-point error, which requires manual computation from the captured P-point tensors.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | P0 error is ~0 for the source's own layer (since forward_pre_hook fires before any computation in the layer) | Common Pitfalls | Low -- this is a mathematical certainty for single-matrix quantization; P0 is the residual input from the previous layer |
| A2 | P-point tensors from both passes are on CPU (tracker hooks move them with .cpu()) | Architecture Patterns | Low -- verified in error_propagation.py source code |
| A3 | The FP16 baseline checkpoint exists at the path provided via --checkpoint | Standard Stack | HIGH -- no checkpoints exist yet in the repo. The planner must add a checkpoint verification task |
| A4 | Validation data split files exist (tierN_val.bin) | Standard Stack | MEDIUM -- real_tiers directory exists but val split may not be complete for all tiers |

## Open Questions

1. **RMSNorm measurement: per-source or consolidated?**
   - What we know: D-06 requires RMSNorm attenuation across all 12 layers. P-point data is captured from every quantized forward pass, which always covers all 12 layers.
   - What's unclear: Should RMSNorm metrics be reported per-source (21 values per layer) or consolidated (1 value per layer from some aggregation strategy)?
   - Recommendation: Report per-source RMSNorm metrics, keyed by source matrix. For the JSON `rmsnorm_attenuation` section, nest under source matrix (extending D-14's flat structure). Phase 5 can then aggregate as needed.

2. **Cross-layer error propagation scope**
   - What we know: Each quantized forward pass captures P-points at ALL 12 layers, not just the source's own layer.
   - What's unclear: Whether to report P-point errors at NON-source layers (e.g., when quantizing layer 0's q_proj, what is the error at layer 5's P-points?).
   - Recommendation: Defer to Phase 5 extended comparison. Report only per-source-layer P-point errors (per D-04). The data is already captured for cross-layer analysis if needed.

3. **Results directory creation**
   - What we know: `results/` directory does not exist in the repository.
   - What's unclear: Whether the script should create it with os.makedirs or require it to exist.
   - Recommendation: Script creates it with os.makedirs(os.path.dirname(output), exist_ok=True).

4. **Checkpoint availability**
   - What we know: `checkpoints/scaled_fp16_baseline/model.pt` does not exist in the repo.
   - What's unclear: Whether the checkpoint is available on the remote server, or whether Phase 3 created it during execution.
   - Recommendation: The planner must add a pre-condition task that verifies checkpoint existence and documents where to find it.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.11+ | Script runtime | 27 | 3.14.5 | -- |
| PyTorch >= 2.3 | Model, hooks, quantization | TBD | -- | -- |
| CUDA | GPU execution | TBD | -- | CPU fallback (slow but functional) |
| sshpass | Remote execution | TBD | -- | Manual rsync |
| data/real_tiers/*_val.bin | Data pipeline | 27 | 4 files | CharTokenizer offline fallback |
| checkpoint .pt file | Model loading | NOT AVAILABLE | -- | BLOCKING -- must be addressed |

**Missing dependencies with no fallback:**
- FP16 baseline checkpoint (.pt file). No checkpoint exists in the current repo. The planner must either (a) run Phase 3 first to produce one, (b) locate one from a prior execution, or (c) generate a minimal checkpoint for trace validation.

**Missing dependencies with fallback:**
- data/real_tiers val split: exists (4 val .bin files). But if missing, the CharTokenizer offline fallback will provide functional-but-less-meaningful data.

## Validation Architecture

> Skipped: `workflow.nyquist_validation` is explicitly `false` in .planning/config.json.

## Security Domain

> Skipped: This phase involves no user input, network requests, authentication, or data storage that would require security enforcement. The script loads a local checkpoint, processes pre-tokenized data, and saves results to a file. No applicable ASVS categories.

## Sources

### Primary (HIGH confidence)
- ErrorPropagationTracker source: `src/analysis/error_propagation.py` -- verified hook architecture, P-point definitions, CPU offloading, compute_p3_p6()
- TransformerLayer source: `src/model/transformer.py:172-194` -- verified P-point placement in forward pass
- FPQuantizer source: `src/quantization/fp_quantizer.py:53-80` -- verified quantize() interface and FP4 E2M1 grid
- validate_theorem1.py source: `src/experiments/validate_theorem1.py` -- verified argparse pattern, model loading, DataLoader construction, tracker lifecycle
- MicroGemmaFPConfig source: `src/model/config.py` -- verified architecture dimensions (12 layers, 768 hidden, 832 input dim with pl_emb)
- load_checkpoint source: `src/experiments/training_utils.py:359-364` -- verified interface
- MultiTierDataset source: `src/experiments/training_utils.py:131-159` -- verified split='val' support
- CONTEXT.md decisions: `.planning/phases/04-error-propagation-trace/04-CONTEXT.md` -- all locked decisions documented

### Secondary (MEDIUM confidence)
- Results/checkpoints directories absent: verified via `ls` on filesystem -- confirms os.makedirs needed
- Val split .bin files exist: verified via `ls data/real_tiers/` -- 4 _val.bin files present

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all components are existing project code
- Architecture: HIGH -- pattern directly follows Phase 3's validate_theorem1.py
- Pitfalls: HIGH -- identified from code inspection and domain knowledge of the existing infrastructure

**Research date:** 2026-05-17
**Valid until:** 2026-06-17 (stable codebase, no fast-moving dependencies)
