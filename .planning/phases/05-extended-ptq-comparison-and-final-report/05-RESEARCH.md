# Phase 5: Extended PTQ Comparison and Final Report - Research

**Researched:** 2026-05-17
**Domain:** PTQ evaluation pipeline orchestration, per-matrix output error measurement, multi-config comparison synthesis, final report generation
**Confidence:** HIGH

## Summary

Phase 5 consumes all prior phase outputs and produces the definitive experimental comparison and final report. Two scripts are needed: `run_full_comparison.py` (long-running, 24 PTQ configs on GPU) and `write_final_report.py` (fast, reads all JSON results and generates REPORT.md).

The comparison script must: (1) load FP16 baseline and cond_regularized checkpoints; (2) apply all 6 quantization methods across 2 formats (FP8, FP4) and 2 checkpoints (24 total configs); (3) for each config, compute PPL (100 validation steps) and per-matrix ||dy||/||y|| (single validation batch via ErrorPropagationTracker); (4) synthesize GPTQ-vs-RTN and Lloyd-Max-vs-uniform comparisons; (5) merge Phase 3 and Phase 4 results into a per-matrix summary table; (6) export `results/full_comparison.json`.

The report script reads `theorem1_validation.json`, `error_propagation_trace.json`, and `full_comparison.json`, and regenerates `docs/REPORT.md` with corrected methodology, revised conclusions, and all numerical results.

**Primary recommendation:** Follow the proven method-dispatch pattern from `phase2_comparison.py`, extended with ErrorPropagationTracker integration from `validate_theorem1.py`. The clean data split already exists (`*_train.bin`/`*_val.bin`). All 6 quantizer classes have working implementations with consistent interfaces.

## Architecture Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| 24-config PTQ method dispatch | Experiment scripts | — | All quantization and evaluation code runs in Python experiment scripts; no server/service tiers involved |
| Per-matrix ||dy||/||y|| measurement | Experiment scripts (ErrorPropagationTracker) | — | Tracker attaches hooks at runtime; error computation offline after single forward pass |
| PPL evaluation | Experiment scripts (evaluate_perplexity) | — | Standard batched evaluation loop on validation split |
| Cross-method comparison synthesis | Experiment scripts | — | Post-hoc analysis after all 24 configs complete; data in-memory, exported as JSON |
| Report generation | Experiment scripts (write_final_report) | — | Reads JSON results files, writes Markdown; no external rendering |
| Phase 3/4 data integration | Report script | — | Reads theorem1_validation.json and error_propagation_trace.json; no re-computation |

## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01:** 24 configurations: 2 checkpoints (FP16 baseline, cond_regularized) x 2 formats (FP8 E4M3, FP4 E2M1) x 6 methods (RTN per-channel, GPTQ per-channel, Lloyd-Max adaptive, Hadamard rotation + RTN, outlier rotation + RTN, MXFP4 block-scaling). All use clean validation split for PPL and training split for calibration.
- **D-02:** Re-use existing quantizer infrastructure: `FPQuantizer` for RTN, `GPTQQuantizer` for GPTQ, `AdaptiveGridQuantizer` for Lloyd-Max, `HadamardRotation` + `FPQuantizer` for Hadamard, `DuQuantStyleQuantizer` for outlier rotation, `MXFP4Quantizer` for MXFP4.
- **D-03:** Both PPL and ||dy||/||y|| collected per config. Tracker attached during PPL evaluation for simultaneous measurement.
- **D-04:** GPTQ vs RTN comparison: 4 configs (2 checkpoints x 2 formats). Report (a) PPL diff, (b) mean ||dy||/||y|| diff, (c) per-matrix delta. Negative delta means GPTQ reduces error.
- **D-05:** GPTQ calibration: training split, 256 samples, seq_len=512 (same as original phase2_comparison.py).
- **D-06:** Lloyd-Max vs uniform: 2 configs (2 checkpoints x FP4 E2M1). Report (a) PPL diff, (b) mean ||dy||/||y|| diff, (c) per-matrix delta.
- **D-07:** Lloyd-Max grids fitted on training split (256 samples). Per-channel scaling. Per-layer grid optimization (not per-matrix).
- **D-08:** Per-matrix summary table: 72 rows, columns {name, layer, type, kappa, ||dW||/||W||, ||dy||/||y||, tightness_ratio, norm_attenuation}. Source: Phase 3 JSON, Phase 5 norm_attenuation column from Phase 4 trace.
- **D-09:** Table printed to stdout + JSON export (`results/per_matrix_summary.json`). Sorted by layer, then type.
- **D-10:** Single comparison script `src/experiments/run_full_comparison.py`: loads both checkpoints, runs 24 configs, computes PPL + ||dy||/||y|| per config, runs GPTQ/RTN and Lloyd-Max/uniform comparisons, loads Phase 3/4 JSON, generates summary table, exports `results/full_comparison.json`.
- **D-11:** Separate report script `src/experiments/write_final_report.py`: reads all JSON results, generates REPORT.md with all required sections.
- **D-12:** argparse interfaces defined for both scripts.
- **D-13:** Checkpoints must exist as pre-condition (`checkpoints/scaled_fp16_baseline/model.pt` and `checkpoints/cond_regularized/model.pt`). Script exits with error if missing.
- **D-14:** v2 visualizations deferred. Phase 5 produces structured JSON for future chart generation.

### Claude's Discretion

- Whether to support re-entrant partial execution (`--configs 0-11` and `--configs 12-23`)
- Exact print table formatting (column widths, decimal places)
- Whether to include per-layer-type subgroup analysis
- Whether to compute cross-checkpoint comparisons beyond basic PPL delta
- Logging verbosity
- Whether to omit Hadamard/outlier rotation methods for FP4 if results are pathological

### Deferred Ideas (OUT OF SCOPE)

- VIS-01/02/03: v2 visualizations (waterfall charts, scatter plots, bar charts)
- ANALYSIS-01: Per-head attention error decomposition
- ANALYSIS-03: Rank stability analysis

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| COMP-01 | Re-run 24-config PTQ comparison with clean data split, reporting both PPL and per-matrix ||dy||/||y|| | All 6 quantizer interfaces are verified working. Clean data split exists (`*_train.bin`/`*_val.bin`). ErrorPropagationTracker.compute_output_error() can compute ||dy||/||y|| given any quantizer object. Pattern for simultaneous PPL + error measurement: attach tracker before PPL eval, capture activations during eval forward passes. |
| COMP-02 | Compare GPTQ column compensation against RTN on output-space error | GPTQQuantizer.quantize_model() accepts a base FPQuantizer and calibration dataloader. compute_output_error() accepts any quantizer — pass `gptq_quantized_model` vs `rtn_quantized_model`. Per-matrix delta computed offline from saved error dicts. |
| COMP-03 | Compare Lloyd-Max adaptive grids against uniform E2M1 on ||dy||/||y|| | AdaptiveGridQuantizer uses calibrate() + quantize_model() pattern with Lloyd-Max. Uniform E2M1 uses FPQuantizer('fp4_e2m1', per_channel=True). Both produce per-matrix errors via same compute_output_error(). |
| REPORT-01 | Generate per-matrix error summary table with all required columns | Phase 3 JSON (theorem1_validation.json) contains kappa, dw_norm, dy_norm_mean, tightness_ratio per matrix. Phase 4 JSON (error_propagation_trace.json) contains norm_attenuation per layer. Combine with Phase 5 ||dy||/||y|| values for complete table. |
| REPORT-02 | Generate error propagation waterfall data for visualization | Waterfall data comes from Phase 4 trace JSON (per-source per-P-point error sequences). Report script reads and formats this data. |
| REPORT-03 | Update REPORT.md with corrected values and revised theoretical assessment | REPORT.md structure defined in CONTEXT.md section "Specific Ideas" (10 sections). Report script writes structured Markdown. Existing ANALYSIS.md provides mathematical derivations to reference. |

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | >= 2.3.0 | All tensor operations, model loading, quantization simulation | Canonical project dependency, all prior phases use it [VERIFIED: codebase] |
| ErrorPropagationTracker | in-repo | Per-matrix ||dy||/||y|| via activation hook capture | Built in Phase 2, used in Phase 3/4/5 [VERIFIED: src/analysis/error_propagation.py] |
| FPQuantizer | in-repo | Simulated FP8/FP4 round-to-nearest quantization | Standard PTQ method, per-channel/per-tensor [VERIFIED: src/quantization/fp_quantizer.py] |
| GPTQQuantizer | in-repo | Hessian-based column compensation PTQ | Established quantizer from Phase 2 [VERIFIED: src/quantization/gptq.py] |
| AdaptiveGridQuantizer | in-repo | Lloyd-Max per-layer adaptive grid optimization | Proprietary method from Phase 2 [VERIFIED: src/quantization/adaptive_grid.py] |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| DuQuantStyleQuantizer | in-repo | Outlier-aware scaling + block-Hadamard rotation + grid quantization | Method 5 (outlier rotation) [VERIFIED: src/quantization/outlier_rotation.py] |
| MXFP4Quantizer | in-repo | Block-scaling MXFP4 (block_size=32) with dynamic per-block scale | Method 6 (MXFP4) [VERIFIED: src/quantization/fp4_grids.py] |
| HadamardRotation | in-repo | Fast Walsh-Hadamard transform for activation/weight homogenization | Method 4 (Hadamard + RTN) [VERIFIED: src/quantization/hadamard.py] |
| evaluate_perplexity | in-repo | Weighted PPL over validation split | Standard eval, all prior phases [VERIFIED: src/experiments/training_utils.py:321-345] |
| load_checkpoint | in-repo | Load model state dict from .pt file | Model loading, all prior phases [VERIFIED: src/experiments/training_utils.py:359-364] |
| get_dataloader | in-repo | Factory with split='train'|'val' support | Data loading, Phase 1 [VERIFIED: src/experiments/training_utils.py:204-227] |
| scipy.stats | >= 1.10 | Pearson r computation with accurate p-values | Optional, falls back to numpy [VERIFIED: src/experiments/validate_theorem1.py:43-73] |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Single comparison script (run_full_comparison.py) | Modify phase2_comparison.py in-place | Cleaner separation — phase2_comparison.py is a historical artifact; dedicated Phase 5 script with correct data split avoids confusion |
| Dedicated report script | Manual REPORT.md editing | Script guarantees consistency between JSON data and report text; manual editing would introduce copy-paste errors |
| Per-matrix summary from in-memory data | From JSON files only | In-memory allows cross-referencing during same execution; JSON export ensures reproducibility |

## Package Legitimacy Audit

> No external packages are installed by Phase 5. All quantization, measurement, and evaluation infrastructure is implemented in-repo. The comparison and report scripts import only existing project modules (`src.model`, `src.quantization`, `src.analysis`, `src.experiments.training_utils`) and standard library (`json`, `argparse`, `os`, `math`, `numpy`). SciPy is an optional dependency (same as Phase 3 — validates as of training data [ASSUMED]).

| Package | Registry | slopcheck | Disposition |
|---------|----------|-----------|-------------|
| scipy (optional) | PyPI | See Phase 3 audit | Optional — Pearson r fallback in pure numpy exists |
| No new packages introduced | N/A | N/A | N/A |

## Architecture Patterns

### System Architecture Diagram

```
                              +-----------------------+
                              |   Data: real_tiers/   |
                              |  *_{train,val}.bin    |
                              +-----------+-----------+
                                          |
                                          v
                    +---------------------+---------------------+
                    |     get_dataloader(split='val')          |
                    |     get_dataloader(split='train')        |
                    +---------------------+---------------------+
                                          |
          +-------------------------------+-------------------------------+
          |                               |                               |
          v                               v                               v
+----------------------+    +------------------------+    +---------------------------+
| Load 2 checkpoints:  |    | For each of 24 configs:|    | Load Phase 3/4 results:   |
| - scaled_fp16_baseline|   | 1. Load fresh model    |    | - theorem1_validation.json |
| - cond_regularized   |    | 2. Apply quantization  |    | - error_propagation_trace.json|
+----------------------+    | 3. Attach tracker      |    +---------------------------+
          |                 | 4. Evaluate PPL (100   |              |
          |                 |    steps) = capture    |              |
          |                 |    activations         |              |
          |                 | 5. compute_output_error|              |
          |                 |    = per-matrix errors |              |
          |                 | 6. Collect results     |              |
          |                 +------------------------+              |
          |                            |                            |
          +----------------------------+----------------------------+
                                       |
                                       v
                    +-----------------------------------------------+
                    |         Comparison Analyses:                  |
                    |  - GPTQ vs RTN (4 pairs)                     |
                    |  - Lloyd-Max vs uniform (2 pairs)             |
                    |  - Per-matrix summary table (72 rows)         |
                    +-----------------------------------------------+
                                       |
                                       v
                    +-----------------------------------------------+
                    |       Export: results/full_comparison.json    |
                    |       Export: results/per_matrix_summary.json |
                    +-----------------------------------------------+
                                       |
                                       v
               +---------------------------------------------------+
               |  write_final_report.py:                           |
               |  Reads 3 JSON files + docs/ANALYSIS.md            |
               |  Writes docs/REPORT.md (10 sections)              |
               +---------------------------------------------------+
```

### Recommended Project Structure

No new files or directories needed beyond:
- `src/experiments/run_full_comparison.py` (new — 24-config comparison orchestration)
- `src/experiments/write_final_report.py` (new — report generation from JSON results)
- `results/full_comparison.json` (generated by comparison script)
- `results/per_matrix_summary.json` (generated by comparison script)
- `docs/REPORT.md` (regenerated by report script)

All other assets already exist: quantizer modules, analysis modules, data, model.

### Pattern 1: Method Dispatch + Evaluation Loop

**What:** Apply one quantization method to a fresh model copy, evaluate PPL and per-matrix error, collect results, clean up. Repeat for all 24 configs. Nested loop: 2 checkpoints -> 2 formats -> 6 methods.

**When to use:** For the 24-config comparison loop.

**Example pattern (from phase2_comparison.py + validate_theorem1.py):**

```python
# Method dispatch — core loop pattern (Source: phase2_comparison.py lines 114-159)
for ckpt_name, ckpt_path in checkpoints.items():
    for fmt_name, fmt_str in formats.items():
        for method_name in methods:
            model = load_model(ckpt_path, device)
            try:
                if method_name == 'rtn':
                    q = FPQuantizer(fmt_str, per_channel=True)
                    apply_ptq_simple(model, q)
                elif method_name == 'gptq':
                    q = FPQuantizer(fmt_str, per_channel=True)
                    calib_loader = get_dataloader(4, 512, 256, data_dir, split='train')
                    gptq = GPTQQuantizer(q)
                    gptq.quantize_model(model, calib_loader, device)

                # Single forward pass captures activations for error computation
                tracker = ErrorPropagationTracker()
                tracker.attach(model)
                # Use validation data for this single pass
                val_batch = next(iter(get_dataloader(1, 512, 1, data_dir, split='val')))
                # This same forward pass is part of the PPL evaluation loop
                # ... OR use a separate single-batch capture

                # Compute per-matrix ||dy||/||y|| for THIS config's quantizer
                q_for_error = FPQuantizer(fmt_str, per_channel=True)
                errors = tracker.compute_output_error(model, q_for_error)

                # Evaluate PPL on validation split
                ppl = evaluate_perplexity(model, val_loader, device, 100)
            finally:
                del model; torch.cuda.empty_cache()
```

### Pattern 2: Simultaneous PPL + Error Measurement

**What:** Attach ErrorPropagationTracker before starting PPL evaluation, run the full 100-step eval, then compute per-matrix errors from the captured activations. The tracker captures activations from the LAST forward pass only (pre-hooks overwrite previous capture), so we need to decide which batch to use.

**Alternative A (CONTEXT.md D-03 recommendation):** Attach tracker during PPL evaluation — but the tracker captures activations on every forward call, overwriting previous ones. Only the last batch's activations are available for error computation. The error computation uses the model's weights as quantized, which is correct.

**Alternative B (safer, recommended):** After PPL evaluation, do a separate single-batch forward pass with tracker attached. This guarantees clean per-matrix error from the quantized model without coupling to the PPL eval loop. Tradeoff: one extra forward pass per config (negligible cost vs 100 eval steps).

**Recommended approach (Alternative B):**

```python
# 1. Evaluate PPL (no tracker needed for this)
ppl = evaluate_perplexity(model, val_loader, device, 100)

# 2. Separate single-batch error measurement
tracker = ErrorPropagationTracker()
tracker.attach(model)
batch = next(iter(val_loader))
with torch.no_grad():
    model(batch['input_ids'].to(device))
tracker.detach()
errors = tracker.compute_output_error(model, quantizer)
```

### Pattern 3: Quantizer Application for Method 4 (Hadamard Rotation + RTN)

**What:** Hadamard rotation is applied as a pre-processing step before quantization and post-processing step after dequantization. The model's weight matrices are rotated in-place, quantized, then unrotated.

**Implementation approach:**

```python
# Hadamard rotation + RTN (Source: hadamard.py hadamard_rotate_weight)
from src.quantization.hadamard import hadamard_rotate_weight

# Rotate weights, quantize, unrotate
for name, param in model.get_quantizable_weights():
    W = param.data
    W_rotated = hadamard_rotate_weight(W)
    W_quantized = quantizer.quantize(W_rotated)
    param.data = hadamard_rotate_weight(W_quantized)  # Inverse
```

**Key insight:** Hadamard rotation is self-inverse (H @ H = nI, so H @ (H @ W) = nW; the scale factor 1/sqrt(n) in fast_hadamard_transform ensures H @ H = I). The apply-then-undo pattern means the quantization distortion happens in rotated space but the final weights remain in FP16 space.

### Pattern 4: Quantizer Application for Method 5 (Outlier Rotation + RTN)

**What:** DuQuantStyleQuantizer handles the full pipeline internally (scale, rotate, quantize, unrotate, unscale). Its `quantize(W)` method does the whole pipeline in one call.

```python
from src.quantization.fp4_grids import FP4_E2M1_GRID
from src.quantization.outlier_rotation import DuQuantStyleQuantizer

duquant = DuQuantStyleQuantizer(FP4_E2M1_GRID, block_size=32)
for name, param in model.get_quantizable_weights():
    if param.dim() >= 2:
        param.data = duquant.quantize(param.data)
```

### Pattern 5: Method Dispatch for Method 6 (MXFP4)

```python
from src.quantization.fp4_grids import MXFP4Quantizer

mx = MXFP4Quantizer(block_size=32)
for name, param in model.get_quantizable_weights():
    if param.dim() >= 2:
        param.data = mx.quantize(param.data)
```

### Anti-Patterns to Avoid

- **Dual-purpose script overloading:** Do not combine the comparison loop and report generation in one script. The comparison takes hours (24 configs x 100 eval steps on GPU); report generation is seconds. Keep them separate per D-10/D-11.
- **Forgetting clean data split:** All Phase 2 scripts used `get_dataloader()` without `split=` parameter, which loaded all .bin files including the validation split for calibration. Phase 5 MUST pass `split='train'` for calibration and `split='val'` for evaluation.
- **FP4 Hadamard/Outlier path:** Hadamard rotation and outlier rotation were designed for FP8. At FP4's higher quantization noise, the rotation benefits may vanish or invert. The CONTEXT.md explicitly flags this (Claude's discretion to skip FP4 variants).
- **Reusing activations across quantizer instances:** `compute_output_error(model, quantizer)` needs a quantizer that matches the applied quantization. If GPTQ was applied, the model's weights are already modified in-place, so passing any quantizer to compute_output_error would mis-measure. Use a separate FPQuantizer for the error computation that matches the applied method.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Per-matrix ||dy||/||y|| | Custom forward pass analysis | ErrorPropagationTracker.compute_output_error() | Takes quantizer object, captures activations via hooks, computes offline. Already verified in Phase 3/4. [VERIFIED: src/analysis/error_propagation.py:289-330] |
| FP format simulation | Custom quantize logic | FPQuantizer (4 formats: E4M3, E5M2, E2M1, E4M3FN) | Simulates exact FP grid with per-channel/per-tensor scaling, deterministic/stochastic. [VERIFIED: src/quantization/fp_quantizer.py] |
| GPTQ column compensation | Manual Hessian inversion + update | GPTQQuantizer | Handles Cholesky damping, progressive stabilization, configurable blocksize. [VERIFIED: src/quantization/gptq.py] |
| Lloyd-Max grid optimization | Manual Lloyd iteration | AdaptiveGridQuantizer (calibrate + quantize_model) | Per-layer Lloyd-Max with optional kappa-weighting, symmetric grid construction. [VERIFIED: src/quantization/adaptive_grid.py] |
| Report generation from JSON | Manual copy-paste into Markdown | write_final_report.py | A single-threaded script guarantees consistency between numerical results and text. |

**Key insight:** Every component Phase 5 needs already exists with verified working interfaces. The primary engineering effort is orchestration — connecting existing parts with correct data splits, not building new infrastructure.

## Runtime State Inventory

> **Phase type:** Greenfield evaluation + report generation. No rename, refactor, or migration. Skipping this section.

## Common Pitfalls

### Pitfall 1: compute_output_error Quantizer Mismatch
**What goes wrong:** GPTQ modifies weights in-place (quantize + compensate). After GPTQ, calling `compute_output_error(model, FPQuantizer(...))` re-quantizes the already-GPTQ-compensated weights, producing an ||dy||/||y|| that reflects double-quantization, not GPTQ's actual output error.
**Why it happens:** `compute_output_error` applies its quantizer argument to each weight matrix during computation, regardless of whether the model's weights are already quantized.
**How to avoid:** For GPTQ, pass the same quantizer type (e.g., `FPQuantizer('fp4_e2m1', per_channel=True)`) to compute_output_error that GPTQ used internally. The weights are already modified, so compute_output_error will re-quantize them and produce the correct final output error. Alternatively, understand that after GPTQ, the model's weights contain compensated values that are NOT the original FP16 weights — the `||(W_compensated_q - W_compensated)||` error is the relevant metric.
**Warning signs:** ||dy||/||y|| values for GPTQ that are orders of magnitude away from RTN values.

### Pitfall 2: Tracker Captures Only Last Batch
**What goes wrong:** ErrorPropagationTracker's pre-hooks overwrite previous activation captures. If attached during the 100-step PPL evaluation loop, only the last batch's activations are available for error computation.
**Why it happens:** Each forward_pre_hook stores `input_args[0].detach().clone()` to a single dict key, overwriting the previous value.
**How to avoid:** Use a separate single-batch forward pass for error measurement after PPL evaluation (Alternative B in Architecture Patterns). This decouples the two measurements and guarantees clean activations.
**Warning signs:** Per-matrix errors that differ significantly between runs with different batch sizes.

### Pitfall 3: Data Leak from Calibration on Validation Data
**What goes wrong:** `get_dataloader()` without `split=` falls back to loading ALL .bin files (train + val). GPTQ/adaptive grid calibration would then observe validation data statistics, making PPL results optimistic (in-sample, not out-of-sample).
**Why it happens:** The original `get_dataloader()` in Phase 1 accepted but did not require the `split=` parameter; the `MultiTierDataset` matched `*_val.bin` AND `*_train.bin` when `split` was empty.
**How to avoid:** Always pass `split='train'` for calibration dataloaders and `split='val'` for evaluation dataloaders.
**Warning signs:** PPL values that are suspiciously close to FP16 baseline for methods that had calibration data access.

### Pitfall 4: Mixed Interface for Per-Channel Scaling in Compute Output Error
**What goes wrong:** compute_output_error receives a quantizer and applies it to each weight matrix independently. For per-channel quantizers, each matrix gets its own per-channel scale. But the ErrorPropagationTracker captures activations x that are BATCHES of sequences, not individual tokens — the output space error ||(W_q - W)x|| / ||Wx|| uses batch activation, which is correct.
**Why it happens:** The pre-hook captures the full batch input to the Linear layer.
**How to avoid:** No action needed — this is the correct behavior. The relative error over a batch of activations gives a single scalar per matrix, which is what the summary table expects.
**Warning signs:** None — this is designed behavior.

## Code Examples

### Model Loading and Checkpoint Verification
```python
# Source: validate_theorem1.py lines 292-297, phase2_comparison.py lines 43-49
from src.model.config import MicroGemmaFPConfig
from src.model.transformer import MicroGemmaFPForCausalLM
from src.experiments.training_utils import load_checkpoint

def load_model(ckpt_path, device):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    config = MicroGemmaFPConfig()
    model = MicroGemmaFPForCausalLM(config).to(device)
    load_checkpoint(model, None, ckpt_path, device)
    model.eval()
    return model
```

### ErrorPropagationTracker Usage for Per-Matrix ||dy||/||y||
```python
# Source: validate_theorem1.py lines 254-276, error_propagation.py lines 289-330
from src.analysis.error_propagation import ErrorPropagationTracker

tracker = ErrorPropagationTracker()
tracker.attach(model)

# Single validation batch
batch = next(iter(val_loader))
input_ids = batch['input_ids'].to(device)
with torch.no_grad():
    model(input_ids)

tracker.detach()

# Compute per-matrix errors for this quantizer
errors = tracker.compute_output_error(model, quantizer)
# errors: dict[str, float] like {'model.layers.0.attention.q_proj': 0.0123, ...}
```

### Method Dispatch Pattern (24 Configs)
```python
# Source: phase2_comparison.py lines 96-159 (adapted for Phase 5)

checkpoints = {
    'fp16_baseline': 'checkpoints/scaled_fp16_baseline/model.pt',
    'cond_regularized': 'checkpoints/cond_regularized/model.pt',
}
formats = {'FP8': 'fp8_e4m3', 'FP4': 'fp4_e2m1'}
methods = ['rtn', 'gptq', 'lloyd_max', 'hadamard', 'outlier', 'mxfp4']

all_results = {}
for ckpt_name, ckpt_path in checkpoints.items():
    for fmt_name, fmt_str in formats.items():
        for method_name in methods:
            # Skip: Hadamard/Outlier for FP4 (per Claude's discretion)
            if fmt_name == 'FP4' and method_name in ('hadamard', 'outlier'):
                continue

            model = load_model(ckpt_path, device)
            q = FPQuantizer(fmt_str, per_channel=True)

            # Apply quantization method
            if method_name == 'rtn':
                apply_rtn(model, q)
            elif method_name == 'gptq':
                calib = get_dataloader(4, 512, 256, data_dir, split='train')
                GPTQQuantizer(q).quantize_model(model, calib, device)
            elif method_name == 'lloyd_max':
                aq = AdaptiveGridQuantizer(kappa_weight=0.0)
                aq.calibrate(model, data_dir, device)  # uses training split
                aq.quantize_model(model)
            elif method_name == 'hadamard':
                apply_hadamard_rtn(model, q)
            elif method_name == 'outlier':
                apply_outlier_rotation(model, fmt_str)
            elif method_name == 'mxfp4':
                apply_mxfp4(model)

            # Evaluate
            val_loader = get_dataloader(8, 512, 100, data_dir, split='val')
            ppl = evaluate_perplexity(model, val_loader, device, 100)

            # Per-matrix error
            tracker = ErrorPropagationTracker()
            tracker.attach(model)
            batch = next(iter(val_loader))
            with torch.no_grad():
                model(batch['input_ids'].to(device))
            tracker.detach()
            errors = tracker.compute_output_error(model, q)

            all_results[key] = {'ppl': ppl, 'errors': errors}
            del model; torch.cuda.empty_cache()
```

### GPTQ Calibration with Clean Data Split
```python
# Source: gptq.py quantize_model, training_utils.py get_dataloader
# Use training split for calibration
calib_loader = get_dataloader(
    batch_size=4, max_seq_len=512, max_steps=256,
    data_dir=data_dir, split='train'
)
# The quantize_model method reads batch['input_ids'] and collects
# activations via forward hooks for Hessian estimation
gptq = GPTQQuantizer(quantizer, blocksize=128)
stats = gptq.quantize_model(model, calib_loader, device)
```

### Hadamard + RTN Application
```python
# Source: hadamard.py hadamard_rotate_weight, phase2_comparison.py apply_ptq_simple
from src.quantization.hadamard import hadamard_rotate_weight

@torch.no_grad()
def apply_hadamard_rtn(model, quantizer):
    for name, param in model.get_quantizable_weights():
        W = param.data
        W_rot = hadamard_rotate_weight(W)
        W_q = quantizer.quantize(W_rot)
        param.data = hadamard_rotate_weight(W_q)
```

### Per-Matrix Summary Table Construction
```python
# Source: validate_theorem1.py lines 342-382 for table construction pattern

# Merge Phase 3 data (kappa, dw_norm, tightness) with Phase 5 data (dy_norm)
# and Phase 4 data (norm_attenuation)
summary_rows = []
for row in phase3_results:  # from theorem1_validation.json
    name = row['name']
    entry = {
        'name': name,
        'layer': row['layer'],
        'type': row['type'],
        'kappa': row['kappa'],
        'dw_norm': row['dw_norm'],
        'dy_norm': phase5_dy.get(name, float('nan')),
        'tightness_ratio': row['tightness_ratio'],
        'norm_attenuation': phase4_attenuation.get(layer_of(name), float('nan')),
    }
    summary_rows.append(entry)

# Sort by layer -> type
type_rank = {'attention': 0, 'ffn': 1, 'global': 2}
summary_rows.sort(key=lambda r: (r['layer'] if r['layer'] >= 0 else 999,
                                  type_rank.get(r['type'], 99)))
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| phase2_comparison.py (pre-clean-split, PPL-only) | run_full_comparison.py (clean split, PPL + per-matrix ||dy||/||y||) | Phase 5 | Corrects data leakage from calibration on eval data; adds output-space error metric |
| Single-script manual calculation | Two-script automated pipeline | Phase 5 | Separates GPU-heavy evaluation from CPU-light report generation |
| Inverse power iteration for kappa | Exact SVD via torch.linalg.svdvals | Phase 3 (from ANALYSIS.md audit) | Exact condition numbers, no estimation bias |

**Deprecated/outdated:**
- `phase2_comparison.py` method of using unsplit data for calibration: Phase 5 should not replicate this behavior.
- PPL-only evaluation of quantization: Phase 5 adds per-matrix ||dy||/||y|| as a layer-level metric, providing more granular insight than PPL alone.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | SciPy is available on remote GPU server (optional — Pearson r fallback exists in pure numpy) | Code Examples, Dependencies | LOW — if scipy missing, fallback to numpy.pearsonr works for primary computation |
| A2 | Checkpoints exist at paths `checkpoints/scaled_fp16_baseline/model.pt` and `checkpoints/cond_regularized/model.pt` | Common Pitfalls | HIGH — if missing, entire phase is blocked. Script must exit with clear error |
| A3 | clean train/val split bin files already exist in `data/real_tiers/` | Standard Stack | MEDIUM — current data inspection confirms they exist |
| A4 | Hadamard rotation and outlier rotation methods may show instability at FP4 | Architecture Patterns | MEDIUM — Claude's discretion to skip these FP4 variants; D-01 defines the list but D-10 says "may be omitted if pathological" |
| A5 | ErrorPropagationTracker activations from PPL eval loop work for simultaneous measurement | Architecture Patterns, Pitfall 2 | MEDIUM — if tracker overwrites activations, separate forward pass is needed |
| A6 | The 72 weight matrices are exactly those from get_quantizable_weights() returning `(proj|embed_tokens|lm_head)` with dim >= 2 | Code Examples | LOW — verified in transformer.py:264-270 |

## Open Questions (RESOLVED)

1. **How should compute_output_error handle GPTQ models?**
   - What we know: After GPTQ, weights are compensated in-place. `compute_output_error` re-quantizes whatever weights are present. If we pass the same quantizer type, it measures ||(W_compensated_q - W_compensated)|| / ||W_compensated * x||, which is the actual output error of GPTQ — exactly what we want.
   - What's unclear: Whether to use the original FP16 activations (captured before any quantization) or re-capture after GPTQ with the compensated weights. Using the original activations is standard (single-pass capture at FP16) and matches what validate_theorem1.py does.
   - **RESOLVED:** Use original FP16 activations (captured once at the start), then compute_output_error with the same quantizer type that was applied. This is consistent with Theorem 1's measurement protocol.

2. **Should Hadamard/Outlier methods compute Hadamard on activations or weights?**
   - What we know: `hadamard_rotate_weight` rotates the WEIGHT matrix. The `HadamardRotation` module rotates ACTIVATIONS at runtime. Both approaches exist in the literature.
   - What's unclear: The CONTEXT.md says "Hadamard rotation + round-to-nearest" without specifying weight vs activation rotation. The `hadamard.py` module provides both.
   - **RESOLVED:** Follow CONTEXT.md D-02 — use weight rotation (apply Hadamard to weight via hadamard_rotate_weight, quantize with FPQuantizer, undo Hadamard).

3. **Should per-matrix error use the SAME validation batch across all configs for consistency?**
   - What we know: Different batches have different activation distributions, so ||dy||/||y|| values vary slightly between batches.
   - **RESOLVED:** Use fixed seed=42 and first validation batch for all 24 configs' error computation, ensuring cross-config comparability.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.11+ | All scripts | Yes | 3.14.5 (local), conda `sle` (remote) | — |
| PyTorch >= 2.3 | Model/quantization | No (local), Yes (remote) | — | Run via ./remote_python.sh |
| CUDA GPU | Model inference | No (local), Yes (remote, 8x RTX 4090) | — | CPU fallback possible but 100x slower |
| NumPy | JSON data handling | Yes (local, remote) | — | — |
| SciPy | Optional Pearson r | [ASSUMED: remote] | — | Pure-numpy fallback in validate_theorem1.py |
| checkpoint (baseline) | Comparison | Unknown | — | Script exits with error if missing |
| checkpoint (cond) | Comparison | Unknown | — | Script exits with error if missing |

**Missing dependencies with no fallback:**
- PyTorch + CUDA GPU for running the comparison (must use remote server)
- FP16 baseline and cond_regularized checkpoints (must exist before Phase 5)

**Missing dependencies with fallback:**
- SciPy: pure-numpy Pearson r fallback exists

## Validation Architecture

> Skipping — `workflow.nyquist_validation` is explicitly `false` in config.json.

## Security Domain

> Skipping — No security-sensitive operations (network, auth, filesystem write of non-sensitive JSON/Markdown only).

## Sources

### Primary (HIGH confidence)
- [VERIFIED: codebase] `src/experiments/phase2_comparison.py` — Method dispatch pattern verified by reading the file.
- [VERIFIED: codebase] `src/experiments/validate_theorem1.py` — ErrorPropagationTracker usage, model loading, table printing, JSON export verified.
- [VERIFIED: codebase] `src/analysis/error_propagation.py` — ErrorPropagationTracker class with attach/compute_output_error/detach verified.
- [VERIFIED: codebase] `src/experiments/training_utils.py` — evaluate_perplexity, load_checkpoint, get_dataloader with split support verified.
- [VERIFIED: codebase] `src/quantization/fp_quantizer.py` — FPQuantizer interface verified.
- [VERIFIED: codebase] `src/quantization/gptq.py` — GPTQQuantizer quantize_model interface verified.
- [VERIFIED: codebase] `src/quantization/adaptive_grid.py` — AdaptiveGridQuantizer calibrate/quantize_model verified.
- [VERIFIED: codebase] `src/quantization/hadamard.py` — HadamardRotation and hadamard_rotate_weight verified.
- [VERIFIED: codebase] `src/quantization/outlier_rotation.py` — DuQuantStyleQuantizer quantize method verified.
- [VERIFIED: codebase] `src/quantization/fp4_grids.py` — MXFP4Quantizer, GridQuantizer, pre-built grids verified.
- [VERIFIED: codebase] `src/analysis/condition.py` — compute_all_condition_numbers, estimate_condition_number verified.
- [VERIFIED: codebase] `src/model/transformer.py` — get_quantizable_weights() verified.
- [VERIFIED: data directory] `data/real_tiers/` — train/val split bin files confirmed present.
- [VERIFIED: codebase] `.planning/phases/05-extended-ptq-comparison-and-final-report/05-CONTEXT.md` — All locked decisions verified.

### Secondary (MEDIUM confidence)
- [CITED: CONTEXT.md] COMP-01 through COMP-03, REPORT-01 through REPORT-03 requirements — refer to CONTEXT.md decisions D-01 through D-14.
- [ASSUMED] SciPy availability on remote server — not verified at runtime, but fallback exists.

### Tertiary (LOW confidence)
- None — all technical claims verified against the actual codebase and data directory.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - all quantizers, analysis tools, and data loading code verified against actual source files.
- Architecture: HIGH - method dispatch, error measurement, and report generation patterns extracted from proven Phase 2/3/4 code.
- Pitfalls: HIGH - all four documented pitfalls are verified against the actual code behavior (tracker overwrite, compute_output_error gotcha, data leakage, per-channel scaling).

**Research date:** 2026-05-17
**Valid until:** 2026-06-17 (stable codebase, no fast-moving dependencies)
