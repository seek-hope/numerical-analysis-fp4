# Phase 5: Extended PTQ Comparison and Final Report - Context

**Gathered:** 2026-05-17
**Status:** Ready for planning

## Phase Boundary

Complete comparison of all PTQ methods under clean conditions using per-matrix output error as the sole evaluation metric, culminating in the final project report. Re-run the 24-config PTQ comparison using the clean data split (Phase 1) and the ErrorPropagationTracker (Phase 2) to report per-matrix ||dy||/||y|| for every configuration. PPL is not used as an evaluation metric — it is confounded by downstream RMSNorm/attention/FFN/lm_head transformations (see REPORT.md Methodology §6). Compare GPTQ column compensation against round-to-nearest on output-space error. Compare Lloyd-Max adaptive grids against uniform E2M1 on ||dy||/||y||. Synthesize all Phase 1-4 results into a comprehensive per-matrix error summary table and update the project report with corrected methodology and final theoretical assessment.

## Implementation Decisions

### 24-Config Comparison Scope (COMP-01)

- **D-01:** The 24 configurations are defined as: 2 checkpoints (FP16 baseline, cond_regularized) × 2 formats (FP8 E4M3, FP4 E2M1) × 6 methods (round-to-nearest per-channel, GPTQ per-channel, Lloyd-Max adaptive, Hadamard rotation + round-to-nearest, outlier rotation + round-to-nearest, MXFP4 block-scaling). All use the clean validation split for evaluation and training split for calibration (GPTQ Hessian, adaptive grid fitting).
- **D-02:** Re-use existing quantizer infrastructure: `FPQuantizer` for round-to-nearest, `GPTQQuantizer` for GPTQ, `AdaptiveGridQuantizer` for Lloyd-Max, `HadamardRotation` + `FPQuantizer` for Hadamard, `DuQuantStyleQuantizer` for outlier rotation, `MXFP4Quantizer` for MXFP4.
- **D-03:** Per-matrix ||dy||/||y|| measured via manual computation from captured FP16 activations (not PPL — see Methodology §6 in REPORT.md for the PPL confound rationale).

### GPTQ vs Round-to-Nearest Comparison (COMP-02)

- **D-04:** For both checkpoints and both formats (4 configs: 2 checkpoints × 2 formats × 1 method-pair), compare GPTQ against round-to-nearest. For each config, report: (a) mean ||dy||/||y|| difference across all 72 matrices, (b) per-matrix ||dy||/||y|| delta (GPTQ_error - RTN_error). Negative values indicate GPTQ reduces output-space error; positive values indicate it increases it.
- **D-05:** Use the same calibration data for GPTQ Hessian computation as the original phase2_comparison.py — training split, 256 samples, seq_len=512.

### Lloyd-Max vs Uniform E2M1 Comparison (COMP-03)

- **D-06:** For both checkpoints (2 configs: 2 checkpoints × FP4 E2M1 × 1 method-pair), compare Lloyd-Max adaptive grids against uniform E2M1. For each checkpoint, report: (a) mean ||dy||/||y|| difference, (b) per-matrix ||dy||/||y|| delta (adaptive_error - uniform_error).
- **D-07:** Lloyd-Max grids fitted on training split (256 samples). Adaptive grid quantization uses per-channel scaling; grid values are per-layer optimized (not per-matrix).

### Per-Matrix Error Summary Table (REPORT-01)

- **D-08:** Single comprehensive table with 72 rows (one per Linear weight matrix) and columns: name, layer, type (attention/ffn/global), κ(W), ||δW||/||W|| (FP4), ||δy||/||y|| (FP4 RTN), tightness_ratio, RMSNorm attenuation (input_norm ratio from Phase 4 trace). Source data: κ and weight error from Phase 3's validate_theorem1.py output (theorem1_validation.json), ||dy||/||y|| from the re-run 24-config comparison, RMSNorm attenuation from Phase 4's trace output (error_propagation_trace.json).
- **D-09:** Table printed to stdout and exported as JSON (`results/per_matrix_summary.json`). Sort by layer → type for logical grouping.

### Script Design

- **D-10:** Single comprehensive comparison script `src/experiments/run_full_comparison.py` that:
  1. Loads FP16 baseline and cond_regularized checkpoints
  2. Runs the 24-config PTQ comparison with clean data split
  3. For each config: computes per-matrix ||dy||/||y|| (single batch via manual computation from captured FP16 activations)
  4. Performs GPTQ vs RTN comparison (COMP-02)
  5. Performs Lloyd-Max vs uniform comparison (COMP-03)
  6. Loads Phase 3 and Phase 4 JSON results
  7. Generates per-matrix error summary table (REPORT-01)
  8. Exports all results as `results/full_comparison.json`
- **D-11:** Separate final report script `src/experiments/write_final_report.py` that:
  1. Reads all JSON results (theorem1_validation.json, error_propagation_trace.json, full_comparison.json)
  2. Generates REPORT.md with: executive summary, methodology (corrected), per-matrix table, GPTQ analysis, Lloyd-Max analysis, RMSNorm analysis, propagation waterfall summary, revised theoretical assessment
  3. References existing `docs/ANALYSIS.md` for mathematical derivations (Theorems 1-4)
- **D-12:** argparse interface for comparison script: `--fp16_checkpoint`, `--cond_checkpoint`, `--data_dir`, `--output`, `--device`, `--batch_size`, `--max_seq_len`, `--eval_steps` (default 100). Report script: `--results_dir` (path to results/), `--output` (default docs/REPORT.md).

### Checkpoint Dependency

- **D-13:** Phase 5 requires two checkpoints: FP16 baseline (`checkpoints/scaled_fp16_baseline/model.pt`) and cond_regularized (`checkpoints/cond_regularized/model.pt`). These must exist before the script runs. If they don't exist, either the training scripts must be re-run first, or the comparison script should exit with a clear error message. This is a pre-condition, not a build step.

### Visualization (Deferred)

- **D-14:** v2 visualization requirements (VIS-01: waterfall chart, VIS-02: κ vs ||dy||/||y|| scatter, VIS-03: RMSNorm bar chart) are deferred. Phase 5 produces the structured JSON data these visualizations need; actual chart generation is a post-project follow-up unless time permits. Phase 5 scripts print formatted tables to stdout — sufficient for the course project report.

### Claude's Discretion

- Whether to run the comparison script as a single pass (all 24 configs in one script execution) or allow re-entrant partial execution (e.g., `--configs 0-11` and `--configs 12-23` for GPU time management)
- Exact print table formatting (column widths, decimal places)
- Whether to include per-layer-type subgroup analysis in the comparison report (attention-only ||dy||/||y||, FFN-only ||dy||/||y||)
- Whether to compute cross-checkpoint comparisons (FP16 baseline vs cond_regularized) on ||dy||/||y||
- Logging verbosity (progress per config, timing info)
- Whether to include the Hadamard and outlier rotation methods if they show instability at this model scale (they may be omitted if results are pathological)

## Canonical References

### Phase 1-4 Outputs (Primary Data Sources)
- `results/theorem1_validation.json` — Phase 3: per-matrix κ, ||dW||/||W||, ||dy||/||y||, tightness_ratio across 3 seeds
- `results/error_propagation_trace.json` — Phase 4: per-source P-point waterfall, RMSNorm attenuation, parallel/orthogonal decomposition
- `Phase 1 dataloaders` — `get_dataloader(split='train')` and `get_dataloader(split='val')` for clean calibration/evaluation

### Measurement Infrastructure
- `src/analysis/error_propagation.py` — ErrorPropagationTracker (attach, compute_output_error, P-point capture)
- `src/analysis/condition.py:55-65` — compute_all_condition_numbers()
- `src/experiments/validate_theorem1.py` — Reference: model loading, tracker usage, results table printing, JSON export

### Quantization Methods
- `src/quantization/fp_quantizer.py` — FPQuantizer (FP8 E4M3, FP4 E2M1, round-to-nearest)
- `src/quantization/gptq.py` — GPTQQuantizer (Hessian calibration, column compensation)
- `src/quantization/adaptive_grid.py` — AdaptiveGridQuantizer (Lloyd-Max per-layer)
- `src/quantization/fp4_grids.py` — MXFP4Quantizer (block-scaling, block_size=32)
- `src/quantization/hadamard.py` — HadamardRotation (Walsh-Hadamard transform)
- `src/quantization/outlier_rotation.py` — DuQuantStyleQuantizer (outlier scaling + Hadamard)

### Existing Comparison Scripts (Templates)
- `src/experiments/phase2_comparison.py` — Original 24-config PTQ comparison (pre-clean-split, PPL-only)
- `src/experiments/final_summary.py` — Summary script loading multiple checkpoints
- `src/experiments/eval_quantization.py` — Unified PTQ evaluation (method dispatch)
- `src/experiments/compare_adaptive_grid.py` — Lloyd-Max grid comparison

### Training Infrastructure
- `src/experiments/training_utils.py:313-337` — evaluate_perplexity() (validation PPL)
- `src/experiments/training_utils.py:351-356` — load_checkpoint()
- `src/experiments/training_utils.py:197-219` — get_dataloader(split='val')

### Model
- `src/model/transformer.py:264-270` — get_quantizable_weights()
- `src/model/config.py:15-87` — MicroGemmaFPConfig

### Requirements
- `.planning/REQUIREMENTS.md` §COMP-01, COMP-02, COMP-03, REPORT-01, REPORT-02, REPORT-03
- `.planning/ROADMAP.md` §"Phase 5: Extended PTQ Comparison and Final Report" — Success criteria (5 items)

### Project Foundation
- `.planning/PROJECT.md` — Core value, constraints, key decisions
- `.planning/STATE.md` — Accumulated context
- `docs/ANALYSIS.md` — Full mathematical derivations (Theorems 1-4)

### Phase Context
- `.planning/phases/03-theorem-1-validation/03-CONTEXT.md` — Theorem 1 validation decisions
- `.planning/phases/04-error-propagation-trace/04-CONTEXT.md` — Error propagation trace decisions

## Existing Code Insights

### Reusable Assets
- `ErrorPropagationTracker` + `compute_output_error()` — Directly callable for per-matrix ||dy||/||y|| during PTQ evaluation
- Existing quantizer classes (`FPQuantizer`, `GPTQQuantizer`, `AdaptiveGridQuantizer`, `MXFP4Quantizer`, `HadamardRotation`, `DuQuantStyleQuantizer`) — All 6 methods have working implementations in `src/quantization/`
- `evaluate_perplexity()` — Standard PPL evaluation with shifted labels, padding masks, weighted loss
- `phase2_comparison.py` — Proven config dispatch logic (apply method → evaluate → report). Can be adapted with: (a) clean split dataloaders, (b) ErrorPropagationTracker for ||dy||/||y|| alongside PPL
- `validate_theorem1.py` — Model loading, tracker attach/detach pattern — copy the proven flow
- `_classify_matrix()` / `_fmt_val()` — Reusable utilities for matrix classification and safe float formatting

### Established Patterns
- argparse + main() for experiment scripts
- `with torch.no_grad()` for all evaluation
- Results returned as dict[str, float] keyed by module path
- JSON export with nested structure
- print() for logging, section separators for visual grouping
- Method dispatch pattern: apply quantization → evaluate → report → loop
- Remote execution via ./remote_python.sh

### Integration Points
- **Phase 3 output** (`theorem1_validation.json`): κ values, weight error, tightness ratios — consumed for REPORT-01 per-matrix summary table
- **Phase 4 output** (`error_propagation_trace.json`): RMSNorm attenuation ratios, P-point waterfalls — consumed for REPORT-01 norm_attenuation column and REPORT-02 waterfall data
- **Phase 1 dataloader**: Clean train/val split — all Phase 5 calibration uses training split, all evaluation uses validation split
- **PTQ method classes**: All in `src/quantization/` — Phase 5 is the definitive consumer of all quantization methods

## Specific Ideas

The 24 configurations, expanded:

| # | Checkpoint | Format | Method |
|---|-----------|--------|--------|
| 1-4 | FP16 baseline | FP8 E4M3 | RTN, GPTQ, Hadamard, Outlier |
| 5-8 | FP16 baseline | FP4 E2M1 | RTN, GPTQ, Lloyd-Max, MXFP4 |
| 9-12 | Cond regularized | FP8 E4M3 | RTN, GPTQ, Hadamard, Outlier |
| 13-16 | Cond regularized | FP4 E2M1 | RTN, GPTQ, Lloyd-Max, MXFP4 |

Plus specific method comparisons:
- GPTQ vs RTN: configs 1vs2, 5vs6, 9vs10, 13vs14 (4 pairs × 2 formats)
- Lloyd-Max vs uniform: configs 7vs5, 15vs13 (2 pairs, FP4 only)

Note: Hadamard rotation and outlier rotation may not apply meaningfully to FP4 (they're FP8-centric techniques). If phase2_comparison.py skips these for FP4, maintain consistency.

Final REPORT.md structure:
1. Executive Summary
2. Methodology (corrected — exact SVD, per-matrix measurement, clean data split, Bonferroni correction)
3. Theorem 1 Validation Results (from Phase 3)
4. Error Propagation Trace (from Phase 4)
5. Extended PTQ Comparison (24-config with both metrics)
6. GPTQ Analysis (column compensation vs output error)
7. Lloyd-Max Analysis (adaptive grids vs uniform)
8. RMSNorm Error Blocking (synthesis of Phase 2 + Phase 4 findings)
9. Revised Theoretical Assessment
10. References (ANALYSIS.md, PROPOSAL.md)

## Deferred Ideas

- **VIS-01/02/03 (v2 visualizations)**: Error waterfall charts, κ vs ||dy||/||y|| scatter plots, RMSNorm bar charts. Phase 5 produces the structured JSON these need — chart generation is post-project.
- **Per-head attention error decomposition (v2 ANALYSIS-01)**: 3× compute cost; not needed for current conclusions.
- **Rank stability analysis (v2 ANALYSIS-03)**: Δσ_distribution measurement — interesting but separate from the core comparison.

---

*Phase: 5-Extended PTQ Comparison and Final Report*
*Context gathered: 2026-05-17*
*Mode: --auto (all gray areas auto-selected, recommended options chosen)*
