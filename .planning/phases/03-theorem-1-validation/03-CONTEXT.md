# Phase 3: Theorem 1 Validation - Context

**Gathered:** 2026-05-17
**Status:** Ready for planning

## Phase Boundary

Test whether Theorem 1's predicted upper bound ||dy||/||y|| <= kappa * ||dW||/||W|| holds empirically at per-matrix granularity across all 72 Linear weight matrices in the Micro-Gemma-FP model. Use the ErrorPropagationTracker (Phase 2) for per-matrix output-space error measurement. Compute Pearson correlation between kappa and ||dy||/||y|| across all matrices with Bonferroni-corrected significance threshold. Run across 3 random seeds with bootstrap confidence intervals. Produce a definitive YES/NO/QUALIFIED answer with supporting statistical evidence.

## Implementation Decisions

### Statistical Computation
- **D-01:** Use `scipy.stats.pearsonr` for Pearson correlation coefficient computation (p-value, r-value in one call). Compute tightness_ratio = (||dy||/||y||) / (kappa * ||dW||/||W||) for each matrix. Bonferroni threshold: alpha = 0.05/72 = 0.00069.
- **D-02:** Bootstrap 95% CI via 10,000 resamples of the 72 (kappa, ||dy||/||y||) pairs. For each resample, compute Pearson r. CI = [2.5th percentile, 97.5th percentile] of bootstrap r distribution. Report mean ± std of r across 3 seeds, with combined bootstrap CI from pooled distributions.

### Multi-Seed Execution
- **D-03:** Run the full measurement pipeline (load checkpoint → attach tracker → single forward pass → compute ||dy||/||y||, kappa, ||dW||/||W||) for each of 3 seeds (42, 123, 456). Use the Phase 1 val split for the forward pass data. Each seed controls: (a) the random seed for data shuffling, (b) the evaluation batch selection.
- **D-04:** Aggregate across seeds: for each matrix, report mean and std of ||dy||/||y||, ||dW||/||W||, and tightness_ratio across the 3 seeds. Compute Pearson r(kappa, mean_||dy||) for the primary result. Report seed-by-seed r values for reproducibility assessment.

### Report Structure
- **D-05:** Single analysis script `src/experiments/validate_theorem1.py` that:
  1. Loads the FP16 baseline checkpoint
  2. Runs measurement pipeline for 3 seeds
  3. Computes Pearson r, Bonferroni threshold, bootstrap CI
  4. Prints the 72-matrix results table (name, layer, type, kappa, ||dW||/||W||, ||dy||/||y||, tightness_ratio)
  5. States the YES/NO/QUALIFIED verdict with supporting statistics
  6. Exports results as JSON for potential visualization
- **D-06:** Verdict rubric:
  - YES: r > 0.5 AND p < 0.00069 AND bootstrap CI excludes 0
  - QUALIFIED: r > 0.2 but doesn't meet all YES criteria (weak correlation, or significance borderline)
  - NO: r < 0.2 OR p > 0.05 (uncorrected) — no meaningful linear relationship

### Kappa and Error Sources
- **D-07:** Kappa(W) computed via `compute_all_condition_numbers()` (exact SVD from condition.py). ||dW||/||W|| computed as `(W_q - W_fp).norm() / W_fp.norm()` using FP4 round-to-nearest. ||dy||/||y|| from `ErrorPropagationTracker.compute_output_error()`.
- **D-08:** Results table uses module path naming convention (D-10 from Phase 2). Layer index and type (attention/fnn/global) parsed from module path for grouping. Matrix type: `q_proj`/`k_proj`/`v_proj`/`o_proj` → "attention", `gate_proj`/`up_proj`/`down_proj` → "ffn", `embed_tokens`/`lm_head` → "global".

### Outputs
- **D-09:** Primary outputs: (1) 72-row results table printed to stdout, (2) statistical summary with verdict, (3) `results/theorem1_validation.json` for downstream visualization (Phase 5).

### Claude's Discretion
- Whether to use scipy.stats or implement Pearson/bootstrapping in pure numpy
- Table formatting (column widths, alignment, sorting order)
- Number of bootstrap resamples (10,000 recommended, adjust for runtime)
- Whether to include per-layer-type subgroup analysis (attention-only, FFN-only correlations)
- Logging verbosity during multi-seed execution

## Canonical References

### Measurement Infrastructure (Phase 2)
- `src/analysis/error_propagation.py` — ErrorPropagationTracker with compute_output_error() and validate_null_measurement()
- `src/experiments/measure_qerror.py` — Reference experiment script (model loading, tracker attach, error compute, results table)
- `src/analysis/condition.py:28-65` — estimate_condition_number(), compute_all_condition_numbers() via exact SVD
- `src/quantization/fp_quantizer.py:53-80` — FPQuantizer for FP4 round-to-nearest

### Model Architecture
- `src/model/transformer.py:264-270` — get_quantizable_weights() naming convention
- `src/model/config.py:15-87` — MicroGemmaFPConfig (architecture dimensions)

### Data
- `src/experiments/training_utils.py:197-219` — get_dataloader(split='val') for validation data
- `src/experiments/training_utils.py:351-356` — load_checkpoint()

### Requirements
- `.planning/REQUIREMENTS.md` §VAL-01, VAL-02, VAL-04 — Full requirement text
- `.planning/ROADMAP.md` §"Phase 3: Theorem 1 Validation" — Success criteria (4 items)

### Project Foundation
- `.planning/PROJECT.md` — Core value, constraints, key decisions
- `.planning/STATE.md` — Accumulated context (Bonferroni correction, single-pass protocol)

## Existing Code Insights

### Reusable Assets
- `ErrorPropagationTracker` (error_propagation.py): attach/detach/compute_output_error/validate_null_measurement — directly callable from Phase 3 script
- `compute_all_condition_numbers()` (condition.py): returns dict[name → float] of kappa values — already validated
- `FPQuantizer('fp4_e2m1', per_channel=True)`: round-to-nearest FP4 — the quantizer for ||dW||/||W|| computation
- `measure_qerror.py`: Existing end-to-end pipeline — can be refactored into reusable functions or used as template
- `get_quantizable_weights()` (transformer.py): canonical 72-matrix list with naming convention

### Established Patterns
- Results returned as `dict[str, float]` keyed by module path
- Experiment scripts follow argparse + main() pattern
- Module path names strip `.weight` suffix for hook storage, use dot-separated paths
- 100-step evaluation with fixed seed for reproducibility
- JSON export for structured data interchange

### Integration Points
- **Phase 2 ErrorPropagationTracker**: Primary measurement tool — Phase 3 is the first consumer
- **FP16 baseline checkpoint**: Model source — loaded via `load_checkpoint()`
- **Validation dataloader**: Data source — `get_dataloader(split='val')`
- **Phase 5**: Will consume the per-matrix results table and JSON for final comparison

## Deferred Ideas

None — discussion stayed within phase scope.

---

*Phase: 3-Theorem 1 Validation*
*Context gathered: 2026-05-17*
*Mode: --auto (all gray areas auto-selected, recommended options chosen)*
