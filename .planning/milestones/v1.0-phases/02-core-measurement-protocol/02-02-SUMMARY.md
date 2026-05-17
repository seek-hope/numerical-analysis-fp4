---
phase: 02-core-measurement-protocol
plan: 02
subsystem: experiments
tags: [measurement, error-propagation, kappa, fp4, condition-number, quantization]

requires:
  - phase: 02-01
    provides: ErrorPropagationTracker hook-based activation capture and offline per-matrix ||dy||/||y|| computation
provides:
  - Runnable measurement experiment script (measure_qerror.py) with full pipeline: model loading, tracker attachment, single forward pass, kappa computation, output-space and weight-space error computation, null validation, results table, and JSON export
affects: [03-correlation-analysis, 05-final-delivery]

tech-stack:
  added: []
  patterns:
    - "measurement experiment script that chains: model loading -> tracker attach -> forward pass -> detach -> kappa -> error compute -> null validation -> table output"
    - "per-matrix dw_norm computed via get_quantizable_weights() and FPQuantizer.quantize(), keyed by module path without .weight suffix"
    - "results table with columns: name, layer, type, kappa, ||dW||/||W||, ||dy||/||y||"

key-files:
  created:
    - src/experiments/measure_qerror.py
  modified: []

key-decisions:
  - "Use .replace('.weight', '') to normalize all dict keys (kappas, errors, dw_norms) to module-path convention"
  - "For results table, NaN kappa/dw_norm values printed as 'N/A' string instead of showing NaN"
  - "dw_norm computed via model.get_quantizable_weights() which filters to 2D weights containing 'proj', 'embed_tokens', or 'lm_head' in name"
  - "Layer index -1 used for global matrices (lm_head, embed_tokens) instead of per-layer indices"

patterns-established:
  - "Measurement experiment pipeline: Model loading -> tracker -> forward pass -> kappa -> output-space error -> weight-space error -> null validation -> results"
  - "Dict key normalization: strip .weight suffix so all per-matrix dicts use module-path keys without parameter suffix"
  - "NaN handling in results display: isinstance(x, float) and (x != x) check for NaN, render as 'N/A'"

requirements-completed: [MEAS-04, VAL-03]

duration: 8min
completed: 2026-05-17
---

# Phase 02 Plan 02: End-to-End Measurement Experiment Summary

**creates `src/experiments/measure_qerror.py` -- a runnable experiment script that loads the Micro-Gemma-FP model from a checkpoint, attaches ErrorPropagationTracker, runs a single forward pass on validation data, computes per-matrix kappa(W) via exact SVD, per-matrix ||dy||/||y|| and ||dW||/||W|| via FP4 round-to-nearest quantization, validates the pipeline with a null measurement, and outputs a structured results table with optional JSON export**

## Performance

- **Duration:** 8 min
- **Started:** 2026-05-17
- **Completed:** 2026-05-17
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments

- Created `src/experiments/measure_qerror.py` with complete end-to-end measurement pipeline
- Pipeline steps: model loading from checkpoint, ErrorPropagationTracker attachment, single forward pass on validation data (batch_size=1, seed=42, split='val'), tracker detach, kappa computation via exact SVD, per-matrix output-space error ||dy||/||y||, per-matrix weight-space relative error ||dW||/||W||, null measurement validation
- CLI interface with --checkpoint (required), --data_dir, --output, --device arguments
- Formatted results table with columns: name, layer, type, kappa, ||dW||/||W||, ||dy||/||y||, plus summary statistics
- Optional JSON export with per-matrix results for Phase 3 consumption

## Task Commits

Each task was committed atomically:

1. **Task 1: Create measurement experiment script with model loading and pipeline execution** - `e0e225d` (feat)
2. **Task 2: Add results table output and JSON export** - `b70bc15` (feat)

## Files Created/Modified

- `src/experiments/measure_qerror.py` - End-to-end per-matrix quantization error measurement pipeline

## Decisions Made

- Used `.replace('.weight', '')` to normalize dict keys across kappas (from `compute_all_condition_numbers`), errors (from `tracker.compute_output_error`), and dw_norms (from `get_quantizable_weights`) -- all three dicts use module-path keys without `.weight` suffix
- NaN values for kappa/dw_norm rendered as `'N/A'` in the table using a NaN check via `isinstance(x, float) and (x != x)`
- Layer index -1 for global matrices (lm_head, embed_tokens); 0-11 for per-layer attention/FFN matrices
- dw_norm only computed for matrices returned by `model.get_quantizable_weights()` (2D weights with 'proj', 'embed_tokens', or 'lm_head' in name), which may be a subset of all captured activations

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## Next Phase Readiness

- `measure_qerror.py` is the runnable entry point for Phase 2 verification
- Phase 3 will consume the JSON output from this script for correlation analysis between kappa, ||dW||/||W||, and ||dy||/||y||
- Remote smoke test needed: `./remote_python.sh src/experiments/measure_qerror.py --help` to verify CLI parsing on GPU server
- Full integration test: `./remote_python.sh src/experiments/measure_qerror.py --checkpoint checkpoints/scaled_fp16_baseline/model.pt --data_dir data/real_tiers --output results/phase02_results.json`

---
*Phase: 02-core-measurement-protocol*
*Completed: 2026-05-17*
