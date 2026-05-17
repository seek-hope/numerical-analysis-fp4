---
phase: 05-extended-ptq-comparison-and-final-report
plan: 01
subsystem: experiments
tags: ptq, comparison, gptq, lloyd-max, hadamard, outlier-rotation, mxfp4, ppl, error-propagation
requires:
  - phase: 03-theorem1-validation
    provides: theorem1_validation.json (per-matrix kappa, dw_norm, tightness_ratio)
  - phase: 04-error-propagation-trace
    provides: error_propagation_trace.json (per-layer norm_attenuation)
  - phase: 02-core-measurement-protocol
    provides: ErrorPropagationTracker, MultiTierDataset, get_dataloader with split support
provides:
  - run_full_comparison.py — definitive 16-20 config PTQ comparison orchestration
  - results/full_comparison.json — structured results for all configs (PPL, per-matrix errors)
  - results/per_matrix_summary.json — 72-row per-matrix table with merged Phase 3/4/5 data
affects: plan 05-02 (final report generation)
tech-stack:
  added: none (uses existing torch, numpy, json)
  patterns:
    - Manual per-matrix ||dy||/||y|| computation (x @ W.T) avoids Pitfall 1 quantizer mismatch
    - One-pass activation capture per checkpoint with fixed seed for cross-config comparability
    - GPTQ calibration on split='train', PPL evaluation on split='val' (clean data split)
key-files:
  created:
    - src/experiments/run_full_comparison.py
  modified: []
key-decisions:
  - "Manual error computation (not compute_output_error) to avoid Pitfall 1 quantizer mismatch for GPTQ/Lloyd-Max in-place methods"
  - "Activations captured ONCE per checkpoint at FP16 before any quantization, using fixed seed for all configs"
  - "GPTQ calibration uses split='train'; PPL evaluation uses split='val' per clean data split protocol"
  - "Lloyd-Max uses kappa_weight=0.0 (uniform) for fair comparison with RTN per D-06/D-07"
  - "Hadamard/Outlier for FP4 gated behind --include_experimental flag (default OFF)"
  - "--config_start/--config_end for re-entrant partial execution via config slicing"
patterns-established:
  - "All 6 quantizer application functions use @torch.no_grad() and accept (model, fmt_str, **kwargs) for uniform dispatch"
  - "Error computation uses manual y_fp = x @ W_fp.T vs y_q = x @ W_q.T for each matrix, comparing original vs quantized weights"
  - "Comparison analyses compute both PPL delta and mean ||dy||/||y|| delta with per-layer-type subgroup breakdown (attention, FFN)"
requirements-completed:
  - COMP-01
  - COMP-02
  - COMP-03
  - REPORT-01
duration: 12min
completed: 2026-05-17
---

# Phase 5 Plan 1: Extended PTQ Comparison Script

**Definitive 16-20 config PTQ comparison script with per-matrix ||dy||/||y||, PPL evaluation, GPTQ-vs-RTN and Lloyd-Max-vs-uniform analyses, merged Phase 3/4/5 per-matrix summary table, and JSON export**

## Performance

- **Duration:** 12 min
- **Started:** 2026-05-17T07:57:00Z
- **Completed:** 2026-05-17T08:09:00Z
- **Tasks:** 2
- **Files created:** 1 (867 lines)

## Accomplishments

- Created `run_full_comparison.py` (867 lines), the Phase 5 definitive PTQ comparison orchestrator
- 6 quantizer application functions: RTN, GPTQ, Lloyd-Max, Hadamard+RTN, Outlier+RTN, MXFP4
- Method dispatch dict with uniform interface for all quantizer methods
- One-pass activation capture at FP16 per checkpoint with fixed seed for cross-config comparability
- Manual per-matrix ||dy||/||y|| computation avoiding Pitfall 1 (quantizer re-application mismatch)
- GPTQ calibration on split='train', PPL evaluation on split='val' (clean data split)
- GPTQ-vs-RTN comparison with per-matrix delta and per-layer-type subgroup breakdown
- Lloyd-Max-vs-uniform E2M1 comparison with per-matrix delta
- 72-row per-matrix summary table merging Phase 3 (kappa/dw_norm), Phase 4 (norm_attenuation), and Phase 5 (dy_norm)
- JSON export to both `results/full_comparison.json` and `results/per_matrix_summary.json`
- Checkpoint existence guard with FileNotFoundError
- Re-entrant partial execution via `--config_start`/`--config_end`
- `--include_experimental` flag to extend Hadamard/Outlier to FP4 format

## Task Commits

Each task was committed atomically:

1. **Task 1: Write run_full_comparison.py skeleton** - `69594e3` (feat)
2. **Task 2: Add main() loop** - `dee81d3` (feat)

**Plan metadata:** `pending` (docs: complete plan — will be committed as part of execution tracking)

## Files Created/Modified

- `src/experiments/run_full_comparison.py` - Full 867-line PTQ comparison orchestrator with 14 functions, method dispatch, comparison analyses, per-matrix summary table, and JSON export

## Decisions Made

- **Manual error computation:** Using `x @ W_fp.T` vs `x @ W_q.T` for each matrix, avoiding `compute_output_error` which re-applies quantizer statelessly (Pitfall 1: GPTQ/Lloyd-Max modify weights in-place and have no stateless quantizer interface)
- **One-pass activation capture:** Activations captured once per checkpoint at FP16 with `torch.manual_seed(args.seed)` for deterministic batch selection across all configs, ensuring cross-config comparability
- **Clean data split:** GPTQ calibration uses `split='train'` (dedicated calibration data); PPL evaluation uses `split='val'` (never sees calibration data)
- **Lloyd-Max without kappa weighting:** `kappa_weight=0.0` for fair comparison with RTN per D-06/D-07 (the kappa-weighted variant was evaluated in Phase 2)
- **Experimental methods gated:** Hadamard and Outlier rotation for FP4 are behind `--include_experimental` flag to keep default run focused on primary comparisons

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None. The file compiles and all imports are structurally valid. Full execution requires PyTorch and remote GPU server (no local GPU available).

## Threat Surface Scan

No new threat surface introduced. The script reads local files only (checkpoints, JSON, data) and writes local JSON files. No network, auth, or external service operations. GPU memory exhaustion mitigated via `del model` + `torch.cuda.empty_cache()` in finally block per config, and `--config_start`/`--config_end` for re-entrant partial execution.

## Stub Tracking

No stubs identified — all quantizer application functions have real implementations, the main loop is complete with error handling, JSON export writes real data structures, and the comparison analyses have actual computation logic.

## Next Phase Readiness

- Ready for Plan 05-02 (final report generation / `write_final_report.py`)
- `results/full_comparison.json` and `results/per_matrix_summary.json` are the expected inputs for the report generation script
- To execute, run on remote GPU server: `./remote_python.sh src/experiments/run_full_comparison.py --data_dir data/real_tiers`

## Self-Check: PASSED

| Check | Status |
|-------|--------|
| run_full_comparison.py exists | FOUND |
| SUMMARY.md exists | FOUND |
| Commit 69594e3 (Task 1) | FOUND |
| Commit dee81d3 (Task 2) | FOUND |
| Python syntax (`py_compile`) | PASS |
| Line count | 867 (min 250 required) |

---
*Phase: 05-extended-ptq-comparison-and-final-report*
*Completed: 2026-05-17*
