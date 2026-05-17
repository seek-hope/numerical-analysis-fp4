---
phase: 03-theorem-1-validation
plan: 01
phase_slug: theorem-1-validation
plan_slug: validate-theorem1
type: execute
subsystem: measurement
requires:
  - Phases 1-2 checkpoint + validation data + ErrorPropagationTracker
provides:
  - Per-matrix Theorem 1 validation pipeline (3 seeds, Bonferroni correction, bootstrap CI)
  - results/theorem1_validation.json for Phase 5 consumption
affects:
  - src/experiments/validate_theorem1.py
tags: [theorem-1, validation, multi-seed, correlation, bootstrap, bonferroni]
key-files:
  created:
    - src/experiments/validate_theorem1.py
    - results/theorem1_validation.json (generated at runtime)
  modified: []
tech-stack:
  added: []
  patterns:
    - "Explicit DataLoader construction with shuffle=True (avoids get_dataloader shuffle=False-for-val bug)"
    - "Seed-independent kappa and ||dW||/||W|| computed once, ||dy||/||y|| per seed"
    - "Key normalization: strip .weight suffix for cross-dict key matching"
    - "Three-tier verdict rubric: YES (r>0.5, p<Bonferroni, CI excludes 0) / QUALIFIED (r>0.2) / NO"
key-decisions:
  - "Bonferroni threshold = 0.05/72 = 0.00069 for 72-matrix multiple comparison correction (per D-01)"
  - "Bootstrap 95% CI via 10,000 resamples of (kappa, dy) pairs, percentile method (per D-02)"
  - "Primary analysis includes only 'proj' matrices (q/k/v/o/gate/up/down), excludes embed_tokens (nn.Embedding, no Linear hook)"
  - "Pure-numpy Pearson fallback with scipy.stats.t CDF when scipy is unavailable"
  - "Verdict rubric from D-06: YES/QUALIFIED/NO with explicit criterion reporting"
metrics:
  duration: 0m
  completed_date: 2026-05-17
  lines_of_code: 641
  functions_defined: 9
  classes_defined: 0
---

# Phase 3 Plan 1: Theorem 1 Multi-Seed Validation Summary

**One-liner:** Build `src/experiments/validate_theorem1.py` -- a multi-seed (42, 123, 456) per-matrix measurement pipeline that computes kappa(W) via exact SVD, ||dW||/||W|| via FP4 E2M1 round-to-nearest, and ||dy||/||y|| via ErrorPropagationTracker across 3 seeds, then computes Pearson r(kappa, mean ||dy||/||y||) with Bonferroni correction (alpha=0.05/72), bootstrap 95% CI (10,000 resamples), seed-by-seed r values, per-layer-type subgroup correlations, and outputs a definitive YES/QUALIFIED/NO verdict with a formatted 72-matrix results table and JSON export.

## Tasks Completed

### Task 1: Build multi-seed measurement pipeline

Created the core measurement script `src/experiments/validate_theorem1.py` with:

- **CLI (argparse):** `--checkpoint` (required), `--data_dir`, `--output`, `--device`, `--n_resamples`, `--batch_size`, `--max_seq_len`
- **Model loading:** MicroGemmaFPConfig defaults, MicroGemmaFPForCausalLM, load_checkpoint
- **Seed-independent pre-loop:**
  - Kappa via `compute_all_condition_numbers()` (exact SVD), keys normalized by stripping `.weight` suffix
  - ||dW||/||W|| via `FPQuantizer(fmt='fp4_e2m1', per_channel=True)` iterating `model.get_quantizable_weights()`, same key normalization
- **3-seed loop (42, 123, 456):**
  - `torch.manual_seed(seed)` before DataLoader creation
  - Explicit `DataLoader(MultiTierDataset(d, l, split='val'), shuffle=True, collate_fn=collate_batch)` -- avoids `get_dataloader()` shuffle=False-for-val bug
  - Fresh `ErrorPropagationTracker()` per seed: attach, forward pass, detach, compute_p3_p6, compute_output_error
- **Per-matrix aggregation:** mean and std of ||dy||/||y|| across seeds, tightness_ratio = mean_dy / max(kappa * dw_norm, 1e-12)
- **Matrix type classification:** layer index from `layers.N` subpath, type from last path segment (attention/ffn/global)
- **Filtering:** Primary 72-matrix analysis includes only `proj` modules (excludes `embed_tokens` which is nn.Embedding with no Linear hook)

### Task 2: Add statistical analysis, verdict, and output formatting

Extended the script with:

- **Pearson correlation:** scipy.stats.pearsonr with pure-numpy fallback using `np.corrcoef` + t-statistic p-value via scipy.stats.t.cdf
- **Bonferroni threshold:** alpha = 0.05 / 72 = 0.000694... (constant used for significance check)
- **Bootstrap CI:** `bootstrap_pearson_ci()` function with n_resamples parameter, resamples (kappa, dy) pairs with replacement, percentile CI [2.5th, 97.5th]
- **Seed-by-seed r values:** separate Pearson r for each seed's ||dy||/||y|| against kappa
- **Subgroup analysis:** attention-only and FFN-only correlations (informational, no Bonferroni)
- **Verdict rubric (YES/QUALIFIED/NO):**
  - YES: r > 0.5 AND p < 0.00069 AND CI excludes 0
  - QUALIFIED: r > 0.2 but not all YES criteria met (reports which criterion failed)
  - NO: r <= 0.2 OR uncorrected p > 0.05
- **72-row results table:** 7 columns (name, layer, type, kappa, ||dW||/||W||, mean ||dy||/||y||, tightness_ratio), sorted by layer then type, NaN/Inf rendered as "N/A"
- **Statistical summary block:** matrix count, r, p (scientific), Bonferroni threshold, bootstrap CI, seed-by-seed r, subgroup correlations, verdict with reason
- **JSON export:** `os.makedirs()` before write, json.dump with indent=2, all required keys (checkpoint, num_matrices, bonferroni_alpha, pearson_r, pearson_p, bootstrap_ci, seed_by_seed_r, subgroup_correlations, verdict, verdict_reason, results[])
- **Edge cases:** NaN handling in table (isnan/isinf check), matrix count mismatch warning, denominator clamp (max 1e-12) in tightness_ratio, sys.exit(0) on completion

## Deviations from Plan

None -- plan executed exactly as written.

## Known Stubs

No stubs found.

## Threat Flags

No threat flags. The script is a pure analysis tool: reads local checkpoints and bin files, performs statistical computation in-process, writes JSON to local disk. No network, no authentication, no untrusted input beyond CLI args (which use argparse built-in type checking).

## Self-Check: PASSED

All verification checks passed:
- `python -m py_compile src/experiments/validate_theorem1.py` exits 0
- File is 641 lines (min 200 requirement met)
- All 8 required imports present (ErrorPropagationTracker, compute_all_condition_numbers, FPQuantizer, load_checkpoint, MultiTierDataset, collate_batch, MicroGemmaFPConfig, MicroGemmaFPForCausalLM)
- `main()` and `parse_args()` exported
- `.replace('.weight', '')` key normalization present (2 occurrences)
- `torch.manual_seed()` in seed loop before DataLoader creation
- `FPQuantizer(fmt='fp4_e2m1', per_channel=True)` used
- `MultiTierDataset(args.data_dir, args.max_seq_len, split='val')` with explicit `DataLoader(shuffle=True, collate_fn=collate_batch)` construction
- Bonferroni threshold `0.05 / 72` present
- Verdict strings "YES", "QUALIFIED", "NO" present in verdict computation
- `json.dump` with `indent=2` for JSON export
- `os.makedirs` before JSON write
- Commit 499bbc1 exists with `feat(03-01):` prefix
