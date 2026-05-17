---
phase: 05-extended-ptq-comparison-and-final-report
verified: 2026-05-17T16:30:00Z
status: gaps_found
score: 0/8 must-haves verified
overrides_applied: 0
gaps:
  - truth: "16+ PTQ configs evaluated with both PPL and per-matrix ||dy||/||y|| on clean data split"
    status: failed
    reason: "run_full_comparison.py exists and compiles but has NEVER been executed. No results/ directory exists. results/full_comparison.json and results/per_matrix_summary.json are both missing. The required checkpoints (checkpoints/scaled_fp16_baseline/model.pt, checkpoints/cond_regularized/model.pt) also do not exist, so the script cannot run even if GPU access were available."
    artifacts:
      - path: "src/experiments/run_full_comparison.py"
        issue: "Script exists (867 lines, compiles) but has never been run. No output files generated."
      - path: "results/"
        issue: "Directory does not exist."
      - path: "results/full_comparison.json"
        issue: "Expected output file does not exist."
      - path: "checkpoints/scaled_fp16_baseline/model.pt"
        issue: "Required checkpoint does not exist."
      - path: "checkpoints/cond_regularized/model.pt"
        issue: "Required checkpoint does not exist."
    missing:
      - "Execute run_full_comparison.py on remote GPU server to produce results/full_comparison.json"
      - "Required checkpoints must first be trained or provided"
  - truth: "GPTQ vs RTN comparison reports per-matrix ||dy||/||y|| delta with direction (+ means GPTQ worse)"
    status: failed
    reason: "Comparison logic exists in run_full_comparison.py (lines 544-603) but has never executed. Requires full_comparison.json which does not exist."
    artifacts:
      - path: "results/full_comparison.json"
        issue: "Missing — comparison output depends on this file"
    missing:
      - "Run run_full_comparison.py to generate the comparison data"
  - truth: "Lloyd-Max vs uniform E2M1 comparison reports per-matrix ||dy||/||y|| delta"
    status: failed
    reason: "Comparison logic exists in run_full_comparison.py (lines 607-660) but has never executed."
    artifacts:
      - path: "results/full_comparison.json"
        issue: "Missing — comparison output depends on this file"
    missing:
      - "Run run_full_comparison.py to generate the comparison data"
  - truth: "Per-matrix summary table with 9 columns generated for all 72 matrices"
    status: failed
    reason: "Logic exists in run_full_comparison.py (lines 665-758, 856-859) but has never executed. results/per_matrix_summary.json does not exist. Additionally, this requires theorem1_validation.json and error_propagation_trace.json from Phases 3 and 4, which also do not exist."
    artifacts:
      - path: "results/per_matrix_summary.json"
        issue: "Expected output file does not exist"
      - path: "results/theorem1_validation.json"
        issue: "Phase 3 data not present — required input"
      - path: "results/error_propagation_trace.json"
        issue: "Phase 4 data not present — required input"
    missing:
      - "Run validate_theorem1.py (Phase 3) to produce theorem1_validation.json"
      - "Run trace_error_propagation.py (Phase 4) to produce error_propagation_trace.json"
      - "Run run_full_comparison.py to produce per_matrix_summary.json"
  - truth: "REPORT.md contains per-matrix summary table with all 9 columns for 72 matrices"
    status: failed
    reason: "docs/REPORT.md exists (7.7KB, last modified 2026-05-08) but is a pre-phase-5 Chinese-language legacy report. It contains 0 occurrences of 'per-matrix', 'summary table', or any required Phase 5 section headings."
    artifacts:
      - path: "docs/REPORT.md"
        issue: "Exists but is the wrong document — pre-phase-5 Chinese-language report, not the output of write_final_report.py"
    missing:
      - "Run write_final_report.py after all JSON data is available to regenerate REPORT.md"
  - truth: "REPORT.md contains error propagation waterfall data sourced from Phase 4 trace"
    status: failed
    reason: "docs/REPORT.md does not contain any error propagation waterfall data."
    missing:
      - "Run write_final_report.py with all JSON inputs present"
  - truth: "REPORT.md updated with corrected kappa values, per-matrix measurements, and revised theoretical assessment"
    status: failed
    reason: "docs/REPORT.md has not been updated. It is the pre-phase-5 version last committed on 2026-05-08."
    missing:
      - "Run write_final_report.py to regenerate REPORT.md with corrected values"
  - truth: "REPORT.md follows the 10-section structure defined in CONTEXT.md"
    status: failed
    reason: "docs/REPORT.md has a completely different structure (Chinese-language sections). None of the 10 required English section headings (Executive Summary, Methodology, Theorem 1 Validation Results, Error Propagation Trace, Extended PTQ Comparison, GPTQ Analysis, Lloyd-Max Analysis, RMSNorm Error Blocking, Revised Theoretical Assessment, References) are present in the file."
    missing:
      - "Rewrite REPORT.md to follow the 10-section structure"
      - "Run write_final_report.py to generate the correctly structured report"
---

# Phase 5: Extended PTQ Comparison and Final Report - Verification Report

**Phase Goal:** Complete comparison of all PTQ methods under clean conditions with both PPL and per-matrix output error, culminating in the final project report
**Verified:** 2026-05-17T16:30:00Z
**Status:** gaps_found
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | 16+ PTQ configs evaluated with both PPL and per-matrix ||dy||/||y|| on clean data split | FAILED | Script exists and compiles but never executed. No results/ directory. No output JSON files. No checkpoints. |
| 2 | GPTQ vs RTN comparison reports per-matrix ||dy||/||y|| delta with direction | FAILED | Comparison logic in code but never executed. Requires full_comparison.json which does not exist. |
| 3 | Lloyd-Max vs uniform E2M1 comparison reports per-matrix ||dy||/||y|| delta | FAILED | Comparison logic in code but never executed. |
| 4 | Per-matrix summary table with 9 columns generated for all 72 matrices | FAILED | per_matrix_summary.json does not exist. Phase 3 and Phase 4 input JSONs also missing. |
| 5 | REPORT.md contains per-matrix summary table with all 9 columns for 72 matrices | FAILED | REPORT.md is pre-phase-5 Chinese-language document, not the output of write_final_report.py |
| 6 | REPORT.md contains error propagation waterfall data sourced from Phase 4 trace | FAILED | REPORT.md has no waterfall data |
| 7 | REPORT.md updated with corrected kappa values, per-matrix measurements, and revised theoretical assessment | FAILED | REPORT.md is stale (last commit 2026-05-08) |
| 8 | REPORT.md follows the 10-section structure defined in CONTEXT.md | FAILED | REPORT.md has a completely different structure |

**Score:** 0/8 truths verified

### Root Cause

Both scripts (`run_full_comparison.py` and `write_final_report.py`) were written and compile correctly, but the computational pipeline was never executed:

1. `run_full_comparison.py` requires PyTorch, GPU, and two pre-trained checkpoints. None of these prerequisites are met in the local environment. The script has never been run on the remote GPU server.
2. `write_final_report.py` requires `full_comparison.json` (output of step 1), which does not exist.
3. `docs/REPORT.md` is a legacy document from before Phase 5 was planned. It has not been updated.

The pattern is: **code infrastructure complete, data pipeline never executed.**

### Required Artifacts

| Artifact | Expected | Status | Details |
| -------- | -------- | ------ | ------- |
| `src/experiments/run_full_comparison.py` | Full PTQ comparison orchestration script, >=250 lines | VERIFIED | 867 lines, compiles, all 10 required functions present |
| `results/full_comparison.json` | Structured results for all configs | MISSING | File does not exist |
| `results/per_matrix_summary.json` | 72-row per-matrix table | MISSING | File does not exist |
| `src/experiments/write_final_report.py` | Automated report generator, >=150 lines | VERIFIED | 1696 lines, compiles, all 13 required functions present |
| `docs/REPORT.md` | Final project report with 10 sections | STUB | File exists but is pre-phase-5 legacy document with wrong content |

### Key Link Verification

| From | To | Via | Status | Details |
| ---- | --- | --- | ------ | ------- |
| run_full_comparison.py | ErrorPropagationTracker | import + attach/detach | WIRED | Line 48 import, line 256 attach/259 detach |
| run_full_comparison.py | get_dataloader(split='train') | calibration dataloader | WIRED | Line 157, GPTQ calibration uses split='train' per D-05 |
| run_full_comparison.py | get_dataloader(split='val') | eval dataloader | WIRED | Line 327, PPL evaluation uses split='val' per D-03 |
| run_full_comparison.py | All 6 quantizer classes | method dispatch | WIRED | Lines 42-47 imports, lines 215-222 METHOD_DISPATCH dict |
| write_final_report.py | theorem1_validation.json | json.load | WIRED | Pattern at line 1622 |
| write_final_report.py | error_propagation_trace.json | json.load | WIRED | Pattern at line 1623 |
| write_final_report.py | full_comparison.json | json.load | WIRED | Pattern at line 1624 |
| write_final_report.py | docs/REPORT.md | file write | WIRED | Pattern at line 1593, write at line 1686 |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
| -------- | ------------- | ------ | ------------------ | ------ |
| run_full_comparison.py | all_results | Method dispatch + PPL eval + error computation | No — never executed | DISCONNECTED |
| run_full_comparison.py | comparisons | GPTQ/RTN and Lloyd-Max/uniform delta computation | No — never executed | DISCONNECTED |
| run_full_comparison.py | per_matrix_summary | Merged Phase 3/4/5 data | No — never executed | DISCONNECTED |
| write_final_report.py | th1_data, trace_data, comp_data | json.load on result files | No — input files missing | DISCONNECTED |
| docs/REPORT.md | N/A | write_final_report.py main() | No — wrong document, not script output | DISCONNECTED |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
| -------- | ------- | ------ | ------ |
| run_full_comparison.py compiles | python -m py_compile | No errors | PASS |
| write_final_report.py compiles | python -m py_compile | No errors | PASS |
| run_full_comparison.py produces output | ./remote_python.sh ... --data_dir data/real_tiers | SKIP — no GPU, no checkpoints | SKIP |
| write_final_report.py generates REPORT.md | python src/experiments/... --results_dir results | SKIP — results/ does not exist | SKIP |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
| ----------- | ----------- | ----------- | ------ | -------- |
| COMP-01 | 05-01 | Re-run 24-config PTQ comparison with clean data split, reporting both PPL and per-matrix ||dy||/||y|| | FAILED | run_full_comparison.py exists but never executed on GPU |
| COMP-02 | 05-01 | GPTQ vs RTN: compare ||dy||/||y|| — quantify whether column compensation reduces output error | FAILED | Comparison logic exists in code but never executed |
| COMP-03 | 05-01 | Lloyd-Max vs uniform: compare per-matrix ||dy||/||y|| — test if distribution-adaptive grids reduce error | FAILED | Comparison logic exists in code but never executed |
| REPORT-01 | 05-01, 05-02 | Generate per-matrix error summary table with all required columns | FAILED | per_matrix_summary.json does not exist |
| REPORT-02 | 05-02 | Generate error propagation waterfall data for visualization | FAILED | REPORT.md does not contain this data |
| REPORT-03 | 05-02 | Update REPORT.md with corrected kappa values, per-matrix measurements, revised theoretical assessment | FAILED | REPORT.md is pre-phase-5 legacy document |

### Anti-Patterns Found

No TBD, FIXME, or XXX markers found in either script file. Both scripts are clean of debt markers.

No stub patterns detected — both scripts contain full, substantive implementations.

### Human Verification Required

1. **REPORT.md content verification** — After running write_final_report.py, human should verify that all 10 sections are present and that numerical values match JSON data.

2. **GPU execution of run_full_comparison.py** — Requires remote execution on the 8x RTX 4090 server and verification that all 16-20 configs complete without errors.

3. **Checkpoint availability** — Requires FP16 baseline and cond_regularized checkpoints. If these don't exist, they must be trained first.

### Gaps Summary

Phase 5 has a fundamental execution gap: the computational pipeline (run_full_comparison.py -> full_comparison.json -> write_final_report.py -> REPORT.md) was never executed. Both scripts are correctly implemented and compile, but they require:

1. Two pre-trained checkpoints (`checkpoints/scaled_fp16_baseline/model.pt` and `checkpoints/cond_regularized/model.pt`) that do not exist locally.
2. GPU server access (8x RTX 4090) to run `run_full_comparison.py`.
3. Phase 3 and Phase 4 JSON outputs (`theorem1_validation.json`, `error_propagation_trace.json`) that also do not exist locally.
4. After steps 1-3 produce the JSON outputs, `write_final_report.py` can regenerate `docs/REPORT.md`.

The existing `docs/REPORT.md` is a pre-phase-5 legacy document from 2026-05-08 with a different structure and in Chinese. It must be regenerated by `write_final_report.py` after all prerequisite data is available.

**To close all gaps:** Execute on remote GPU server:
```bash
# Step 1: Ensure checkpoints exist (or train them)
# Step 2: Run Phase 3 (validate_theorem1.py) -> theorem1_validation.json
# Step 3: Run Phase 4 (trace_error_propagation.py) -> error_propagation_trace.json
# Step 4: Run Phase 5 comparison
./remote_python.sh src/experiments/run_full_comparison.py --data_dir data/real_tiers
# Step 5: Generate final report
python src/experiments/write_final_report.py
```

---

*Verified: 2026-05-17T16:30:00Z*
*Verifier: Claude (gsd-verifier)*
