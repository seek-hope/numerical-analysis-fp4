---
phase: 04-error-propagation-trace
plan: 01
subsystem: experiments
tags: [error-propagation, trace, rmsnorm, waterfall, json-export]
provides: [TRACE-01, TRACE-02, TRACE-03]
key-files:
  created:
    - src/experiments/trace_error_propagation.py
    - results/error_propagation_trace.json (runtime)
decisions:
  - Per-matrix quantization for source attribution (D-01): quantize one matrix at a time, 21 total
  - Two-pass methodology (D-03): 1 FP16 reference pass + 21 per-source quantized passes
  - Waterfall sequence (D-05): [P0, P1, P2, P3, P4, P5, P6] per source matrix
  - RMSNorm across ALL 12 layers (D-06): both input_norm and post_attn_norm
  - Vector projection decomposition (D-08): parallel = |<d,y>|/||y||, orthogonal = ||d-proj||/||y||
  - FP4 E2M1 per-channel round-to-nearest (D-10): standard quantizer from Phase 2/3
  - JSON export with three sections (D-14): trace, rmsnorm_attenuation, rmsnorm_decomposition
duration: "~20 minutes"
completed: "2026-05-17T07:15:00Z"
---

# Phase 04 Error Propagation Trace Plan 01 Summary

One-liner: Create trace_error_propagation.py -- a 630-line experiment script that quantizes 21 individual FP4 weight matrices across layers 0/5/11, runs per-source error waterfall measurement through 7 P-points, computes RMSNorm attenuation and parallel/orthogonal decomposition across all 12 layers, and exports structured JSON to `results/error_propagation_trace.json`.

## Tasks

| Task | Name                                          | Commit   |
| ---- | --------------------------------------------- | -------- |
| 1    | Script skeleton, FP16 pass, per-source loop   | fd5043c  |
| 2    | RMSNorm metrics, waterfall tables, JSON export | (included in Task 1 commit) |

## Key Files Created

| File | Description |
|------|-------------|
| `src/experiments/trace_error_propagation.py` | Complete error propagation trace experiment script (630 lines) |
| `results/error_propagation_trace.json` | Structured trace data generated at runtime |

## Deviations from Plan

None -- plan executed exactly as written.

## Key Implementation Details

### Script Structure
- Follows `validate_theorem1.py` pattern: shebang, docstring, imports, constants, helper functions, `parse_args()`, `main()`, `if __name__ == "__main__": main()`
- 6 argparse arguments: --checkpoint (required), --data_dir, --output, --device, --batch_size, --max_seq_len

### Measurement Protocol
- **FP16 reference pass**: Single forward pass with `ErrorPropagationTracker` captures clean P-points for all 12 layers
- **Per-source loop**: For each of 21 source matrices (7 per layer 0/5/11): save original weight -> quantize single matrix in FP4 E2M1 per-channel -> fresh tracker -> forward pass -> compute P-point errors relative to FP16 reference -> restore weight
- **try/finally**: Weight restoration guaranteed even on exceptions (Pitfall 1 mitigation)
- **Single batch**: Same `input_ids` reused for all 22 forward passes (Pitfall 5 mitigation)

### TRACE-01: P-Point Error Waterfall
- P-point errors: `||q - ref|| / ||ref||` for source's own layer only (D-04)
- Waterfall sequence: [P0_err, P1_err, P2_err, P3_err, P4_err, P5_err, P6_err] (D-05)
- P0 expected ~0 for source layer (pre-hook fires before weight is used)

### TRACE-02: RMSNorm Attenuation
- Both `input_norm` (P0->P1) and `post_attn_norm` (P3->P4) across all 12 layers (D-06)
- Ratio = `||d_post|| / ||d_pre||`; NaN when pre-error < 1e-8 (D-07)
- Reported per-source-matrix for all 12 layers

### TRACE-03: RMSNorm Error Decomposition
- Vector projection: `parallel = |<d,y>|/||y||`, `orthogonal = ||d - proj||/||y||` (D-08)
- Pythagorean identity verification: `|total^2 - (parallel^2 + orthogonal^2)| < 1e-6`

### Pitfalls Addressed
1. Weight restoration failure -> try/finally blocks
2. Tracker state accumulation -> fresh ErrorPropagationTracker per quantized pass
3. P0 ~zero for source layer -> documented expectation, flagged in output
4. P3/P6 computation timing -> compute_p3_p6() called after detach() on quantized tracker
5. Sequence length mismatch -> same batch reused for all 22 passes
6. Projection norm invariance -> dimensionless ratios verified with Pythagorean identity

## Verification

```bash
python -m py_compile src/experiments/trace_error_propagation.py
```

All 14 Task 1 behavior tests pass:
- py_compile exits 0
- TRACED_LAYERS = [0, 5, 11]
- FPQuantizer(fmt='fp4_e2m1', per_channel=True)
- ErrorPropagationTracker imported and used (3 occurrences)
- get_quantizable_weights called (1 occurrence)
- _classify_matrix defined and used (2 occurrences)
- try/finally blocks with original_weight restoration (2/1/2)
- compute_p3_p6 called (2 occurrences)
- tracker._p_points accessed (4 occurrences)
- waterfall generated (8 occurrences)
- split='val' used (1 occurrence)

All 15 Task 2 verification tests pass:
- py_compile exits 0
- d_pre_norm, d_post_norm, parallel, orthogonal, pythagorean all present
- json.dump and os.makedirs used
- "trace", "rmsnorm_attenuation", "rmsnorm_decomposition" keys present
- _print_waterfall_tables, _export_json, _compute_rmsnorm_metrics defined
- sys.exit(0) called at end

## Threat Surface

No new threat surface introduced. The script loads a local checkpoint, processes pre-tokenized .bin data shards, performs in-memory tensor computation, and writes results to a local JSON file. No network access, user input, or authentication.

## Known Stubs

None -- all functionality is fully implemented.

## Self-Check

```
python -m py_compile src/experiments/trace_error_propagation.py -> 0 (PASSED)
```

- [X] `src/experiments/trace_error_propagation.py` exists (630 lines, >= 400 min)
- [X] Script passes `python -m py_compile`
- [X] All 6 argparse arguments present
- [X] TRACED_LAYERS = [0, 5, 11]
- [X] FPQuantizer uses `fmt='fp4_e2m1', per_channel=True`
- [X] 21 source matrices selected from get_quantizable_weights
- [X] try/finally weight restoration with original_weight.clone()
- [X] compute_p3_p6() called on reference and per-source trackers
- [X] Waterfall sequence [P0..P6] per source
- [X] RMSNorm attenuation for input_norm and post_attn_norm, all 12 layers
- [X] RMSNorm decomposition with parallel/orthogonal + Pythagorean verification
- [X] Formatted waterfall tables printed to stdout
- [X] JSON export with three sections (trace, rmsnorm_attenuation, rmsnorm_decomposition)
- [X] os.makedirs before JSON file write
- [X] All D-01 through D-14 decisions implemented
- [X] All Pitfalls 1-6 mitigated in code
