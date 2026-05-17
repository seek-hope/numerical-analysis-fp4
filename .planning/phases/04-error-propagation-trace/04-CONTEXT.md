# Phase 4: Error Propagation Trace - Context

**Gathered:** 2026-05-17
**Status:** Ready for planning

## Phase Boundary

Map the full journey of quantization error through the Transformer layer pipeline to identify where error amplifies and where it attenuates. Trace per-source-matrix quantization error through all 6 per-layer measurement points (P0→P1→P2→P4→P5→P6) for representative layers 0 (early), 5 (mid), and 11 (late). Measure RMSNorm attenuation ratios across all 12 layers, decomposing the error into parallel (along signal direction) and orthogonal components. Export structured error waterfall data for downstream visualization and final reporting in Phase 5.

## Implementation Decisions

### Quantization Scope and Measurement Strategy (TRACE-01)

- **D-01:** Per-matrix quantization for source attribution. Quantize one weight matrix at a time for layers 0, 5, and 11 (7 matrices × 3 layers = 21 sources), re-running forward pass for each to trace that source's error footprint through P-points. This isolates error attribution — critical for understanding which matrices dominate error propagation.
- **D-02:** All other matrices not being traced stay at FP16. Only the single target matrix is quantized per forward pass. This prevents cascading confound from simultaneous quantization (consistent with the single-source isolation principle established in Phase 2).
- **D-03:** Two-pass methodology per source matrix: (1) one FP16 reference forward pass stores clean P-point states for all layers, (2) for each source matrix, a quantized forward pass stores perturbed P-point states. Error at each P-point = ||p_q - p_fp16|| / ||p_fp16||. FP16 reference pass is run once and reused across all 21 quantized passes.

### P-Point Error Computation (TRACE-01)

- **D-04:** For each source matrix in layers 0/5/11, measure ||d||/||y|| at all 6 P-points of that source's own layer only (not cross-layer — a layer N's weight quantization primarily affects layer N's P-points, and cross-layer propagation is already captured by the fact that layer N+1's input IS layer N's output).
- **D-05:** Error waterfall data: for each source matrix in each traced layer, produce the sequence [P0_err, P1_err, P2_err, P3_err, P4_err, P5_err, P6_err] showing how error magnitude changes at each pipeline stage. P3 and P6 are computed from residuals (P3=P0+P2, P6=P3+P5), using the already-implemented compute_p3_p6().

### RMSNorm Attenuation Measurement (TRACE-02)

- **D-06:** Measure ||d_post||/||d_pre|| for both input_norm and post_attn_norm across ALL 12 layers (not just layers 0/5/11). Data is already captured by P-point hooks — input_norm sits between P0 and P1, post_attn_norm sits between P3 and P4.
- **D-07:** RMSNorm input error d_pre = y_pre_q - y_pre_fp16, RMSNorm output error d_post = y_post_q - y_post_fp16. Attenuation ratio = ||d_post||/||d_pre||. Ratio < 1 means RMSNorm attenuates error; ratio > 1 means it amplifies.

### RMSNorm Error Decomposition (TRACE-03)

- **D-08:** Vector projection method for parallel/orthogonal decomposition. For each RMSNorm with clean output y and error vector d:
  - parallel_component = |<d, y>| / ||y|| (error along signal direction, normalized)
  - orthogonal_component = ||d - (d·y/||y||²)y|| / ||y|| (error perpendicular to signal)
  - Verify: total² = parallel² + orthogonal² (Pythagorean identity — validates decomposition correctness)
- **D-09:** Both components reported for input_norm and post_attn_norm across all 12 layers. This tests the hypothesis that RMSNorm attenuates orthogonal error more than parallel error (since RMSNorm normalizes by RMS of the signal, which is dominated by the parallel component).

### Quantization Configuration

- **D-10:** Use FP4 E2M1 round-to-nearest per-channel quantization (standard FPQuantizer(fmt='fp4_e2m1', per_channel=True)) for all trace measurements. This is the canonical quantizer established in Phase 2/Phase 3 — consistency enables cross-phase comparison.
- **D-11:** No GPTQ compensation for propagation tracing — GPTQ's column compensation violates per-matrix independence (same principle as Phase 3's D-07).

### Script Design

- **D-12:** Single experiment script `src/experiments/trace_error_propagation.py` that:
  1. Loads the FP16 baseline checkpoint
  2. Runs the FP16 reference pass with tracker attached, saves all P-points and G-points
  3. For each source matrix in layers 0/5/11: quantizes that one matrix, runs forward pass with tracker, computes P-point errors relative to FP16 reference
  4. Computes RMSNorm attenuation and decomposition across all 12 layers
  5. Prints per-source error waterfall tables
  6. Exports all data as `results/error_propagation_trace.json`
- **D-13:** argparse interface consistent with Phase 3's validate_theorem1.py: --checkpoint, --data_dir, --output, --device, --batch_size, --max_seq_len. Uses validation data split (split='val').

### Data Output Format

- **D-14:** JSON structure with three top-level sections:
  ```json
  {
    "checkpoint": "path/to/checkpoint.pt",
    "num_selected_layers": 3,
    "selected_layers": [0, 5, 11],
    "trace": {
      "layer_0": [
        {
          "source_matrix": "model.layers.0.attention.q_proj",
          "matrix_type": "attention",
          "p_points": {"P0": 0.0123, "P1": 0.0089, ...},
          "waterfall": [0.0123, 0.0089, ...]
        }
      ]
    },
    "rmsnorm_attenuation": {
      "layer_0": {
        "input_norm": {"d_pre_norm": ..., "d_post_norm": ..., "ratio": ...},
        "post_attn_norm": {"d_pre_norm": ..., "d_post_norm": ..., "ratio": ...}
      }
    },
    "rmsnorm_decomposition": {
      "layer_0": {
        "input_norm": {"parallel": ..., "orthogonal": ..., "total": ...},
        "post_attn_norm": {"parallel": ..., "orthogonal": ..., "total": ...}
      }
    }
  }
  ```

### Claude's Discretion

- Whether to extend full per-matrix trace to all 12 layers (not just 0/5/11) — the infrastructure supports it since P-points are captured for all layers regardless
- Exact print table format for waterfall data (column widths, decimal places, sorting)
- Whether to compute cross-layer error propagation (e.g., does layer N's q_proj error show up at layer N+1's P0?) — interesting but potentially adds noise
- Single-seed execution (seed=42) for efficiency; multi-seed belongs to Phase 5 extended comparison if needed
- Whether to include a "no-quantization" null control trace alongside quantized traces

## Canonical References

### Measurement Infrastructure (Phase 2)
- `src/analysis/error_propagation.py` — ErrorPropagationTracker with P-point hooks (P0-P6), G-point hooks (G0-G2), compute_p3_p6(), compute_output_error()
- `src/analysis/condition.py:55-65` — compute_all_condition_numbers() for per-matrix kappa (context only — not directly used in trace, but referenced for interpreting which matrices are expected to produce more error)

### Theorem 1 Validation (Phase 3)
- `src/experiments/validate_theorem1.py` — Reference for model loading, tracker attachment, per-matrix quantization, results table printing, JSON export patterns
- `.planning/phases/03-theorem-1-validation/03-CONTEXT.md` — Phase 3 decisions (per-matrix granularity, Bonferroni correction, multi-seed methodology)

### Error Propagation Analysis
- `src/analysis/lipschitz.py` — Existing Lipschitz propagation analysis (may provide baseline predictions for error amplification through FFN/attention)
- `src/analysis/sensitivity.py` — Per-layer sensitivity reports (context for expected RMSNorm attenuation patterns)

### Model Architecture
- `src/model/transformer.py:172-194` — TransformerLayer.forward() — defines exact P-point placement in forward pass
- `src/model/transformer.py:264-270` — get_quantizable_weights() — canonical weight naming convention
- `src/model/config.py:15-87` — MicroGemmaFPConfig — architecture dimensions

### Quantization
- `src/quantization/fp_quantizer.py:53-80` — FPQuantizer for FP4 round-to-nearest

### Data
- `src/experiments/training_utils.py:197-219` — get_dataloader(split='val') for validation data
- `src/experiments/training_utils.py:351-356` — load_checkpoint()

### Requirements
- `.planning/REQUIREMENTS.md` §TRACE-01, TRACE-02, TRACE-03 — Full requirement text
- `.planning/ROADMAP.md` §"Phase 4: Error Propagation Trace" — Success criteria (4 items)

### Project Foundation
- `.planning/PROJECT.md` — Core value, constraints, key decisions
- `.planning/STATE.md` — Accumulated context (single-pass protocol, Bonferroni correction)
- `.planning/phases/02-core-measurement-protocol/02-CONTEXT.md` — Phase 2 decisions (hook architecture, P-point definitions, activation storage)

## Existing Code Insights

### Reusable Assets
- `ErrorPropagationTracker` (error_propagation.py): Already registers P0-P6 hooks on all 12 layers, captures fp16 P-point states, computes P3/P6. The FP16 reference pass is a direct use of existing attach/forward/detach flow. No modification needed.
- `compute_p3_p6()` (error_propagation.py:170-200): Computes residual-add outputs from captured P-points — directly reusable.
- `_classify_matrix()` (validate_theorem1.py:107-132): Parses module path into (layer_index, matrix_type) — reusable for grouping trace results.
- `FPQuantizer(fmt='fp4_e2m1', per_channel=True)`: The standard quantizer for per-matrix quantization in the forward pass.
- `get_quantizable_weights()` (transformer.py): Lists all 72 Linear weight matrices — used to identify which matrices to individually quantize.
- `load_checkpoint()` (training_utils.py): Model loading from checkpoint.
- `_make_input_hook` / `_make_p_hook` / `_make_g_hook` (error_propagation.py): Factory functions for hook creation — reusable pattern for trace-specific hooks if needed.

### Established Patterns
- argparse + main() pattern for experiment scripts
- `with torch.no_grad()` for all evaluation/measurement passes
- Results returned as dict[str, float] keyed by module path
- JSON export with nested structure for complex results
- print() for logging, section separators for visual grouping
- Device auto-detection: `torch.device('cuda' if torch.cuda.is_available() else 'cpu')`

### Integration Points
- **ErrorPropagationTracker**: Primary measurement tool. Phase 4 extends its usage from per-matrix output error (Phase 3) to per-source P-point error waterfall tracing. No tracker code changes expected — all new logic is in the trace script.
- **validate_theorem1.py patterns**: Model loading, dataloader construction, checkpoint loading — copy the proven patterns.
- **Phase 5 (Extended PTQ Comparison)**: Will consume the RMSNorm attenuation data and error waterfall data from this phase's JSON output for the final comparison report.
- **Transformer model**: Hook target (unchanged — all instrumentation is external).

## Specific Ideas

Measurement points are fully defined by Phase 2's ErrorPropagationTracker:
- P0: hidden_states before input_norm (residual input to layer)
- P1: hidden_states after input_norm, before attention
- P2: attention output, before residual add
- P3: post-attention residual (P0 + P2, computed)
- P4: hidden_states after post_attn_norm, before FFN
- P5: FFN output, before residual add
- P6: post-FFN residual (P3 + P5, computed)

The 7 quantizable matrices per layer for per-source tracing:
- Attention: q_proj [768×768], k_proj [256×768], v_proj [256×768], o_proj [768×256]
- FFN: gate_proj [3072×768], up_proj [3072×768], down_proj [768×3072]

RMSNorm attenuation sits at two points per layer:
- Between P0→P1: layer.input_norm (first RMSNorm)
- Between P3→P4: layer.post_attn_norm (second RMSNorm)

The key numerical analysis question this phase answers: Do large-kappa matrices (like o_proj with kappa~16000) produce larger errors that survive RMSNorm attenuation, or does RMSNorm effectively erase all error regardless of source?

## Deferred Ideas

None — all auto-selected decisions align with prior phase decisions and roadmap requirements.

---

*Phase: 4-Error Propagation Trace*
*Context gathered: 2026-05-17*
*Mode: --auto (all gray areas auto-selected, recommended options chosen)*
