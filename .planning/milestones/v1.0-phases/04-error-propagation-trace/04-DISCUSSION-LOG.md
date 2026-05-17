# Phase 4: Error Propagation Trace - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-17
**Phase:** 04-error-propagation-trace
**Areas discussed:** Quantization scope, P-point error computation, RMSNorm decomposition, Script design

---

## Quantization Scope for Error Propagation Trace

| Option | Description | Selected |
|--------|-------------|----------|
| Per-matrix quantization (one at a time) | Quantize each of 7 weight matrices individually in layers 0/5/11, re-run forward pass per source. 21 quantized passes. Isolates source attribution. | ✓ |
| Per-layer quantization | Quantize all 7 matrices in a layer simultaneously. 3 quantized passes. Shows per-layer aggregate but can't attribute to specific matrices. | |
| All-at-once quantization | Quantize all 72 matrices. 1 quantized pass. Shows real-world cascade but no source attribution. | |

**Auto-selected:** Per-matrix quantization (recommended) — TRACE-01's requirement phrasing "For each quantized weight matrix" implies per-source attribution. This matches Phase 3's per-matrix granularity philosophy.

---

## P-Point Error Computation Method

| Option | Description | Selected |
|--------|-------------|----------|
| Two-pass (FP16 ref + quantized per source) | Store FP16 reference P-points once, re-run quantized forward per source matrix, compute difference. 1 + 21 = 22 total forward passes. | ✓ |
| Stored activations from Phase 3 | Reuse Phase 3's stored P-point activations. Would need to re-run anyway since Phase 3 only captured clean activations. | |

**Auto-selected:** Two-pass approach — most direct and matches the ||dy||/||y|| formalism from all prior phases. FP16 reference pass runs once and is reused.

---

## RMSNorm Error Decomposition

| Option | Description | Selected |
|--------|-------------|----------|
| Vector projection | Project error vector d onto signal direction y: parallel = |<d,y>|/||y||, orthogonal = ||d - (d·y/||y||²)y||/||y|| | ✓ |
| SVD-based decomposition | Align with principal components of the signal. More expensive, no clear benefit for this analysis. | |

**Auto-selected:** Vector projection — simpler, faster, directly answers "does RMSNorm attenuate or redirect error?" Pythagorean identity provides built-in validation.

---

## Script Design

| Option | Description | Selected |
|--------|-------------|----------|
| Single comprehensive script | `trace_error_propagation.py` — TRACE-01, TRACE-02, TRACE-03 in one script with modular functions | ✓ |
| Separate scripts per subtask | `trace_waterfall.py`, `measure_rmsnorm.py`, `decompose_error.py` — more modular but more files | |

**Auto-selected:** Single script — follows Phase 3's validate_theorem1.py pattern of one comprehensive script per phase.

---

## Claude's Discretion

- Whether to extend full trace from 3 layers to all 12 (P-point data available for all)
- Print table formatting (column widths, decimal places)
- Cross-layer error propagation measurement (layer N's error visible at layer N+1's P0)
- Single-seed execution (seed=42) vs multi-seed
- Null control trace inclusion

## Deferred Ideas

None — discussion stayed within phase scope. All auto-selected decisions align with prior context.
