# Phase 2: Core Measurement Protocol - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-17
**Phase:** 2-Core Measurement Protocol
**Areas discussed:** Hook Registration Architecture, Activation Storage, Error Computation Pipeline, Kappa Computation, Module Naming Convention, Null Measurement, Evaluation Batch

---

## Hook Registration Architecture

| Option | Description | Selected |
|--------|-------------|----------|
| Pre-hooks on nn.Linear + forward hooks on submodules | `register_forward_pre_hook` on each nn.Linear for x capture; `register_forward_hook` on input_norm/attention/post_attn_norm/ffn for P-points. External to transformer.py. | ✓ |
| Single unified forward hook on TransformerLayer | One hook per layer, manually parse hidden_states at each sub-step. Requires knowledge of internal forward logic. | |
| Monkey-patch TransformerLayer.forward | Wrap forward method to insert measurement points. More invasive, risk of breaking quantization hooks. | |

**Auto-selected choice:** Pre-hooks on nn.Linear + forward hooks on submodules
**Rationale:** Cleanest separation — x-capture hooks target Linear directly (always correct regardless of how modules compose), P-point hooks target specific submodules. No modification to transformer.py satisfies the success criterion. Matches the GPTQ activation collection pattern already in the codebase.

---

## Per-Matrix Activation Shape

| Option | Description | Selected |
|--------|-------------|----------|
| Full input tensor x | Save complete `[batch*seq_len, in_features]` tensor from pre-hook. Enables exact ||dy||/||y|| computation offline. ~113 MB for 72 matrices. | ✓ |
| Pre-computed norms only | Compute ||x|| and x^T·x in hook, discard full tensor. Lighter memory but loses per-token granularity for Phase 4. | |

**Auto-selected choice:** Full input tensor x
**Rationale:** Single-pass capture saves exactly one set of activations — 113 MB on CPU is negligible. Full tensors enable exact ||(W_q - W)x|| computation without approximation. P-point hooks also need full hidden_states for Phase 4's propagation analysis.

---

## Error Computation Pipeline

| Option | Description | Selected |
|--------|-------------|----------|
| Offline compute from saved x | After forward pass: for each matrix, load W_fp, quantize to W_q, compute y_fp = x @ W_fp.T, y_q = x @ W_q.T, return ||y_q - y_fp|| / ||y_fp||. | ✓ |
| In-hook computation | Compute ||dy||/||y|| during the forward pass. Requires quantizing weights before or during the pass. Mixes measurement with forward execution. | |

**Auto-selected choice:** Offline compute from saved x
**Rationale:** Single-pass capture means the forward pass is clean FP16 — no quantization happens during it. This eliminates cascading confound (quantization error from one layer propagating to the next layer's x). The offline computation isolates each matrix's error independently.

---

## Kappa Computation

| Option | Description | Selected |
|--------|-------------|----------|
| Reuse existing `estimate_condition_number()` | Call `compute_all_condition_numbers()` from condition.py. Exact SVD already validated. No implementation needed. | ✓ |
| Build κ into ErrorPropagationTracker | Add κ computation as a tracker method. Duplicates existing code. | |

**Auto-selected choice:** Reuse existing function
**Rationale:** condition.py already has battle-tested exact SVD with `clamp(min=1e-12)` safety. `compute_all_condition_numbers()` iterates all dim>=2 params. The ErrorPropagationTracker focuses on what's new: activation capture and ||dy||/||y||. Kappa is a separate diagnostic pass.

---

## Module Naming Convention

| Option | Description | Selected |
|--------|-------------|----------|
| Reuse `get_quantizable_weights()` naming | Dot-separated parameter paths: `model.layers.0.attention.q_proj.weight`. Strip `.weight` for hook key. | ✓ |
| Custom flat naming scheme | Assign sequential IDs (W_00 through W_71). Simpler but loses structural information. | |

**Auto-selected choice:** Reuse existing convention
**Rationale:** The dot-separated path encodes layer index, block type, and projection type — essential for Phase 3's per-layer-type correlation analysis and Phase 4's propagation tracing. Already understood by all existing tooling.

---

## Null Measurement

| Option | Description | Selected |
|--------|-------------|----------|
| Identity quantization | W_q = W_fp (no-op quantizer). Verify max ||dy||/||y|| < 1e-5 across all matrices. | ✓ |
| Skip null measurement in Phase 2 | Defer to Phase 3 validation. Risk: measurement bug discovered late. | |

**Auto-selected choice:** Identity quantization
**Rationale:** Success criterion #4 explicitly requires null measurement with < 1e-5 threshold. This is a pipeline integrity check — if it fails, everything downstream is invalid. Implementing as `tracker.validate_null_measurement()` makes it a one-liner in the experiment script.

---

## Evaluation Batch

| Option | Description | Selected |
|--------|-------------|----------|
| Single batch (batch_size=1, seq_len=512) | One forward pass, 72 activation tensors, ~113 MB CPU. Single scalar per matrix. | ✓ |
| Full 100-step evaluation | 100 forward passes, 7200 activation tensors, ~11 GB. Mean ± std per matrix. | |

**Auto-selected choice:** Single batch
**Rationale:** Phase 2 builds the measurement infrastructure — a single batch proves the pipeline works and satisfies success criteria #2 ("single FP16 forward pass"). Multi-step statistical aggregation is Phase 3's responsibility (VAL-01 through VAL-04 require 3-seed, Bonferroni correction). Keep Phase 2 scope tight.

---

## Claude's Discretion

- Hook cleanup strategy (remove handles in detach() vs store and discard)
- GPU vs CPU for error computation (GPU faster for large matrices, CPU safer for memory)
- Error message format for null measurement violation reporting
- Whether to store P-point tensors per-token or as full batch tensor
- Logging verbosity during hook registration (module count, hook placement confirmation)

## Deferred Ideas

None — discussion stayed within phase scope. All decisions align with PRIOR STATE.md accumulated context (single-pass protocol, round-to-nearest for Theorem 1 test).
