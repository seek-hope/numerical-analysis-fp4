# Phase 2: Core Measurement Protocol - Context

**Gathered:** 2026-05-17
**Status:** Ready for planning

## Phase Boundary

Implement `ErrorPropagationTracker` in `src/analysis/error_propagation.py` — a hook-based measurement system that captures per-matrix pre-activation tensors x from a single FP16 forward pass and computes ||(W_q − W)x|| / ||Wx|| offline for all 72 Linear weight matrices. The tracker also registers measurement points P0-P6 (per-layer) and G0-G2 (global) for downstream propagation tracing in Phase 4. Does NOT modify `transformer.py` — all instrumentation is external via PyTorch forward hooks.

## Implementation Decisions

### Hook Registration Architecture
- **D-01:** Use `register_forward_pre_hook` on each `nn.Linear` module to capture pre-activation tensors x (the input to `F.linear(x, W)`). Iterate `model.named_modules()` externally to find all Linear layers without touching transformer.py.
- **D-02:** Use `register_forward_hook` on `input_norm`, `attention`, `post_attn_norm`, and `ffn` submodules to capture hidden_states at measurement points P0-P6. P3 computed as `P0 + P2` (residual add), P6 computed as `residual_at_P3 + P5`. G0 hooks on `embed_tokens`, G1 on final `norm`, G2 on `lm_head`.
- **D-03:** Single `ErrorPropagationTracker` class with `attach(model)` / `detach()` API. Internally manages two hook subsystems: per-Linear pre-hooks for x capture, and per-submodule forward hooks for P-point tracking. Shared results dict accessible after forward pass.

### Activation Storage
- **D-04:** In each pre-hook callback, detach and clone the input tensor, move to CPU immediately to avoid GPU memory pressure. Store as `dict[module_path → cpu_tensor]` where module_path uses the dot-separated naming convention (e.g., `model.layers.0.attention.q_proj`). No list — single forward pass = single tensor per matrix.
- **D-05:** For P-point hooks, store hidden_states tensor (detached, cloned, CPU) keyed by `{layer_idx}_{point_id}` (e.g., `5_P0`). Also store for the batch — a single pass captures all P-point states.

### Error Computation Pipeline
- **D-06:** After forward pass completes, `tracker.compute_output_error(model, quantizer)` iterates saved activations. For each (name, x_cpu): load W_fp from model, quantize to W_q via `quantizer.quantize(W_fp)`, move x to same device as weight, compute `y_fp = x @ W_fp.T`, `y_q = x @ W_q.T`, compute `norm(y_q - y_fp) / norm(y_fp)`. Return `dict[name → float]`.
- **D-07:** The quantizer used for error computation is the standard `FPQuantizer(fmt='fp4_e2m1', per_channel=True)` with deterministic round-to-nearest. No GPTQ compensation — per-matrix independence is required for Theorem 1 testing (per STATE.md accumulated context).

### Kappa Computation
- **D-08:** Reuse `estimate_condition_number()` from `src/analysis/condition.py` (exact SVD via `torch.linalg.svdvals`). Call `compute_all_condition_numbers(model)` for the full 72-matrix κ report. Kappa computation is separate from activation capture — no hooks needed.
- **D-09:** The ErrorPropagationTracker focuses on activation capture and ||dy||/||y|| computation. Kappa is a separate pass, not part of the forward hook system.

### Module Naming Convention
- **D-10:** Reuse `get_quantizable_weights()` naming convention (dot-separated parameter paths like `model.layers.0.attention.q_proj.weight`). Hook storage keys strip `.weight` suffix, using module path (e.g., `model.layers.0.attention.q_proj`). This maps directly to weight names for offline computation.

### Null Measurement
- **D-11:** Null measurement implemented as: load FP16 model, run forward pass with tracker attached, then "quantize" with identity (W_q = W_fp). Verify all ||dy||/||y|| < 1e-5. Implement as a dedicated method `tracker.validate_null_measurement(model)` that reports the max observed error.

### Evaluation Batch
- **D-12:** Single evaluation batch (batch_size=1, seq_len=512, seed=42) used for activation capture. This keeps memory manageable (~113 MB for 72 activation tensors) and satisfies "single-pass" requirement. Multi-step aggregation (100 steps) belongs to Phase 3.

### Claude's Discretion
- Exact hook cleanup strategy (remove hooks in detach() vs store handles)
- Whether to compute ||dy||/||y|| on GPU (faster for large matrices) or CPU (safer for memory)
- Error message format for null measurement violation reporting
- Whether to store P-point tensors per-token or as full batch tensor
- Logging verbosity during hook registration (module count, hook placement confirmation)

## Canonical References

### Model Architecture
- `src/model/transformer.py:172-194` — `TransformerLayer.forward()` — defines where P0-P6 measurement points sit in the forward pass
- `src/model/transformer.py:201-226` — `MicroGemmaFPModel.forward()` — defines G0 (after embed_tokens), G1 (after final norm)
- `src/model/transformer.py:229-256` — `MicroGemmaFPForCausalLM.forward()` — defines G2 (after lm_head), and label shifting for loss
- `src/model/transformer.py:264-270` — `get_quantizable_weights()` — the canonical list of 72 quantizable weight matrices with naming convention
- `src/model/config.py:15-87` — `MicroGemmaFPConfig` — architecture dimensions (hidden_size=768, intermediate_size=3072, 12 layers, GQA 4:1)

### Existing Analysis
- `src/analysis/condition.py:28-39` — `estimate_singular_values()` — exact SVD for κ(W) computation (already validated, reuse directly)
- `src/analysis/condition.py:42-52` — `estimate_condition_number()` — κ = σ_max / σ_min via exact SVD
- `src/analysis/condition.py:55-65` — `compute_all_condition_numbers()` — iterate all params dim>=2, compute κ for each

### Existing Quantization
- `src/quantization/fp_quantizer.py:53-80` — `FPQuantizer.__init__()` and `quantize()` — the quantizer used for W → W_q in error computation
- `src/quantization/gptq.py:137-191` — `GPTQQuantizer._collect_activations()` — existing forward-hook pattern for activation capture (analogous approach)

### Requirements
- `.planning/REQUIREMENTS.md` §MEAS-01 through MEAS-04, VAL-03 — Full requirement text for this phase
- `.planning/ROADMAP.md` §"Phase 2: Core Measurement Protocol" — Success criteria (5 items)

### Prior Phase Context
- `.planning/phases/01-clean-data-split/01-CONTEXT.md` — Data split decisions (train/val separation, dataloader API)
- `.planning/STATE.md` — Accumulated context: single-pass protocol, round-to-nearest for Theorem 1, Bonferroni correction

### Project Foundation
- `.planning/PROJECT.md` — Core value, constraints, key decisions (exact SVD, per-matrix granularity, output MSE metric)
- `.planning/REQUIREMENTS.md` — Full v1 requirements traceability matrix

## Existing Code Insights

### Reusable Assets
- `estimate_condition_number()` / `estimate_singular_values()` (`condition.py:28-52`): Exact SVD-based κ computation. Proven correct (previously fixed from power iteration). Reuse directly — no modification needed.
- `FPQuantizer.quantize()` (`fp_quantizer.py:79+`): Deterministic round-to-nearest FP4 quantization. The quantizer for W → W_q in error computation.
- `get_quantizable_weights()` (`transformer.py:264-270`): Returns list of (name, param) for all 72 quantizable matrices. Use this naming convention for hook storage keys.
- `GPTQQuantizer._collect_activations()` (`gptq.py:152+`): Existing pattern for collecting activations via forward hooks. Same approach, different storage target (pre-activation x instead of layer input).
- `compute_all_condition_numbers()` (`condition.py:55-65`): Iterate all params, compute κ. Can be called directly after model loading.
- `MicroGemmaFPConfig` (`config.py:15-87`): Dataclass with all architecture dimensions — no need to hardcode layer count or hidden size.

### Established Patterns
- All hooks use `@torch.no_grad()` context — measurement is inference-only, no gradient tracking.
- Hooks are registered via `register_forward_hook(handle)` and stored for cleanup — follow GPTQ's handle management pattern.
- Tensors are detached and cloned before storing to avoid retaining autograd graph.
- Module identification uses `model.named_modules()` with string matching on module class (e.g., `isinstance(module, nn.Linear)`).
- Results returned as `dict[str, float]` keyed by parameter/module name — consistent with `compute_all_condition_numbers()` output format.

### Integration Points
- **Transformer model** (`transformer.py`): The sole target for hook registration. No modification to the file — all instrumentation is external.
- **`get_dataloader(split='val')`** (`training_utils.py`): Provides evaluation data. Phase 2 uses a single validation batch for activation capture.
- **`load_checkpoint()`** (`training_utils.py:351-356`): Load FP16 baseline checkpoint for measurement.
- **`src/analysis/`** package: ErrorPropagationTracker lives here alongside condition.py, lipschitz.py, sensitivity.py — consistent with the analysis layer's purpose.
- **Downstream consumers**: Phase 3 (Theorem 1 Validation) will call `tracker.compute_output_error()` and combine with κ values. Phase 4 (Error Propagation Trace) will read P-point data from the same tracker.

## Specific Ideas

The 72 weight matrices are distributed as:
- Per layer (×12): q_proj, k_proj, v_proj, o_proj (4 attention) + gate_proj, up_proj, down_proj (3 FFN) = 7 per layer × 12 = 84
- Global: embed_tokens (1) + lm_head (1, tied with embed_tokens) = effectively 1 additional
- Total unique: 12 × 7 + 1 = 85 Linear modules, but lm_head is tied to embed_tokens.weight, and `get_quantizable_weights()` may exclude some based on naming. The exact count will be reported by the tracker on attach.

Measurement point definitions (matching ROADMAP P0-P6/G0-G2):
- P0: hidden_states before input_norm (residual input to layer)
- P1: hidden_states after input_norm, before attention
- P2: attention output, before residual add
- P3: post-attention residual (P0 + P2)
- P4: hidden_states after post_attn_norm, before FFN
- P5: FFN output, before residual add
- P6: post-FFN residual (P3 + P5)
- G0: after embed_tokens
- G1: after final RMSNorm
- G2: after lm_head

## Deferred Ideas

None — discussion stayed within phase scope. All auto-selected decisions align with prior STATE.md accumulated context and PROJECT.md key decisions.

---

*Phase: 2-Core Measurement Protocol*
*Context gathered: 2026-05-17*
*Mode: --auto (all gray areas auto-selected, recommended options chosen)*
