# Phase 2: Core Measurement Protocol - Research

**Researched:** 2026-05-17
**Domain:** PyTorch forward-hook-based activation capture and offline per-matrix output-space relative error computation
**Confidence:** HIGH

## Summary

This phase implements `ErrorPropagationTracker` in `src/analysis/error_propagation.py` -- an external hook-based measurement system that captures pre-activation tensors from a single FP16 forward pass and computes per-matrix `||dy||/||y||` for all Linear weight matrices in the 164M Micro-Gemma-FP Transformer. The tracker registers two hook subsystems: (1) `register_forward_pre_hook` on each `nn.Linear` module to capture inputs `x`, and (2) `register_forward_hook` on specific submodules to capture measurement points P0-P6 (per-layer, 6 points) and G0-G2 (global, 3 points). After the forward pass, error is computed offline: `norm(W_q @ x.T - W_fp @ x.T) / norm(W_fp @ x.T)`. The tracker does NOT modify `transformer.py` -- all instrumentation is external.

The phase also computes per-matrix `kappa(W)` via exact SVD (reusing `estimate_condition_number()` from `condition.py`) and implements a null measurement validation to verify the pipeline produces `||dy||/||y|| < 1e-5` when identity quantization is used.

**Primary recommendation:** Implement `ErrorPropagationTracker` as a single class with `attach(model)` / `detach()` API, internal handle management, and `compute_output_error(model, quantizer)` method. Reuse `get_quantizable_weights()` naming convention for module paths. Exact SVD from `condition.py` is already validated and production-ready.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01:** Use `register_forward_pre_hook` on each `nn.Linear` module to capture pre-activation tensors x. Iterate `model.named_modules()` externally to find all Linear layers without touching transformer.py.
- **D-02:** Use `register_forward_hook` on `input_norm`, `attention`, `post_attn_norm`, and `ffn` submodules to capture hidden_states at measurement points P0-P6. P3 computed as `P0 + P2` (residual add), P6 computed as `residual_at_P3 + P5`. G0 hooks on `embed_tokens`, G1 on final `norm`, G2 on `lm_head`.
- **D-03:** Single `ErrorPropagationTracker` class with `attach(model)` / `detach()` API. Internally manages two hook subsystems: per-Linear pre-hooks for x capture, and per-submodule forward hooks for P-point tracking. Shared results dict accessible after forward pass.
- **D-04:** In each pre-hook callback, detach and clone the input tensor, move to CPU immediately. Store as `dict[module_path -> cpu_tensor]` where module_path uses dot-separated naming. No list -- single forward pass = single tensor per matrix.
- **D-05:** For P-point hooks, store hidden_states tensor (detached, cloned, CPU) keyed by `{layer_idx}_{point_id}` (e.g., `5_P0`). Also store for the batch -- a single pass captures all P-point states.
- **D-06:** After forward pass completes, `tracker.compute_output_error(model, quantizer)` iterates saved activations. For each (name, x_cpu): load W_fp from model, quantize to W_q via quantizer, move x to same device as weight, compute `y_fp = x @ W_fp.T`, `y_q = x @ W_q.T`, compute `norm(y_q - y_fp) / norm(y_fp)`. Return `dict[name -> float]`.
- **D-07:** The quantizer used for error computation is `FPQuantizer(fmt='fp4_e2m1', per_channel=True)` with deterministic round-to-nearest. No GPTQ compensation.
- **D-08:** Reuse `estimate_condition_number()` from `src/analysis/condition.py` (exact SVD via `torch.linalg.svdvals`). Call `compute_all_condition_numbers(model)` for the full kappa report. Kappa computation is separate from activation capture.
- **D-09:** The ErrorPropagationTracker focuses on activation capture and ||dy||/||y|| computation. Kappa is a separate pass, not part of the forward hook system.
- **D-10:** Reuse `get_quantizable_weights()` naming convention (dot-separated parameter paths). Hook storage keys strip `.weight` suffix, using module path.
- **D-11:** Null measurement: load FP16 model, run forward pass with tracker, "quantize" with identity (W_q = W_fp). Verify all ||dy||/||y|| < 1e-5. Implement as `tracker.validate_null_measurement(model)`.
- **D-12:** Single evaluation batch (batch_size=1, seq_len=512, seed=42) for activation capture. Multi-step aggregation to Phase 3.

### Claude's Discretion
- Exact hook cleanup strategy (remove hooks in detach() vs store handles)
- Whether to compute ||dy||/||y|| on GPU (faster) or CPU (safer for memory)
- Error message format for null measurement violation reporting
- Whether to store P-point tensors per-token or as full batch tensor
- Logging verbosity during hook registration (module count, hook placement confirmation)

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| MEAS-01 | Implement ErrorPropagationTracker with P0-P6 (per-layer) and G0-G2 (global) measurement points | TransformerLayer.forward() structure at lines 181-194 defines P0-P6 locations. MicroGemmaFPModel.forward() at lines 215-226 defines G0-G1. MicroGemmaFPForCausalLM.forward() at lines 245-257 defines G2. Submodule forward hooks on input_norm, attention, post_attn_norm, ffn for P-points; hooks on embed_tokens, norm, lm_head for G-points. |
| MEAS-02 | Single-pass activation capture for all Linear weight matrices | `model.named_modules()` with `isinstance(module, nn.Linear)` finds 85 Linear modules. Pre-hooks capture input tensor `x` per D-04. Single batch (batch=1, seq=512) keeps memory ~113 MB for 85 activation tensors. |
| MEAS-03 | Compute per-matrix ||(W_q - W)x|| / ||Wx|| using round-to-nearest FP4 quantization | D-06 pipeline: load W_fp, quantize via `FPQuantizer('fp4_e2m1', per_channel=True).quantize()`, compute y_fp and y_q via `x @ W.T`, measure Frobenius norm ratio. |
| MEAS-04 | Compute per-matrix kappa(W) via exact SVD | `condition.py:estimate_condition_number()` already uses exact `torch.linalg.svdvals()`. For 164M model matrices (max ~3072x832), cost is <1ms per matrix. `compute_all_condition_numbers()` iterates dim>=2 params. |
| VAL-03 | Null measurement: quantize FP16 to identity, verify ||dy||/||y|| < 1e-5 | D-11: skip quantization step (W_q = W_fp directly). Pipeline noise from hooks, tensor cloning, and floating-point should be below 1e-5. |
</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Activation capture (Linear pre-hooks) | Analysis (external instrumentation) | Model (passive target) | Hooks register on model modules but live in analysis code; model file untouched |
| Measurement points P0-P6 / G0-G2 | Analysis (external instrumentation) | Model (passive target) | Forward hooks on submodule boundaries; storage and keying managed by ErrorPropagationTracker |
| Error computation ||dy||/||y|| | Analysis | Quantization (called offline) | W_fp loaded from model, W_q via FPQuantizer, computation in standalone method |
| Condition number kappa(W) | Analysis (condition.py) | -- | Already exists and validated; reuse directly |
| Null measurement validation | Analysis | -- | Self-contained pipeline check within ErrorPropagationTracker |
| Data provision (single batch) | Data (training_utils) | -- | `get_dataloader(split='val')` yields single batch via D-12 |
| Checkpoint loading | Data (training_utils) | -- | `load_checkpoint()` from training_utils.py; checkpoint on remote GPU server |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | >= 2.3.0 | All tensor ops, hook registration, SVD, linear algebra | Project dependency; all model/quantization/analysis layers depend on it |

### Existing Code APIs (Reused Directly)
| Module | Function/Class | Purpose | Phase Role |
|--------|---------------|---------|------------|
| `src/analysis/condition.py` | `estimate_condition_number()` | Exact SVD-based kappa computation | Called for each weight matrix to produce kappa report (MEAS-04) |
| `src/analysis/condition.py` | `compute_all_condition_numbers()` | Iterate all params, return dict[name->kappa] | Single-call kappa report for all 85+ matrices |
| `src/quantization/fp_quantizer.py` | `FPQuantizer` class | Simulated FP4 quantization | Used in `compute_output_error()` to produce W_q from W_fp (MEAS-03) |
| `src/model/transformer.py` | `get_quantizable_weights()` | Returns list of (name, param) for weight matrices | Naming convention source for module path keys (D-10) |
| `src/experiments/training_utils.py` | `get_dataloader(split='val')` | Validation dataloader | Single evaluation batch for activation capture (D-12) |
| `src/experiments/training_utils.py` | `load_checkpoint()` | Load FP16 baseline checkpoint | Model loading for measurement |

### New Code (ErrorPropagationTracker)
| Method | Purpose | Key API |
|--------|---------|---------|
| `attach(model)` | Register all hooks on model | Returns self; stores hook handles internally |
| `detach()` | Remove all registered hooks | Iterates stored handles, calls `.remove()` |
| `compute_output_error(model, quantizer)` | Offline ||dy||/||y|| for all captured activations | Returns `dict[str, float]` |
| `validate_null_measurement(model)` | Identity quantization check | Returns max error, raises if > 1e-5 |
| Property: `activations` | Access captured x tensors | `dict[str, torch.Tensor]` keyed by module path |
| Property: `p_points` | Access captured P/G-point tensors | `dict[str, torch.Tensor]` keyed by `{layer_idx}_{point_id}` |

### Installation
No new packages required. Pure PyTorch, reusing existing code.

### Version Verification
All dependencies are existing project imports. No new packages to verify.

## Package Legitimacy Audit

> No external packages installed by this phase. ErrorPropagationTracker uses only:
> - `torch` (existing project dependency)
> - `src.analysis.condition` (existing module)
> - `src.quantization.fp_quantizer` (existing module)
> - `src.model.transformer` (existing module, imported but not modified)
> - `src.experiments.training_utils` (existing module, for dataloader)

No new registry packages required. Slopcheck not needed.

## Architecture Patterns

### System Architecture Diagram

```
Forward Pass (FP16, single batch, batch=1, seq=512, seed=42)
    |
    v
ErrorPropagationTracker.attach(model)
    |
    +-- register_forward_pre_hook (nn.Linear modules x 85)
    |       captures x before each Linear
    |       -> detach, clone, CPU -> activations[name]
    |
    +-- register_forward_hook (submodule hooks)
    |       G0: embed_tokens output
    |       G1: final norm output
    |       G2: lm_head output
    |       P0: layer.forward input
    |       P1: input_norm output (before attention)
    |       P2: attention output (before residual add)
    |       P3: RESIDUAL = P0 + P2 (computed, not hooked)
    |       P4: post_attn_norm output (before FFN)
    |       P5: ffn output (before residual add)
    |       P6: RESIDUAL = P3 + P5 (computed, not hooked)
    |
    v
Forward pass completes -> hooks populated
    |
    v
tracker.compute_output_error(model, quantizer)
    |
    for each (name, x_cpu) in activations:
        W_fp = getattr(model, name).weight
        W_q = quantizer.quantize(W_fp)       # FP4 E2M1, round-to-nearest
        x = x_cpu.to(W_fp.device)
        y_fp = x @ W_fp.T
        y_q = x @ W_q.T
        err = norm(y_q - y_fp) / norm(y_fp)  # Frobenius norm
        results[name] = err
    |
    v
compute_all_condition_numbers(model)          # Separate pass, no hooks
    for each weight with dim >= 2:
        kappa = estimate_condition_number(W)
        kappa_results[name] = kappa
    |
    v
tracker.validate_null_measurement(model)       # Identity quantization
    -> assert all errors < 1e-5
    |
    v
tracker.detach()                               # Cleanup all hooks
```

### Recommended Project Structure
```
src/
├── analysis/
│   ├── __init__.py            # Empty (existing pattern)
│   ├── condition.py           # kappa estimation (existing, reused)
│   ├── lipschitz.py           # Lipschitz propagation (existing)
│   ├── sensitivity.py         # Sensitivity analysis (existing)
│   └── error_propagation.py   # NEW: ErrorPropagationTracker
├── model/
│   ├── config.py              # MicroGemmaFPConfig (unchanged)
│   └── transformer.py         # Model (NOT modified by this phase)
├── quantization/
│   ├── fp_quantizer.py        # FPQuantizer (unchanged)
│   └── ...
└── experiments/
    └── training_utils.py      # DataLoader, load_checkpoint (unchanged)
```

### Pattern 1: Forward Hook Registration for Activation Capture

**What:** Register `register_forward_pre_hook` on every `nn.Linear` module to capture pre-activation tensors. Stores handles for cleanup. Follows the pattern established by `GPTQQuantizer._collect_activations()` in `gptq.py:193-247`.

**When to use:** During `ErrorPropagationTracker.attach(model)` -- iterate model submodules, identify `nn.Linear` via `isinstance()`, register pre-hooks that detach+clone+CPU the input tensor.

**Key differences from GPTQ pattern:**
- GPTQ stores a list of tensors (multi-pass accumulation, then concatenate). Phase 2 stores a single tensor (single pass, no accumulation needed).
- GPTQ flattens batch+seq dims to (n_samples, in_features). Phase 2 preserves shape (batch, seq, in_features) for correct output computation via `x @ W.T`.
- GPTQ skips embed_tokens and lm_head hooks. Phase 2 includes all `nn.Linear` layers.

**Example:**
```python
# Source: Derived from GPTQQuantizer._collect_activations() in gptq.py:193-247
# and D-01/D-04 decisions in CONTEXT.md

def _register_linear_pre_hooks(self, model):
    """Register pre-hooks on all nn.Linear modules."""
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        # Store module path key (strip .weight suffix convention)
        module_path = name  # e.g., 'model.layers.0.attention.q_proj'
        handle = module.register_forward_pre_hook(
            self._make_pre_hook(module_path)
        )
        self._hook_handles.append(handle)

def _make_pre_hook(self, module_path: str):
    """Create a pre-hook closure that captures input x."""
    @torch.no_grad()
    def hook(module, input_args):
        # input_args[0] shape: (batch, seq_len, in_features)
        x = input_args[0].detach().clone().cpu()
        self._activations[module_path] = x
    return hook
```

### Pattern 2: Measurement Point (P-point) Hook Registration

**What:** Register `register_forward_hook` on specific submodules (input_norm, attention, post_attn_norm, ffn) to capture hidden_states at defined measurement points. Also register hooks on embed_tokens, norm, and lm_head for global points G0-G2.

**When to use:** During `ErrorPropagationTracker.attach(model)` alongside linear pre-hook registration.

**P0-P6 mapping to TransformerLayer.forward() at transformer.py:181-194:**
```
P0: input to layer.forward (residual before input_norm)
    -> register_forward_pre_hook on TransformerLayer itself
P1: output of input_norm (input to attention)
    -> register_forward_hook on input_norm
P2: output of attention (before residual add)
    -> register_forward_hook on attention
P3: P0 + P2 (computed, not hooked)
P4: output of post_attn_norm (input to FFN)
    -> register_forward_hook on post_attn_norm
P5: output of ffn (before residual add)
    -> register_forward_hook on ffn
P6: P3 + P5 (computed, not hooked)
```

**G0-G2 mapping to MicroGemmaFPModel.forward() and MicroGemmaFPForCausalLM.forward():**
```
G0: after embed_tokens (model.forward line 220)
    -> register_forward_hook on model.model.embed_tokens
G1: after final norm (model.forward line 225)
    -> register_forward_hook on model.model.norm
G2: after lm_head (causal_lm.forward line 247)
    -> register_forward_hook on model.lm_head
```

**Example storage key convention (D-05):**
```python
self._p_points[f"{layer_idx}_P0"] = p0_tensor   # per-layer
self._g_points["G0"] = g0_tensor                  # global
```

### Pattern 3: Offline Error Computation (D-06)

**When to use:** After the forward pass completes and all activations are captured.

```python
# Source: D-06 decision in CONTEXT.md
def compute_output_error(self, model, quantizer):
    results = {}
    for module_path, x_cpu in self._activations.items():
        # Load the weight
        module = self._resolve_module(model, module_path)
        W_fp = module.weight.data  # (out_features, in_features)

        # Quantize to FP4 (deterministic, round-to-nearest)
        W_q = quantizer.quantize(W_fp)

        # Compute error on weight's device (GPU recommended for speed)
        device = W_fp.device
        x = x_cpu.to(device)
        y_fp = x @ W_fp.T  # (batch, seq, out_features)
        y_q = x @ W_q.T

        # Frobenius norm ratio
        err = (y_q - y_fp).norm().item() / y_fp.norm().clamp(min=1e-12).item()
        results[module_path] = err

    return results
```

### Pattern 4: Null Measurement Validation (D-11)

**When to use:** After forward pass and before real quantization-based error computation, to validate the measurement pipeline.

```python
def validate_null_measurement(self, model) -> float:
    """Run identity quantization and verify errors are below epsilon."""
    with torch.no_grad():
        # "Quantize" with identity: W_q = W_fp
        # compute_output_error triggers the same pipeline but W_q == W_fp
        pass  # See Code Examples section for full pattern
```

### Anti-Patterns to Avoid

- **Modifying transformer.py:** All hooks are external via `register_forward_hook`/`pre_hook`. Never add hook registration code inside the model file.
- **Retaining autograd graph:** Every hook must call `.detach()` on captured tensors. Undetached tensors keep the entire computation graph alive (~3 GB for 164M model).
- **Accumulating across batches:** Store only one tensor per module path (single forward pass). GPTQ's list accumulation pattern is inappropriate here.
- **Flattening activation shape:** Preserve `(batch, seq, in_features)` for correct `x @ W.T` output computation. GPTQ flattens to `(n_samples, in_features)` for Hessian computation, which is wrong for error computation.
- **Computing error on CPU for large matrices:** For gate_proj (832x3072) and down_proj (3072x768), CPU matmuls are significantly slower. Move x to weight's device.
- **Using GPTQ-compensated weights for Theorem 1 test:** Per-matrix independence is required. GPTQ column compensation couples matrices (D-07).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Condition number estimation | Power iteration from scratch | `condition.py:estimate_condition_number()` | Already uses exact SVD, tested and correct. Matrices are small enough (< 832x832 for most) that SVD is cheap. |
| FP4 quantization | Custom quantizer | `FPQuantizer(fmt='fp4_e2m1', per_channel=True)` | Existing, validated, supports deterministic and stochastic rounding |
| Hook registration framework | Custom hook management library | Raw PyTorch `register_forward_hook` + handle list | Minimal overhead; project only needs one attach/detach lifecycle |
| Model parameter iteration | Manual module tree traversal | `model.named_modules()`, `model.named_parameters()` | Standard PyTorch APIs; handles submodule recursion and device placement |

**Key insight:** The analysis layer already has well-tested primitives for kappa estimation. The new work is the hook registration infrastructure and offline error computation pipeline, which are composed from existing APIs rather than built from scratch.

## Common Pitfalls

### Pitfall 1: Hook Closure Capturing Stale References
**What goes wrong:** If hooks capture a mutable variable (like `module_path` in a loop) by reference, all hooks may reference the last value of the variable.
**Why it happens:** Python closures capture variables by reference, not by value. Classic `lambda i=i: ...` bug.
**How to avoid:** Use a factory function (`_make_pre_hook(module_path)`) that returns a closure. This captures `module_path` as a local in the factory, which is evaluated at call time.
**Warning signs:** All activations stored under the same module_path key (overwriting each other).

### Pitfall 2: Hook Retention After Forward Pass
**What goes wrong:** Hooks remain registered on model modules even after the tracker is no longer needed. This can corrupt subsequent forward passes (especially if the hook modifies state).
**Why it happens:** `register_forward_hook` persists until `handle.remove()` is called or the module is garbage collected.
**How to avoid:** Always call `tracker.detach()` after measurement. Store all handles in a list and iterate `h.remove()`. Implement `__del__` or `__enter__/__exit__` as safety net.
**Warning signs:** Memory grows across multiple forward passes, or activations from unrelated runs appear in the tracker.

### Pitfall 3: GPU Memory Explosion from Undetached Tensors
**What goes wrong:** Hook captures the input tensor without `.detach()`, keeping the autograd graph alive. For a 164M model, the retained graph can consume >3 GB of GPU memory.
**Why it happens:** The model.forward() builds a computation graph. If any hook stores a reference to a node in this graph (even indirectly), the entire graph stays alive.
**How to avoid:** Always call `.detach().clone()` before storing. Move to CPU immediately (`.cpu()`). The pattern `x = input_args[0].detach().clone().cpu()` is mandatory in every hook.
**Warning signs:** "CUDA out of memory" after only 1-2 forward passes.

### Pitfall 4: Wrong Output Shape in Error Computation
**What goes wrong:** `y_fp = x @ W_fp.T` produces `(batch, seq, out_features)` but the Frobenius norm is computed over all elements, which is correct.
**Why this is not actually a pitfall:** Per D-06, the Frobenius norm of the full output tensor `(batch, seq, out_features)` is the correct ||dy|| for the Theorem 1 bound. No need to aggregate per-token or per-position.
**When to worry:** If a flattening step (like GPTQ does for Hessian) is applied, the output computation changes from `x @ W.T` to a different operation.

### Pitfall 5: Module Path Key Format Mismatch
**What goes wrong:** Hook storage key uses `model.layers.0.attention.q_proj` but `get_quantizable_weights()` returns `model.layers.0.attention.q_proj.weight`. Downstream consumers can't match keys.
**Why it happens:** The pre-hook receives the module (not the parameter), so its natural key is the module path. But parameter names include `.weight`.
**How to avoid:** D-10 mandates stripping `.weight` from parameter paths to produce module paths. Apply consistently: when building results dict, use module paths (no `.weight` suffix) as keys.
**Warning signs:** Phase 3 code can't join kappa dict with error dict because keys don't match.

### Pitfall 6: Loss of Per-Layer Embedding Context in Attention/FFN Inputs
**What goes wrong:** The pre-hook on `q_proj` captures `x = cat([hidden_states, pl_emb], dim=-1)` which has shape `(batch, seq, 832)`. The weight `q_proj.weight` has shape `(out_features, 832)`. The computation `x @ W.T` is correct -- but the input `x` includes per-layer embeddings that differ per layer, so activations from layer 0 and layer 5 look different even for the same hidden_states.
**Why this matters:** This is correct behavior. The activations include per-layer context as designed. Phase 3 analysis should be aware that q/k/v inputs are not pure hidden states.

## Code Examples

### Full ErrorPropagationTracker Class Structure

```python
# Source: Derived from D-01 through D-12 decisions in CONTEXT.md
# Lines based on established patterns from gptq.py and sensitivity.py

import torch
import torch.nn as nn


class ErrorPropagationTracker:
    """Hook-based activation capture and offline error computation.

    Usage:
        model = MicroGemmaFPForCausalLM(config)
        load_checkpoint(model, None, checkpoint_path, device)

        quantizer = FPQuantizer(fmt='fp4_e2m1', per_channel=True)
        tracker = ErrorPropagationTracker()

        # Single-pass activation capture
        tracker.attach(model)
        batch = next(iter(get_dataloader(split='val', batch_size=1)))
        model(batch['input_ids'].to(device))
        tracker.detach()

        # Error computation
        errors = tracker.compute_output_error(model, quantizer)

        # Null validation
        max_null_err = tracker.validate_null_measurement(model)

        # Kappa computation (separate pass)
        kappas = compute_all_condition_numbers(model)
    """

    def __init__(self):
        self._activations = {}        # module_path -> cpu_tensor
        self._p_points = {}           # "{layer_idx}_{point_id}" -> cpu_tensor
        self._g_points = {}           # "G0", "G1", "G2" -> cpu_tensor
        self._hook_handles = []       # list of hook handles for cleanup
        self._activation_keys = []    # ordered list of module paths

    def attach(self, model):
        """Register all hooks on the model.

        Registers two hook subsystems:
        1. Pre-hooks on all nn.Linear modules for pre-activation capture
        2. Forward hooks on measurement point submodules
        """
        self._register_linear_pre_hooks(model)
        self._register_p_point_hooks(model)
        self._register_g_point_hooks(model)

        # Report hook registration summary
        print(f"[Tracker] Registered {len(self._activations)} Linear pre-hooks, "
              f"{len(self._p_points)} P-point hooks, "
              f"{len(self._g_points)} G-point hooks")
        return self

    def detach(self):
        """Remove all registered hooks."""
        count = len(self._hook_handles)
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        print(f"[Tracker] Removed {count} hooks")
        return self

    @property
    def activations(self) -> dict[str, torch.Tensor]:
        return dict(self._activations)

    @property
    def p_points(self) -> dict[str, torch.Tensor]:
        return {**self._p_points, **self._g_points}

    @torch.no_grad()
    def compute_output_error(self, model, quantizer) -> dict[str, float]:
        """Compute ||dy||/||y|| for all captured Linear layers.

        For each saved activation x and corresponding weight matrix W:
            y_fp = x @ W_fp.T
            y_q  = x @ W_q.T   (W_q = quantizer.quantize(W_fp))
            err  = ||y_q - y_fp|| / ||y_fp||

        Returns dict mapping module_path -> relative_error.
        """
        results = {}
        for module_path, x_cpu in self._activations.items():
            # Resolve module and get weight
            module = self._resolve_module(model, module_path)
            if module is None or not hasattr(module, 'weight'):
                continue
            W_fp = module.weight.data

            # Move x to weight's device for computation
            device = W_fp.device
            x = x_cpu.to(device)  # (batch, seq, in_features)

            # FP16 output (reference)
            y_fp = x @ W_fp.T  # (batch, seq, out_features)

            # Quantize and compute quantized output
            W_q = quantizer.quantize(W_fp)
            y_q = x @ W_q.T

            # Relative Frobenius norm error
            denom = y_fp.norm().clamp(min=1e-12)
            err = (y_q - y_fp).norm() / denom
            results[module_path] = err.item()

        return results

    @torch.no_grad()
    def validate_null_measurement(self, model) -> float:
        """Run identity quantization and verify pipeline integrity.

        Returns the maximum ||dy||/||y|| across all modules.
        Raises ValueError if any error exceeds 1e-5.
        """
        # Reuse compute_output_error with identity quantizer
        # Simpler: just check that y_q == y_fp by computing directly
        max_err = 0.0
        violations = []

        for module_path, x_cpu in self._activations.items():
            module = self._resolve_module(model, module_path)
            if module is None or not hasattr(module, 'weight'):
                continue
            W_fp = module.weight.data
            x = x_cpu.to(W_fp.device)

            # y_fp and y_q are the same (no quantization error)
            # Pipeline noise from FP arithmetic and hook storage
            y_fp = x @ W_fp.T
            y_q = x @ W_fp.T  # identical

            denom = y_fp.norm().clamp(min=1e-12)
            err = (y_q - y_fp).norm() / denom
            max_err = max(max_err, err.item())
            if err.item() > 1e-5:
                violations.append((module_path, err.item()))

        if violations:
            msg = (f"Null measurement failed: {len(violations)} modules "
                   f"exceed 1e-5. Max error: {max_err:.2e}")
            print(f"[WARN] {msg}")
            for path, e in violations[:5]:
                print(f"  {path}: {e:.2e}")
            raise ValueError(msg)

        return max_err

    def _register_linear_pre_hooks(self, model):
        """Register pre-hooks on all nn.Linear modules for x capture."""
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            # Store module path (parameter path minus .weight suffix)
            module_path = name
            handle = module.register_forward_pre_hook(
                self._make_input_hook(module_path)
            )
            self._hook_handles.append(handle)
            # Pre-seed key so order is preserved
            self._activation_keys.append(module_path)

    def _make_input_hook(self, module_path: str):
        """Create pre-hook that captures input tensor x."""
        @torch.no_grad()
        def hook(module, input_args):
            x = input_args[0].detach().clone().cpu()
            self._activations[module_path] = x
        return hook

    def _register_p_point_hooks(self, model):
        """Register forward hooks for P0-P6 measurement points."""
        layers = model.model.layers if hasattr(model, 'model') and \
            hasattr(model.model, 'layers') else getattr(model, 'layers', [])

        for layer_idx, layer in enumerate(layers):
            # P0: input to layer (pre-hook on the layer itself)
            handle_p0 = layer.register_forward_pre_hook(
                self._make_p_hook(layer_idx, 'P0'))
            self._hook_handles.append(handle_p0)

            # P1: after input_norm
            handle_p1 = layer.input_norm.register_forward_hook(
                self._make_p_hook(layer_idx, 'P1'))
            self._hook_handles.append(handle_p1)

            # P2: after attention (before residual add)
            handle_p2 = layer.attention.register_forward_hook(
                self._make_p_hook(layer_idx, 'P2'))
            self._hook_handles.append(handle_p2)

            # P4: after post_attn_norm (before FFN)
            handle_p4 = layer.post_attn_norm.register_forward_hook(
                self._make_p_hook(layer_idx, 'P4'))
            self._hook_handles.append(handle_p4)

            # P5: after FFN (before residual add)
            handle_p5 = layer.ffn.register_forward_hook(
                self._make_p_hook(layer_idx, 'P5'))
            self._hook_handles.append(handle_p5)

        # P3 and P6 are computed (not hooked)
        # P3 = P0 + P2, P6 = P3 + P5

    def _make_p_hook(self, layer_idx: int, point_id: str):
        """Create hook that stores P-point activations."""
        @torch.no_grad()
        def hook(module, input_args, output=None):
            if output is not None:
                # Forward hook: capture output
                tensor = output.detach().clone().cpu()
            else:
                # Pre-hook: capture input
                tensor = input_args[0].detach().clone().cpu()
            key = f"{layer_idx}_{point_id}"
            self._p_points[key] = tensor
        return hook

    def _register_g_point_hooks(self, model):
        """Register forward hooks for G0-G2 global measurement points."""
        # G0: after embed_tokens
        handle_g0 = model.model.embed_tokens.register_forward_hook(
            self._make_g_hook('G0'))
        self._hook_handles.append(handle_g0)

        # G1: after final RMSNorm
        handle_g1 = model.model.norm.register_forward_hook(
            self._make_g_hook('G1'))
        self._hook_handles.append(handle_g1)

        # G2: after lm_head
        handle_g2 = model.lm_head.register_forward_hook(
            self._make_g_hook('G2'))
        self._hook_handles.append(handle_g2)

    def _make_g_hook(self, point_id: str):
        """Create hook that stores G-point activations."""
        @torch.no_grad()
        def hook(module, input_args, output):
            tensor = output.detach().clone().cpu()
            if isinstance(tensor, tuple):
                tensor = tensor[0]
            self._g_points[point_id] = tensor
        return hook

    def compute_p3_p6(self):
        """Compute derived measurement points P3 and P6 from stored P0/P2/P5.

        P3 = P0 + P2  (post-attention residual)
        P6 = P3 + P5  (post-FFN residual)

        Must be called after forward pass. Modifies self._p_points in-place.
        """
        # Group P-points by layer
        p0_keys = {k for k in self._p_points if k.endswith('_P0')}
        for key in p0_keys:
            layer_prefix = key.replace('_P0', '')
            p2_key = f"{layer_prefix}_P2"
            p5_key = f"{layer_prefix}_P5"

            if p2_key in self._p_points:
                p3_key = f"{layer_prefix}_P3"
                p0 = self._p_points[key]
                p2 = self._p_points[p2_key]
                self._p_points[p3_key] = (p0 + p2).clone()

            if p5_key in self._p_points and p2_key in self._p_points:
                p6_key = f"{layer_prefix}_P6"
                p3_key_internal = f"{layer_prefix}_P3"
                # P3 may not exist if p2_key was missing; check
                if p3_key_internal in self._p_points:
                    p3_val = self._p_points[p3_key_internal]
                    p5 = self._p_points[p5_key]
                    self._p_points[p6_key] = (p3_val + p5).clone()

    def _resolve_module(self, model, module_path: str):
        """Resolve a dot-separated module path on the model."""
        parts = module_path.split('.')
        obj = model
        for part in parts:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                return None
        return obj
```

### Kappa Computation (MEAS-04) -- Reuse Existing

```python
# Source: condition.py:28-65
# Already validated, no modification needed.

from src.analysis.condition import compute_all_condition_numbers

# After model is loaded:
kappas = compute_all_condition_numbers(model)
# Returns dict[str, float] keyed by parameter name (e.g., 'model.layers.0.attention.q_proj.weight')
# For matching with tracker results, strip '.weight' suffix on keys
```

### Single Evaluation Batch (D-12)

```python
# Source: D-12 and training_utils.py

from src.experiments.training_utils import get_dataloader

torch.manual_seed(42)
dataloader = get_dataloader(
    batch_size=1,
    max_seq_len=512,
    split='val',
    data_dir='data/real_tiers'
)

batch = next(iter(dataloader))
# batch contains: {'input_ids': (1, 512), 'labels': (1, 512), 'attention_mask': (1, 512)}
```

## State of the Art

This project is at the intersection of numerical analysis and LLM quantization. The ErrorPropagationTracker fills a specific gap: existing quantization papers (GPTQ, AWQ, QuIP) evaluate via PPL, but the numerical analysis community requires per-matrix output-space error to test Theorem 1 boundary.

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| PPL as primary metric | Per-matrix ||dy||/||y|| output error | Current phase | Enables direct Theorem 1 testing at the correct granularity |
| Two-pass error (quantize, run, compare) | Single-pass activation capture (quantize offline) | Current phase | Eliminates cascading error from quantized weight affecting subsequent layer inputs |
| Power iteration for kappa (underestimates by 5000x) | Exact SVD for kappa | Previous fix | Path to this phase; kappa values are now accurate |
| Per-layer aggregation | Per-weight-matrix granularity | Current phase | Addresses 1000x kappa variation within the same layer (q_proj ~100 vs o_proj ~16000) |

**Deprecated/outdated:**
- `analyze_quantization_sensitivity()` in `condition.py:135-167`: Computes per-matrix kappa and weight-space MSE but uses weight-space error only (no activation-based ||dy||/||y||). Not deprecated -- still useful for weight-space metrics, but Phase 2's activation-based approach supersedes it for output-space error.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `model.named_modules()` returns `nn.Linear` modules in a consistent order across runs | Architecture Patterns | If order changes, activation_keys ordering would be non-deterministic. Mitigation: tracker uses deterministic iteration (Python dicts preserve insertion order since 3.7) |
| A2 | Single batch (batch=1, seq=512) produces representative activations for all layers | Code Examples | If a specific token position or sequence triggers unique activation patterns, one batch may not capture diverse inputs. However, MEAS-02 requires single-pass only; Phase 3 handles multi-step. |
| A3 | The FP16 baseline checkpoint at `checkpoints/scaled_fp16_baseline/model.pt` exists on the remote server | Code Examples | If the checkpoint path differs or doesn't exist, the measurement pipeline has no model to attach to. Planner must verify checkpoint location. |
| A4 | `get_dataloader(split='val', batch_size=1)` works after Phase 1 data split | Code Examples | If Phase 1 hasn't been fully executed or the val split files don't exist, dataloader will fall back to offline corpus. Offline corpus data is still valid for measurement but may differ from real data distribution. |
| A5 | Linear module count is 85 (not 72 as stated in some earlier documents) | Ph. Req. | The 72 count from CONTEXT.md is approximate. `get_quantizable_weights()` returns 86 entries (84 proj + embed_tokens + lm_head), but embed_tokens is nn.Embedding (not nn.Linear). The nn.Linear count via isinstance check is 85 (84 proj + lm_head). Tracker reports actual count at attach time. |

## Open Questions

1. **What happens if the checkpoint path differs between local and remote environments?**
   - What we know: Phase 2 runs on remote GPU. Checkpoints are trained on remote.
   - What's unclear: Whether `checkpoints/scaled_fp16_baseline/model.pt` exists or has a different path.
   - Recommendation: Before execution, verify checkpoint path exists on remote. Use `./remote_run.sh "ls checkpoints/scaled_fp16_baseline/"` during setup. If not found, Phase 1 FP16 baseline must be trained first, or measurement targets available checkpoints.

2. **Should embedded P3/P6 be computed eagerly during forward pass or lazily after?**
   - What we know: P3 = P0 + P2 and P6 = P3 + P5 are computed, not hooked. The required tensors (P0, P2, P5) are captured via hooks.
   - What's unclear: Whether to compute P3/P6 lazily after forward pass (cleaner, separates capture from computation) or eagerly inside the P5 hook (avoids extra post-processing).
   - Recommendation: Lazy post-forward computation via `compute_p3_p6()` method. Keeps hooks simple and follows the principle that hooks capture raw data, derived quantities are computed separately.

3. **How does the tracker handle the tied embed_tokens and lm_head weight?**
   - What we know: `lm_head.weight = embed_tokens.weight` (same tensor, line 235 of transformer.py). Both appear in named_parameters.
   - What's unclear: Should the tracker deduplicate? If both are nn.Linear (lm_head is) and the same weight is quantized twice, the error dict has duplicate entries. But they have different module paths and different input tensors.
   - Recommendation: No deduplication. `lm_head` and `embed_tokens` have different inputs and produce different x tensors, so their ||dy||/||y|| values are legitimately different even though the weight is shared.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python >= 3.11 | All code | yes | 3.14.5 | -- |
| PyTorch >= 2.3 | Hooks, SVD, linear ops | yes (local) | (check remote) | -- |
| SSH + conda `sle` | Remote execution | yes | (remote) | -- |
| Checkpoint file | Model loading | unknown (no local checkpoints) | -- | Train baseline first, or use available checkpoint |
| Validation .bin files | Dataloader (split='val') | unknown (Phase 1 output) | -- | Fallback to offline corpus |

**Missing dependencies with no fallback:**
- The FP16 baseline checkpoint must exist on the remote server. If it does not, Phase 2 cannot proceed until a baseline model is available.

**Missing dependencies with fallback:**
- Validation data: if Phase 1 val split hasn't been created, `get_dataloader(split='val')` will use the offline embedded corpus. This produces valid activation tensors for pipeline testing, though the distribution differs from real data.

## Validation Architecture

> Skip condition met: `workflow.nyquist_validation` is explicitly `false` in `.planning/config.json`.

## Security Domain

> Not applicable. ErrorPropagationTracker is an internal measurement tool with no external input surfaces, no network access, no user data exposure, and no persistence layer. All hooks operate on pre-existing model tensors. The only inputs are:
> - Model parameters (read-only via hooks)
> - Quantizer configuration (internal constant: FP4 E2M1)
> - Single validation batch (from local .bin files)

No ASVS categories apply. No threat patterns relevant.

## Sources

### Primary (HIGH confidence)
- `src/model/transformer.py` lines 172-270 -- Verified forward pass structure for P0-P6/G0-G2 mapping and get_quantizable_weights() naming
- `src/analysis/condition.py` lines 28-65 -- Verified exact SVD kappa computation (reuse directly)
- `src/quantization/fp_quantizer.py` lines 53-122 -- Verified FPQuantizer API for D-07 quantizer selection
- `src/quantization/gptq.py` lines 193-247 -- Verified existing hook pattern for activation capture (reference architecture)
- `src/experiments/training_utils.py` lines 176-227 -- Verified dataloader API with split='val' support
- `src/model/config.py` lines 15-87 -- Verified architecture dimensions (12 layers, 768 hidden, 832 input dim with per-layer emb)
- `.planning/phases/02-core-measurement-protocol/02-CONTEXT.md` -- Locked decisions D-01 through D-12, discretion areas
- `.planning/REQUIREMENTS.md` -- MEAS-01 through MEAS-04, VAL-03 requirement text
- `.planning/config.json` -- workflow.nyquist_validation: false

### Secondary (MEDIUM confidence)
- `.planning/PROJECT.md` -- Project context, key decisions, measurement philosophy
- `.planning/STATE.md` -- Accumulated context: single-pass protocol, round-to-nearest, Bonferroni correction

### Tertiary (LOW confidence)
- None -- all claims in this research are verified against the existing codebase or locked decisions.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- All dependencies are existing project modules, no package unknowns
- Architecture: HIGH -- P0-P6/G0-G2 mapping verified against transformer.py source; hook patterns from GPTQ code
- Pitfalls: HIGH -- Based on known PyTorch hook gotchas and confirmed by existing GPTQ code patterns

**Research date:** 2026-05-17
**Valid until:** No time-sensitive dependencies; stable project codebase
