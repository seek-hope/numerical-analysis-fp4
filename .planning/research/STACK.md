# Technology Stack: Per-Layer Quantization Error Measurement Infrastructure

**Project:** Numerical Analysis for FP4 Transformer Quantization
**Researched:** 2026-05-17
**Domain:** Per-weight-matrix error propagation measurement (Theorem 1 validation, RMSNorm blocking measurement, full error trace)
**Mode:** Ecosystem

---

## Core Finding: The Gap

Every published LLM quantization paper surveyed (GPTQ, SmoothQuant, LLM.int8(), QuIP, SparseGPT, ZeroQuant, Outlier Suppression) evaluates quantization quality exclusively through **end-to-end perplexity (PPL) and zero-shot accuracy**. None of them systematically report per-layer relative output error (||dy||/||y||) or per-weight-matrix propagation ratios. The project's Theorem 1 validation protocol is novel in the quantization literature.

The closest precedent is **SparseGPT** (Frantar & Alistarh, 2023), which reports "layer-wise squared error relative to exact reconstruction" in its appendix (Figure 11) — but only to validate their approximation quality, not as a standard evaluation metric.

**Implication:** There is no off-the-shelf measurement infrastructure to adopt. Every component must be purpose-built using PyTorch hooks, numerical analysis routines, and custom evaluation loops.

---

## Recommended Measurement Infrastructure

### Core Measurement Framework

| Component | Technology | Version | Purpose | Rationale |
|-----------|-----------|---------|---------|-----------|
| Forward hook capture | PyTorch `register_forward_hook` | >=2.3 | Capture pre/post activations at 6+ points per layer | Already in codebase; used by GPTQ, SparseGPT, SmoothQuant for Hessian/scale calibration; the established pattern for per-layer analysis |
| Relative error computation | PyTorch tensor ops (`.norm()`, `/`) | — | Compute ||dy||/||y|| per matrix | Theorem 1 uses relative error; all matrix perturbation theory norms are consistent with PyTorch's default Frobenius/2-norms |
| Layer-wise MSE | PyTorch `F.mse_loss` | — | Track raw output deviation alongside relative error | Complements relative error; MSE is monotonic in PPL and more interpretable |
| Signal-to-noise ratio | PyTorch tensor ops | — | SNR(y, y_hat) = 10*log10(var(y)/var(dy)) | Scale-free error metric used in DSP literature; useful for cross-layer comparison |
| Condition number (exact SVD) | PyTorch `torch.linalg.svd` | >=2.0 | Compute kappa(W) = sigma_max / sigma_min | Already implemented; exact SVD on up to 768x4096 matrices is feasible (100x cheaper than a single eval step) |
| Lipschitz propagation | PyTorch `torch.autograd.functional.jacobian` | >=2.0 | Measure per-layer Lipschitz constants | Already implemented in `src/analysis/lipschitz.py` |
| Bootstrap CI | NumPy `np.random.choice` + manual resampling | >=1.21 | Compute PPL confidence intervals | Standard non-parametric bootstrapping; no library dependency |

### Recommended Metrics Table

For each Linear weight matrix (q_proj, k_proj, v_proj, out_proj, fc1, fc2) and each Norm module (pre-RMSNorm, post-RMSNorm for attention and FFN), track:

| Metric | Symbol | Computation | Reported Across | Why |
|--------|--------|-------------|-----------------|-----|
| Relative output error | ||dy||/||y|| | norm(y_hat - y) / norm(y) | Per matrix, per layer, aggregated by type | Theorem 1 prediction target |
| Absolute output MSE | MSE(y, y_hat) | mean((y_hat - y)^2) | Per matrix, per layer | Monotonic in PPL; easier to interpret |
| SNR | SNR(y, y_hat) | 10*log10(var(y)/var(dy)) | Per matrix, per layer | Scale-free; comparable across different matrix sizes |
| Error amplification ratio | ||dy_out||/||dy_in|| | norm(post-quant error) / norm(pre-quant error) | Per module (especially RMSNorm) | Directly tests Theorem 2 (RMSNorm blocking) |
| Condition number | kappa(W) | sigma_max / sigma_min | Per matrix, aggregated by type | Theorem 1's predicted error multiplier |
| Normalized weight error | ||dW||/||W|| | norm(W_quant - W) / norm(W) | Per matrix | Theorem 1's perturbation term |

### Observation Points Per Layer (6+ hooks)

```
Layer ℓ input (hidden states from previous layer)
  |
  v── [HOOK 1] pre-RMSNorm (attn)          → measure ||dy|| before normalization
  |
  RMSNorm_attn
  |
  v── [HOOK 2] post-RMSNorm (attn)         → measure attenuation ratio: ||dy_post||/||dy_pre||
  |
  v── [HOOK 3] into q_proj, k_proj, v_proj → per-matrix ||dy||/||y|| for Q, K, V weights
  |
  Attention
  |
  v── [HOOK 4] before out_proj             → per-matrix ||dy||/||y|| for out_proj weight
  |
  out_proj → residual_add
  |
  v── [HOOK 5] pre-RMSNorm (ffn)           → second RMSNorm entry
  |
  RMSNorm_ffn
  |
  v── [HOOK 6] post-RMSNorm (ffn)          → second attenuation ratio
  |
  v── [HOOK 7] fc1, fc2                    → per-matrix ||dy||/||y|| for FFN weights
  |
  FFN → residual_add
  |
  v── [HOOK 8] post-layer output           → layer-level error after residual
```

---

## Hook Infrastructure Design

### Pattern: The GPTQ-Standard Closure (Verified)

The GPTQ repository (Frantar et al., 2023) and SmoothQuant (Xiao et al., 2023) both use the same closure pattern for per-submodule hooks. This is the verified standard:

```python
# Pattern A: Hook with closure for named storage (for submodule-level capture)
def make_error_hook(name, storage_dict):
    """Create a forward hook that captures pre-quantization output for later comparison."""
    def hook(module, input, output):
        # input is always a tuple; output is passed as-is
        storage_dict[name] = {
            'output': output.detach().clone(),  # CRITICAL: clone to avoid in-place modification
            'input': input[0].detach() if isinstance(input, tuple) else input.detach(),
        }
    return hook

# Registration
storage = {}
handles = []
for name, mod in model.named_modules():
    if isinstance(mod, (nn.Linear, RMSNorm)):
        handles.append(mod.register_forward_hook(make_error_hook(name, storage)))

try:
    model(batch)  # Forward pass captures all activations
    # storage now contains all layer outputs
finally:
    for h in handles:
        h.remove()
```

### Critical Hook Best Practices (from GPTQ codebase + PyTorch docs)

| Practice | Why | Source |
|----------|-----|--------|
| Always `.detach()` output before storing | Prevents building computation graph for activations | PyTorch docs; GPTQ uses `.data` |
| Always `.clone()` stored tensors | Outputs may be modified in-place by subsequent operations (e.g., in-place attention masking) | PyTorch docs; common bug |
| Remove hooks after use with `.remove()` | Hooks persist and fire on every forward call; can cause memory leaks | PyTorch docs |
| `input` is always a tuple | Even for single-input modules; always index with `input[0]` | PyTorch docs (verified); GPTQ uses `inp[0].data` |
| Store hooks in a list for cleanup | `for h in handles: h.remove()` is the standard pattern | GPTQ, SmoothQuant codebases |
| Use `functools.partial` or closures | Avoids mutable default argument bugs; preserves per-submodule state | SmoothQuant uses `partial(stat_input_hook, name=name)` |
| Detach from graph before computing MSE | Use `.data` or `.detach()` to avoid autograd overhead for analysis | GPTQ uses `.data` throughout |

### Pattern B: Two-Pass Capture (for pre-quant vs post-quant comparison)

To measure ||dy||/||y|| per matrix, run TWO forward passes and compare:

```python
def capture_activations(model, batch, hook_filter=None):
    """Run forward pass and capture all Linear outputs."""
    storage = {}
    handles = []
    for name, mod in model.named_modules():
        if hook_filter and not hook_filter(name, mod):
            continue
        if isinstance(mod, nn.Linear):
            handles.append(mod.register_forward_hook(
                make_error_hook(name, storage)
            ))
    
    with torch.no_grad():
        model(batch)
    
    for h in handles:
        h.remove()
    return storage

# Pass 1: FP16 baseline activations
fp16_out = capture_activations(model_fp16, batch)

# Pass 2: Quantized model activations
quantize_weights(model)  # Apply PTQ
quant_out = capture_activations(model_quant, batch)

# Compute per-matrix relative error
for name in fp16_out:
    y = fp16_out[name]['output']   # Original output  
    y_hat = quant_out[name]['output']  # Quantized output
    dy = y_hat - y
    rel_error = dy.norm() / y.norm()
    print(f"{name}: ||dy||/||y|| = {rel_error:.6f}")
```

### Memory Budget for Hook Storage

For the 164M Micro-Gemma-FP model (batch_size=2, seq_len=2048, hidden=768):

| Hook Point | Tensor Shape | Elements | Memory (FP32) |
|------------|-------------|----------|---------------|
| Per Linear output | (2, 2048, 768) | 3,145,728 | 12 MB |
| 48 Linear layers | 48 x 12 MB | — | 576 MB |
| Per RMSNorm output | (2, 2048, 768) | 3,145,728 | 12 MB |
| 24 RMSNorm layers | 24 x 12 MB | — | 288 MB |
| **Total per pass** | — | — | **~864 MB** |

Two passes (pre-quant + post-quant) with simultaneous storage: **~1.7 GB** — acceptable on 24 GB GPU.

**Optimization:** If memory is tight, store only norms, not full tensors:
```python
storage[name] = {'output_norm': output.norm().item()}
```
This drops to negligible memory (a few float values per hook).

---

## Evaluation Protocol (Statistical Rigor)

### Current State of the Field

| Paper | Seeds | Confidence Intervals | Per-Layer Metrics |
|-------|-------|---------------------|-------------------|
| GPTQ (2023) | 1 (implicit) | No | No (only E2E PPL) |
| SmoothQuant (2023) | 1 (seed=42) | No | No |
| LLM.int8 (2022) | 1 | No | No |
| QuIP (2023) | Multiple (for incoherence RNG) | No | Proxy loss only |
| SparseGPT (2023) | 5 (for calibration ablation) | Std reported | Layer-wise MSE (appendix) |
| **This project** | **3+** | **Bootstrap CI for PPL** | **Per-matrix rel error + MSE** |

### Recommended Protocol

**Paired evaluation** (compare across methods on same data):

```
For each PTQ method:
  1. Encode: method, quantizer config, random seed used
  2. For each of 3+ seeds:
     a. Sub-sample calibration data with this seed
     b. Run PTQ calibration (GPTQ, Lloyd-Max, etc.)
     c. Run 100-step evaluation
     d. Record: PPL, per-matrix ||dy||/||y||, per-matrix MSE, layer-level error ratios
  3. Report: mean +/- std across seeds for all metrics
  4. Flag: any metric where std > mean * 0.1 as "high variance"  
```

**Bootstrap confidence intervals for PPL:**

```python
def bootstrap_ppl(losses: torch.Tensor, n_resamples=1000):
    """Compute 95% CI for PPL from per-token losses."""
    # losses: (n_tokens,) tensor of individual token losses
    n = len(losses)
    bootstrap_ppls = []
    for _ in range(n_resamples):
        idx = np.random.choice(n, n, replace=True)
        sample_ppl = torch.exp(losses[idx].mean()).item()
        bootstrap_ppls.append(sample_ppl)
    ci_low = np.percentile(bootstrap_ppls, 2.5)
    ci_high = np.percentile(bootstrap_ppls, 97.5)
    return ci_low, ci_high
```

**For comparing methods:** report the standard approach of mean +/- std across the 3 runs. Paired t-tests within-seed (each seed sees both methods) control for calibration-data variation.

### Data Splitting (Critical)

The statistical validity of all error measurements depends on clean train/val/test splits. As documented in `docs/ANALYSIS.md`, all PTQ calibration data must come from the **train** split, and all evaluation from the **val** split. No overlap.

```
data/real_tiers/
  tier1_c4_train.bin      # calibration data source
  tier1_c4_val.bin        # evaluation data source
  tier2_fineweb_train.bin
  tier2_fineweb_val.bin
  ...
```

Implemented split ratio: 95% train / 5% val per tier.

---

## Alternatives Considered

### Hook Pattern Alternatives

| Approach | Recommended? | Why Not |
|----------|-------------|---------|
| `register_forward_hook` closure pattern | YES | Standard across GPTQ, SmoothQuant, SparseGPT; minimal overhead |
| `torch.fx` symbolic tracing | NO | Overkill for activation capture; adds graph management complexity |
| Monkey-patching `forward()` | NO | More invasive; requires re-wrapping each module; GPTQ paper tried this and moved to hooks |
| Custom wrapper nn.Module subclass | NO | Would require model surgery; breaks HuggingFace compatibility |
| Register hook on all modules (`model.named_modules()`) | YES | Standard; simple filter by `isinstance` check |
| Register hook per-weight-matrix individually | YES (preferred for theorem validation) | Gives named storage keys matching weight matrix identity |

### Metric Alternatives

| Metric | Recommended? | Why |
|--------|-------------|-----|
| ||dy||/||y|| (relative output error) | YES | Direct Theorem 1 metric |
| MSE between pre/post quant outputs | YES | Complimentary; more widely understood |
| SNR | MAYBE | Useful for cross-layer comparison but less standard |
| Cosine similarity between outputs | NO | Insensitive to magnitude; Theorem 1 predicts magnitude change |
| KL divergence of output distributions | NO | Only meaningful at final layer; Theorem 1 bounds L2 norm |
| Perplexity | YES (as secondary) | Standard in field; too far downstream from perturbation |
| Proxy loss tr((Ŵ-W)H(Ŵ-W)^T) | MAYBE | GPTQ/QuIP use this; useful but requires Hessian per matrix |

### Error Measurement Flow Alternatives

| Approach | Recommended? | Why |
|----------|-------------|-----|
| Two-pass: FP16 then quantized, compare hooks | YES | Clean measurement; separate passes avoid interference |
| Single-pass: quantize and compare within one forward | NO | Complicates architecture; need both paths simultaneously |
| Single-pass: only quantized, compare vs. stored baseline | MAYBE | Saves memory but requires storing FP16 activations from prior run |

---

## Statistical Validity Stack

| Concern | Tool/Method | Implementation |
|---------|-------------|----------------|
| Seed management | Python `random` + `torch.manual_seed` + `np.random.seed` | Already partially done (seed=42); extend to seed set {42, 43, 44} |
| Calibration sampling | Random subset of training data with fixed seed | GPTQ standard: 128 x 2048-token segments |
| Evaluation sampling | Fixed evaluation set (no overlap with calibration) | 100 steps from val split (800 sequences) |
| PPL CI | Bootstrap resampling of per-token losses | ~15 LOC; 1000 resamples |
| Method comparison | Paired within-seed, plus mean +/- std across seeds | Standard approach in ML literature |
| Multiple hypothesis correction | Bonferroni or Benjamini-Hochberg if comparing >5 methods | Optional; note in report if used |

---

## Sources

1. **GPTQ codebase (hook pattern standard)** — `gptq.py` and `opt.py` from IST-DASLab/gptq (GitHub): uses `register_forward_hook` closure pattern to capture per-Layer activations for Hessian accumulation. L57-72 of `opt.py` shows the `add_batch` closure pattern. This is the most directly relevant pattern for the project. [HIGH confidence — verified via raw source]
2. **SmoothQuant codebase (hook pattern)** — `smoothquant/calibration.py` from mit-han-lab/smoothquant (GitHub): uses `register_forward_hook` with `functools.partial` to capture activation min/max per Linear layer. [HIGH confidence — verified via raw source]
3. **PyTorch docs — `register_forward_hook` signature** — docs.pytorch.org: `hook(module, input, output)` where `input` is always a tuple, `output` is as-is. Returns `RemovableHandle` with `.remove()`. [HIGH confidence — verified via Context7/docs.pytorch.org]
4. **QuIP paper (per-layer proxy loss)** — Chee et al., 2023, arXiv:2307.13304: reports "proxy loss" = tr((W_hat - W) H (W_hat - W)^T ) averaged per model dimension. Closest thing to per-layer metric in published quantization papers. [MEDIUM confidence — verified via PDF text extraction, Table 14]
5. **SparseGPT paper (layer-wise MSE appendix)** — Frantar & Alistarh, 2023, arXiv:2301.00774: Section A.1 (Appendix) reports "layer-wise squared error of SparseGPT relative to exact reconstruction" per layer type (q, k, v, out, fc1, fc2), Figure 11. Only published LLM quantization paper with per-layer error visualization. [HIGH confidence — verified via PDF text extraction]
6. **SparseGPT seed sensitivity** — Frantar & Alistarh, 2023, Appendix A: "Sensitivity to Random Seeds" reports 5-run result: `13.52 +/- 0.075` for 50% pruning, demonstrating that calibration sensitivity to data sampling is low but measurable. [HIGH confidence — verified via PDF text extraction]
7. **Bootstrap confidence intervals for PPL** — Standard non-parametric technique (Efron & Tibshirani, 1993); used in language modeling literature. No library dependency needed. [HIGH confidence — verified methodology]
8. **GPTQ calibration protocol** — Frantar et al., 2023, arXiv:2210.17323: 128 random 2048-token segments from C4 training data. [HIGH confidence — verified via PDF text extraction, line 1412]

---

## Installation Requirements

The existing codebase already has all core dependencies:

```bash
# Already installed:
# PyTorch >= 2.3, NumPy, HuggingFace Transformers, etc.

# No additional libraries needed for the measurement infrastructure itself.
# Bootstrap CI uses numpy.random.choice (already available).
```

All measurement infrastructure uses standard PyTorch tensor operations and `torch.linalg` — no new dependencies.

---

## Dependencies This Researches

- **`src/experiments/`** — The measurement protocol will be implemented as a new experiment script. Pattern: `eval_error_propagation.py` or similar.
- **`src/model/transformer.py`** — Hooks fire automatically on any submodule; no model modifications needed.
- **`src/analysis/condition.py`** — Condition number computation feeds into per-matrix analysis.
- **`src/quantization/*.py`** — PTQ methods are invoked once before measurement pass.
- **`src/experiments/training_utils.py`** — `get_dataloader()` must accept `split='train'|'val'` to implement data separation.
