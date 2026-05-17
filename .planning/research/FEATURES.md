# Feature Landscape: Quantization Error Measurement Protocol

**Domain:** Numerical analysis-driven Transformer quantization error measurement
**Researched:** 2026-05-17

## Overview

This document defines what the measurement protocol (MEAS-01 through MEAS-05 in
PROJECT.md) must do. Features are categorized by criticality: table stakes
(missing = protocol is useless), differentiators (makes this protocol uniquely
valuable for testing numerical analysis predictions), and anti-features
(deliberately excluded to maintain focus).

The primary design constraint: **PPL is the wrong metric for testing Theorem 1**
(PROJECT.md, Key Finding). The protocol must measure at the point where the
theory makes predictions -- the Linear layer output -- and then trace how that
error propagates through the rest of the forward pass.

---

## Table Stakes

Features without which the protocol fails its research purpose.

### TS-1: Per-Matrix Output Error Measurement

**Feature:** For every quantizable weight matrix \(W\) in the model, measure the
relative output error \(\|\hat{y} - y\| / \|y\|\) when \(W\) is quantized to \(W
+ \delta W\) while all other weights remain at FP16.

**Why required:** This is the direct test of Theorem 1. The theory predicts
\(\|\delta y\| / \|y\| \leq \kappa(W) \cdot \|\delta W\| / \|W\|\). Without this
measurement, the protocol cannot validate or falsify the theoretical bound.

**Implementation points (12 layers x 7 matrices = 84 matrices):**
- Attention: `q_proj`, `k_proj`, `v_proj`, `o_proj` (4 per layer, 48 total)
- FFN: `gate_proj`, `up_proj`, `down_proj` (3 per layer, 36 total)
- Embedding: `embed_tokens` (1)
- Output: `lm_head` (1, tied to embed_tokens -- measure once)
- Per-layer embeddings: 12 small (64-dim) embeddings

**Granularity:** Per-tensor (one scalar per matrix). This is the correct level
for \(\kappa(W)\) correlation because \(\kappa\) is a matrix-level property.

**Complexity:** Medium. Requires forward hooks on every Linear layer, or a
two-pass approach: clean forward pass to capture reference activations, then
quantized forward pass to capture perturbed activations.

**Dependency:** Requires FP16 baseline checkpoint (already exists at
`checkpoints/scaled_fp16_baseline/model.pt`).

---

### TS-2: Condition Number Correlation

**Feature:** For each matrix measured in TS-1, compute the Pearson correlation
\(r(\kappa(W), \|\delta y\| / \|y\|)\) and the Spearman rank correlation to
check if the relationship is monotonic but nonlinear.

**Why required:** This is the central numerical analysis claim -- that
ill-conditioned weight matrices amplify quantization error more. The existing
code (`validate_theory.py`) already computes \(\kappa(W)\) per layer, but
correlates it with **PPL degradation** not with **per-matrix output error**.
The protocol must correlate at the correct level.

**Sub-feature - Aggregation layer:** After per-matrix measurement, average
within each layer (4 attention + 3 FFN = 7 matrices) and repeat the correlation
at layer level. Compare: does aggregation wash out signal?

**Complexity:** Low. \(\kappa(W)\) computation via exact SVD already exists in
`condition.py:28-38`. Matrices are \(\leq 832 \times 832\), SVD costs <1ms each.

---

### TS-3: Full Error Propagation Trace

**Feature:** Insert measurement hooks at every structural boundary in each
Transformer layer to trace how quantization error flows from weight matrix
through normalization, attention, FFN, and residual connections.

**Measurement points per layer (10 points):**

| # | Point | Label | What It Measures |
|---|-------|-------|------------------|
| 1 | Layer input | `layer_{i}_input` | Reference: hidden states entering layer |
| 2 | Pre-input-norm | `layer_{i}_pre_input_norm` | Before input_norm (same as layer input) |
| 3 | Post-input-norm | `layer_{i}_post_input_norm` | After input_norm, before attention |
| 4 | Attention raw output | `layer_{i}_attn_raw` | After `o_proj`, before residual add |
| 5 | Post-attention residual | `layer_{i}_post_attn_residual` | `residual + attn_output` |
| 6 | Pre-post-attn-norm | `layer_{i}_pre_post_attn_norm` | Before `post_attn_norm` (same as #5) |
| 7 | Post-post-attn-norm | `layer_{i}_post_post_attn_norm` | After `post_attn_norm`, before FFN |
| 8 | FFN raw output | `layer_{i}_ffn_raw` | After `down_proj`, before residual add |
| 9 | Post-FFN residual | `layer_{i}_post_ffn` | `residual + ffn_output` (layer output) |
| 10 | Layer output | `layer_{i}_output` | Same as #9; explicit alias |

**Global points (5):**

| # | Point | Label |
|---|-------|-------|
| 11 | Embedding output | `embed_output` |
| 12 | Final norm input | `pre_final_norm` |
| 13 | Final norm output | `post_final_norm` |
| 14 | Logits (lm_head output) | `logits` |

**Total: 125 measurement points per forward pass** (12 layers x 10 + 5 global).

**Why required:** Without tracing, the protocol cannot answer "where does the
error go?" The key finding from Phase 1 was that RMSNorm blocks error
propagation -- but that was measured at PPL level, not at the level of signal
norm through the layer stack. This trace makes the mechanism directly visible.

**Complexity:** Medium-high. Cannot use simple PyTorch `register_forward_hook`
on all points because the hook API captures module input/output, but many
points (residual adds, pre-norm) are inside `forward()` methods. Two approaches:

- **Approach A (recommended):** Instrument `TransformerLayer.forward()` with
  explicit capture calls into a shared `MeasurementContext` object. Does not
  require modifying external code, just adding capture lines to the forward
  method.
- **Approach B (non-invasive):** Wrap each submodule with pre/post hooks and
  also insert dummy modules at residual points. More complex, fragile, and
  harder to validate.

---

### TS-4: Data Export for Downstream Analysis

**Feature:** Write all measurement results to structured data files that can be
directly loaded by analysis tools (Pandas, NumPy, matplotlib, seaborn) without
manual reformatting.

**Output format:** Two files per experiment run:

1. **`{experiment_name}_measurements.parquet`** (or equivalently structured JSON):
   - One row per (experiment_config, layer_idx, measurement_point, metric)
   - Columns: `experiment`, `layer_idx`, `layer_type`, `point_name`, `matrix_name`,
     `kappa`, `||delta_y||/||y||`, `||delta_W||/||W||`, `error_predicted`,
     `rmsnorm_ratio`, `n_tokens`, `seed`
   - Parquet preferred for cross-platform, columnar, with compression

2. **`{experiment_name}_correlations.json`**:
   - Summary statistics: Pearson/Spearman r for each correlation examined
   - Per-matrix detailed results for scatter plots
   - PPL values (FP16 baseline, quantized, delta)

**Why required:** The protocol generates thousands of data points. JSON-only
output creates unmanageably large files and requires custom parsing for every
analysis script. A columnar format with a consistent schema enables rapid
iterative analysis.

**Complexity:** Low. Pandas `to_parquet()` handles serialization. The schema
design is the main work.

---

### TS-5: Same-Input Paired Comparison

**Feature:** The clean (FP16) and quantized forward passes must process exactly
the same input batch with the same random seed, ensuring that output error
measurements reflect only the weight perturbation, not input sampling noise.

**Why required:** A batch of 8 sequences x 512 tokens has ~4000 tokens. Token
sampling variance between runs can introduce output norm differences on the
same order as quantization error for low-sensitivity layers. Paired comparison
eliminates this confound.

**Implementation:** Load one batch, run two forward passes:
```python
batch = next(iter(loader))
with torch.no_grad():
    y_clean = model_clean(batch)
    y_quant = model_quant(batch)
errors = {point: (y_quant[point] - y_clean[point]).norm(dim=-1)
          / y_clean[point].norm(dim=-1)}
```

**Complexity:** Low. Standard paired experimental design.

---

### TS-6: Multi-Seed Statistical Rigor

**Feature:** Repeat the full measurement protocol across 3 random seeds (42, 73,
99) and report mean +/- std for all error metrics. Include pairwise effect
sizes (Cohen's d) for comparisons between strategies.

**Why required:** Per the ANALYSIS.md design review (Issue 3: no variance
estimates), all prior results are single-run point estimates. The measurement
protocol is an opportunity to establish statistical rigor from the start.

**Complexity:** Low (repeated runs) to medium (effect size computation). Three
seeds x 100 eval steps = 300 total evaluation steps, which is ~3 hours on GPU.

---

## Differentiators

Features that make this protocol uniquely suited to testing numerical analysis
theory in Transformers.

### D-1: RMSNorm Attenuation Profile at Every Normalization Point

**Feature:** At every RMSNorm in the model (input_norm, post_attn_norm, q_norm,
k_norm, final_norm), measure two quantities:

1. **Norm ratio**: \(\|\delta_{\text{post}}\| / \|\delta_{\text{pre}}\|\) --
   how much the error magnitude shrinks (or grows) through normalization.
2. **Direction change**: Cosine similarity
   \(\cos(\theta) = \delta_{\text{pre}}^T \delta_{\text{post}} / (\|\delta_{\text{pre}}\| \cdot \|\delta_{\text{post}}\|)\)
   -- whether the error structure is transformed.

**Why differentiating:** The existing analysis (ANALYSIS.md, Theorem 2) proves
that RMSNorm projects error onto the component orthogonal to the signal,
preventing it from propagating to the next layer's RMSNorm. But this was never
directly measured. The QK-norms (q_norm, k_norm) are also RMSNorm -- do they
contribute to error blocking too? Measuring attenuation at every point answers
this.

**Total RMSNorm points per layer:**
- `input_norm` (pre-attention): 2 points (pre, post) = 1 ratio
- `q_norm`, `k_norm` (per-head QK-norm): 2 points each = 2 ratios (note: per-head,
  so 12 x 64-dim vectors = 768-dim total)
- `post_attn_norm` (pre-FFN): 2 points = 1 ratio
- Global: `final_norm` = 1 ratio
Total: ~5 ratios per layer x 12 = 60, plus 1 global = 61 RMSNorm attenuation
measurements per forward pass.

**Complexity:** Medium. The q_norm and k_norm operate on reshaped tensors
(batch, seq, heads, head_dim). Must capture before and after with correct shape
handling.

---

### D-2: Per-Linear-Layer Error Waterfall

**Feature:** For each layer, produce a waterfall chart showing error magnitude
at each of the 10 measurement points, with color encoding for amplification
(red) vs attenuation (blue) relative to the previous point.

**Why differentiating:** A single waterfall chart for one layer makes the
mechanism visible at a glance:
- Weight quantization error enters at point 4 (attention output) and point 8
  (FFN output).
- RMSNorm at point 3 should *reduce* any input error (attenuation).
- Residual add at points 5 and 9 should mix error with clean residual --
  does it dilute or preserve?

This is the primary diagnostic for "where does the error go?" and is what the
current PPL-based analysis cannot show.

**Data output:** Per-layer, per-point mean and std error, organized for stacked
bar (waterfall) visualization.

**Complexity:** Low. The trace from TS-3 directly feeds this.

---

### D-3: Per-Matrix Error Decomposition (Not Just Per-Layer)

**Feature:** Measure \(\|\delta y\| / \|y\|\) **individually** for each of the 84
Linear weight matrices (12 layers x 7 matrices). Do not aggregate to layer level
before correlation analysis.

**Why differentiating:** The existing code (`sensitivity.py`) aggregates \(\kappa\)
across all matrices in a layer using `avg_kappa`. But \(\kappa(W)\) varies by
1000x between `q_proj` (\(\kappa \sim 100\)) and `o_proj` (\(\kappa \sim 16000\))
within the same layer (PROJECT.md, Key Decision). Aggregating washes out the
very signal the theory predicts.

Per-matrix measurement gives 84 data points for the correlation analysis instead
of 12, providing statistical power to detect relationships even if individual
effect sizes are small.

**Complexity:** Medium. Requires quantizing one matrix at a time while keeping
all other weights at FP16, then running a forward pass. 84 forward passes x 100
eval steps = 8400 steps, which is expensive. **Optimization:** Measure all 84
matrices in a single forward pass by using paired clean/quant models and
capturing intermediate activations at each Linear output via hooks. This
reduces to 1-2 forward passes total.

---

### D-4: Error Distribution Shape Analysis (Beyond Mean)

**Feature:** For each measurement point, report not just the mean relative error
but the full distribution across tokens and batch positions:
- Mean (standard)
- Median (robust to outliers)
- Standard deviation
- 90th and 99th percentiles
- Max (worst-case)
- Skewness (is error concentrated in a few tokens?)
- Outlier fraction: proportion of tokens with error > 3\(\sigma\) above mean

**Why differentiating:** Mean relative error can look small while individual
tokens experience catastrophic error. The theoretical bound
\(\|\delta y\| / \|y\| \leq \kappa \cdot \|\delta W\| / \|W\|\) is a
**per-instance** bound, not a bound on the mean. If the bound is tight but only
affects 1% of tokens, the mean hides this. PPL is sensitive to outlier tokens
with very high loss, so distribution shape connects directly to the metric that
matters downstream.

**Implementation:** Compute error per token (batch x seq_len), then compute
distribution statistics across the batch+seq dimensions.

**Complexity:** Medium. Requires per-token storage for each measurement point
instead of scalar reduction. Memory: 8 batch x 512 seq x 768 hidden x 4 bytes =
12MB per point, x 125 points = 1.5GB for full trace. Must aggregate on-the-fly
with streaming statistics to avoid OOM.

---

### D-5: Residual Stream Error Tracking

**Feature:** Separate the error entering the residual stream from attention vs.
FFN at each layer. Track the residual stream's cumulative error as a running
total through the network.

**Why differentiating:** The Transformer architecture is:
\[
\text{layer output}_\ell = \text{embedding} + \sum_{k=0}^\ell \text{attn}_k + \sum_{k=0}^\ell \text{ffn}_k
\]
Quantization error in layer \(k\)'s attention or FFN adds directly to this
running sum. The residual stream is where error accumulates. Tracking it
separately from the normalization branches reveals:
- Do early-layer errors persist in the residual (additive) or get corrected by
  later layers (subtractive)?
- Does the residual stream's error grow linearly with layer count or sub-linearly?

**Implementation:** After measuring error at each `attn_raw` and `ffn_raw`
point, maintain a running sum for the residual stream. Compare the actual
`layer_{i}_output` error (from paired clean/quant forward) with the sum of
individual errors -- the difference is the interaction effect (error in one
layer affecting later layers' computation).

**Complexity:** Medium. Requires careful propagation of error through the
residual graph. The interaction effect decomposition is novel and requires
two additional forward passes with isolated perturbations.

---

### D-6: Per-Matrix \(\kappa(W)\) Rank Stability Analysis

**Feature:** Across all 84 matrices, compute the rank ordering by \(\kappa(W)\)
and compare with rank ordering by \(\|\delta y\| / \|y\|\). Report the rank
correlation (Spearman) and the fraction of matrices where the theory correctly
predicts "top-10 most sensitive" and "bottom-10 least sensitive."

**Why differentiating:** Pearson correlation is dominated by the high-\(\kappa\)
tail (a few o_proj matrices with \(\kappa\sim 16000\) vs most with \(\kappa<500\)).
Rank correlation and top-k accuracy are more practical metrics: they ask "does
the theory tell us which matrices to worry about?" not "does the theory predict
the exact error magnitude?" This directly informs mixed-precision decisions.

**Complexity:** Low (derived from TS-2 data).

---

## Anti-Features

Features explicitly excluded from the measurement protocol.

### AF-1: Activation Quantization Measurement

**Exclude because:** PROJECT.md explicitly lists activation quantization as out
of scope. The protocol measures error from **weight quantization only**. Adding
activation quantization would double the measurement space (each layer has
multiple activation tensors per forward pass) and confound the weight-perturbation
signal that Theorem 1 predicts.

**What to do instead:** If activation quantization is later added, create a
separate protocol. This protocol is purpose-built for weight quantization error.

---

### AF-2: Per-Head Attention Pattern Error

**Exclude because:** The numerical analysis theory (\(\kappa(W)\) bounds for
linear systems) applies to matrix multiplication, not to attention pattern
computation. Per-head attention pattern changes from quantization are an
interesting downstream effect but are not predicted by the core theory being
tested.

**What to do instead:** Include per-head *output* error (after SDPA, before
`o_proj`) as a measurement point in TS-3, but do not break it into individual
head patterns. The norm across all heads is sufficient.

---

### AF-3: Hardware Timing / Throughput

**Exclude because:** This is a numerical analysis project, not a systems
performance project. Wall-clock time, FLOPS utilization, memory bandwidth,
and GPU kernel launch overhead are irrelevant to the research question of
whether \(\kappa(W)\) predicts quantization error propagation.

**What to do instead:** If deployment metrics are needed later, use a separate
performance profiling tool (e.g., PyTorch Profiler, NVIDIA Nsight). Do not add
timing instrumentation to the measurement protocol.

---

### AF-4: Full Per-Channel Error Heatmap

**Exclude because:** A per-channel heatmap (768 output channels x 84 matrices =
64,512 cells per forward pass) generates an overwhelming volume of data that
lacks a clear theoretical interpretation. Theorem 1 operates at the matrix
level; per-channel analysis would require a per-channel condition number, which
does not exist for the weight matrix as a whole.

**What to do instead:** If channel-level analysis is needed, run a targeted
experiment on a single high-\(\kappa\) matrix with per-channel error reporting.
Do not include in the standard protocol.

---

### AF-5: Training Dynamics / QAT Tracking

**Exclude because:** The protocol measures **PTQ** error on a fixed baseline
model. QAT training dynamics (how error evolves during training with STE
gradients) operate on different timescales (thousands of steps vs. single
forward pass) and involve gradient approximation errors that are not covered by
Theorem 1.

**What to do instead:** If QAT error evolution is studied, it should be a
separate experiment with its own measurement protocol (measuring error at
checkpoint intervals during training).

---

### AF-6: Real-Time Streaming Measurement

**Exclude because:** The protocol operates at batch level on cached `.bin` data
files. There is no streaming data pipeline, no online learning, and no need for
real-time inference metrics. Streaming measurement would add latency-sensitive
logic that complicates the code without contributing to the research goal.

**What to do instead:** Batch measurement is sufficient. The protocol processes
100 evaluation steps (~800 sequences) in minutes, which provides adequate token
count for stable error estimates.

---

### AF-7: Per-Layer PPL as a Primary Metric

**Exclude because:** The central finding driving this protocol is that **PPL is
the wrong metric** for testing Theorem 1. Per-layer PPL (quantize layer i,
measure PPL change) aggregates error through the entire remainder of the model,
obscuring the per-matrix signal. The protocol keeps PPL as a secondary sanity
check (to confirm that error measurements correspond to real degradation) but
does not use PPL correlation as evidence for or against Theorem 1.

**What to do instead:** Use per-matrix output error (TS-1) as the primary metric
for Theorem 1 validation. Use per-layer PPL only as a downstream context
indicator.

---

## Feature Dependencies

```
TS-1 (Per-matrix output error)  ──┬── TS-2 (κ correlation)
                                   ├── D-3 (per-matrix decomposition)
                                   └── D-6 (rank stability)

TS-3 (Full propagation trace)  ────┬── D-1 (RMSNorm attenuation)
                                   ├── D-2 (error waterfall)
                                   ├── D-4 (distribution shape)
                                   └── D-5 (residual tracking)

TS-4 (Data export)  ─────────────── D-1, D-2, D-3, D-5 (all need output format)

TS-5 (Paired comparison)  ───────── TS-1, TS-3 (required for clean/quant alignment)

TS-6 (Multi-seed)  ──────────────── All (statistical rigor applies globally)
```

## MVP Recommendation

The minimum viable protocol that can produce publishable results:

**Phase A (Must implement in order):**

1. **TS-5 + TS-1 + TS-3**: Implement the paired forward-pass measurement with
   hooks at all 125 measurement points. This is the core infrastructure.
2. **TS-4**: Add data export to structured format. Without this, the data cannot
   be analyzed.
3. **TS-2**: Compute per-matrix \(\kappa(W)\) correlation with output error.
   This is the primary deliverable.

**Phase B (High value, add next):**

4. **D-1**: Add RMSNorm attenuation measurement at all normalization points.
   This directly validates Theorem 2 and is the most impactful differentiator.
5. **D-4**: Add distribution shape statistics (percentiles, outlier fraction).
   This connects per-token error distribution to PPL degradation.
6. **TS-6**: Add multi-seed repetition for statistical rigor.

**Phase C (Nice to have):**

7. **D-2**: Waterfall visualization data (derived from D-1 data).
8. **D-5**: Residual stream tracking (requires additional forward passes).
9. **D-3 + D-6**: Per-matrix decomposition and rank stability (derived from TS-1
   data, no extra forward passes needed).

**Defer indefinitely:**
- All anti-features (AF-1 through AF-7)
- Anything that requires modifying training procedures or QAT pipelines

## Sources

- PROJECT.md (project requirements, key findings, active tasks)
- ANALYSIS.md (Theorem 1-4 derivations, experimental design review, RMSNorm analysis)
- `src/model/transformer.py` (TransformerLayer, RMSNorm, Attention, FFN structure)
- `src/analysis/condition.py` (κ(W) computation, SVD, exact vs. surrogate)
- `src/analysis/lipschitz.py` (Lipschitz propagation, cascade factor computation)
- `src/analysis/sensitivity.py` (per-layer sensitivity aggregation)
- `src/experiments/validate_theory.py` (existing P1/P2/P3 validation protocol)
- `src/experiments/validate_rmsnorm.py` (RMSNorm ablation, block ratio measurement)
- Domain knowledge: quantization error propagation measurement in Transformer architectures (GPTQ literature, SmoothQuant, SpQR, QuIP#)
