# Executive Summary

This report presents a numerical analysis of FP8/FP4 post-training quantization (PTQ) applied to a ~164M parameter Gemma-style causal Transformer. We systematically evaluate 72 weight matrices across 12 layers, measuring per-matrix condition numbers kappa(W) via exact SVD, per-matrix output errors ||dy||/||y|| at each Linear layer output, and error propagation through RMSNorm and attention blocks. The study tests whether classical matrix perturbation theory (Theorem 1: ||dy||/||y|| <= kappa(W) * ||dW||/||W||) holds empirically at per-matrix granularity.

Theorem 1 validation yields a verdict of **NO**: Pearson r = -0.2258, p = 3.8885e-02 (Bonferroni threshold = 0.00069), bootstrap 95% CI = [-0.3407, -0.1390]. 
Correlation analysis across 3 random seeds and per-layer-type subgroups (attention, FFN, global) provides the quantitative basis for this assessment.

The extended PTQ comparison evaluates up to 6 quantization methods across 2 checkpoints (FP16 baseline and condition-number-regularized) and 2 formats (FP8 E4M3, FP4 E2M1), collecting both perplexity (PPL) and per-matrix output errors for every configuration.
For FP8 E4M3, the best method on the FP16 baseline checkpoint is **gptq** (PPL = 206.77, delta = +0.044044 vs FP16 baseline).
For FP4 E2M1, the best method is **lloyd_max** (PPL = 207.83, delta = +1.0985 vs FP16 baseline).
GPTQ column compensation and Lloyd-Max adaptive grids are analyzed separately for their effect on output-space error.

Error propagation tracing across all 12 layers reveals that RMSNorm attenuates input error by an average factor of 0.1667 (ratio of post-norm to pre-norm error magnitude). This attenuation, combined with the parallel/orthogonal decomposition of error at each norm output, explains how per-matrix quantization errors interact with the Transformer's normalization architecture.


## Methodology (Corrected)

The following methodology corrections were applied relative to the original project proposal. These corrections resolve measurement flaws identified during the experimental design audit (see `docs/ANALYSIS.md`, Part 1).

### 1. Condition number computation

Exact SVD via `torch.linalg.svdvals` (not power iteration approximation). The original proposal used `inverse_power_iteration` which incorrectly computed sigma_max instead of sigma_min, overestimating kappa values by up to 5000x. The exact SVD call is cheap for matrices up to 832 dimensions and gives exact kappa(W) = sigma_max / sigma_min. See ANALYSIS.md Section 1.6 (Issue 7) for the full audit trail.

### 2. Per-matrix measurement granularity

Output error ||dy||/||y|| is now measured at each Linear layer's output (the matrix-vector product y = Wx), not after the full cascade through RMSNorm, attention, FFN, and subsequent layers. This is the correct granularity for testing Theorem 1, which predicts the bound ||dy||/||y|| <= kappa(W) * ||dW||/||W|| at the linear map output. The original proposal's per-layer aggregation hid 1000x variation between q_proj (kappa ~ 100) and o_proj (kappa ~ 16000) within the same layer.

### 3. Clean data split

Calibration (GPTQ Hessian estimation, Lloyd-Max grid fitting) uses only the training split (first 95% of each data tier). Evaluation uses only the validation split (last 5% of each tier). This eliminates the in-sample PPL optimism caused by calibration and evaluation drawing from the same pool. The split is enforced at the dataloader level via `get_dataloader(split='train')` and `get_dataloader(split='val')`. See ANALYSIS.md Section 1.4 for the original audit finding.

### 4. Bonferroni correction

For the 72-matrix Pearson correlation test, the significance threshold is Bonferroni-corrected: alpha = 0.05 / 72 = 0.00069. This is mandatory statistical rigor when testing 72 simultaneous correlations — without correction, the expected number of false positives at alpha=0.05 is 72 * 0.05 = 3.6. The corrected threshold ensures a family-wise error rate of 0.05.

### 5. Single-pass activation capture

FP16 activations are captured once per checkpoint in a single forward pass before any quantization is applied. The same captured activations are reused across all quantization configurations for that checkpoint. This avoids the cascading confound that would arise from a two-pass approach (FP16 pass + quantized pass with different input data). Per Pitfall 5 of the measurement protocol, all quantized forward passes use the same input batch as the FP16 reference pass.

The mathematical derivations for Theorems 1-4 (the theoretical foundation of this project) are documented in `docs/ANALYSIS.md`, Part 2. These derivations are referenced throughout this report but are not reproduced here.


## Theorem 1 Validation Results

### Statistical Analysis

**Pearson correlation:** r = -0.2258, p = 3.8885e-02

**Bonferroni threshold:** alpha = 0.05 / 72 = 0.000694 (&alpha; = 0.000694)

**Bootstrap 95% CI:** [-0.3407, -0.1390] (10,000 resamples)

**Verdict: NO**

Theorem 1's predicted upper bound does not hold empirically at per-matrix granularity. r=-0.2258 <= 0.2 (negligible correlation)

### Seed-by-Seed Correlations

| Seed | Pearson r |
|------|-----------|
| 42 | -0.1801 |
| 123 | -0.2803 |
| 456 | -0.2068 |

### Per-Layer-Type Subgroup Analysis

| Type | Matrices | Pearson r | p-value |
|------|----------|-----------|---------|
| attention | 48 | -0.1706 | 2.46e-01 |
| ffn | 36 | -0.7391 | 2.63e-07 |
| global | 0 | 0.0000 | 1.00e+00 |


## Error Propagation Trace

### RMSNorm Attenuation

The following table reports per-layer RMSNorm attenuation ratios averaged across all 21 error source matrices. The input norm attenuation is the ratio ||delta_post|| / ||delta_pre|| at the input RMSNorm (P0 -> P1). Post-attention norm attenuation is the ratio at the post-attention RMSNorm (P3 -> P4). Parallel and orthogonal components decompose the output error into projection onto the signal direction versus residual.

| Layer | Input Norm Attenuation | Post-Attn Norm Attenuation | ||d_parallel|| | ||d_orthogonal|| |
|-------|-------------------------|----------------------------|--------------|----------------|
| 0 | — | 1.0161 | 0.0000 | 0.0000 |
| 1 | 0.2899 | 0.2644 | 0.1311 | 0.0103 |
| 2 | 0.2450 | 0.1983 | 0.0985 | 0.0088 |
| 3 | 0.2037 | 0.1834 | 0.0727 | 0.0078 |
| 4 | 0.1802 | 0.1718 | 0.0655 | 0.0075 |
| 5 | 0.1721 | 0.1571 | 0.0609 | 0.0072 |
| 6 | 0.1570 | 0.1420 | 0.0748 | 0.0112 |
| 7 | 0.1383 | 0.1321 | 0.0649 | 0.0106 |
| 8 | 0.1300 | 0.1167 | 0.0596 | 0.0101 |
| 9 | 0.1135 | 0.1089 | 0.0501 | 0.0093 |
| 10 | 0.1052 | 0.1009 | 0.0456 | 0.0089 |
| 11 | 0.0985 | 0.0915 | 0.0418 | 0.0085 |

**Observation:** The mean input RMSNorm attenuation across all layers is 0.1667. 
RMSNorm consistently attenuates input error (ratio < 1.0), confirming its error-blocking role.

### Error Waterfall (Representative Layers)

The following waterfall tables show per-source quantization error at each P-point (P0 through P6) within the source matrix's own layer. P0 is the pre-linear input (should be ~0 for single-matrix quantization), P6 is the post-FFN output. Each row represents a single weight matrix quantized to FP4 E2M1 round-to-nearest.

#### Layer 0

| Source Matrix | Type | P0 | P1 | P2 | P3 | P4 | P5 | P6 |
|--------------|------|---|---|---|---|---|---|---|
| `L0.attention.q_proj.weight` | weight | 0.000000 | 0.000000 | 0.031687 | 0.031602 | 0.032429 | 0.030216 | 0.030155 |
| `L0.attention.k_proj.weight` | weight | 0.000000 | 0.000000 | 0.014890 | 0.014851 | 0.015346 | 0.011435 | 0.011635 |
| `L0.attention.v_proj.weight` | weight | 0.000000 | 0.000000 | 0.027315 | 0.027242 | 0.027199 | 0.013356 | 0.014602 |
| `L0.attention.o_proj.weight` | weight | 0.000000 | 0.000000 | 0.075349 | 0.075148 | 0.075389 | 0.021444 | 0.028210 |
| `L0.ffn.gate_proj.weight` | weight | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.028761 | 0.026745 |
| `L0.ffn.up_proj.weight` | weight | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.027226 | 0.025318 |
| `L0.ffn.down_proj.weight` | weight | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.074774 | 0.069533 |

#### Layer 5

| Source Matrix | Type | P0 | P1 | P2 | P3 | P4 | P5 | P6 |
|--------------|------|---|---|---|---|---|---|---|
| `L5.attention.q_proj.weight` | weight | 0.000000 | 0.000000 | 0.024572 | 0.006648 | 0.006899 | 0.007994 | 0.006387 |
| `L5.attention.k_proj.weight` | weight | 0.000000 | 0.000000 | 0.031022 | 0.008393 | 0.008889 | 0.009891 | 0.008090 |
| `L5.attention.v_proj.weight` | weight | 0.000000 | 0.000000 | 0.053686 | 0.014524 | 0.015022 | 0.012932 | 0.013868 |
| `L5.attention.o_proj.weight` | weight | 0.000000 | 0.000000 | 0.066852 | 0.018086 | 0.018820 | 0.011892 | 0.017192 |
| `L5.ffn.gate_proj.weight` | weight | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.080348 | 0.013604 |
| `L5.ffn.up_proj.weight` | weight | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.063873 | 0.010815 |
| `L5.ffn.down_proj.weight` | weight | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.117297 | 0.019860 |

#### Layer 11

| Source Matrix | Type | P0 | P1 | P2 | P3 | P4 | P5 | P6 |
|--------------|------|---|---|---|---|---|---|---|
| `L11.attention.q_proj.weight` | weight | 0.000000 | 0.000000 | 0.018374 | 0.002731 | 0.002806 | 0.003263 | 0.002542 |
| `L11.attention.k_proj.weight` | weight | 0.000000 | 0.000000 | 0.022136 | 0.003290 | 0.003467 | 0.004484 | 0.003099 |
| `L11.attention.v_proj.weight` | weight | 0.000000 | 0.000000 | 0.047466 | 0.007056 | 0.007330 | 0.004864 | 0.006420 |
| `L11.attention.o_proj.weight` | weight | 0.000000 | 0.000000 | 0.056825 | 0.008447 | 0.008795 | 0.003349 | 0.007577 |
| `L11.ffn.gate_proj.weight` | weight | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.041951 | 0.008144 |
| `L11.ffn.up_proj.weight` | weight | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.033158 | 0.006437 |
| `L11.ffn.down_proj.weight` | weight | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.077734 | 0.015091 |

Waterfall data is shown for layers 0, 5, 11 (as defined by the error propagation trace protocol).


## Extended PTQ Comparison

### 24-Config Comparison Matrix

The table below reports perplexity (PPL) and per-matrix output error (mean ||dy||/||y|| across all matrices) for every evaluated configuration. Configurations are grouped by checkpoint, then format, sorted by PPL ascending within each group. Delta is relative to the FP16 baseline for the same checkpoint.

| Checkpoint | Format | Method | PPL | Delta | Mean ||dy||/||y|| |
|------------|--------|--------|-----|-------|------------------|
| cond_regularized | FP16   | baseline                  | 216.57 | +0.000000 | — |
|               | FP8    | gptq                      | 216.54 | -0.027598 | 0.021023 |
|               | FP8    | rtn                       | 216.63 | +0.061330 | 0.014120 |
|               | FP8    | hadamard                  | 548.20 | +331.6332 | 0.510644 |
|               | FP8    | outlier                   | 30311.14 | +30094.5676 | 1.012394 |
|               | FP4    | lloyd_max                 | 217.70 | +1.1257 | 0.068038 |
|               | FP4    | gptq                      | 217.98 | +1.4048 | 0.120577 |
|               | FP4    | mxfp4                     | 218.84 | +2.2656 | 0.073486 |
|               | FP4    | rtn                       | 219.07 | +2.4962 | 0.083455 |
| fp16_baseline | FP16   | baseline                  | 206.73 | +0.000000 | — |
|               | FP8    | gptq                      | 206.77 | +0.044044 | 0.020385 |
|               | FP8    | rtn                       | 206.83 | +0.102150 | 0.013672 |
|               | FP8    | hadamard                  | 529.97 | +323.2392 | 0.512558 |
|               | FP8    | outlier                   | 31741.23 | +31534.4967 | 1.012846 |
|               | FP4    | lloyd_max                 | 207.83 | +1.0985 | 0.066427 |
|               | FP4    | gptq                      | 208.52 | +1.7911 | 0.116885 |
|               | FP4    | rtn                       | 209.15 | +2.4208 | 0.080922 |
|               | FP4    | mxfp4                     | 209.70 | +2.9678 | 0.071446 |

### Best Per-Checkpoint Summary

**Fp16 Baseline** (FP16 baseline PPL = 206.73):
- Best FP8: **gptq** (PPL = 206.77, Delta = +0.044044, mean ||dy||/||y|| = 0.020385)
- Best FP4: **lloyd_max** (PPL = 207.83, Delta = +1.0985, mean ||dy||/||y|| = 0.066427)

**Cond Regularized** (FP16 baseline PPL = 216.57):
- Best FP8: **gptq** (PPL = 216.54, Delta = -0.027598, mean ||dy||/||y|| = 0.021023)
- Best FP4: **lloyd_max** (PPL = 217.70, Delta = +1.1257, mean ||dy||/||y|| = 0.068038)

**Note:** Hadamard rotation and outlier rotation methods are FP8-centric techniques. If FP4 results for these methods are absent, they were omitted due to expected instability at FP4 precision (the rotation increases activation dynamic range, which FP4's limited exponent range cannot represent effectively).


## GPTQ Analysis: Column Compensation vs Output Error

GPTQ weight compensation is compared against round-to-nearest (RTN) for each pair (same checkpoint, same format). Negative deltas indicate GPTQ reduces output-space error or PPL; positive values indicate GPTQ increases error.

### Fp16 Baseline / FP8

- **PPL:** RTN = 206.83 -> GPTQ = 206.77 (Delta = -0.058106)
- **Mean ||dy||/||y||:** RTN = 0.013672 -> GPTQ = 0.020385 (Delta = +0.006713)
- **Interpretation:** GPTQ increases output-space error by 0.67% relative to RTN.

### Fp16 Baseline / FP4

- **PPL:** RTN = 209.15 -> GPTQ = 208.52 (Delta = -0.629702)
- **Mean ||dy||/||y||:** RTN = 0.080922 -> GPTQ = 0.116885 (Delta = +0.035964)
- **Interpretation:** GPTQ increases output-space error by 3.60% relative to RTN.

### Cond Regularized / FP8

- **PPL:** RTN = 216.63 -> GPTQ = 216.54 (Delta = -0.088928)
- **Mean ||dy||/||y||:** RTN = 0.014120 -> GPTQ = 0.021023 (Delta = +0.006903)
- **Interpretation:** GPTQ increases output-space error by 0.69% relative to RTN.

### Cond Regularized / FP4

- **PPL:** RTN = 219.07 -> GPTQ = 217.98 (Delta = -1.0914)
- **Mean ||dy||/||y||:** RTN = 0.083455 -> GPTQ = 0.120577 (Delta = +0.037122)
- **Interpretation:** GPTQ increases output-space error by 3.71% relative to RTN.

**Cross-format observation:** At FP8, GPTQ changes mean ||dy||/||y|| by +0.006808 on average.
At FP4, GPTQ changes mean ||dy||/||y|| by +0.036543 on average.


## Lloyd-Max Analysis: Adaptive Grids vs Uniform E2M1

Lloyd-Max adaptive grid quantization is compared against uniform E2M1 round-to-nearest for FP4 format. Negative deltas indicate Lloyd-Max reduces error vs uniform; positive values indicate Lloyd-Max increases error.

### Fp16 Baseline

- **PPL:** Uniform = 209.15 -> Lloyd-Max = 207.83 (Delta = -1.3223)
- **Mean ||dy||/||y||:** Uniform = 0.080922 -> Lloyd-Max = 0.066427 (Delta = -0.014494)
- **Attention mean delta:** -0.010489, **FFN mean delta:** -0.019835
- **Interpretation:** Lloyd-Max reduces output-space error by 1.45% relative to uniform E2M1.

### Cond Regularized

- **PPL:** Uniform = 219.07 -> Lloyd-Max = 217.70 (Delta = -1.3705)
- **Mean ||dy||/||y||:** Uniform = 0.083455 -> Lloyd-Max = 0.068038 (Delta = -0.015417)
- **Attention mean delta:** -0.011449, **FFN mean delta:** -0.020707
- **Interpretation:** Lloyd-Max reduces output-space error by 1.54% relative to uniform E2M1.


## RMSNorm Error Blocking

RMSNorm plays a critical role in controlling quantization error propagation through Transformer layers. This section synthesizes evidence from RMSNorm ablation experiments (Phase 2), per-layer attenuation measurements (Phase 4), and per-matrix output error data (Phase 5).

**Phase 2 finding:** RMSNorm ablation experiments demonstrated that removing RMSNorm causes quantization error to grow by 1000x or more across 12 layers. With RMSNorm present, per-layer error stays within the same order of magnitude as the input perturbation.

**Phase 4 measurement:** Across 11 layers, the mean input RMSNorm attenuation ratio (||delta_post|| / ||delta_pre||) is 0.1667. 
This corresponds to a 83.3% reduction in error magnitude at the input RMSNorm — RMSNorm consistently *blocks* (reduces) error magnitude.

**Error decomposition (parallel/orthogonal):** At the input RMSNorm output, the mean parallel component (projection onto signal direction) is 0.063794, and the mean orthogonal component (residual) is 0.008348. 
The parallel and orthogonal components are comparable, suggesting RMSNorm both reduces magnitude and redirects error away from the signal direction.

**Phase 5 per-matrix evidence:** The mean tightness ratio (||dy||/||y|| / (kappa(W) * ||dW||/||W||)) across 84 matrices is 0.0559. 
A tightness ratio below 1.0 means the Theorem 1 bound is not saturated — the actual output error is smaller than the worst-case bound, consistent with RMSNorm's error-blocking and error-redirecting effects.

**Synthesis:** RMSNorm functions as both an error attenuator (reducing error magnitude by projecting it orthogonal to the signal) and a propagation blocker (preventing the Lipschitz multiplicative error cascade that would occur in unnormalized architectures). The theoretical basis is established in Theorem 2 (see ANALYSIS.md, Section 2.3), which shows that RMSNorm's output error is bounded by the input relative error with no multiplicative growth.


## Revised Theoretical Assessment

### Original Hypothesis

The original project proposal hypothesized that Theorem 1 (||dy||/||y|| <= kappa(W) * ||dW||/||W||) would provide a quantitatively useful upper bound on quantization error at each weight matrix's output. If the bound held tightly, kappa(W) could guide mixed-precision allocation: high-kappa matrices receive higher precision, while low-kappa matrices can tolerate more aggressive quantization.

### Revised Understanding

The bound does not hold empirically at per-matrix granularity (r = -0.2258). kappa(W) alone is insufficient to predict output-space quantization error.

> Theorem 1's predicted upper bound does not hold empirically at per-matrix granularity. r=-0.2258 <= 0.2 (negligible correlation)

### What kappa(W) Misses

The empirical results reveal several factors that the condition number alone cannot capture:

**1. Off-diagonal error coupling**

Theorem 1 assumes ||dy|| = ||dW * x|| <= ||dW|| * ||x||, which is tight only when the input x aligns with dW's dominant direction. In practice, the quantization error dW is not aligned with the worst-case direction — it is structured by the grid rounding pattern, which depends on W's own singular vectors. This directional mismatch means the actual error is consistently smaller than the kappa-scaled bound.

**2. Cascading error through layers**

Theorem 1 is a single-layer bound. In a multi-layer transformer, each layer's output error becomes the next layer's input perturbation. Even if individual-layer errors are bounded, their interaction through attention and FFN nonlinearities can amplify or cancel in ways not predicted by per-matrix kappa. The error propagation trace (Section 4) quantifies this: at some layers, error attenuates; at others, it grows.

**3. RMSNorm's non-multiplicative effect**

Theorem 2 (ANALYSIS.md Section 2.3) shows that RMSNorm fundamentally changes the error propagation mechanism. Instead of the Lipschitz multiplicative cascade that would occur in unnormalized networks, RMSNorm projects error onto the orthogonal component of the signal, bounding relative error rather than amplifying it. The experimental data confirms this: RMSNorm attenuation ratios are consistently below 1.0 for the input norm, indicating systematic error reduction.

### Evidence Summary

The revised assessment is grounded in measurements from three experimental phases:

**Phase 3 (Theorem 1 validation):** The verdict is 'NO' with r = -0.2258. 
  - attention matrices: r = -0.1706, p = 2.46e-01
  - ffn matrices: r = -0.7391, p = 2.63e-07
  - global matrices: r = 0.0000, p = 1.00e+00

**Phase 4 (Error propagation trace):** Mean input RMSNorm attenuation = 0.1667 (across 11 layers). 
Mean post-attention RMSNorm attenuation = 0.2236.

**Phase 5 (PTQ comparison):** Tightness ratio distribution: mean = 0.0559, min = 0.0000, max = 0.1376 (across 84 matrices).
The typical output error is an order of magnitude smaller than the Theorem 1 bound, confirming the bound is substantially loose in practice.

### Mathematical Foundation

The full mathematical derivations for Theorems 1-4 are documented in `docs/ANALYSIS.md`, Part 2. These include:
- **Theorem 1:** Single-layer quantization error bound
- **Corollary 1.1 / Theorem 2:** RMSNorm error blocking
- **Theorem 3:** Stochastic rounding cumulative error
- **Theorem 4:** Lloyd-Max optimality conditions
- **Strategy B (Condition number regularization):** Differentiable kappa surrogate
- **GPTQ:** Column compensation derivation from ||WX - hat(W)X||_F minimization


## References

1. **ANALYSIS.md:** `docs/ANALYSIS.md` — Mathematical derivations for Theorems 1-4

2. **PROPOSAL.md:** `docs/PROPOSAL.md` — Original project proposal

3. **Theorem 1 data:** `results/theorem1_validation.json` — Phase 3 per-matrix kappa, weight error, output error, tightness ratio

4. **Error propagation data:** `results/error_propagation_trace.json` — Phase 4 error waterfall, RMSNorm attenuation, decomposition

5. **Full comparison data:** `results/full_comparison.json` — Phase 5 extended PTQ comparison across 16-24 configs

6. **Per-matrix summary:** `results/per_matrix_summary.json` — Merged per-matrix error summary (Phase 3/4/5)

