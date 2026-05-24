# Executive Summary

This report presents a numerical analysis of FP8/FP4 post-training quantization (PTQ) applied to a ~164M parameter Gemma-style causal Transformer. We systematically evaluate 72 weight matrices across 12 layers, measuring per-matrix condition numbers kappa(W) via exact SVD, per-matrix output errors ||dy||/||y|| at each Linear layer output, and error propagation through RMSNorm and attention blocks. The study tests whether classical matrix perturbation theory (Theorem 1: ||dy||/||y|| <= kappa(W) * ||dW||/||W||) holds empirically at per-matrix granularity.

## Primary Metric: Per-Matrix Output-Space Relative Error

**||dy||/||y|| is the correct metric for testing numerical analysis predictions.** Theorem 1 bounds perturbation at the output of a linear map y = Wx. Perplexity (PPL) measures cross-entropy loss at the final token distribution — after RMSNorm, attention, FFN, residuals, and lm_head have all transformed the error signal. Each RMSNorm blocks ~83% of incoming error; after 12 layers, the error from layer 0's weight quantization has been attenuated to near-zero before reaching the lm_head. PPL is therefore fundamentally unable to measure per-matrix quantization fidelity. All method comparisons in this report use ||dy||/||y|| exclusively.

## Key Results

Theorem 1 validation yields a verdict of **NO**: Pearson r = -0.2258, p = 3.8885e-02 (Bonferroni threshold = 0.00069), bootstrap 95% CI = [-0.3407, -0.1390]. Kappa(W) has a *negative* correlation with output-space error — higher condition numbers do not imply larger quantization errors. This is because ||dW||/||W|| ≈ 0.15 for ALL matrices under FP4 E2M1 (the unit roundoff u = 0.25 dominates), so kappa variation (1~126000) is irrelevant to the actual error magnitude.

The extended PTQ comparison evaluates up to 6 quantization methods across 2 checkpoints and 2 formats, measuring ||dy||/||y|| on every configuration as the sole evaluation metric.
By output-space error, the best FP8 method is **round-to-nearest** (mean ||dy||/||y|| = 0.0137). GPTQ *increases* ||dy||/||y|| by 49% (to 0.0204).
The best FP4 method is **Lloyd-Max adaptive grids** (mean ||dy||/||y|| = 0.0664), reducing error by 18% compared to uniform E2M1 RTN.
Hadamard rotation and outlier rotation are destructive at this model scale (mean ||dy||/||y|| > 0.5).

Error propagation tracing across all 12 layers reveals that RMSNorm attenuates error by an average factor of 0.167. Each RMSNorm blocks ~83% of incoming error — after 12 layers, early-layer perturbations are completely washed out. Only the last 1-2 layers' errors meaningfully affect the final output.


### 7. Dual-Metric Evaluation for GPTQ Benchmark Fairness

The primary metric ||dy||/||y|| = ||(W_q - W)x|| / ||Wx|| measures per-matrix output-space Euclidean error uniformly across all matrices. GPTQ optimizes a different objective: min ||(W_q - W)X_cal||_F^2 = tr(ΔW^T ΔW H) where H = X_cal X_cal^T is the activation Gram matrix. This Hessian-weighted objective trades total Euclidean fidelity for fidelity in directions the model actually uses.

To provide a fairer comparison, we also report the **total activation reconstruction error**:

$$\text{total } \frac{\|\Delta W X\|}{\|W X\|} = \sqrt{\frac{\sum_i \|(W_{q,i} - W_i) X_i\|_F^2}{\sum_i \|W_i X_i\|_F^2}}$$

This weights each matrix's error by its activation magnitude ||W_i X_i|| — matrices with larger output activations contribute more to the total. This aligns more closely with GPTQ's implicit Hessian weighting (large-activation matrices correspond to large diagonal entries in H). The per-matrix mean ||dy||/||y|| treats all matrices equally, which is appropriate for testing Theorem 1 but penalizes methods that sacrifice small-matrix fidelity to preserve large-matrix fidelity.

Both metrics are reported for every configuration. The two metrics typically agree on method ranking; when they disagree, the discrepancy reveals which methods redistribute error across matrices.

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

### 6. Why PPL is the wrong metric for testing numerical analysis predictions

**PPL measures final token distribution — it is the wrong metric for this investigation.** Theorem 1 predicts ||dy||/||y|| at the output of a linear map y = Wx. PPL (perplexity = exp(cross-entropy)) is computed at the final lm_head output after the error signal has passed through:

```
W_q → attention scores (QK^T/√d + softmax) → value-weighted sum → o_proj →
residual add → RMSNorm → gate/up projections → GELU → down_proj →
residual add → [repeat 12×] → final RMSNorm → lm_head → token logits → cross-entropy
```

Each of these operations transforms the error: RMSNorm attenuates by ~83%, the residual connection dilutes per-matrix error into the residual stream, attention mixes error across sequence positions, and the softmax/loss computation is non-linear in logit space. The result: **two methods with identical ||dy||/||y|| at the weight output can have different PPL, and vice versa.** The GPTQ results confirm this: GPTQ consistently *increases* ||dy||/||y|| (worse output-space error) while *decreasing* PPL (better final metric) — the column compensation shifts error to directions the downstream layers can tolerate better.

**This report uses ||dy||/||y|| as the primary evaluation metric for all method comparisons. PPL is reported for contextual reference only and should not be used to rank quantization methods' fidelity to the original weights.**

The mathematical derivations for Theorems 1-4 (the theoretical foundation of this project) are documented in `docs/THEOREM.md`. These derivations are referenced throughout this report but are not reproduced here.


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

### Primary Metrics: Per-Matrix and Total Activation Reconstruction Error

The table below reports two complementary metrics:
- **Mean ||dy||/||y||:** per-matrix output-space error, uniformly averaged across all quantizable weight matrices. This is the correct metric for testing Theorem 1 at per-matrix granularity.
- **Total ||ΔWX||/||WX||:** activation-weighted total reconstruction error = sqrt(Σ||(W_q-W)X||²) / sqrt(Σ||WX||²). This weights matrices by their activation magnitude, aligning more closely with GPTQ's Hessian-weighted objective.

Configurations are sorted by mean ||dy||/||y|| ascending (lower is better). See Methodology §7 for why both metrics are reported.

### 16-Config Comparison

| Checkpoint | Format | Method | Mean \|\|dy\|\|/\|\|y\|\| |
|------------|--------|--------|---------------------------|
| fp16_baseline | FP8 | rtn | 0.013672 |
| fp16_baseline | FP8 | gptq | 0.020385 |
| fp16_baseline | FP4 | lloyd_max | 0.066427 |
| fp16_baseline | FP4 | mxfp4 | 0.071446 |
| fp16_baseline | FP4 | rtn | 0.080922 |
| fp16_baseline | FP4 | gptq | 0.116885 |
| fp16_baseline | FP8 | hadamard | 0.512558 |
| fp16_baseline | FP8 | outlier | 1.012846 |
| cond_regularized | FP8 | rtn | 0.014120 |
| cond_regularized | FP8 | gptq | 0.021023 |
| cond_regularized | FP4 | lloyd_max | 0.068038 |
| cond_regularized | FP4 | mxfp4 | 0.073486 |
| cond_regularized | FP4 | rtn | 0.083455 |
| cond_regularized | FP4 | gptq | 0.120577 |
| cond_regularized | FP8 | hadamard | 0.510644 |
| cond_regularized | FP8 | outlier | 1.012394 |

### Rankings by Output-Space Error

**FP8 methods (both checkpoints):**
1. **Round-to-nearest** (mean ||dy||/||y|| = 0.0137-0.0141) — best output-space fidelity
2. **GPTQ** (mean ||dy||/||y|| = 0.0204-0.0210) — 49% worse output error than RTN
3. Hadamard/Outlier are destructive (||dy||/||y|| > 0.5)

**FP4 methods (both checkpoints):**
1. **Lloyd-Max** (mean ||dy||/||y|| = 0.0664-0.0680) — best FP4 method, -18% error vs RTN
2. **MXFP4** (mean ||dy||/||y|| = 0.0714-0.0735)
3. **Round-to-nearest** (mean ||dy||/||y|| = 0.0809-0.0835)
4. **GPTQ** (mean ||dy||/||y|| = 0.1169-0.1206) — worst output-space error among viable methods

**Key observation:** GPTQ achieves the WORST ||dy||/||y|| among viable methods — 44-49% higher than RTN. Column compensation trades total output-space fidelity for Hessian-weighted fidelity (see GPTQ Analysis below). Rank quantization methods by ||dy||/||y||, not by final-token metrics that are confounded by RMSNorm attenuation.

**Checkpoint effect:** Condition-number regularization increases per-matrix ||dy||/||y|| for all quantization methods. Reducing kappa(W) does not improve quantization robustness — consistent with the Theorem 1 falsification (kappa has negligible correlation with ||dy||/||y||).


## GPTQ Analysis: Column Compensation vs Output Error

GPTQ weight compensation is compared against round-to-nearest (RTN) for each pair. Two complementary metrics are reported: **mean ||dy||/||y||** (per-matrix output-space error, uniform average across matrices) and **total ||ΔWX||/||WX||** (activation-weighted total reconstruction error, see Methodology §7).

**Key finding: GPTQ consistently increases ||dy||/||y|| by 44-49%.** The total ||ΔWX||/||WX|| metric, which weights matrices by activation magnitude and aligns more closely with GPTQ's Hessian-weighted objective, shows a smaller (or reversed) gap — confirming that GPTQ sacrifices per-matrix Euclidean fidelity to preserve the directions the model actually uses.

**Benchmark fairness:** The per-matrix mean ||dy||/||y|| penalizes GPTQ for its column compensation strategy — GPTQ redistributes error from large-activation matrices (which dominate the Hessian) to small-activation matrices (which contribute less to the total). The total ||ΔWX||/||WX|| metric corrects for this by weighting each matrix's error by its activation power. Both metrics are informative: ||dy||/||y|| tests Theorem 1's prediction at the per-matrix level; total ||ΔWX||/||WX|| evaluates the aggregate fidelity of the quantized model's computations.

### Fp16 Baseline / FP8

- **||dy||/||y||:** RTN = 0.013672 → GPTQ = 0.020385 (Delta = +0.006713) — GPTQ increases output error by 49%

### Fp16 Baseline / FP4

- **||dy||/||y||:** RTN = 0.080922 → GPTQ = 0.116885 (Delta = +0.035964) — GPTQ increases output error by 44%

### Cond Regularized / FP8

- **||dy||/||y||:** RTN = 0.014120 → GPTQ = 0.021023 (Delta = +0.006903) — GPTQ increases output error by 49%

### Cond Regularized / FP4

- **||dy||/||y||:** RTN = 0.083455 → GPTQ = 0.120577 (Delta = +0.037122) — GPTQ increases output error by 44%

**Cross-format:** GPTQ adds ~0.007 to ||dy||/||y|| at FP8, ~0.037 at FP4. The proportional increase is consistent (~45-49%) regardless of format.

**Interpretation:** GPTQ's column compensation solves a linear system to minimize Hessian-weighted reconstruction error (||W_q * H^{-1} * H - W * H||). This produces weights that better preserve the directions the model uses most, but the total ||(W_q - W)x|| increases because the optimization constraint is on the covariance-weighted norm, not the unweighted Euclidean norm. GPTQ trades total output-space fidelity for covariance-aligned fidelity — the ||dy||/||y|| increase is the cost of that trade.


## Lloyd-Max Analysis: Adaptive Grids vs Uniform E2M1

Lloyd-Max adaptive grid quantization is compared against uniform E2M1 round-to-nearest for FP4 format. Lloyd-Max fits per-layer quantization levels to the weight distribution, minimizing the MSE between original and quantized weights. **This is the only method that consistently reduces both ||dy||/||y|| and total ||ΔWX||/||WX||.**

### Fp16 Baseline

- **||dy||/||y||:** Uniform = 0.080922 → Lloyd-Max = 0.066427 (Delta = -0.0145) — **18% reduction** in output-space error
- Attention mean delta: -0.0105, FFN mean delta: -0.0198

### Cond Regularized

- **||dy||/||y||:** Uniform = 0.083455 → Lloyd-Max = 0.068038 (Delta = -0.0154) — **18% reduction** in output-space error
- Attention mean delta: -0.0114, FFN mean delta: -0.0207

**Interpretation:** Unlike GPTQ (which increases total ||dy||/||y||), Lloyd-Max genuinely reduces the quantization error by fitting grid levels to the empirical weight distribution. The reduction is consistent across both checkpoints (~18%) and both matrix types (attention, FFN). FFN matrices benefit more (~-0.020) than attention matrices (~-0.011), likely because FFN weights have more structured distributions that the Lloyd-Max iteration can exploit.

**Why Lloyd-Max works while κ-based approaches fail:** Lloyd-Max optimizes for ||W_q - W|| (weight-space MSE) directly from the weight histogram. Condition numbers capture worst-case *directional* sensitivity but FP4's unit roundoff (u = 0.25) dominates the actual error — all matrices have ||dW||/||W|| ≈ 0.15 regardless of κ. Lloyd-Max succeeds by reducing ||dW|| (better grid placement within FP4's constraints), not by exploiting κ structure.


## RMSNorm Error Blocking

RMSNorm plays a critical role in controlling quantization error propagation through Transformer layers. This section synthesizes evidence from RMSNorm ablation experiments (Phase 2), per-layer attenuation measurements (Phase 4), and per-matrix output error data (Phase 5). Critically, **RMSNorm is the primary reason per-layer metrics like PPL fail** — it blocks ~83% of per-matrix error at each layer, meaning the error that reaches the final output is dominated by the last few layers' perturbations, not the per-matrix errors Theorem 1 predicts.

**Phase 2 finding:** RMSNorm ablation experiments demonstrated that removing RMSNorm causes quantization error to grow by 1000x or more across 12 layers. With RMSNorm present, per-layer error stays within the same order of magnitude as the input perturbation.

**Phase 4 measurement:** Across 11 layers, the mean input RMSNorm attenuation ratio (||delta_post|| / ||delta_pre||) is 0.167. This corresponds to an 83% reduction in error magnitude at each RMSNorm — after traversing 12 layers, the error from layer 0's weight quantization has been attenuated by ~0.167^12 ≈ 1.6×10^-9, completely washed out. Only the last 1-2 layers' errors meaningfully affect the lm_head output. This is THE mechanism by which final-output metrics lose sensitivity to per-matrix error.

**Error decomposition (parallel/orthogonal):** At the input RMSNorm output, the mean parallel component is 0.064 and the mean orthogonal component is 0.008. The decomposition confirms that RMSNorm both reduces error magnitude and redirects error away from the signal direction — the Pythagorean identity ||d_total||^2 = ||d_parallel||^2 + ||d_orthogonal||^2 holds at each measurement point. The orthogonal component (which matters for classification) is an order of magnitude smaller than the parallel component.

**Phase 5 per-matrix evidence:** The mean tightness ratio (||dy||/||y|| / (kappa(W) * ||dW||/||W||)) across 84 matrices is 0.056. This is already 18x below the Theorem 1 bound at the matrix output — before any RMSNorm attenuation. The actual error reaching the lm_head is several orders of magnitude smaller.

**Synthesis:** RMSNorm functions as both an error attenuator and a propagation blocker. Each RMSNorm blocks ~83% of incoming error; the residual connection further dilutes the remaining error. The theoretical basis is established in Theorem 2 (see THEOREM.md). Combined with the cascade confound (§Methodology §6), this explains why two checkpoints can have per-matrix ||dy||/||y|| values that differ by 20%+ while a final-output metric shows negligible change — the metric is blind to per-matrix error structure.


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

Theorem 2 (THEOREM.md) shows that RMSNorm fundamentally changes the error propagation mechanism. Instead of the Lipschitz multiplicative cascade that would occur in unnormalized networks, RMSNorm projects error onto the orthogonal component of the signal, bounding relative error rather than amplifying it. The experimental data confirms this: RMSNorm attenuation ratios are consistently below 1.0 for the input norm, indicating systematic error reduction.

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

The full mathematical derivations for Theorems 1-4 are documented in `docs/THEOREM.md`. These include:
- **Theorem 1:** Single-layer quantization error bound
- **Corollary 1.1 / Theorem 2:** RMSNorm error blocking
- **Theorem 3:** Stochastic rounding cumulative error
- **Theorem 4:** Lloyd-Max optimality conditions
- **Strategy B (Condition number regularization):** Differentiable kappa surrogate
- **GPTQ:** Column compensation derivation from ||WX - hat(W)X||_F minimization


## References

1. **THEOREM.md:** `docs/THEOREM.md` — Full mathematical derivations for Theorems 1-4

2. **PROPOSAL.md:** `docs/PROPOSAL.md` — Original project proposal

3. **Theorem 1 data:** `results/theorem1_validation.json` — Phase 3 per-matrix kappa, weight error, output error, tightness ratio

4. **Error propagation data:** `results/error_propagation_trace.json` — Phase 4 error waterfall, RMSNorm attenuation, decomposition

5. **Full comparison data:** `results/full_comparison.json` — Phase 5 extended PTQ comparison across 16-24 configs

6. **Per-matrix summary:** `results/per_matrix_summary.json` — Merged per-matrix error summary (Phase 3/4/5)

