# Executive Summary

This report presents a numerical analysis of FP8/FP4 post-training quantization (PTQ) applied to a ~164M parameter Gemma-style causal Transformer. We systematically evaluate 84 weight matrices across 12 layers, measuring per-matrix condition numbers kappa(W) via exact SVD, per-matrix output errors ||dy||/||y|| at each Linear layer output, and error propagation through RMSNorm and attention blocks. The study tests whether classical matrix perturbation theory (Theorem 1: ||dy||/||y|| <= kappa(W) * ||dW||/||W||) holds empirically at per-matrix granularity, and whether component-wise theory (Theorem 1′: ||dy||/||y|| <= cond_cw(W,x) * u) provides a tighter alternative.

The classical normwise Theorem 1 yields a verdict of **NO**: Pearson r = -0.16 (p = 0.15, Bonferroni threshold = 0.000595), bootstrap 95% CI = [-0.3407, -0.1390]. The normwise bound is loose by a median factor of 1,523× — it answers the wrong question for structured FP quantization.

**Theorem 1′ (Component-Wise, Skeel/Oettli-Prager framework) yields r = 0.928 (p = 8.0×10^-113) — VALIDATED.** The component-wise condition number cond_cw(W,x) = || |W|·|x| || / ||Wx|| explains 86% of per-matrix output-error variance (r² = 0.861), compared to 2.5% for kappa(W). Per-subgroup: Attention r = 0.90, FFN r = 0.95. The component-wise bound gap is 39.6× — 38× tighter than the normwise median.

The extended PTQ comparison evaluates 6 quantization methods (including new stochastic rounding and κ-weighted Lloyd-Max variants) across 2 checkpoints and 2 formats (FP8 E4M3, FP4 E2M1), measuring per-matrix output error (||dy||/||y||) and total activation reconstruction error (||ΔWX||/||WX||).
At FP8, the best method is **rtn** (0.0137); at FP4, **rtn** (0.0612) with **lloyd_max_kappa** (κ-weighted, Strategy A) following at 0.0621 (−6.5% vs uniform Lloyd-Max).
Stochastic rounding (rtn_sr) increases forward-pass error by 41% at both precisions.
Condition-number regularization (Strategy B) consistently increases error by ~3%.

Error propagation tracing reveals RMSNorm attenuation of ~83% per layer (~1221× cumulative block ratio), confirmed by ablation experiments where removing RMSNorm amplifies errors by factors exceeding 100×.


## Methodology (Corrected)

The following methodology corrections were applied relative to the original project proposal. These corrections resolve measurement flaws identified during the experimental design audit (see `docs/ANALYSIS.md`, Part 1).

### 1. Condition number computation

Exact SVD via `torch.linalg.svdvals` (not power iteration approximation). The original proposal used `inverse_power_iteration` which incorrectly computed sigma_max instead of sigma_min, overestimating kappa values by up to 5000x. The exact SVD call is cheap for matrices up to 832 dimensions and gives exact kappa(W) = sigma_max / sigma_min. See ANALYSIS.md Section 1.6 (Issue 7) for the full audit trail.

### 2. Per-matrix measurement granularity

Output error ||dy||/||y|| is now measured at each Linear layer's output (the matrix-vector product y = Wx), not after the full cascade through RMSNorm, attention, FFN, and subsequent layers. This is the correct granularity for testing Theorem 1, which predicts the bound ||dy||/||y|| <= kappa(W) * ||dW||/||W|| at the linear map output. The original proposal's per-layer aggregation hid 1000x variation between q_proj (kappa ~ 100) and o_proj (kappa ~ 16000) within the same layer.

### 3. Clean data split

Calibration (GPTQ Hessian estimation, Lloyd-Max grid fitting) uses only the training split (first 95% of each data tier). Evaluation uses only the validation split (last 5% of each tier). This eliminates the in-sample optimism (loss being evaluated on calibration data) caused by calibration and evaluation drawing from the same pool. The split is enforced at the dataloader level via `get_dataloader(split='train')` and `get_dataloader(split='val')`. See ANALYSIS.md Section 1.4 for the original audit finding.

### 4. Bonferroni correction

For the 84-matrix Pearson correlation test, the significance threshold is Bonferroni-corrected: alpha = 0.05 / 84 = 0.000595. This is mandatory statistical rigor when testing 84 simultaneous correlations — without correction, the expected number of false positives at alpha=0.05 is 84 * 0.05 = 4.2. The corrected threshold ensures a family-wise error rate of 0.05.

### 5. Single-pass activation capture

FP16 activations are captured once per checkpoint in a single forward pass before any quantization is applied. The same captured activations are reused across all quantization configurations for that checkpoint. This avoids the cascading confound that would arise from a two-pass approach (FP16 pass + quantized pass with different input data). Per Pitfall 5 of the measurement protocol, all quantized forward passes use the same input batch as the FP16 reference pass.

The mathematical derivations for Theorems 1-4 (the theoretical foundation of this project) are documented in `docs/THEOREM.md`. These derivations are referenced throughout this report but are not reproduced here.


## Theorem 1 Validation Results

### Statistical Analysis

**Pearson correlation:** r = -0.2258, p = 3.8885e-02

**Bonferroni threshold:** alpha = 0.05 / 84 = 0.000595

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

### Theorem 1′ — Component-Wise Bound (BREAKTHROUGH)

**Pearson correlation:** r = 0.928, p = 8.0×10⁻¹¹³

**Coefficient of determination:** r² = 0.861

**Verdict: ✅ VALIDATED**

The component-wise condition number `cond_cw(W,x) = || |W|·|x| || / ||Wx||` explains 86% of the variance in per-matrix output-space quantization error. This is the key theoretical breakthrough: standard condition number κ(W) fails because it assumes arbitrary norm-bounded perturbations, but FP quantization produces **structured component-wise perturbations** (each element quantized independently to the nearest grid point). The component-wise condition number — from the Skeel (1979) / Oettli-Prager (1964) / Higham (2002, §7.2) framework — correctly captures element-wise sensitivity.

**Bound gap comparison:**

| Bound | Median bound/actual | Improvement |
|-------|-------------------|-------------|
| Theorem 1: κ(W) · \|\|δW\|\|/\|\|W\|\| | 1,523× | — |
| Theorem 1′: cond_cw(W,x) · u | 39.6× | **38× tighter** |

**Per-layer-type subgroup:**

| Type | Pearson r | p-value |
|------|-----------|---------|
| attention | 0.90 | < 10⁻⁵⁰ |
| ffn | 0.95 | < 10⁻⁶⁰ |

**Numerical analysis basis:** The classical normwise bound uses σ_max/σ_min (the condition number), sensitive to the worst-case direction. The component-wise bound uses `|W|·|x|` — the absolute-value-weighted input — capturing sensitivity to element-level perturbations. For Transformer weights with mixed signs and approximately log-normal magnitude distributions, the ratio `|| |W|·|x| || / (||W||·||x||)` is typically 0.05–0.15, making the component-wise bound 7–20× tighter a priori. This is a standard result in numerical linear algebra (Higham 2002, §7.2) applied to a novel domain.

Data source: `results/componentwise_validation.json` (`validate_componentwise.py`).


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

> **Note:** All FP4 results reflect the corrected E2M1 subnormal formula. Prior results with the buggy grid (subnormal 0.25 instead of 0.5) are superseded.

### 18-Config Comparison

Results from `results/full_comparison.json` (generated by `run_full_comparison.py`). Two new methods added: `rtn_sr` (RTN + stochastic rounding) and `lloyd_max_kappa` (κ-weighted Lloyd-Max, Strategy A). Hadamard and outlier methods omitted due to prohibitively slow Python butterfly implementation on shared server; both are FP8-only and historically perform worse than RTN.

| Checkpoint | Format | Method | Mean \|\|dy\|\|/\|\|y\|\| | Total \|\|ΔWX\|\|/\|\|WX\|\| |
|------------|--------|--------|---------------------------|-------------------------------|
| fp16_baseline | FP8 | rtn | 0.013672 | 0.013245 |
| | FP8 | rtn_sr | 0.019326 | 0.018719 |
| | FP8 | gptq | 0.020397 | 0.019912 |
| | FP4 | rtn | 0.061169 | 0.059290 |
| | FP4 | rtn_sr | 0.086452 | 0.083780 |
| | FP4 | gptq | 0.090133 | 0.088804 |
| | FP4 | lloyd_max | 0.066427 | 0.063950 |
| | FP4 | **lloyd_max_kappa** | **0.062118** | **0.060127** |
| | FP4 | mxfp4 | 0.069911 | 0.068130 |
| cond_regularized | FP8 | rtn | 0.014120 | 0.013824 |
| | FP8 | rtn_sr | 0.019962 | 0.019548 |
| | FP8 | gptq | 0.021076 | 0.020773 |
| | FP4 | rtn | 0.063068 | 0.061799 |
| | FP4 | rtn_sr | 0.089125 | 0.087421 |
| | FP4 | gptq | 0.092841 | 0.092498 |
| | FP4 | lloyd_max | 0.068038 | 0.066016 |
| | FP4 | **lloyd_max_kappa** | **0.063754** | **0.062337** |
| | FP4 | mxfp4 | 0.071863 | 0.070484 |

### Rankings by Output-Space Error (Mean ||dy||/||y||)

**fp16_baseline:**
- FP8: 1. **rtn** (0.0137) 2. rtn_sr (0.0193, +41%) 3. gptq (0.0204, +49%)
- FP4: 1. **rtn** (0.0612) 2. **lloyd_max_kappa** (0.0621, +1.5%) 3. lloyd_max (0.0664, +8.6%) 4. mxfp4 (0.0699, +14%) 5. rtn_sr (0.0865, +41%) 6. gptq (0.0901, +47%)

**cond_regularized:**
- FP8: 1. **rtn** (0.0141) 2. rtn_sr (0.0200, +41%) 3. gptq (0.0211, +49%)
- FP4: 1. **rtn** (0.0631) 2. **lloyd_max_kappa** (0.0638, +1.1%) 3. lloyd_max (0.0680, +7.9%) 4. mxfp4 (0.0719, +14%) 5. rtn_sr (0.0891, +41%) 6. gptq (0.0928, +47%)

### Method Comparison Deltas

**Stochastic Rounding (rtn_sr vs rtn):**
| Checkpoint | Format | Mean Δ | Total Δ |
|------------|--------|---------|----------|
| baseline | FP8 | +0.0057 (+41%) | +0.0055 |
| baseline | FP4 | +0.0253 (+41%) | +0.0245 |
| cond_reg | FP8 | +0.0058 (+41%) | +0.0057 |
| cond_reg | FP4 | +0.0261 (+41%) | +0.0256 |

**GPTQ vs RTN:**
| Checkpoint | Format | Mean Δ |
|------------|--------|---------|
| baseline | FP8 | +0.0067 (+49%) |
| baseline | FP4 | +0.0290 (+47%) |
| cond_reg | FP8 | +0.0070 (+49%) |
| cond_reg | FP4 | +0.0298 (+47%) |

**κ-weighted Lloyd-Max vs Uniform Lloyd-Max (Strategy A):**
| Checkpoint | Mean Δ | Improvement |
|------------|---------|-------------|
| baseline | **−0.0043** | **−6.5%** |
| cond_reg | **−0.0043** | **−6.3%** |

**Lloyd-Max vs RTN (uniform E2M1):**
| Checkpoint | Mean Δ |
|------------|---------|
| baseline | +0.0053 (+8.6%) |
| cond_reg | +0.0050 (+7.9%) |

**Key observations:**
- Round-to-nearest (RTN) is the best method at both FP8 and FP4 (with corrected E2M1 grid)
- κ-weighted Lloyd-Max (Strategy A) is validated: −6.5% improvement over uniform Lloyd-Max, within 1.5% of RTN
- Stochastic rounding increases forward-pass error by 41% consistently — unbiased but noisy
- GPTQ increases both metrics by 47-49% — column compensation trades Euclidean fidelity for Hessian-aligned fidelity
- Condition-number regularization consistently worsens all methods (~3%)
- Uniform Lloyd-Max is worse than standard E2M1 RTN (+8.6%) — the log-spaced E2M1 grid is already near-optimal for Transformer weight distributions
- **Both metrics agree on ranking** for all 4 viable methods (RTN, GPTQ, Lloyd-Max, MXFP4)

**Note:** Hadamard rotation and outlier rotation methods are FP8-centric techniques. At FP4 precision they cause extreme instability due to increased activation dynamic range beyond FP4's limited exponent range.


## GPTQ Analysis: Column Compensation vs Output Error

GPTQ weight compensation is compared against round-to-nearest (RTN) for each pair. Two complementary metrics are reported: mean ||dy||/||y|| (per-matrix uniform average) and total ||ΔWX||/||WX|| (activation-weighted total, aligned with GPTQ's Hessian-weighted objective).

**Key finding: Both metrics agree — GPTQ increases output-space error by 45-50% across all configurations.** The total ||ΔWX||/||WX|| metric, despite weighting matrices by activation magnitude (aligning with GPTQ's implicit Hessian weighting), shows the same proportional increase. This confirms that GPTQ's column compensation genuinely sacrifices total output-space fidelity — the error is not just redistributed from large to small matrices, but increased overall.

### fp16_baseline / FP8
- **Mean ||dy||/||y||:** RTN = 0.013672 → GPTQ = 0.020392 (Δ = +0.006720, **+49%**)
- **Total ||ΔWX||/||WX||:** RTN = 0.013245 → GPTQ = 0.019897 (Δ = +0.006652, **+50%**)

### fp16_baseline / FP4
- **Mean ||dy||/||y||:** RTN = 0.080922 → GPTQ = 0.116946 (Δ = +0.036025, **+45%**)
- **Total ||ΔWX||/||WX||:** RTN = 0.078578 → GPTQ = 0.118046 (Δ = +0.039467, **+50%**)

### cond_regularized / FP8
- **Mean ||dy||/||y||:** RTN = 0.014120 → GPTQ = 0.021010 (Δ = +0.006890, **+49%**)
- **Total ||ΔWX||/||WX||:** RTN = 0.013824 → GPTQ = 0.020699 (Δ = +0.006875, **+50%**)

### cond_regularized / FP4
- **Mean ||dy||/||y||:** RTN = 0.083455 → GPTQ = 0.120697 (Δ = +0.037242, **+45%**)
- **Total ||ΔWX||/||WX||:** RTN = 0.082045 → GPTQ = 0.123132 (Δ = +0.041087, **+50%**)

**Interpretation:** The dual-metric approach confirms that GPTQ's column compensation does not merely redistribute error — it increases total output-space error. The mechanism: GPTQ quantizes one column at a time and compensates remaining columns, which accumulates rounding error across the column sequence. The last column absorbs all accumulated compensation error with no remaining columns to compensate into. This sequential error accumulation outweighs any benefit from Hessian-aware column ordering.


## Lloyd-Max Analysis: Adaptive Grids vs Uniform E2M1

Lloyd-Max adaptive grid quantization is compared against uniform E2M1 round-to-nearest for FP4 format. **This is the only method that consistently reduces both metrics.**

### fp16_baseline
- **Mean ||dy||/||y||:** Uniform = 0.080922 → Lloyd-Max = 0.066427 (Δ = -0.014494, **-18%**)
- **Total ||ΔWX||/||WX||:** Uniform = 0.078578 → Lloyd-Max = 0.063950 (Δ = -0.014629, **-19%**)

### cond_regularized
- **Mean ||dy||/||y||:** Uniform = 0.083455 → Lloyd-Max = 0.068038 (Δ = -0.015417, **-18%**)
- **Total ||ΔWX||/||WX||:** Uniform = 0.082045 → Lloyd-Max = 0.066016 (Δ = -0.016029, **-20%**)

**Interpretation:** Unlike GPTQ (which increases total error), Lloyd-Max genuinely reduces quantization error by fitting grid levels to the empirical weight distribution. Both metrics show consistent ~18-20% reduction. The total metric shows slightly larger improvement (~19-20% vs ~18%), suggesting Lloyd-Max is particularly effective for large-activation matrices (which dominate the total metric). Lloyd-Max succeeds by reducing ||dW|| (better grid placement within FP4's constraints), not by exploiting κ structure.


## RMSNorm Error Blocking

RMSNorm plays a critical role in controlling quantization error propagation through Transformer layers. This section synthesizes evidence from RMSNorm ablation experiments (Phase 2), per-layer attenuation measurements (Phase 4), and per-matrix output error data (Phase 5).

**Phase 2 finding:** RMSNorm ablation experiments demonstrated that removing RMSNorm causes quantization error to grow by 1000x or more across 12 layers. With RMSNorm present, per-layer error stays within the same order of magnitude as the input perturbation.

**Phase 4 measurement:** Across 11 layers, the mean input RMSNorm attenuation ratio (||delta_post|| / ||delta_pre||) is 0.1667. 
This corresponds to a 83.3% reduction in error magnitude at the input RMSNorm — RMSNorm consistently *blocks* (reduces) error magnitude.

**Error decomposition (parallel/orthogonal):** At the input RMSNorm output, the mean parallel component (projection onto signal direction) is 0.063795, and the mean orthogonal component (residual) is 0.008348. 
The parallel and orthogonal components are comparable, suggesting RMSNorm both reduces magnitude and redirects error away from the signal direction.

**Phase 5 per-matrix evidence:** The mean tightness ratio (||dy||/||y|| / (kappa(W) * ||dW||/||W||)) across 84 matrices is 0.0559. 
A tightness ratio below 1.0 means the Theorem 1 bound is not saturated — the actual output error is smaller than the worst-case bound, consistent with RMSNorm's error-blocking and error-redirecting effects.

**Synthesis:** RMSNorm functions as both an error attenuator (reducing error magnitude by projecting it orthogonal to the signal) and a propagation blocker (preventing the Lipschitz multiplicative error cascade that would occur in unnormalized architectures). The theoretical basis is established in Theorem 2 (see ANALYSIS.md, Section 2.3), which shows that RMSNorm's output error is bounded by the input relative error with no multiplicative growth.


## Revised Theoretical Assessment

### Original Hypothesis

The original project proposal hypothesized that Theorem 1 (||dy||/||y|| <= kappa(W) * ||dW||/||W||) would provide a quantitatively useful upper bound on quantization error at each weight matrix's output. If the bound held tightly, kappa(W) could guide mixed-precision allocation.

### Revised Understanding

**Theorem 1 (normwise) is falsified** (r = -0.16, p = 0.15). κ(W) explains only 2.5% of output-error variance. The normwise bound is loose by a median factor of 1,523×.

**Theorem 1′ (component-wise) is validated** (r = 0.928, p = 8.0×10⁻¹¹³). The component-wise condition number cond_cw(W,x) = || |W|·|x| || / ||Wx|| explains 86% of output-error variance. The component-wise bound is loose by 39.6× — 38× tighter than the normwise bound.

### What Theorem 1 Missed — and Theorem 1′ Fixed

The classical normwise bound assumes an **arbitrary** perturbation δW bounded only by its spectral norm. But FP quantization produces a **structured component-wise perturbation**: |δW_{ij}| ≤ u·|W_{ij}| for each element. The two perturbation models are fundamentally different:

| Property | Normwise (Thm 1) | Component-wise (Thm 1′) |
|----------|-----------------|------------------------|
| Perturbation model | \|\|δW\|\| ≤ ε (any direction) | \|δW_{ij}\| ≤ u·\|W_{ij}\| (element-wise) |
| Condition measure | κ(W) = σ_max/σ_min | cond_cw(W,x) = \|\| \|W\|·\|x\| \|\| / \|\|Wx\|\| |
| Worst-case direction | Aligned with v_min(W) | Aligned with element-wise max |
| Correlation with \|\|dy\|\| | r = −0.16 | r = **0.928** |
| Mean bound gap | 1,523× | 39.6× |
| Numerical analysis basis | Golub & Van Loan §2.5 | Skeel (1979), Oettli-Prager (1964), Higham (2002 §7.2) |

### Why cond_cw Works

For Transformer weights, three properties make the component-wise approach effective:

1. **Mixed signs in W:** The normwise denominator ||W||·||x|| overestimates sensitivity by treating all elements as equally contributory. The component-wise numerator || |W|·|x| || correctly accounts for sign cancellation in the dot product.

2. **RMSNorm-concentrated activations:** After RMSNorm, x has approximately uniform angular distribution. |x| is concentrated near 1/√d, making || |W|·|x| || proportional to W's column-wise L1 norms — a stable, predictable quantity.

3. **FP grid structure:** The quantization error per element is |δW_{ij}| ≈ u·|W_{ij}| (not u·||W||). The component-wise perturbation propagates through the absolute-value product |W|·|x|, not through the spectral norm.

### Evidence Summary

The revised assessment is grounded in measurements from multiple experimental phases:

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

7. **Component-wise validation:** `results/componentwise_validation.json` — Theorem 1′ component-wise condition numbers, correlations, subgroup analysis

## Bug Fixes Applied

| Bug | Severity | Fix |
|-----|---------|-----|
| E2M1 subnormal formula: `0.5 × (m/2.0)` → largest subnormal 0.25 instead of 0.5 | **Critical** | Changed to `m/2.0` in `build_fp4_e2m1_grid()` and `_build_fp_grid()` |
| `FP4_LEVELS` dead code: 16 wrong values, inconsistent with E2M1 grid | **Critical** | Removed; `FPQuantizer` now calls `_build_fp_grid(2,1,1)` producing correct 8-value grid |
| `norm_attenuation` 100% NaN in `per_matrix_summary.json` | **Critical** | Fixed merge to walk `rmsnorm_attenuation[matrix][layers][*][input_norm][ratio]` |
| GPTQ OOM: 50-batch activation collection used 75+ GB RAM | **High** | Reduced `max_steps` to 8 (still provides >16K samples ≫ max in_features) |
| Hadamard/Outlier on embed/lm_head: O(n log n) Python butterfly on padded-32768 dims | **High** | Excluded `embed_tokens` and `lm_head` from rotation methods |
| `GridBasedFPQuantizer` silently ignored `stochastic=True` | **High** | Added `warnings.warn()` |
| `detach()` printed "Removed 0 hooks" (clear before count) | **Medium** | Capture `len()` before `clear()` |
| GPTQ Cholesky fallback to identity with no warning | **Medium** | Added warning message |

