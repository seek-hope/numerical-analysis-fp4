# Domain Pitfalls: Quantization Error Measurement in Transformers

**Domain:** Numerical analysis of FP4 weight quantization in Transformer models
**Researched:** 2026-05-17
**Overall confidence:** MEDIUM (based on established numerical analysis principles and codebase audit; some sources could not be refreshed via web search)

## Pitfall Classification

| Severity | Label | Meaning |
|----------|-------|---------|
| CRITICAL | Causes invalid measurements or wrong conclusions | Must fix before protocol design |
| HIGH | Biases results or adds systematic error | Must document and mitigate |
| MODERATE | Reduces measurement precision or interpretability | Should address in protocol |
| LOW | Edge case or minor interpretation issue | Acknowledge, document |

---

## 1. PyTorch Forward Hook Pitfalls for Measurement

### PITFALL 1.1: Double Counting from Shared Weight Tensors

**Severity:** CRITICAL

**What goes wrong:** `MicroGemmaFPForCausalLM` ties `self.lm_head.weight = self.model.embed_tokens.weight` (transformer.py line 235). These are the SAME tensor object. If you register hooks on both `embed_tokens` and `lm_head`, or if you enumerate all Linear layers and quantize the shared weight twice, the perturbation is applied twice to the same tensor. The second quantization overwrites the first, and measurement sees only the final state.

**Why it happens:** The weight tying is explicit in `__init__`. Enumeration patterns like `named_parameters()` or `named_modules()` that filter by `nn.Linear` will find both references but hit the same tensor.

**Consequences:**
- Double quantization corrupts the weight
- ||y|| comparisons between "clean" and "quantized" passes use different baselines
- Statistics for lm_head and embed_tokens are identical, inflating the apparent sample size

**Prevention:**
- Skip `embed_tokens` AND `lm_head` when quantizing (they are the same tensor)
- In measurement, only hook one of them
- The current code correctly skips both (`if 'embed' in name.lower() or 'lm_head' in name.lower()`), but verify this in all measurement paths

**Detection:**
- Assert tensor identity: `assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()`
- Count unique weight matrices: `len(set(p.data_ptr() for p in model.parameters() if p.dim() >= 2))`

**Phase to address:** MEAS-01 (per-weight-matrix Theorem 1 validation)

---

### PITFALL 1.2: Hook Captures Input, Not Output (Wrong Quantity for Theorem 1)

**Severity:** CRITICAL

**What goes wrong:** `nn.Linear` forward hooks receive `(input, output)` — but the input is `(x,)`, not `Wx`. Theorem 1 bounds ||y_hat - y||/||y|| where y = Wx. If you hook the input (pre-linear) and output (post-linear), you can compute this correctly. But if you hook the input and compute on x (as `gptq.py:_collect_activations` does), you are capturing calibration data, not the output error.

**Why it happens:** `gptq.py` intentionally hooks inputs for Hessian computation (line 206: `inp[0].detach()`). This is correct for GPTQ. But for Theorem 1 validation, the hook must capture both pre-activation x and post-activation y = Wx to compute ||(W_q - W)x||/||Wx||.

**Consequences:** If the measurement protocol uses the same hook pattern as `_collect_activations`, it captures the wrong quantity. The ratio ||(W_q - W)x||/||Wx|| requires y = Wx, which is the OUTPUT, not the input.

**Prevention:**
- Use TWO hooks per weight matrix: one on the input to capture x, one on the output to capture y
- Or compute y = Wx manually after quantization (clean y from saved x, quantized y from quantized W, same x)
- The latter approach (recompute y from saved x) avoids the need for output hooks and is more controlled

**Detection:**
- If the measurement script only registers hooks on input, it is computing ||W_q - W||/||W|| (weight-space error), not ||(W_q - W)x||/||Wx|| (output-space error)

**Phase to address:** MEAS-01

---

### PITFALL 1.3: Hook Ordering and Registration Lifecycle

**Severity:** HIGH

**What goes wrong:** Multiple hooks on the same module fire in registration order. If you register hooks, then modify model weights, then run a forward pass, the hooks fire on the modified model. If you register hooks on submodules (e.g., each Linear inside Attention), they fire during the parent module's forward pass. If you accidentally register duplicate hooks (e.g., by calling a function twice), each hook fires separately for the same forward pass, causing double-counting.

**Why it happens:**
- Hooks persist until explicitly removed (`.remove()`)
- `register_forward_hook` does not check for duplicates
- Closures in hook factories capture state by reference, not value — if you mutate a variable the closure references, the hook sees the mutated value

**Specific issues in this codebase:**
- `gptq.py` registers hooks, runs model, then removes hooks (lines 214-234). This pattern is correct.
- But after `weight.data.copy_(W_q)`, any already-registered hooks will fire on the quantized weight's forward pass. The order is: register hooks -> run calibration -> remove hooks -> quantize weights -> (separate evaluation pass without hooks). This is safe. But if hooks remain during quantization, they fire on partially-quantized states.

**Prevention:**
- Always remove hooks immediately after the measurement pass
- Use `with` pattern or `try/finally` to guarantee cleanup
- Do not interleave hook registration and weight modification
- Verify: assert all hooks are removed before modifying weights

**Detection:**
- Check `module._forward_hooks` dict before and after measurement
- Assert empty before weight modification

**Phase to address:** MEAS-02 (full error propagation trace)

---

### PITFALL 1.4: Memory Accumulation in Hook Collectors

**Severity:** HIGH

**What goes wrong:** The common pattern of appending tensors to a list in a hook and concatenating later (as in `gptq.py` lines 208-211) accumulates all activations in GPU memory until the forward pass completes and they can be moved to CPU. For the 164M model with 12 layers:
- Each activation: batch=8, seq=512, dim=768 = ~12.6M FP32 values = ~50 MB
- 12 layers x 6 weight matrices = 72 activations x 50 MB = ~3.6 GB for one batch
- Over 50 calibration steps, if not offloaded per-step: 180 GB

**Why it happens:** `gptq.py` appends `x_flat.cpu()` so data moves to CPU each step. This is correct and prevents OOM. But for measurement protocols that need GPU-resident activations (e.g., for computing ||(W_q - W)x|| on GPU), the memory accumulates rapidly.

**Consequences:**
- GPU OOM for long calibration runs
- Silent fallback to smaller batch size changes activation statistics
- The current gptq.py correctly saves to CPU, but new measurement protocols might skip this for speed

**Prevention:**
- Keep the `.cpu()` offload pattern
- For per-matrix Theorem 1: process one matrix at a time, compute ||(W_q - W)x||/||Wx|| immediately, discard x
- Batch-wise averaging: compute running statistics instead of storing all activations

**Detection:**
- Add memory monitor: `torch.cuda.max_memory_allocated()` before/after measurement pass

**Phase to address:** MEAS-01, MEAS-02

---

### PITFALL 1.5: Hooks Under `torch.no_grad()` vs Autograd Graph

**Severity:** MODERATE

**What goes wrong:** Forward hooks fire regardless of `torch.no_grad()` context. The tensors captured by hooks may or may not be part of the autograd graph. `gptq.py` correctly calls `.detach()` on captured activations (line 207). But measurement hooks that do NOT detach will retain the entire computation graph, causing memory leaks.

**Why it happens:** PyTorch hooks receive the actual tensor from the forward pass. If the forward pass was under `no_grad()`, the tensors have `requires_grad=False` anyway. But measurement scripts sometimes omit `no_grad()` to keep the option of gradient-based analysis.

**Consequences:**
- Memory grows unboundedly if hooks accumulate graph nodes
- `backward()` on captured tensors would backprop through the entire model

**Prevention:**
- Always `detach()` in hooks, even under `no_grad()` (defense in depth)
- Explicitly wrap measurement forward passes in `@torch.no_grad()`

**Phase to address:** All MEAS phases

---

## 2. Statistical Pitfalls in Per-Layer Metric Analysis

### PITFALL 2.1: Multiple Comparison Problem Across 72 Weight Matrices

**Severity:** CRITICAL

**What goes wrong:** The planned protocol (MEAS-01) tests correlation between kappa(W) and ||delta_y||/||y|| for each of ~72 weight matrices (12 layers x 6 projections). Running 72 hypothesis tests at alpha=0.05 yields ~3.6 expected false positives by chance alone. If the protocol reports "significant correlations" without correction, results are unreliable.

**Why it happens:** Standard practice in deep learning papers reports per-layer metrics without multiple-testing correction. Reviewer culture rarely enforces it because per-layer analysis is considered "exploratory."

**Consequences:**
- Claiming "kappa(W) correlates with ||delta_y||/||y|| in 5 of 12 layers" when 3 of those are chance
- Selecting layers with high correlation for mixed-precision decisions overfits to noise

**Prevention:**
- Report ALL 72 results, not a cherry-picked subset
- Apply Bonferroni correction: threshold = 0.05 / 72 = 0.00069
- Or use False Discovery Rate (Benjamini-Hochberg) which is less conservative
- Better: report effect sizes (Pearson r), not just p-values
- Even better: permutation test — shuffle layer assignments and compare observed r against null distribution

**Detection:**
- Count how many tests are reported as "significant" and check if correction was applied
- If the protocol describes correlation without mentioning correction, flag immediately

**Phase to address:** MEAS-01

---

### PITFALL 2.2: Simpson's Paradox — Aggregation Masks Per-Matrix Effects

**Severity:** CRITICAL

**What goes wrong:** q_proj and o_proj within the same layer differ in condition number by up to 160x (k ~100 for q_proj vs k ~16,000 for o_proj). If you average kappa across all matrices in a layer before correlating with ||delta_y||/||y||, you lose the signal. The within-layer variation is larger than the between-layer variation.

**Why it happens:** `sensitivity.py` (lines 33-40) averages kappa across all weight matrices per layer. The planned MEAS-01 protocol should analyze per-matrix, but the existing infrastructure encourages aggregation.

**Consequences:**
- Correlation analysis shows r ~ 0 because averaging destroys the within-layer slope
- Mixed-precision assignments averaged across a layer assign FP8 to everything or nothing
- The real effect (different sensitivities within a layer) is invisible

**Prevention:**
- Analyze at the weight-matrix granularity, not layer level
- For each of the 12 layers, run 6 separate tests (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)
- Report: "within-layer kappa range = [min, max]" for each layer
- Visualize: scatter plot of kappa(W) vs ||delta_y||/||y|| with points colored by projection type

**Detection:**
- If the protocol mentions "per-layer" correlation, check if aggregation is happening
- Variance decomposition: what fraction of total variance is within-layer vs between-layer?

**Phase to address:** MEAS-01

---

### PITFALL 2.3: Confounding by Layer Depth

**Severity:** HIGH

**What goes wrong:** Layer depth confounds the kappa-||delta_y||/||y|| relationship because:
1. Early layers process smaller activations (not yet amplified by residual stream)
2. Late layers process larger activations
3. ||y|| = ||Wx|| depends on x, which depends on depth
4. If kappa(W) also varies systematically with depth (it does: o_proj kappa grows from ~5000 in layer 0 to ~16,000 in layer 11), the correlation between kappa and ||delta_y||/||y|| is partially driven by depth, not by the theoretical mechanism

**Why it happens:** Theorem 1 predicts ||delta_y||/||y|| <= kappa(W) * ||delta_W||/||W||. The bound involves kappa(W), but the measured ratio also depends on x (the input to that layer). x changes with depth, and ||delta_W||/||W|| may also change (FP4 quantization error depends on weight scale, which varies per layer).

**Consequences:**
- Apparent "kappa predicts error" effect is actually "depth predicts both kappa and input scale"
- Controlling for depth removes the correlation
- The theoretical claim is about the bound, not the actual value — the bound may be loose for all layers

**Prevention:**
- Partial correlation: measure kappa-||delta_y||/||y|| correlation controlling for layer index
- Within-layer analysis: for each projection TYPE (all q_proj across all layers), are layers with higher kappa showing higher ||delta_y||/||y||? This removes depth as a confound because all q_proj see similar activation scales.
- Use the SAME input x for all weight matrices (theoretical approach: use random Gaussian x with same spectral properties as real activations, then compare with real activations)

**Detection:**
- Regress ||delta_y||/||y|| on layer_idx: is there a significant slope?
- If yes, the confound is present. Check if kappa also varies with layer_idx.
- Confounding is present if both kappa and the outcome vary with depth, AND controlling for depth reduces the kappa-outcome correlation

**Phase to address:** MEAS-01

---

### PITFALL 2.4: Confounding by Activation Magnitude (Denominator Effect)

**Severity:** HIGH

**What goes wrong:** ||delta_y||/||y|| = ||(W_q - W)x|| / ||Wx||. The denominator ||Wx|| varies per layer based on:
- Activation scales (attention outputs vs FFN outputs)
- GQA structure (q_proj/k_proj/v_proj see different activation statistics than o_proj)
- Per-layer token embeddings (PL embeddings add 64 dims to the input)
- Sliding vs full attention layer types

A layer with naturally small ||Wx|| will have inflated ||delta_y||/||y|| even if kappa is low and ||delta_W|| is small, simply because the denominator is small.

**Why it happens:** The same ||delta_W|| ||x|| numerator divided by a smaller ||Wx|| denominator yields a larger ratio, even when the underlying matrix perturbation is the same size. This is not a bug — it's the mathematical definition — but it means ||delta_y||/||y|| conflates two effects: sensitivity to perturbation (kappa) and output magnitude.

**Consequences:**
- Layers with small output norms appear "more sensitive" when they are just operating in a different regime
- The kappa-||delta_y||/||y|| relationship is contaminated by variation in ||Wx||

**Prevention:**
- Report numerator (||delta_y||) and denominator (||y||) separately, not just the ratio
- Normalize by ||x|| instead of ||y||: ||(W_q - W)x|| / ||x|| measures absolute perturbation per unit input, removing denominator confounding. Compare: ||(W_q - W)x||/||x|| vs kappa * ||delta_W||/||W|| * (something).
- Actually, Theorem 1 uses ||y|| in the denominator precisely because ||y|| >= sigma_min * ||x||. The bound is kappa * ||delta_W||/||W||, which does NOT depend on ||x||. So measuring ||delta_y||/||x|| and comparing to kappa * ||delta_W||/||W|| is equally valid.

**Detection:**
- Coefficient of variation of ||Wx|| across layers: CV = std(||Wx||) / mean(||Wx||)
- High CV > 0.5 means denominator effect is significant

**Phase to address:** MEAS-01

---

### PITFALL 2.5: P-hacking Through Layer Subset Selection

**Severity:** HIGH

**What goes wrong:** With 72 weight matrices, you can find subsets that support any narrative:
- "kappa predicts error in q_proj but not o_proj" (true by chance in some datasets)
- "Layers 3-7 show correlation, but 0-2 and 8-11 don't" (may be real or noise)
- "Sliding layers obey Theorem 1, full layers don't" (sample size: 8 vs 4, low power)

**Why it happens:** Exploratory data analysis is valid, but reporting post-hoc subgroups as confirmatory findings without correction is p-hacking. The multiple-comparison problem (Pitfall 2.1) compounds with subgroup analysis.

**Consequences:** False "discoveries" about which layers are sensitivity-dominated vs noise-dominated. Mixed-precision recommendations based on spurious subgroups.

**Prevention:**
- Pre-register the analysis plan: which layer types, which projections, and the exact correlation formula
- Use hold-out: randomly split layers into exploration (66%) and confirmation (33%) sets
- For all claims about subgroups, report total N, effect size, and confidence interval

**Phase to address:** MEAS-01, MEAS-05

---

## 3. Numerical Stability of ||delta_y||/||y||

### PITFALL 3.1: Division by Near-Zero ||y||

**Severity:** CRITICAL

**What goes wrong:** For some inputs x, the output y = Wx can be near-zero. This can happen:
- When x is near-orthogonal to the row space of W
- In dead attention heads (attention outputs zero for certain tokens)
- In intermediate FFN activations that are gated to near-zero (gate_proj * up_proj -> small values)

When ||y|| is near zero, ||delta_y||/||y|| blows up numerically even if ||delta_y|| is tiny.

**Why it happens:** The transformer's combination of GELU gating, attention softmax, and residual connections creates sparse activation patterns. ||y|| can be 1e-6 while typical values are 1e-1, giving a relative error of 1e5 for a negligible absolute error.

**Consequences:**
- A few tokens dominate the average ||delta_y||/||y||
- The ratio becomes high-variance and not reproducible across different calibration data
- Theorem 1's bound (kappa * ||delta_W||/||W||) may still hold (because sigma_min * ||x|| bounds ||y|| from below), but the measured value is meaningless

**Prevention:**
- Filter out tokens where ||y|| < epsilon * mean(||y||) across tokens (epsilon = 1e-3 or similar)
- Report both filtered and unfiltered metrics
- Use median instead of mean for relative error across tokens
- Also report absolute error ||delta_y|| alongside the ratio

**Detection:**
- Histogram of ||y|| across tokens: are there outliers at 1e-6?
- Coefficient of variation of ||y|| across tokens
- Compare mean vs median of ||delta_y||/||y||: large divergence indicates tail dominance

**Phase to address:** MEAS-01

---

### PITFALL 3.2: Noise Dominance When ||delta_W|| is Small (FP8 Case)

**Severity:** HIGH

**What goes wrong:** For FP8 quantization (E4M3, unit roundoff u = 2^{-4} = 0.0625, but effective relative error per weight is ~u/sqrt(12) ~ 0.018), ||delta_W||/||W|| is about 0.02. For well-conditioned matrices (kappa ~ 10), the predicted bound is kappa * ||delta_W||/||W|| ~ 0.2. The measured ||delta_y||/||y|| of ~0.05 is well above the FP32 noise floor, so measurement is fine.

But for FP4 quantization, the same analysis applies differently because:
- FP4 unit roundoff u = 0.25 (E2M1)
- ||delta_W||/||W|| ~ u/sqrt(12) * sqrt(d) ~ 0.07 * 28 = ~2 for d=768
  Wait, this is wrong. ||delta_W||/||W|| is the relative perturbation in Frobenius norm. For FP4 E2M1 with per-channel scaling, the max relative error per weight is ~u/2 = 0.125, but actual distribution matters. Let me reconsider...

The key point: when ||delta_W||/||W|| is very small (FP8, or for small weight values in FP4), the measured ||delta_y||/||y|| may be dominated by floating-point rounding noise in the computation of y_q and y rather than quantization error.

**Why it happens:** y = Wx and y_q = W_q x are both FP32 operations in this simulation. For FP8 quantization, W_q differs from W by ~1-2% per element. The difference (W_q - W)x is computed in FP32 with no accumulation error. So noise dominance is unlikely for FP8. However, if measuring single-token relative error, FP32 roundoff (~1e-7 relative) vs quantization signal (~1e-2) is fine.

The real concern: if using FP16 activations, the FP16 accumulation noise (~1e-4 relative) may compete with FP8 quantization signal (~1e-2 relative). Still fine for FP8, but:

For a weight matrix where quantization is very accurate (W and W_q are nearly identical, e.g., small weights in FP4 that round to zero), ||delta_y|| is at FP32 noise level while ||y|| is at activation level — the ratio is ~1e-7, well below the predicted bound.

**Consequences:** Not a practical issue for FP4 quantization (where ||delta_W||/||W|| is large enough to dominate noise). But for null measurements (no quantization, verify noise floor), the ratio should be ~1e-7, not 0.

**Prevention:**
- Include a "null measurement" pass: measure ||(W - W)x||/||Wx|| where W is the ORIGINAL weight (no quantization). This should give ~1e-7.
- If null measurement > 1e-5, there is numerical instability in your measurement code.
- The entire measurement protocol should be validated with a no-quantization control.

**Phase to address:** MEAS-01 should include a null measurement control in the same script.

---

### PITFALL 3.3: Using Different x for y and y_q (Cascading Confound)

**Severity:** CRITICAL

**What goes wrong:** To measure per-matrix ||delta_y||/||y|| for Theorem 1, you need:
- y = Wx (clean weight, clean input)
- y_q = W_q x (quantized weight, SAME input x)

BUT: if you quantize the full model and run it end-to-end, then compare outputs at layer L between clean and quantized runs, you are not measuring ||(W_q - W)x||/||Wx|| for that specific layer. You are measuring the cascade of all upstream quantization errors. The input x to layer L differs between the two runs because earlier layers were also quantized.

**Why it happens:** The intuitive but wrong approach: "run clean model, capture all activations. Run quantized model, capture all activations. Compare per-layer." This gives ||delta_y_total||/||y_total||, not ||(W_q - W)x||/||Wx||. Theorem 1 only predicts the single-matrix bound.

**Consequences:**
- Measured ||delta_y||/||y|| is a confounded mix of the target layer's quantization AND all upstream errors
- The claimed correlation with kappa(W) is diluted because upstream errors dominate
- This is exactly what happened with PPL as the metric: the signal from each individual layer's quantization is lost in the cascade

**Prevention:**
- Option A (correct): replace ONE weight matrix at a time. Run model forward once with that single quantized weight. Compare output at that layer only. This isolates the per-matrix effect.
- Option B (approximate): run clean forward, save all x and y. Then for each matrix, compute W_q and directly compute ||(W_q - W)x||/||Wx|| using the SAVED x. No need to run the model again.
- Option B is preferred because it avoids cascading completely. It's also faster (one forward pass, 72 matrix-level computations).

**Detection:**
- If the protocol runs two full model forward passes (clean and quantized) and compares per-layer activations, it has this pitfall.
- Check: does the protocol modify weights BETWEEN the two forward passes?
- If yes, and if more than one weight is modified, cascading confound exists.

**Phase to address:** MEAS-01 — the protocol must use Option B (compute from saved activations), not Option A (two forward passes).

---

### PITFALL 3.4: Signal-to-Noise Ratio of the Perturbation Direction

**Severity:** MODERATE

**What goes wrong:** Theorem 1's bound is tight when delta_W aligns with the left singular vectors of W (specifically u_min). Random quantization noise may or may not align with this direction. If it doesn't, the bound is loose, and ||delta_y||/||y|| is much smaller than kappa * ||delta_W||/||W|| predicts. This is not a measurement error, but a misinterpretation risk.

**Why it happens:** The bound is:
||delta_y||/||y|| <= kappa * ||delta_W||/||W||

For random (or quantization-structured) delta_W, the left side is typically much smaller than the bound. The actual ratio depends on the alignment between delta_W and the singular vectors of W. Uniform quantization noise is largely isotropic, so it does not preferentially align with any singular vector.

**Consequences:**
- Finding that ||delta_y||/||y|| << kappa * ||delta_W||/||W|| is EXPECTED, not a violation of Theorem 1
- The bound is an upper bound, not an equality
- Over-interpreting the gap between measured ratio and predicted bound as "kappa is irrelevant" is wrong

**Prevention:**
- Compute and report the empirical tightness ratio: ||delta_y||/||y|| / (kappa * ||delta_W||/||W||)
- Theoretical range: (1/kappa) to 1.0 (lower bound: delta_W aligned with sigma_max, upper bound: aligned with sigma_min)
- For random quantization noise, expect ~1/kappa or sqrt(1/kappa)
- Flag layers where the tightness ratio > 0.5 as "structurally aligned perturbation"
- Report the alignment angle: cos(theta) = <delta_W vec, u_min v_min^T vec> / ||delta_W||
  - Actually, proper analysis: compute ||(W_q - W) v_min|| / (||W_q - W|| * ||v_min||) to see if the perturbation aligns with the worst-case direction

**Phase to address:** MEAS-01 should include alignment analysis, not just magnitude comparison.

---

## 4. RMSNorm Measurement Pitfalls

### PITFALL 4.1: The Unit-RMS Denominator Illusion

**Severity:** CRITICAL

**What goes wrong:** RMSNorm always produces output with RMS = sqrt(d) (approximately, up to learnable weight factor). This means ||RMSNorm(y)|| is essentially CONSTANT for all inputs. The relative error after RMSNorm:

||delta_output|| / ||RMSNorm(y)|| = ||delta_output|| / sqrt(d)

is really just a scaled version of the absolute error ||delta_output||. It is NOT a genuine "relative error" in the same sense as ||delta_y||/||y|| before normalization.

**Why it happens:** RMSNorm divides by RMS(x), which normalizes the scale. After normalization, the output always has RMS ~ sqrt(d) (assuming weight ~ 1). The denominator in the relative error formula is therefore constant, making "relative error" after RMSNorm equivalent to absolute error divided by a constant.

**Consequences:**
- Claiming "RMSNorm reduces relative error by factor X" is misleading — the relative error after RMSNorm is defined differently than before RMSNorm
- Before RMSNorm: ||delta||/||y|| is truly relative (denominator depends on input)
- After RMSNorm: ||delta||/sqrt(d) is effectively absolute (denominator is constant)
- The comparison is apples-to-oranges

**Prevention:**
- Report RMSNorm's effect as: output_absolute_error / input_relative_error
- Or: compare the propagation MULTIPLIER: ||delta_output|| / ||delta_input|| (how much the absolute error changes, accounting for the norm compression)
- The current formulation in ANALYSIS.md (Theorem 2) correctly derives:
  - ||delta_output|| <= sqrt(d) * ||delta|| / ||y||
  - i.e., absolute output error <= sqrt(d) * relative input error
- The blocking ratio of 1221x was computed as: (error without RMSNorm) / (error with RMSNorm). If both are measured with the same metric, this ratio is meaningful. But ensure both sides use the SAME type of error (absolute or relative).

**Detection:**
- If the protocol reports "relative error after RMSNorm" using the definition ||delta||/||RMSNorm(y)||, flag it
- Check: is the denominator input-dependent or near-constant?
- Compare: var(||RMSNorm(y)||) vs var(||y||) across tokens — RMSNorm output norm should have near-zero variance

**Phase to address:** MEAS-03 (RMSNorm attenuation measurement)

---

### PITFALL 4.2: Parallel vs Orthogonal Decomposition is Necessary for RMSNorm

**Severity:** HIGH

**What goes wrong:** The RMSNorm blocking analysis (Theorem 2 in ANALYSIS.md) shows that RMSNorm projects delta onto the orthogonal complement of y:

delta_output ~ (I - yy^T/||y||^2) delta = delta_perp

If delta is parallel to y, RMSNorm removes it entirely. If delta is orthogonal to y, RMSNorm preserves it fully. The blocking ratio depends on the angle between delta and y, which varies by layer and by quantization method.

**Why it happens:** RMSNorm's error compression is NOT isotropic. It selectively removes the component of the error that is parallel to the signal. This is well-understood theoretically (Theorem 2) but easy to ignore in measurement.

**Consequences:**
- The 1221x blocking ratio is an average that hides layer-to-layer variation
- Attention layers produce structured errors (due to softmax inducing token-token correlations), so delta_attn may be more parallel to its input than FFN errors
- Quantization methods that produce structured errors (GPTQ, which compensates column-by-column) may see different blocking ratios than simple round-to-nearest
- A low blocking ratio doesn't mean RMSNorm is broken — it may just mean the error is naturally orthogonal to the signal

**Prevention:**
- Decompose delta into parallel and orthogonal components: delta_par = (y^T delta / ||y||^2) * y, delta_perp = delta - delta_par
- Report: ||delta_par||, ||delta_perp||, and the angle between delta and y
- Measure blocking ratio PER LAYER TYPE (attention vs FFN, sliding vs full)
- Test: does GPTQ produce more orthogonal errors than round-to-nearest?

**Detection:**
- Compute ||delta - delta_perp|| and compare to ||delta||
- If ||delta_par|| ~ 0, the error is naturally orthogonal and RMSNorm can't block what's already orthogonal
- Low ||delta_par|| before RMSNorm = high blocking ratio regardless of RMSNorm

**Phase to address:** MEAS-03

---

### PITFALL 4.3: RMSNorm Weight Drift During QAT

**Severity:** MODERATE

**What goes wrong:** RMSNorm's learnable `weight` parameter (transformer.py line 19: `self.weight = nn.Parameter(torch.ones(dim))`) is initialized to ones. During QAT, these weights may drift. The unit-RMS property depends on weight ~ 1. If weight drifts significantly, the denominator sqrt(d) in the relative error formula changes.

**Why it happens:** QAT training with quantization noise may adjust RMSNorm weights to compensate. This is a legitimate learned adaptation, but it changes the measurement reference.

**Consequences:**
- The blocking ratio measured on a QAT model may differ from PTQ because RMSNorm weights have drifted
- The "unit-RMS" assumption used in derivations is violated for drifted weights
- Comparing blocking ratios between PTQ (on FP16 baseline) and QAT is not apples-to-apples

**Prevention:**
- Before measuring RMSNorm attenuation, log the actual RMSNorm weights: mean(weight), std(weight)
- Normalize the measurement: ||delta_output|| / ||RMSNorm(y)|| = ||delta_output|| / sqrt(mean(weight^2) * d)
- For the theoretical analysis, incorporate the actual weight values

**Detection:**
- Check if RMSNorm weights differ from 1.0 by more than 1%
- If weights have drifted, the simplified analysis (unit weight assumption) is inaccurate

**Phase to address:** MEAS-03

---

### PITFALL 4.4: RMSNorm Blocks Error, But at What Cost?

**Severity:** MODERATE

**What goes wrong:** RMSNorm blocks error propagation by re-normalizing. But re-normalization itself introduces a non-linearity that distorts the signal. The output after RMSNorm is:
- In the error-free case: y_tilde = y / RMS(y) * weight
- With error: (y + delta) / RMS(y + delta) * weight

The re-normalization creates a SECOND source of output change: even if delta = 0 in the weight, the change in RMS(y+delta) due to delta in a DIFFERENT part of the vector causes non-linear cross-talk. The Taylor expansion (Theorem 2) captures this to first order, but higher-order terms matter for large delta.

**Why it happens:** RMSNorm's denominator RMS(y+delta) depends on ALL elements of y+delta. A large delta in one element changes the normalization factor for ALL elements, spreading the error. This is the "cross-term" or "interference" effect.

**Consequences:**
- For large delta (FP4 quantization, u=0.25), the first-order Taylor approximation may underestimate the true error after RMSNorm
- The cross-talk means RMSNorm can actually AMPLIFY error in some components while attenuating it in others
- The net effect (1st order: attenuation; 2nd order: cross-talk) depends on delta magnitude

**Prevention:**
- Validate the Taylor approximation: compute ||delta_output_exact - delta_output_1st_order|| / ||delta_output_exact||
- If high-order terms exceed 10% of the total effect, the linear approximation is insufficient
- For FP4 quantization, this check is essential because u=0.25 is large

**Phase to address:** MEAS-03

---

## 5. Mixed Precision Measurement Fragility

### PITFALL 5.1: FP16 Accumulation Noise vs FP4 Quantization Signal

**Severity:** HIGH

**What goes wrong:** In mixed-precision execution, the model's activations accumulate in FP16 (or on RTX 4090, TF32). PyTorch's default matmul precision on Ampere GPUs uses TF32 for the multiply and FP32 for the accumulate. But `F.scaled_dot_product_attention` may internally use FP16. The "clean" y = Wx already contains FP16 roundoff error.

FP16 has unit roundoff u_16 = 2^{-11} = 4.88e-4 (10 mantissa bits + implicit leading 1). FP4 E2M1 has u_4 = 0.25. The ratio u_4 / u_16 = 512. So FP4 quantization adds noise ~512x larger than FP16 arithmetic noise — easily distinguishable.

BUT: for FP8 quantization (E4M3, u_8 = 2^{-4} = 0.0625), the ratio u_8 / u_16 = 128. Still distinguishable, but the margin is smaller.

**Why it happens:** The clean model output y is computed in FP16/TF32 precision. The quantization comparison assumes y is exact. It is not — it already contains ~1e-4 relative error from accumulation.

**Consequences:**
- The null measurement (W_q = W) should produce ||delta_y||/||y|| at the FP16 noise level (~1e-4), not 0
- For FP8 quantization where ||delta_y||/||y|| may be ~1e-3, the signal-to-noise ratio is only ~10x
- For layers where quantization is very accurate (many weights round to original value), FP16 noise may dominate

**Prevention:**
- Always include a null measurement (W_q = W, run through same pipeline) to establish noise floor
- Report the SNR: ||delta_y_quantized|| / ||delta_y_null|| for each layer
- For precise measurements, run the entire measurement in FP32 (disable autocast, set matmul precision)
- In the existing codebase: the measurements already run in FP32 simulation mode (weights are FP32 tensors with quantized values), so FP16 accumulation in the forward pass is only relevant for the attention computation

**Detection:**
- The null measurement noise floor should be checked before any real measurement
- If null ||delta_y||/||y|| > 1e-5, investigate accumulation precision

**Phase to address:** MEAS-01, MEAS-05

---

### PITFALL 5.2: Simulated FP4 vs Hardware FP4 — Systematic Underestimate

**Severity:** HIGH

**What goes wrong:** The current code simulates FP4 quantization by rounding FP32 weight values to the nearest FP4-representable value, then storing them in FP32 tensors. The forward pass computes W_q * x where W_q is FP32 and x is FP32. On real hardware, both W_q and x would be FP4, and the accumulation would be in FP16 or FP32.

The simulation underestimates error because:
1. Input x is FP32, not FP4 — no input quantization error
2. The matmul W_q * x is computed in FP32, not FP4 — no accumulation error
3. Residual connections accumulate in FP32, not FP16 — no accumulation error in the residual stream

**Why it happens:** PyTorch does not support FP4 data types. The simulation is FP32-with-restricted-values, which preserves precision for the computation while restricting the values.

**Consequences:**
- Measured ||delta_y||/||y|| is the weight-quantization-only error, not the full hardware FP4 error
- The comparison with Theorem 1 (which predicts weight-quantization-only error) is valid
- But any claim about "PPL at FP4 precision" is optimistic relative to real hardware

**Prevention:**
- Explicitly state: "FP4 simulated in FP32 arithmetic. Real hardware FP4 would additionally include input quantization and accumulation error, estimated to increase ||delta_y||/||y|| by 10-50%."
- For a more realistic estimate: add synthetic input quantization noise (round activations to FP4 before the matmul)
- Use the simulation as a controlled test of WEIGHT quantization theory, not as a proxy for hardware

**Detection:**
- If the protocol claims "FP4 accuracy" without qualifying the simulation, flag it
- Check whether input quantization is applied anywhere in the measurement setup

**Phase to address:** MEAS-01 (documentation limitation), future work (hardware study)

---

### PITFALL 5.3: Per-Channel Scaling Changes the Perturbation Structure

**Severity:** HIGH

**What goes wrong:** `gptq.py` and `fp_quantizer.py` use per-channel quantization: each row of W is scaled by its own max magnitude, quantized in normalized space, then denormalized. This means:

W_q = row_scale * Q(W / row_scale)

The perturbation delta_W = W_q - W has STRUCTURE: rows with larger dynamic range suffer different quantization errors than rows with smaller dynamic range. The Frobenius norm ||delta_W||/||W|| is a single number that hides this structure.

When measuring ||delta_y||/||y||:
- y = Wx involves the original row scaling
- y_q = W_q x involves quantized row scaling
- The ratio is automatically per-row correct because y and y_q use the same x

**Why it happens:** Per-channel quantization is standard and effective. But the measurement of ||delta_W|| (Frobenius norm of the weight perturbation) does not capture the per-row structure. ||delta_y||/||y|| does capture it because it operates in output space.

**Consequences:**
- Comparing ||delta_y||/||y|| to kappa * ||delta_W||/||W|| is an unfair comparison: the left side incorporates per-channel structure, the right side uses a norm that hides it
- The predicted bound kappa * ||delta_W||/||W|| may be systematically loose because ||delta_W||_F is inflated by row scaling differences that don't affect the output

**Prevention:**
- Compute the WEIGHTED perturbation: ||S^{-1} delta_W||/||S^{-1} W|| where S = diag(row_scale)
- This normalizes the perturbation to the per-channel normalized space where quantization actually occurs
- Or: report both unweighted and weighted kappa * ||delta_W||/||W||
- Best: directly compare ||delta_y||/||y|| to kappa * ||delta_W_normalized||/||W_normalized|| where W_normalized = W / row_scale

**Detection:**
- If per-channel quantization is used and ||delta_W|| is computed as Frobenius norm of unnormalized weights, the metric conflates channel scaling with quantization error
- Compare: ||delta_W_raw|| vs ||delta_W_normalized|| — they should differ by max(row_scale)/min(row_scale) factor

**Phase to address:** MEAS-01

---

### PITFALL 5.4: GPTQ Compensation Invalidates the Per-Matrix Independence Assumption

**Severity:** CRITICAL

**What goes wrong:** GPTQ compensates quantization error in column j by updating columns j+1,...,n (gptq.py line 111: `W[:, j+1:] -= error * scale`). After GPTQ compensation, a weight matrix is NOT independently quantized — the later columns contain corrections for earlier columns. Theorem 1 assumes delta_W is an independent perturbation. GPTQ's delta_W is structured with column dependencies.

For the downstream layers: after GPTQ modifies column j+1 of layer L's weight, the input to layer L+1 changes (because layer L's output changes). But GPTQ's compensation is designed to preserve layer L's output, not to be benign for downstream layers.

**Why it happens:** GPTQ's objective is to minimize ||WX - W_q X||_F for a fixed calibration dataset X. It succeeds at this: output MSE at layer L drops dramatically with GPTQ. But:
- The compensated columns may have very different weights than the original, just in directions that don't affect the calibration outputs
- These directions may matter for NON-calibration data (out-of-distribution robustness)
- Theorem 1's bound assumes delta_W is the quantization error, but GPTQ adds structured compensation that is not simple quantization error

**Consequences:**
- For GPTQ-compensated weights, ||delta_W||/||W|| is meaningful but kappa(W) * ||delta_W||/||W|| may be a GROSS overestimate because:
  - delta_W is designed to be in the nullspace of X (i.e., delta_W * X ~ 0)
  - But delta_W may be large in directions outside the span of X
  - kappa(W) captures worst-case amplification over ALL directions, not just those present in X
- The measured ||delta_y||/||y|| on calibration data is small, but on held-out data may be much larger

**Prevention:**
- Measure ||delta_y||/||y|| on HELD-OUT data (not the calibration data used by GPTQ)
- Report the ratio: (||delta_y||_heldout / ||y||_heldout) / (||delta_y||_calib / ||y||_calib)
- Compute the out-of-distribution amplification factor
- For Theorem 1 validation, use simple round-to-nearest (no GPTQ) to test the theory cleanly

**Detection:**
- If GPTQ is enabled, the per-matrix ||delta_y||/||y|| will be suspiciously small on calibration data
- Compare with random noise injection of same ||delta_W||_F: GPTQ's output error should be much smaller, confirming the structured compensation

**Phase to address:** MEAS-01 (use round-to-nearest for Theorem 1), MEAS-05 (include GPTQ as separate comparison)

---

## 6. Additional Pitfalls from Codebase Audit

### PITFALL 6.1: Sensitivity Analysis Computes Weight-Space Bound, Not Output Error

**Severity:** CRITICAL

**What goes wrong:** `condition.py` function `analyze_quantization_sensitivity()` (lines 135-167) computes:
```python
rel_error = ||W_q - W|| / ||W||
predicted_output_error = kappa * rel_error
```
But this computes kappa * ||delta_W||/||W||, which is the WEIGHTED PERTURBATION bound from Theorem 1. It does NOT measure the actual ||delta_y||/||y||. The function never runs the model forward or computes y = Wx. It entirely skips the output-space measurement.

**Why it happens:** The function name says "quantization sensitivity" but it computes only the weight-space bound. This would be a valid comparison if the code also computed actual ||delta_y||/||y||, but it doesn't.

**Consequences:**
- Everything in `analyze_quantization_sensitivity` is a THEORETICAL bound, not an EMPIRICAL measurement
- Any correlation analysis using this function is correlating kappa with itself (since predicted_output_error is just kappa * rel_error, and rel_error is roughly constant across matrices)
- The function should be renamed or rewritten for MEAS-01

**Prevention:**
- For MEAS-01: implement a new function that runs saved activations x through clean and quantized weights, computing actual ||(W_q - W)x||/||Wx||
- Keep the weight-space bound as a comparison, but clearly label it as "theoretical bound" not "measured error"

**Detection:**
- Does ANY code in the current codebase compute ||(W_q - W)x||/||Wx|| for actual model activations x? (Answer: NO, not in the analysis module — gptq.py computes MSE on y, but not the relative norm ratio. `validate_rmsnorm.py` computes similar metrics but for RMSNorm.)
- This is the core gap that MEAS-01 is designed to fill.

**Phase to address:** MEAS-01 (this is the primary measurement gap)

---

### PITFALL 6.2: Data Leakage Between Calibration and Evaluation

**Severity:** CRITICAL

**What goes wrong:** As documented in `ANALYSIS.md` Section 1.4 and `PROJECT.md`: there is no train/val split. `get_dataloader()` loads all .bin files with `shuffle=True`. GPTQ calibration (50-100 steps) and PPL evaluation (100 steps) sample from the same pool. This means:

1. GPTQ's Hessian estimation uses evaluation data statistics
2. Adaptive grid calibration uses evaluation data statistics
3. The evaluation PPL is not a true held-out measurement

**Why it happens:** The data pipeline was built for training, where no held-out set is needed. The evaluation protocol reused the same `get_dataloader()` without implementing a split.

**Consequences:**
- All quantitative evaluation results are optimistic (calibration leakage inflates apparent performance)
- The effect is largest for data-adaptive methods (GPTQ, Lloyd-Max adaptive grids)
- Per-matrix ||delta_y||/||y|| measured on calibration data may not generalize
- This invalidates any claim about "best method" ranking

**Prevention:**
- Implement train/val split: hold out 5% of each tier for evaluation
- The split must happen at the .bin file level, not in the DataLoader shuffle
- MEAS-01's saved activations should use HELD-OUT data for the forward pass

**Detection:**
- `ANALYSIS.md` identifies this clearly: "PTQ calibration data (GPTQ Hessian, Lloyd-Max grid optimization) and evaluation data are drawn from the same pool"
- Check: does `_collect_activations` or the evaluation loop have explicit train/val split logic? (Answer: NO — this is the pending fix)

**Phase to address:** MEAS-04 (data split implementation), MEAS-05 (rerun with clean split)

---

### PITFALL 6.3: Single-Seed, Single-Run Reporting

**Severity:** HIGH

**What goes wrong:** All reported results in REPORT.md are single-run point estimates with no variance estimate. The planned "3 seeds" protocol in PROPOSAL.md Section 3.4 was never executed.

**Why it happens:** The 164M model training takes ~8 hours on 8x RTX 4090. Repeating 3x for each of 24 PTQ configurations is expensive. But for per-matrix measurement (MEAS-01), the cost is much lower (one forward pass), so variance estimation should be standard.

**Consequences:**
- Cannot distinguish between "method A is genuinely better" and "method A won the seed lottery"
- Any correlation or ranking is unstable
- The PPL difference between methods (e.g., 638.7 vs 655.1) cannot be evaluated for significance

**Prevention:**
- For MEAS-01: run on 3 different random seeds of calibration data (different subsets of held-out data)
- Report: mean +- std of ||delta_y||/||y||, kappa, and their correlation
- For correlation analysis: report confidence intervals via bootstrap (resample tokens with replacement, recompute correlation 1000x)

**Detection:**
- If any measurement result is reported without standard deviation, flag as point estimate
- Bootstrap confidence intervals are especially important for correlation coefficients (which are non-normally distributed)

**Phase to address:** MEAS-01, MEAS-05

---

### PITFALL 6.4: Lipschitz Propagation Factor Uses Spectral Norm, Not Actual Error Dynamics

**Severity:** MODERATE

**What goes wrong:** `lipschitz.py` computes L_k as the max spectral norm of any weight matrix in layer k, then computes propagation_factor = product of downstream L_k. This assumes worst-case error growth. Actual error propagation is typically much milder because:
1. Errors don't align with the worst-case direction at each layer
2. GELU activation compresses large negative inputs to near-zero
3. Attention softmax concentrates on high-scoring tokens, limiting error spread

The propagation factor of ~4000 for layer 0 is a worst-case bound, not an expected value.

**Consequences:**
- The predicted output error (sensitivity.py line 51): predicted_impact = avg_kappa * total_mse * prop_factor is a WORST-CASE bound, not an expected value
- Comparing actual ||delta_y||/||y|| to this bound will show huge discrepancies (bound is billions of times larger)
- This does NOT invalidate the theory, but misinterpreting it as "the theory is wrong" is a pitfall

**Prevention:**
- Rename: "propagation_factor" is really "worst_case_propagation_factor" or "propagation_bound"
- Report actual measured propagation alongside the bound
- Compute empirical propagation: for each layer, what fraction of its error reaches the output? Measure directly via cascade experiment.

**Phase to address:** MEAS-02

---

### PITFALL 6.5: GQA and Sliding/Full Attention Create Heterogeneous Measurement Conditions

**Severity:** MODERATE

**What goes wrong:** The model uses GQA (4:1 ratio), sliding attention (layers 0,1,3,4,6,7,9,10) and full attention (layers 2,5,8,11). The head dimension differs: sliding heads are d_k=64, full attention heads are d_k=128. This creates heterogeneous measurement conditions:

- q_proj/k_proj/v_proj in sliding layers: output dim = 768 (12 heads x 64 dim)
- q_proj in full layers: output dim = 1536 (12 heads x 128 dim)
- o_proj input dim = concatenated head outputs = head_dim * num_heads (varies by type)

This means ||y|| (denominator in relative error) has different expected magnitudes for different layer types purely based on dimensionality.

**Why it happens:** The architecture alternates sliding and full attention layers. Weight matrix dimensions change between these types. The measurement protocol must account for this.

**Consequences:**
- Aggregating ||delta_y||/||y|| across sliding and full layers mixes measurements with different dimensionalities
- The denominator ||y|| for full attention q_proj is sqrt(1536/768) = 1.4x larger than for sliding q_proj, purely due to dimension
- This artifact should not be interpreted as a "layer type" effect

**Prevention:**
- Normalize ||delta_y||/||y|| by sqrt(dim) when comparing across different-sized layers
- Or: report per-unit-dimension error: ||delta_y|| / (||y|| * sqrt(dim_y))
- Analyze sliding and full layers separately, not pooled
- The current config (`config.py`) already labels layer types — use this stratification

**Detection:**
- Compare the scatter of ||y|| between sliding and full layers: if systematically different, stratification is needed
- Check: does the kappa-||delta_y||/||y|| correlation differ when controlling for layer type?

**Phase to address:** MEAS-01

---

### PITFALL 6.6: PL Embeddings Change the Linear Layer Input Dimension

**Severity:** MODERATE

**What goes wrong:** Each transformer layer concatenates `hidden_states` (768-dim) with `per_layer_embeddings` (64-dim) before projecting (transformer.py lines 118, 163). This means:
- q_proj, k_proj, v_proj, gate_proj, up_proj input dim = 768 + 64 = 832
- o_proj, down_proj output/input dim = 768

The 832-dim input includes 64 dims of token embedding that are repeated per-layer. These PL embeddings may have different spectral properties than the hidden states.

**Consequences:**
- The "x" in Theorem 1 (||delta_y||/||y|| <= kappa * ||delta_W||/||W||) for q_proj/k_proj/v_proj is a concatenation of hidden states and PL embeddings
- The PL embeddings may be lower-rank or have different singular value distribution than hidden states
- kappa(W) for a 832x768 matrix reflects the combined input space, not just the hidden states
- ||delta_y||/||y|| for these layers depends on both components, potentially diluting or amplifying the correlation

**Prevention:**
- When decomposing results, distinguish between:
  - Layers with PL-embedded inputs (q/k/v/gate/up projections) — input dim 832
  - Layers with pure hidden state inputs (o/down projections) — input dim 768
- For Theorem 1 validation, consider analyzing the 832-dim projections separately
- Option: compute kappa for the submatrix acting on hidden states vs PL embeddings separately

**Detection:**
- If a weight matrix has dim 832 (not 768), it involves PL embeddings
- Check: does the correlation between kappa and ||delta_y||/||y|| differ between 832-dim and 768-dim matrices?

**Phase to address:** MEAS-01

---

## 7. Meta-Pitfalls in Protocol Design

### PITFALL 7.1: Measuring Too Many Things Without Prioritization

**Severity:** HIGH

**What goes wrong:** The planned protocol includes MEAS-01 through MEAS-05: per-matrix Theorem 1, full error propagation trace, RMSNorm attenuation, data split fix, and rerun comparison. This is ambitious. The risk: the protocol produces so many numbers that no clear conclusion emerges.

**Why it happens:** The project is in its final phase with a 4-week timeline. The temptation is to measure everything to "be thorough." But without a crisp hypothesis, the measurement campaign generates noise.

**Consequences:**
- Results table with 72 correlation coefficients and no clear takeaway
- Difficulty writing the final report because there's no "headline finding"
- Wasted compute time on low-value measurements

**Prevention:**
- Define the PRIMARY hypothesis before measurement: "kappa(W) predicts ||delta_y||/||y|| at the individual matrix level" (YES/NO)
- Define the SECONDARY question: "Does RMSNorm change the relationship?"
- Define the TERTIARY question: "Which projection types are most/least sensitive?"
- All other measurements (per-layer, per-type breakdowns) are exploratory and should be labeled as such
- Pre-register the primary analysis plan, including which correlation test, which correction, and what constitutes "confirmation"

**Phase to address:** Before any measurement execution — the protocol design phase

---

### PITFALL 7.2: Confirmation Bias in Reporting

**Severity:** HIGH

**What goes wrong:** The project has a theoretical prediction (Theorem 1) that the team already believes is falsified (based on PPL results). The measurement protocol may unconsciously be designed to confirm the falsification rather than test it fairly.

Signs of confirmation bias:
- Using the exact same data split/corruption that made PPL fail (but now fixed by MEAS-04)
- Emphasizing layers where theory fails, downplaying layers where it works
- Setting significance bars differently for "supporting" vs "contradicting" evidence

**Why it happens:** The team has invested in the theoretical framework and is now in "damage control" mode after the PPL falsification. The natural psychological response is to show WHY the theory failed, not to give it a fair test.

**Consequences:**
- The protocol creates a self-fulfilling "theory is wrong" narrative
- Overlooked: maybe kappa predicts ||delta_y||/||y|| for some projection types but not others, which is still a useful finding
- The RMSNorm blocking finding (1221x) already proves that error propagation is strongly architecture-dependent — this is a positive result, not a failure

**Prevention:**
- State possible outcomes before measuring: "If r > 0.3 for >50% of matrices, Theorem 1 is partially validated. If r < 0.1 for >80% of matrices, Theorem 1 is not supported at this granularity."
- Report ALL 72 results, including those that support the theory
- The most interesting scientific outcome may be that kappa predicts error for some projection types (q_proj) but not others (o_proj) — this is a finding, not a failure

**Phase to address:** Protocol design — pre-register outcomes before seeing the data

---

## Summary: Phase-Specific Warnings

| Phase Topic | Critical Pitfall | Mitigation |
|-------------|------------------|------------|
| MEAS-01 (per-matrix Theorem 1) | Compute ||(W_q - W)x||/||Wx|| from SAVED activations, not two forward passes | One clean forward pass, store activations, compute per-matrix offline |
| MEAS-01 | Multiple comparison problem across 72 matrices | Bonferroni correction or FDR; report effect sizes not just p-values |
| MEAS-01 | Per-channel scaling confounds ||delta_W||/||W|| | Compute in normalized space; report both weighted and unweighted |
| MEAS-01 | GPTQ compensation invalidates per-matrix independence | Use round-to-nearest for Theorem 1, GPTQ as separate comparison |
| MEAS-01 | Null measurement not included | Always run W_q = W control to establish noise floor |
| MEAS-02 (error propagation) | Hook ordering and memory accumulation | Offload to CPU per-batch; verify hook cleanup |
| MEAS-02 | Lipschitz product overestimates actual propagation | Bound vs expectation: report both |
| MEAS-03 (RMSNorm) | Unit-RMS denominator illusion | "Relative error after RMSNorm" is actually absolute/sqrt(d) |
| MEAS-03 | Parallel/orthogonal decomposition needed | Measure angle between delta and y, not just magnitude ratio |
| MEAS-04 (data split) | Incomplete split causes calibration leak | Implement at .bin file creation, not DataLoader level |
| MEAS-05 (rerun) | Single-run reporting | Minimum 3 seeds; report mean +- std |

## Pitfalls Already Encountered (for reference)

These are documented in `docs/ANALYSIS.md` and `PROJECT.md`:

1. **Inverse power iteration computed sigma_max, not sigma_min** — kappa underestimated by ~5000x. FIXED: replaced with exact SVD.
2. **PPL is too far downstream from weight perturbation** — PPL can't capture per-matrix ||delta_y||/||y||. FIXED: new measurement protocol (MEAS-01) uses direct layer output hooking.
3. **No train/val split** — calibration data leaks into evaluation. FIXING: MEAS-04.
4. **Condition number regularization made quantization worse** — not necessarily a measurement error; may be a genuine theoretical limitation. Needs investigation in MEAS-01.
5. **FP4 unit roundoff initially wrong** (0.0625 -> 0.25). FIXED.
6. **Single-run point estimates without variance**. FIXING: MEAS-05.
7. **3-seed protocol planned but not executed**. FIXING: MEAS-05.

## Sources

- Codebase audit of `condition.py`, `sensitivity.py`, `lipschitz.py`, `gptq.py`, `transformer.py`, `config.py`, `training_utils.py`, `adaptive_grid.py`
- `docs/ANALYSIS.md` (experimental design audit and mathematical derivations)
- `.planning/PROJECT.md` (project scope and active tasks)
- Established numerical analysis principles (Higham 2002, Accuracy and Stability of Numerical Algorithms)
- Established statistical principles (Bonferroni correction, Simpson's paradox, confounding variables)
- PyTorch 2.x forward hook documentation (general patterns verified against codebase usage)
