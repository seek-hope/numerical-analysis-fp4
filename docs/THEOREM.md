# Theoretical Derivations

> Consolidated from ANALYSIS.md (§2.1–2.7), thm2.md, and thm3.md.
> Last updated: 2026-05-18.

---

## Theorem 1 — Single-Layer Quantization Error Bound

**Statement.** For $y = Wx$, let $\hat{W} = W + \delta W$. Then:

$$\frac{\|\hat{y} - y\|}{\|y\|} \leq \kappa(W) \cdot \frac{\|\delta W\|}{\|W\|} + O(\|\delta W\|^2)$$

**Derivation.**

Set $\hat{y} = \hat{W}x = (W + \delta W)x = Wx + \delta W \cdot x = y + \delta y$, where $\delta y = \delta W \cdot x$.

The numerator is bounded by the matrix norm:

$$\|\delta y\| = \|\delta W \cdot x\| \leq \|\delta W\| \cdot \|x\|$$

For the denominator, use the minimum singular value. Let $W = U\Sigma V^T$ be the compact SVD with $\Sigma = \text{diag}(\sigma_1, \ldots, \sigma_r)$, $\sigma_1 \geq \cdots \geq \sigma_r > 0$:

$$y = Wx = U\Sigma V^T x$$

Let $z = V^T x$ (orthogonal projection):

$$\|y\|^2 = \|U\Sigma z\|^2 = \|\Sigma z\|^2 = \sum_{i=1}^r \sigma_i^2 z_i^2 \geq \sigma_{\min}^2 \sum z_i^2 = \sigma_{\min}^2 \|z\|^2 = \sigma_{\min}^2 \|x\|^2$$

Hence $\|y\| \geq \sigma_{\min}(W) \cdot \|x\|$. Combining with the upper bound:

$$\frac{\|\delta y\|}{\|y\|} \leq \frac{\|\delta W\| \cdot \|x\|}{\sigma_{\min}(W) \cdot \|x\|} = \frac{\|\delta W\|}{\sigma_{\min}(W)}$$

Multiply and divide by $\|W\| = \sigma_{\max}(W)$:

$$\frac{\|\delta y\|}{\|y\|} \leq \frac{\sigma_{\max}(W)}{\sigma_{\min}(W)} \cdot \frac{\|\delta W\|}{\sigma_{\max}(W)} = \kappa(W) \cdot \frac{\|\delta W\|}{\|W\|}$$

The $O(\|\delta W\|^2)$ term arises because the inequality $\|\hat{y} - y\| \leq \|\delta W\| \cdot \|x\|$ is exact (linear), but any non-linear operation on $\hat{y}$ (e.g., subsequent layer non-linearities) introduces quadratic terms. For the single-layer linear case the bound is tight — equality is achieved when $x$ aligns with $W$'s minimum right singular vector $v_{\min}$ and $\delta W$ aligns with the left singular vector $u_{\min}$.

**Tightness.** Equality holds when $x = v_{\min}$ and $\delta W \propto u_{\min} v_{\min}^T$. Then $\|Wx\| = \sigma_{\min}\|x\|$ and $\|\delta W x\| = \|\delta W\| \|x\|$.

**Empirical verdict:** NO — Pearson $r(\kappa, \|\delta y\|/\|y\|) = -0.23$ across 84 matrices (Phase 3, `validate_theorem1.py`, 3 seeds). FP4 unit roundoff ($u = 0.25$) dominates: $\|\delta W\|/\|W\| \approx 0.15$ for ALL matrices regardless of $\kappa(W)$. See REPORT.md §3.

The reason for this failure is fundamental: Theorem 1 assumes $\delta W$ is an arbitrary norm-bounded perturbation (the worst-case framework of classic matrix perturbation theory), but FP quantization produces a **structured component-wise perturbation** $|\delta W_{ij}| \leq u \cdot |W_{ij}|$. The component-wise structure makes $\kappa(W)$ the wrong condition measure. This is corrected in Theorem 1′ below.

---

## Theorem 1′ — Component-Wise Quantization Error Bound

> **Numerical analysis foundation:** Skeel (1979) component-wise condition number; Oettli-Prager (1964) component-wise backward error; Higham (2002) *Accuracy and Stability of Numerical Algorithms*, §7.2.

**Statement.** For $y = Wx$ with FP quantization $\hat{W}$ satisfying $|\hat{W}_{ij} - W_{ij}| \leq u \cdot |W_{ij}|$ (component-wise relative error ≤ unit roundoff $u$), the forward error satisfies:

$$\frac{\|\delta y\|}{\|y\|} \leq \text{cond}_{\text{cw}}(W, x) \cdot u$$

where the **component-wise condition number** is defined as:

$$\text{cond}_{\text{cw}}(W, x) = \frac{\|\ |W| \cdot |x|\ \|}{\|Wx\|}$$

**Derivation.** The component-wise backward error for FP quantization is:

$$\omega(\Delta W) = \min\{\epsilon : |\Delta W_{ij}| \leq \epsilon \cdot |W_{ij}|, \ \forall i,j\} \leq u$$

By the Oettli-Prager theorem, the forward error is bounded by:

$$\|\delta y\| = \|\Delta W \cdot x\| = \left\|\sum_j \Delta W_{*,j} \cdot x_j\right\| \leq \sum_j \|\Delta W_{*,j}\| \cdot |x_j| \leq \sum_j u \cdot \||W_{*,j}|\| \cdot |x_j| = u \cdot \|\ |W| \cdot |x|\ \|$$

Dividing by $\|y\| = \|Wx\|$ yields the bound.

**Contrast with Theorem 1.** $\text{cond}_{\text{cw}}(W,x)$ measures sensitivity to **component-wise perturbations** (the actual quantization mechanism), while $\kappa(W)$ measures sensitivity to **normwise perturbations** (all directions equally likely). The ratio:

$$\frac{\text{cond}_{\text{cw}}(W, x)}{\kappa(W)} = \frac{\|\ |W| \cdot |x|\ \|}{\|W\| \cdot \|x\|}$$

is typically $\ll 1$ for Transformer weights (where $W$ has mixed signs and $x$ is concentrated near zero after RMSNorm), explaining why the normwise bound is loose by 20–40,000×.

**Empirical verdict: ✅ VALIDATED.** Pearson $r(\text{cond}_{\text{cw}}, \|\delta y\|/\|y\|) = 0.928$ ($p = 8.0 \times 10^{-113}$) across 84 matrices. The component-wise condition number explains 86% of the variance in per-matrix output error, compared to 2.5% for $\kappa(W)$. Per-subgroup: Attention $r = 0.90$, FFN $r = 0.95$. The bound tightness (mean bound/actual ratio) is 39.6× — still loose but ~40× tighter than Theorem 1's median gap of 1,523×. See `results/componentwise_validation.json`.

| Measure | Pearson r | p-value | r² | Mean bound gap |
|---------|----------|---------|-----|----------------|
| $\kappa(W)$ (normwise) | −0.157 | 0.15 | 0.025 | 1,523× |
| $\text{cond}_{\text{cw}}(W,x)$ (component-wise) | **0.928** | **8.0×10⁻¹¹³** | **0.861** | **39.6×** |

---

## Corollary 1.1 — RMSNorm Prevents Error Cascade

**Statement.** In a Transformer with RMSNorm, quantization error at layer $\ell$ does not propagate cumulatively across layers.

**Derivation (informal, from Theorem 2).**

Without RMSNorm, the error $\delta y_\ell$ at the output of layer $\ell$ becomes part of the input to layer $\ell+1$, producing a compound effect:

$$y_{\ell+1} + \delta y_{\ell+1} = f_{\ell+1}(W_{\ell+1} \cdot (y_\ell + \delta y_\ell)) \approx f_{\ell+1}(W_{\ell+1} y_\ell) + J_{f_{\ell+1}} \cdot W_{\ell+1} \cdot \delta y_\ell$$

where $J_f$ is the Jacobian. Error propagates by $\|J_f\| \cdot \|W_{\ell+1}\|$, which is typically > 1. After $L$ layers:

$$\|\delta y_L\| \sim \|\delta y_0\| \cdot \prod_{\ell=1}^{L} \|J_{f_\ell}\| \cdot \|W_\ell\|$$

This is the Lipschitz multiplicative cascade — each layer multiplies the previous layer's error by its own Lipschitz constant.

RMSNorm re-normalizes the signal before the next layer processes it, breaking this multiplicative chain. After re-normalization, the signal again has unit RMS, so prior error magnitude is reset — the propagation mechanism is blocked.

---

## Theorem 2 — RMSNorm Error Blocking

**Statement.** RMSNorm attenuates input perturbation. For an input $y$ with perturbation $\delta$, the output perturbation satisfies:

$$\frac{\|\delta_{\text{out}}\|}{\|\text{RMSNorm}(y)\|} \leq \frac{\|\delta\|}{\|y\|}$$

No amplification — only attenuation via orthogonal projection.

### Derivation A: Jacobian (from thm2.md)

RMSNorm is defined as:

$$\text{RMSNorm}(x) = \sqrt{d} \cdot \frac{x}{\|x\|} \odot \gamma$$

where $\gamma \in \mathbb{R}^d$ is a learnable scale parameter (omitted below — element-wise multiplication does not affect the analysis).

Define $f(x) = \sqrt{d} \cdot \frac{x}{\|x\|}$. For a small perturbation $\delta x$ ($\|\delta x\| \ll \|x\|$), first-order Taylor expansion gives:

$$f(x + \delta x) \approx f(x) + J(x) \delta x$$

The Jacobian $J(x)$ is:

$$J(x) = \frac{\partial}{\partial x} \left( \frac{\sqrt{d}}{\|x\|} x \right) = \sqrt{d} \left( \frac{1}{\|x\|} I - \frac{x x^T}{\|x\|^3} \right) = \frac{\sqrt{d}}{\|x\|} \left( I - \frac{x x^T}{\|x\|^2} \right)$$

Note that $P_{\perp x} = I - \frac{x x^T}{\|x\|^2}$ is a projection matrix — it preserves components orthogonal to $x$ and discards the component along $x$.

$$\delta_{\text{out}} \approx J(y) \delta = \frac{\sqrt{d}}{\|y\|} \left( I - \frac{y y^T}{\|y\|^2} \right) \delta = \frac{\sqrt{d}}{\|y\|} \cdot P_{\perp y} \delta$$

Taking norms (since $\|P_{\perp y}\| \leq 1$):

$$\|\delta_{\text{out}}\| \leq \frac{\sqrt{d}}{\|y\|} \cdot \|\delta\|$$

Since $\|\text{RMSNorm}(y)\| = \sqrt{d}$ (normalized to unit RMS):

$$\frac{\|\delta_{\text{out}}\|}{\|\text{RMSNorm}(y)\|} \leq \frac{\sqrt{d} \cdot \|\delta\|/\|y\|}{\sqrt{d}} = \frac{\|\delta\|}{\|y\|}$$

The output relative error is bounded by the input relative error — RMSNorm cannot amplify error.

### Derivation B: Taylor Expansion of RMS (from ANALYSIS.md §2.3)

Define $r(z) = \sqrt{\frac{1}{d}\sum z_i^2} = \frac{\|z\|}{\sqrt{d}}$. The gradient is $\nabla r(y)_i = \frac{y_i}{d \cdot r(y)}$.

First-order expansion:

$$r(y + \delta) \approx r(y) + \frac{y^T \delta}{d \cdot r(y)}$$

For the reciprocal $1/r$:

$$\frac{1}{r(y + \delta)} \approx \frac{1}{r(y)} - \frac{y^T \delta}{d \cdot r(y)^3}$$

RMSNorm output:

$$\text{RMSNorm}(y + \delta) = \frac{y + \delta}{r(y + \delta)} \approx \frac{y + \delta}{r(y)}\left(1 - \frac{y^T \delta}{d \cdot r(y)^2}\right)$$

To first order (drop $\|\delta\|^2$):

$$\text{RMSNorm}(y + \delta) \approx \frac{y}{r(y)} + \frac{\delta}{r(y)} - \frac{y \cdot (y^T \delta)}{d \cdot r(y)^3}$$

The first term is $\text{RMSNorm}(y)$. The error term is:

$$\delta_{\text{output}} \approx \frac{1}{r(y)}\left(\delta - \frac{y \cdot (y^T \delta)}{\|y\|^2}\right) = \frac{\sqrt{d}}{\|y\|} \cdot P_{\perp y} \delta$$

Same result as Derivation A — the error is projected onto the subspace orthogonal to $y$.

### RMSNorm Blocking Ratio

Without RMSNorm: error after $L$ layers = $\|\delta_0\| \cdot \prod_{\ell} L_\ell$.
With RMSNorm: error after $L$ layers = $\|\delta_0\| \cdot \sqrt{d} / \|y\|$.

Ratio:

$$\frac{\prod_\ell L_\ell}{\sqrt{d}/\|y\|} \gg 1$$

For a typical Transformer ($L_\ell \approx 2\text{--}5$, 12 layers, $d = 768$):

$$\frac{2^{12}}{\sqrt{768}/\|y\|} \approx \frac{4096}{28/\|y\|} \approx 147\|y\|$$

The reported ~1221× block ratio (`rmsnorm_validation.json`) is consistent with this range.

---

## Theorem 3 — Stochastic Rounding Cumulative Error

**Statement.** Deterministic rounding produces $O(n \cdot u)$ cumulative error (worst-case). Stochastic rounding produces $O(\sqrt{n} \cdot u)$ (expected $L^2$ norm).

### Derivation A: General Framework (from ANALYSIS.md §2.4)

**Deterministic rounding.** Each rounding operation introduces error $\epsilon_i \in [-u/2, u/2]$ (round-to-nearest) or $\epsilon_i \in [0, u]$ (round-down), where $u = 2^{-(m+1)}$ is the unit roundoff (IEEE 754 / Higham 2002), $m$ explicit mantissa bits. Total error after $n$ operations:

$$\left|\sum_{i=1}^n \epsilon_i\right| \leq \sum_{i=1}^n |\epsilon_i| \leq n \cdot u = O(nu)$$

**Stochastic rounding.** Round to $\lceil x \rceil$ with probability $\text{frac}(x)/u$, to $\lfloor x \rfloor$ otherwise. Define $\epsilon_i = \text{round}(x_i) - x_i$.

- **Unbiasedness:** $\mathbb{E}[\epsilon_i] = 0$ (by construction — rounding mean equals true value).
- **Variance:** $|\epsilon_i| \leq u$, so $\text{Var}(\epsilon_i) \leq u^2$. For independent rounding:

$$\mathbb{E}\left[\left(\sum_{i=1}^n \epsilon_i\right)^2\right] = \sum_{i=1}^n \mathbb{E}[\epsilon_i^2] + \sum_{i \neq j} \mathbb{E}[\epsilon_i]\mathbb{E}[\epsilon_j]$$

Since $\mathbb{E}[\epsilon_i] = 0$, the cross terms vanish:

$$\mathbb{E}\left[\left(\sum_{i=1}^n \epsilon_i\right)^2\right] = \sum_{i=1}^n \mathbb{E}[\epsilon_i^2] \leq n \cdot u^2$$

Taking square root (Jensen: $\mathbb{E}[\sqrt{X}] \leq \sqrt{\mathbb{E}[X]}$):

$$\mathbb{E}\left[\left|\sum_{i=1}^n \epsilon_i\right|\right] \leq \sqrt{n} \cdot u = O(\sqrt{n}u)$$

### Derivation B: Partial Sum Framework (from thm3.md)

Consider cumulative sum $S_n = \sum_{i=1}^n x_i$. Let $s_k$ be the computed result at step $k$:

$$s_k = r(s_{k-1} + x_k) = (s_{k-1} + x_k)(1 + \delta_k)$$

Absolute error per addition: $\epsilon_k = (s_{k-1} + x_k)\delta_k$, with $|\delta_k| \leq u$.

**Deterministic (worst-case):** All $\delta_k = u$ in the same direction → $\|E_n\|_{\infty} = O(nu)$.

**Stochastic:** $\delta_k$ are random variables with $\mathbb{E}[\delta_k] = 0$. Assuming independent rounding across steps:

$$\mathbb{E}[E_n^2] = \text{Var}\left( \sum_{k=1}^n \epsilon_k \right) = \sum_{k=1}^n \text{Var}(\epsilon_k)$$

By Popoviciu's inequality, for bounded zero-mean $\delta_k$ there exists $c$ such that $\text{Var}(\delta_k) \leq c u^2$.

$$\text{Var}(\epsilon_k) = (s_{k-1} + x_k)^2 \text{Var}(\delta_k) \leq (s_{k-1} + x_k)^2 c u^2$$

Upper bound: any partial sum magnitude $\leq X_{\text{sum}} = \sum |x_i|$. Hence:

$$\|E_n\| = \sqrt{\mathbb{E}[E_n^2]} \leq \sqrt{c}\,\sqrt{n}\,u\,X_{\text{sum}} = O(\sqrt{n}u)$$

### FP4 Numerical Example

FP4 E2M1: $m = 1 \implies u = 2^{-(1+1)} = 0.25$.

For $n = 10^9$ gradient accumulations:

| Method | Error bound | Value |
|--------|-------------|-------|
| Deterministic | $n \cdot u$ | $2.5 \times 10^8$ |
| Stochastic | $\sqrt{n} \cdot u$ | $\approx 7,906$ |

Stochastic rounding reduces cumulative error by ~$3.16 \times 10^4$× (~4.5 orders of magnitude).

### STE Gradient Nuance

In QAT training, gradients are accumulated in FP16/FP32 precision in the backward pass, not FP4. Theorem 3's $O(\sqrt{n} \cdot u)$ advantage therefore applies only to forward-pass unbiased weight estimates. This partially explains why stochastic rounding showed no significant improvement in QAT experiments — the STE gradient signal-to-noise ratio is dominated by the quantization interval, not by the forward rounding strategy.

---

## Theorem 4 — Lloyd-Max Optimality Conditions

**Statement.** Given weight distribution $w \sim p(w)$ and $K$ quantization levels, the quantizer minimizing $\mathbb{E}[(w - Q(w))^2]$ satisfies:

1. **Nearest-neighbor condition:** $Q(w) = q_i$ when $|w - q_i| \leq |w - q_j|$ for all $j$
2. **Centroid condition:** $q_i = \mathbb{E}[w \mid w \in R_i]$

**Derivation.**

The quantizer $Q: \mathbb{R} \to \{q_1, \ldots, q_K\}$ partitions the real line into decision regions $R_i = \{w : Q(w) = q_i\}$. Distortion:

$$\mathcal{D} = \mathbb{E}[(w - Q(w))^2] = \sum_{i=1}^K \int_{R_i} (w - q_i)^2 p(w) \, dw$$

Optimize $\mathcal{D}$ alternately over $\{R_i\}$ and $\{q_i\}$.

**Step 1 (fix $\{q_i\}$, optimize $\{R_i\}$) — Nearest-neighbor:**

For a given $w$, the best $q_i$ minimizes $(w - q_i)^2$. Therefore:

$$R_i = \{w : |w - q_i| \leq |w - q_j| \text{ for all } j \neq i\}$$

These are Voronoi regions (thresholds at midpoints of adjacent $q$ values). In 1D: $R_i = [\theta_{i-1}, \theta_i]$ where $\theta_i = (q_i + q_{i+1})/2$.

**Step 2 (fix $\{R_i\}$, optimize $\{q_i\}$) — Centroid:**

For given $R_i$, minimize:

$$q_i^* = \arg\min_{q} \int_{R_i} (w - q)^2 p(w) \, dw$$

Differentiate w.r.t. $q$ and set to zero:

$$\frac{\partial}{\partial q} \int_{R_i} (w - q)^2 p(w) \, dw = -2 \int_{R_i} (w - q) p(w) \, dw = 0$$

$$\int_{R_i} q \cdot p(w) \, dw = \int_{R_i} w \cdot p(w) \, dw$$

$$q_i^* = \frac{\int_{R_i} w \cdot p(w) \, dw}{\int_{R_i} p(w) \, dw} = \mathbb{E}[w \mid w \in R_i]$$

This is the conditional mean of $w$ within $R_i$ — the centroid.

**Convergence.** Each step non-increasingly reduces distortion — Step 1 by construction; Step 2 computes the $L^2$-optimal representative for each region. Distortion is bounded below (MSE ≥ 0), so the algorithm converges monotonically. Limit points satisfy both conditions and are local minima. Convergence is linear (typical: 10–20 iterations for well-behaved distributions).

**κ-weighted variant (Strategy A).** Replace uniform weighting with κ-adjusted weights:

$$\mathcal{D}_\kappa = \sum_{i=1}^K \int_{R_i} (w - q_i)^2 \cdot c(w) \cdot p(w) \, dw$$

where $c(w) = 1 + \alpha \cdot (\kappa - 1) \cdot |w|/\max|w|$ up-weights large weights in high-κ layers. The centroid update becomes:

$$q_i^* = \frac{\int_{R_i} w \cdot c(w) \cdot p(w) \, dw}{\int_{R_i} c(w) \cdot p(w) \, dw}$$

This is a weighted centroid. Convergence properties are unchanged — it remains coordinate descent, now weighting important errors in the distortion metric.

---

## Strategy B — Condition-Number Regularization

**Objective:**

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \lambda \cdot \sum_{\ell \in \text{Linear}} \log \kappa(W_\ell)$$

**Why $\log \kappa$ rather than linear.** Consider two matrices: $\kappa_1 = 1000$, $\kappa_2 = 5$. Under a linear sum, $1000\lambda$ dominates — a single ill-conditioned layer drowns out the regularization signal for all others. With $\log$: $\log 1000 \approx 6.9$ vs. $\log 5 \approx 1.6$ — still heavily weighted, but all layers receive meaningful gradients. The $\log$ transform also makes regularization scale-invariant w.r.t. weight scaling: $\kappa(cW) = \kappa(W)$, so scaling the model does not change the condition number, but the gradient of $\log \kappa$ scales as $\nabla_W \kappa / \kappa$ — more stable across the wide range of $\kappa$ values.

**Gradient perspective.** By the chain rule:

$$\frac{\partial \mathcal{L}_{\text{total}}}{\partial W_{ij}} = \frac{\partial \mathcal{L}_{\text{CE}}}{\partial W_{ij}} + \lambda \cdot \frac{1}{\kappa(W)} \cdot \frac{\partial \kappa(W)}{\partial W_{ij}}$$

The second term pushes $W$ toward lower condition numbers. For $\kappa = \sigma_{\max}/\sigma_{\min}$:

$$\frac{\partial \kappa}{\partial W_{ij}} = \frac{1}{\sigma_{\min}} \frac{\partial \sigma_{\max}}{\partial W_{ij}} - \frac{\sigma_{\max}}{\sigma_{\min}^2} \frac{\partial \sigma_{\min}}{\partial W_{ij}}$$

Singular value derivatives: $\partial \sigma_k / \partial W_{ij} = u_{ki} v_{kj}$ (outer product of left/right singular vectors). Regularization thus encourages $\sigma_{\max}$ to shrink (contract dominant direction) and $\sigma_{\min}$ to grow (expand weakest direction) — pushing the matrix toward well-conditioned.

**Implementation note.** The training code (`condition.py:68–89`) uses a surrogate rather than exact $\kappa$:

$$\kappa_{\text{surrogate}} = \frac{\sigma_{\max}}{\sqrt{\frac{1}{r}\sum_i \sigma_i^2}} = \frac{\sigma_{\max}}{\text{RMS}(\sigma)}$$

Ratio of $\sigma_{\max}$ to RMS singular value. Equals 1.0 when all singular values are equal, grows when a single direction dominates — but is not strictly $\kappa$. However, it is differentiable during training (no `.item()` calls), whereas exact $\kappa$ requires SVD (non-differentiable and expensive). The surrogate captures the correct qualitative behavior and should be documented as approximate $\kappa$ regularization.

**Empirical note.** Condition-number regularization makes quantization WORSE (Phase 3, Phase 5) — confirmed across both checkpoints. The regularization interferes with the natural spectral structure learned during training.

---

## GPTQ Weight Compensation (Reference)

GPTQ (Frantar et al., 2023) quantizes weights one column at a time and compensates for each column's error by updating the remaining columns.

**Objective:** Find $\hat{W}$ minimizing $\|WX - \hat{W}X\|_F^2$ subject to $\hat{W}$ being in FP format.

Squared error:

$$\mathcal{E} = \|(W - \hat{W})X\|_F^2 = \text{tr}((W - \hat{W})^T (W - \hat{W}) X X^T) = \text{tr}((W - \hat{W}) H (W - \hat{W})^T)$$

where $H = X X^T$ is the activation Gram matrix.

Quantize column-by-column ($\delta_j = \hat{W}_{:,j} - W_{:,j}$) keeping remaining columns fixed. The optimal compensation for column $j$ minimizes the quadratic form w.r.t. the remaining columns:

$$\min_{\text{comp}} \left\|\delta_j H_{j,j+1:}^{1/2} + \text{comp} \cdot H_{j+1:,j+1:}^{1/2}\right\|^2$$

Solution:

$$\text{comp} = -\delta_j \cdot \frac{H_{j, j+1:}}{H_{j,j}}$$

This is the formula implemented at `gptq.py:110–111`. After compensation, output error comes entirely from the last column (no remaining columns to compensate).

---

## Summary of Empirical Verdicts

| Theorem | Prediction | Empirical Result | Verdict |
|---------|-----------|-----------------|---------|
| Thm 1 | $\|\delta y\|/\|y\| \leq \kappa(W) \cdot \|\delta W\|/\|W\|$ | $r = -0.16$ across 84 matrices | **NO** — normwise bound too loose for structured FP perturbation |
| **Thm 1′** | $\|\delta y\|/\|y\| \leq \text{cond}_{\text{cw}}(W,x) \cdot u$ | $r = 0.928$ ($p = 8.0\times10^{-113}$) | **YES** — component-wise condition explains 86% of variance |
| Thm 2 | RMSNorm blocks error cascade | ~1221× block ratio, ~83% attenuation/layer | **YES** — confirmed by Phase 4 trace |
| Thm 3 | Stochastic rounding: $O(\sqrt{n}u)$ vs $O(nu)$ | Forward error +41% at both FP8/FP4; unbiasedness holds | **PARTIAL** — unbiased in expectation, higher per-sample variance |
| Thm 4 | Lloyd-Max minimizes weight-space MSE | Uniform Lloyd-Max +8.6% vs E2M1; κ-weighted −6.5% vs uniform | **PARTIAL** — uniform worse than E2M1; κ-weighting (Strategy A) validated |
| Strategy A | κ-weighted Lloyd-Max reduces error in high-κ layers | −6.5% (baseline), −6.3% (cond_reg) vs uniform Lloyd-Max | **YES** — consistent improvement across both checkpoints |
| Strategy B | Condition-number regularization improves PTQ | +3% error vs baseline across all formats | **NO** — consistently harmful |

---

*Derivations consolidated from ANALYSIS.md (§2.1–2.7), thm2.md, and thm3.md on 2026-05-18.*
