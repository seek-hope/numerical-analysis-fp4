# Phase 3: Theorem 1 Validation - Research

**Researched:** 2026-05-17
**Domain:** Statistical validation of matrix perturbation theory in quantized Transformers
**Confidence:** HIGH

## Summary

This phase tests whether Theorem 1's predicted upper bound `||dy||/||y|| <= kappa * ||dW||/||W||` holds empirically at per-matrix granularity across all 72 Linear weight matrices in the Micro-Gemma-FP model. It consumes the ErrorPropagationTracker (Phase 2) for output-space error measurement, `compute_all_condition_numbers()` for condition numbers, and `FPQuantizer` for weight-space error. The core statistical work involves Pearson correlation with Bonferroni correction and bootstrap confidence intervals across 3 random seeds.

**Primary recommendation:** Build `src/experiments/validate_theorem1.py` as a single self-contained script following the pattern of `measure_qerror.py`. Use `scipy.stats.pearsonr` (available on remote as scipy 1.17.1) for correlation, pure numpy for bootstrap resampling. All required infrastructure (ErrorPropagationTracker, condition.py, FPQuantizer) is already validated from Phase 2.

**Critical prerequisite:** Sync the Phase 1 data split files to remote before running -- `*_val.bin` files exist locally but have not been synced to the GPU server.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01:** Use `scipy.stats.pearsonr` for Pearson correlation coefficient computation (p-value, r-value in one call). Compute tightness_ratio = (||dy||/||y||) / (kappa * ||dW||/||W||) for each matrix. Bonferroni threshold: alpha = 0.05/72 = 0.00069.
- **D-02:** Bootstrap 95% CI via 10,000 resamples of the 72 (kappa, ||dy||/||y||) pairs. For each resample, compute Pearson r. CI = [2.5th percentile, 97.5th percentile] of bootstrap r distribution. Report mean +/- std of r across 3 seeds, with combined bootstrap CI from pooled distributions.
- **D-03:** Run the full measurement pipeline (load checkpoint, attach tracker, single forward pass, compute ||dy||/||y||, kappa, ||dW||/||W||) for each of 3 seeds (42, 123, 456). Use the Phase 1 val split for the forward pass data. Each seed controls: (a) the random seed for data shuffling, (b) the evaluation batch selection.
- **D-04:** Aggregate across seeds: for each matrix, report mean and std of ||dy||/||y||, ||dW||/||W||, and tightness_ratio across the 3 seeds. Compute Pearson r(kappa, mean_||dy||) for the primary result. Report seed-by-seed r values for reproducibility assessment.
- **D-05:** Single analysis script `src/experiments/validate_theorem1.py` that: (1) loads the FP16 baseline checkpoint, (2) runs measurement pipeline for 3 seeds, (3) computes Pearson r, Bonferroni threshold, bootstrap CI, (4) prints the 72-matrix results table (name, layer, type, kappa, ||dW||/||W||, ||dy||/||y||, tightness_ratio), (5) states the YES/NO/QUALIFIED verdict with supporting statistics, (6) exports results as JSON for potential visualization.
- **D-06:** Verdict rubric: YES: r > 0.5 AND p < 0.00069 AND bootstrap CI excludes 0. QUALIFIED: r > 0.2 but doesn't meet all YES criteria. NO: r < 0.2 OR p > 0.05 (uncorrected) -- no meaningful linear relationship.
- **D-07:** Kappa(W) computed via `compute_all_condition_numbers()` (exact SVD from condition.py). ||dW||/||W|| computed as `(W_q - W_fp).norm() / W_fp.norm()` using FP4 round-to-nearest. ||dy||/||y|| from `ErrorPropagationTracker.compute_output_error()`.
- **D-08:** Results table uses module path naming convention. Layer index and type (attention/fnn/global) parsed from module path for grouping. Matrix type: `q_proj`/`k_proj`/`v_proj`/`o_proj` -> "attention", `gate_proj`/`up_proj`/`down_proj` -> "ffn", `embed_tokens`/`lm_head` -> "global".
- **D-09:** Primary outputs: (1) 72-row results table printed to stdout, (2) statistical summary with verdict, (3) `results/theorem1_validation.json` for downstream visualization (Phase 5).

### Claude's Discretion
- Whether to use scipy.stats or implement Pearson/bootstrapping in pure numpy
- Table formatting (column widths, alignment, sorting order)
- Number of bootstrap resamples (10,000 recommended, adjust for runtime)
- Whether to include per-layer-type subgroup analysis (attention-only, FFN-only correlations)
- Logging verbosity during multi-seed execution

### Deferred Ideas (OUT OF SCOPE)
- None
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| VAL-01 | For each of 72 Linear weight matrices, report (kappa, ||dy||/||y||, ||dW||/||W||, tightness_ratio) | ErrorPropagationTracker.compute_output_error() captures ||dy||/||y||. compute_all_condition_numbers() provides kappa via exact SVD. FPQuantizer('fp4_e2m1', per_channel=True).quantize() provides W_q for ||dW||/||W||. All validated in Phase 2. |
| VAL-02 | Compute Pearson r(kappa, ||dy||/||y||) with Bonferroni correction (alpha=0.05/72=0.00069) | scipy.stats.pearsonr available on remote (scipy 1.17.1). Pure numpy fallback via np.corrcoef is 5 lines. Bonferroni threshold is a constant comparison. |
| VAL-04 | Measure with 3 seeds (42, 123, 456), report mean+std, bootstrap 95% CI | Bootstrap CI implemented in ~15 lines of numpy: 10,000 resamples of 72 pairs with replacement, compute Pearson r each iteration, percentile CI. Seeds control data batch selection via torch.manual_seed(). |
</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Theorem 1 statistical validation | Analysis script (experiment) | -- | Standalone script that composes Phase 2 infrastructure (tracker, condition.py, quantizer) to produce a single verdict. No model modifications. |
| Per-matrix error measurement | ErrorPropagationTracker (Phase 2) | -- | Already implemented and validated. Phase 3 only calls compute_output_error() and compute_all_condition_numbers(). |
| Pearson correlation + bootstrapping | Analysis script (inline) | -- | Statistical computation is ~50 lines of Python (scipy + numpy). No separate statistics module needed. |
| Multi-seed orchestration | Analysis script (loop) | -- | 3-iteration loop: set seed, get dataloader, run forward pass, collect results. Single model load, detach/reattach tracker between seeds. |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| scipy.stats | 1.17.1 (remote) | Pearson r with p-value | D-01 locked decision. scipy.stats.pearsonr returns (r, p_value) in one call. Available in remote `sle` conda env. |
| numpy | >= 1.26.0 | Bootstrap resampling, percentile CI | Already in requirements.txt. No new install needed. |
| PyTorch | >= 2.3.0 | Model loading, forward pass, tensor ops | Core framework. Already in requirements.txt. |

### Supporting (Existing Code Reused)
| Component | Module Location | Purpose | Verified |
|-----------|-----------------|---------|----------|
| ErrorPropagationTracker | `src/analysis/error_propagation.py` | Activation capture, compute_output_error() | Yes -- Phase 2 |
| compute_all_condition_numbers() | `src/analysis/condition.py:55` | Kappa via exact SVD | Yes -- Phase 2 |
| FPQuantizer | `src/quantization/fp_quantizer.py` | FP4 round-to-nearest for ||dW||/||W|| | Yes -- Phase 2 |
| MicroGemmaFPConfig | `src/model/config.py` | Model architecture definition | Yes -- entire project |
| load_checkpoint() | `src/experiments/training_utils.py:359` | Checkpoint loading | Yes -- Phase 2 |
| get_dataloader() | `src/experiments/training_utils.py:204` | Validation data access | Yes -- Phase 1 |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| scipy.stats.pearsonr | Pure numpy (np.corrcoef) | Pure numpy is ~5 lines and avoids any dependency concern. scipy gives p-value directly, numpy requires manual t-distribution computation. Either works -- Claude's discretion. |
| Bootstrap CI | Parametric CI (Fisher z-transform) | Fisher z-transform assumes bivariate normality which may not hold. Bootstrap is distribution-free and preferred for correlation confidence intervals. |

**Version verification:**
- scipy on remote: `python -c "import scipy; print(scipy.__version__)"` -> 1.17.1 [VERIFIED: remote execution 2026-05-17]
- numpy: already in requirements.txt, used throughout project [VERIFIED: git grep numpy in src/]
- torch: already in requirements.txt [VERIFIED: git grep torch in src/]

## Package Legitimacy Audit

> No new packages are introduced in this phase. All dependencies (torch, numpy, scipy) are already present on the remote GPU server in the `sle` conda environment. The phase uses existing infrastructure from Phase 2 without adding any new external dependencies.

No packages to audit. This phase is purely composing existing code (ErrorPropagationTracker, condition.py, FPQuantizer) with standard library statistics (scipy, numpy).

## Architecture Patterns

### System Architecture Diagram

```
                    ┌─────────────────────────────────────────────┐
                    │         validate_theorem1.py                │
                    │                                             │
  ┌───────────┐    │  ┌────────┐   ┌────────────┐   ┌────────┐  │
  │ Checkpoint│───▶│  │ Load   │──▶│ 3-Seed Loop│──▶│ Report │  │
  │ model.pt  │    │  │ Model  │   │ (42,123,456)│   │ Verdict│  │
  └───────────┘    │  └────────┘   └─────┬──────┘   └────────┘  │
                   │                     │                       │
                   │              ┌──────┴──────┐                │
                   │              │ Per-Seed     │                │
                   │              │ Pipeline     │                │
                   │              └──────┬──────┘                │
                   │                     │                       │
                   │              ┌──────┴──────────────────┐     │
                   │              │                        │      │
                   │      ┌───────▼──────┐      ┌──────────▼──┐  │
                   │      │ Set seed,    │      │ Attach      │  │
                   │      │ get val batch│─────▶│ ErrorProp-  │  │
                   │      └───────┬──────┘      │ agation-    │  │
                   │              │              │ Tracker     │  │
                   │              │              └──────┬──────┘  │
                   │              │                     │         │
                   │              │              ┌──────▼──────┐ │
                   │              │              │ Single FP16 │ │
                   │              │◄─────────────│ forward pass│ │
                   │              │              └──────┬──────┘ │
                   │              │                     │         │
                   │              │              ┌──────┴──────┐ │
                   │              │              │ Detach,     │ │
                   │              │              │ compute     │ │
                   │              │◄─────────────│ P3/P6       │ │
                   │              │              └─────────────┘ │
                   │              │                              │
                   │              │  Offline per-matrix:         │
                   │              │  ┌──────────────────────┐    │
                   │              │  │ kappa (condition.py) │    │
                   │              │  │ ||dW||/||W|| (FPQ)   │    │
                   │              │  │ ||dy||/||y|| (tracker)│   │
                   │              │  │ tightness_ratio      │    │
                   │              │  └──────────────────────┘    │
                   │              │         Store per-seed       │
                   │              │         results dict         │
                   │              └──────────────────────────────┘
                   │
                   │   ┌──────────────────────────────────┐
                   │   │ Aggregation Phase                │
                   │   │ ┌──────────────────────────────┐ │
                   │   │ │ Per-matrix: mean+std across  │ │
                   │   │ │ 3 seeds for each metric      │ │
                   │   │ │ Pearson r(kappa, mean_||dy||)│ │
                   │   │ │ Bootstrap CI (10k resamples) │ │
                   │   │ │ Bonferroni p-threshold check │ │
                   │   │ │ Verdict (YES/QUALIFIED/NO)   │ │
                   │   │ └──────────────────────────────┘ │
                   │   └──────────────────────────────────┘
                   │
                   │   Outputs:
                   │   ├─ 72-row results table (stdout)
                   │   └─ results/theorem1_validation.json
                   └─────────────────────────────────────────────┘
```

### Data Flow

1. **Start**: Load FP16 baseline checkpoint -> model on GPU
2. **For each seed** (42, 123, 456):
   - torch.manual_seed(seed) -> affects dataloader batch selection
   - get_dataloader(split='val') -> yields batch of validation data
   - tracker.attach(model) -> registers hooks on all nn.Linear modules
   - model(input_ids) -> single forward pass captures activations
   - tracker.detach() -> removes hooks
   - compute_p3_p6() -> computes residual-add measurement points
   - Quantize each weight matrix: W_q = FPQuantizer.quantize(W_fp) -> store ||dW||/||W||
   - tracker.compute_output_error(model, quantizer) -> store ||dy||/||y||
   - compute_all_condition_numbers(model) -> store kappa
   - Build per-seed results dict
3. **Aggregate**: Mean+std of each metric per matrix across 3 seeds
4. **Statistical test**: Pearson r(kappa, mean_||dy||/||y||), bootstrap CI, Bonferroni check
5. **Verdict**: Apply three-tier rubric
6. **Output**: Print table, export JSON

### Recommended Project Structure

No new files or directories needed -- this phase adds a single experiment script:
```
src/experiments/validate_theorem1.py  # 200-300 lines, single analysis script
```

### Pattern 1: Single Analysis Script with Multi-Seed Loop

**What:** Load model once, iterate over seeds, for each seed set RNG state, run measurement pipeline, collect results, aggregate post-hoc.

**When to use:** When the measurement is deterministic given the input batch (round-to-nearest quantization, no stochasticity) and only the batch varies by seed.

**Key design decisions:**
- Load model once to avoid redundant I/O (checkpoint is ~670MB for 164M model)
- Detach and re-attach tracker between seeds to avoid accumulating activations from previous passes
- Store per-seed results as a list of 3 dicts, then aggregate

**Example flow:**
```python
model = load_model(checkpoint_path)
seeds = [42, 123, 456]
all_results = []

for seed in seeds:
    torch.manual_seed(seed)
    tracker = ErrorPropagationTracker()
    tracker.attach(model)
    
    dataloader = get_dataloader(split='val', batch_size=1, max_seq_len=512)
    batch = next(iter(dataloader))
    with torch.no_grad():
        model(batch['input_ids'].to(device))
    
    tracker.detach()
    
    # Compute metrics
    kappas = compute_all_condition_numbers(model)  # strip .weight suffix
    errors = tracker.compute_output_error(model, quantizer)
    dw_norms = compute_dw_norms(model, quantizer)
    
    all_results.append({
        'seed': seed,
        'kappas': kappas,
        'errors': errors,
        'dw_norms': dw_norms,
    })
```

### Anti-Patterns to Avoid

- **Loading model 3 times**: The checkpoint loading is expensive (~seconds). Load once, loop seeds within the same process.
- **Using same batch for all seeds**: The purpose of multiple seeds is batch variation. Each seed must produce a different batch. This means either (a) the dataloader must be recreated per seed, or (b) we manually index different batches.
- **Computing kappa 3 times when it is seed-independent**: Kappa(W) depends only on W, which is the same across seeds (same checkpoint). Compute once and reuse. However, ||dy||/||y|| depends on the input batch x, which varies by seed.
- **Accumulating activations across seeds**: Each forward pass with the tracker attached accumulates activations. Always detach between seeds.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Pearson correlation | Manual formula | scipy.stats.pearsonr or np.corrcoef | One function call, asymptotically correct p-value. Manual t-distribution p-value computation is error-prone. |
| Bootstrap CI | Manual derivation | np.random.choice + np.percentile | 3 lines of numpy for the core, no edge cases. |
| Condition number | Power iteration | compute_all_condition_numbers() (exact SVD) | Already validated in Phase 2. Exact SVD is cheap for 832x832 matrices. |
| Output-space error | Manual hook registration | ErrorPropagationTracker | Already validated with null measurement in Phase 2. Built specifically for this phase. |

**Key insight:** Every computational primitive needed for this phase already exists in the codebase or standard Python data stack. The only "new" code is the orchestration logic and statistical aggregation. Avoid reimplementing any of the measurement infrastructure.

## Runtime State Inventory

> This section omitted -- Phase 3 is a greenfield analysis script, not a rename/refactor/migration phase. No runtime state changes.

## Common Pitfalls

### Pitfall 1: Module Path Name Mismatch Between Systems
**What goes wrong:** `compute_all_condition_numbers()` returns keys with `.weight` suffix (e.g., `model.layers.0.attention.q_proj.weight`), but `ErrorPropagationTracker` stores activations keyed by module path without `.weight` (e.g., `model.layers.0.attention.q_proj`). Results dicts cannot be merged -- matrices silently dropped from analysis.

**Root cause:** `named_parameters()` includes parameter name suffixes; hook registration uses module path without suffixes.

**How to avoid:** Follow `measure_qerror.py` line 98: `kappas = {k.replace('.weight', ''): v for k, v in kappas_raw.items()}`. Apply the same normalization to `dw_norms` keys from `get_quantizable_weights()`.

**Warning signs:** Count of matrices differs between kappa dict (72), errors dict (72), and dw_norms dict (72). Any mismatch means some matrices were skipped.

### Pitfall 2: Data Split Files Not Synced to Remote
**What goes wrong:** Script starts, `get_dataloader(split='val')` finds no `*_val.bin` files on the remote, silently falls back to offline embedded corpus. The resulting activations and errors are computed on toy data, not the real 4.24B token corpus. Results are useless.

**Root cause:** Phase 1 data split files exist locally (`data/real_tiers/tier1_c4_val.bin`, etc.) but are not synced to the remote GPU server. The `sync.sh` excludes `wandb` and `__pycache__` but should include the new files.

**How to avoid:** Run `./sync.sh` before the first remote execution of `validate_theorem1.py`. Verify with `./remote_run.sh "ls /home/bi_group2/Projects/Numerical_Analysis/data/real_tiers/"` that `*_val.bin` files exist.

**Warning signs:** Script prints "[DATA] split=val matched 0 files" or uses CharTokenizer fallback.

### Pitfall 3: Same Validation Batch Across All 3 Seeds
**What goes wrong:** Dataloader is created once, and `next(iter(dataloader))` is called 3 times, each time getting a **different** batch if shuffle=True, or the **same** batch if shuffle=False. If seeds control nothing because the dataloader state is shared, the 3 "replicates" are just measurement noise, not true seed-to-seed variation.

**Root cause:** The dataloader iterator is consumed sequentially across seeds. The first seed gets the first batch; subsequent seeds get subsequent batches -- but the seed value may not affect which batch is drawn if shuffle=False for validation.

**How to avoid:** For each seed, (1) call `torch.manual_seed(seed)`, (2) create a fresh dataloader with `DataLoader(shuffle=True)` for batch randomization, (3) call `next(iter(dataloader))`. For validation data with shuffle=True, different seeds will produce different batch permutations.

**Reference:** The `MultiTierDataset.__len__()` depends on shard sizes. For val split on 4 tiers with 5% each, validation data is ~212M tokens. A batch of 512 tokens with shuffle=True and a specific seed will select a pseudorandom starting position. Each seed produces a different batch.

### Pitfall 4: Tightness Ratio Overflow / Division by Zero
**What goes wrong:** Some matrices may have kappa near zero (well-conditioned) or ||dW||/||W|| near zero (quantization maps weight to itself). The tightness_ratio = (||dy||/||y||) / (kappa * ||dW||/||W||) can produce Inf or NaN.

**Root cause:** Numerical edge cases: (1) kappa can be extremely large (e.g., o_proj), but not zero. (2) ||dW||/||W|| can be very small if FP4 quantization happens to preserve a weight exactly. (3) ||dy||/||y|| can also be very small for matrices where the input activations x are near-zero.

**How to avoid:** Clamp denominator: `denom = max(kappa * dw_norm, 1e-12)`. Report any matrices with denominator < 1e-10 separately in a footnote.

**Warning signs:** Infinite or NaN values in results table.

### Pitfall 5: Weight Norm vs Module Path
**What goes wrong:** `get_quantizable_weights()` returns weights including `embed_tokens` and `lm_head`. But `ErrorPropagationTracker` captures activations for all nn.Linear modules, which includes attention/FFN projections but NOT `embed_tokens` (nn.Embedding, not nn.Linear) and potentially NOT `lm_head` (nn.Linear, yes).

**Root cause:** `model.embed_tokens` is `nn.Embedding`, not `nn.Linear`. The tracker only hooks nn.Linear modules. So the 72 matrices from `get_quantizable_weights()` include `embed_tokens` but the tracker will not capture activations for it. `lm_head` IS nn.Linear and will be captured.

**How to avoid:** Handle the embed_tokens case separately if needed. For ||dy||/||y|| on embed_tokens, manual computation is required (or use the G0 global point). The 72-matrix table should note which matrices use direct measurement vs global point derivation.

### Pitfall 6: Pearson r(p-value) Interpretation with Highly Correlated Data
**What goes wrong:** The 72 weight matrices are not independent -- they come from 12 layers with correlated architectures. Very high p-values may be due to structural dependencies in the data, not lack of correlation.

**Root cause:** Weight matrices within the same layer (q_proj, k_proj, v_proj, etc.) share the same input activations and have correlated spectral properties. This violates the independence assumption underlying the Pearson p-value.

**How to avoid:** The Bonferroni correction already accounts for 72 comparisons. Report the bootstrap CI as the primary evidence (distribution-free). Note in the verdict that the p-value is approximate due to architectural dependencies. The bootstrap CI is more robust since it resamples pairs without assuming independence.

## Code Examples

### Example 1: Pearson Correlation with scipy

```python
import numpy as np
from scipy.stats import pearsonr

# kappa_values: array of 72 kappa values
# error_values: array of 72 ||dy||/||y|| values
r, p_value = pearsonr(kappa_values, error_values)

# Bonferroni threshold
alpha = 0.05 / 72  # = 0.000694

print(f"Pearson r = {r:.4f}, p = {p_value:.6e}")
print(f"Significant at Bonferroni threshold {alpha:.6f}? {'YES' if p_value < alpha else 'NO'}")
```
Source: [VERIFIED: scipy.stats.pearsonr from scipy 1.17.1 on remote]

### Example 2: Bootstrap 95% CI for Pearson r

```python
import numpy as np

def bootstrap_pearson_ci(kappa, error, n_resamples=10000, ci_level=0.95):
    """Bootstrap 95% CI for Pearson r(kappa, error).
    
    Args:
        kappa: array of kappa values (n_matrices,)
        error: array of ||dy||/||y|| values (n_matrices,)
        n_resamples: number of bootstrap iterations
        ci_level: confidence level (default 0.95 = 95%)
    
    Returns:
        (r_lower, r_upper): CI bounds
        r_values: array of all bootstrap r values (for diagnostics)
    """
    n = len(kappa)
    data = np.column_stack([kappa, error])
    r_values = np.zeros(n_resamples)
    
    for i in range(n_resamples):
        idx = np.random.choice(n, n, replace=True)
        sample = data[idx]
        r_values[i] = np.corrcoef(sample[:, 0], sample[:, 1])[0, 1]
    
    alpha = 1 - ci_level
    lower = np.percentile(r_values, 100 * alpha / 2)
    upper = np.percentile(r_values, 100 * (1 - alpha / 2))
    
    return lower, upper, r_values
```
Source: [ASSUMED based on standard bootstrap methodology]

### Example 3: Matrix Type Classification from Module Path

```python
def classify_matrix(module_path: str) -> tuple[int, str]:
    """Parse module path to extract layer index and matrix type.
    
    Returns:
        (layer_idx, matrix_type) where matrix_type is
        'attention', 'ffn', or 'global'.
    """
    parts = module_path.split('.')
    
    # Extract layer index
    if 'layers' in parts:
        layer_idx = int(parts[parts.index('layers') + 1])
    else:
        layer_idx = -1  # global: embed_tokens, lm_head
    
    # Classify by last segment
    last = parts[-1] if parts else ''
    if last in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
        matrix_type = 'attention'
    elif last in ('gate_proj', 'up_proj', 'down_proj'):
        matrix_type = 'ffn'
    elif last in ('embed_tokens', 'lm_head'):
        matrix_type = 'global'
    else:
        matrix_type = 'unknown'
    
    return layer_idx, matrix_type
```
Source: [VERIFIED: measure_qerror.py lines 139-148]

### Example 4: Verdict Application

```python
def compute_verdict(r, p_value, ci_lower, ci_upper):
    """Apply three-tier verdict rubric (D-06).
    
    Returns:
        (verdict, reason): e.g. ('YES', 'r=0.72 > 0.5, p=1e-8 < 0.00069, CI=[0.58,0.82] excludes 0')
    """
    bonferroni_alpha = 0.05 / 72
    
    if r > 0.5 and p_value < bonferroni_alpha and ci_lower > 0:
        return 'YES', f"Strong correlation: r={r:.4f}, p={p_value:.2e}, CI=[{ci_lower:.4f},{ci_upper:.4f}]"
    elif r > 0.2:
        return 'QUALIFIED', (f"Weak correlation: r={r:.4f}. "
                             f"{'YES' if p_value < bonferroni_alpha else 'NO (p>0.00069)'} on significance. "
                             f"CI=[{ci_lower:.4f},{ci_upper:.4f}]")
    else:
        return 'NO', (f"No meaningful correlation: r={r:.4f}. "
                      f"p={p_value:.4f} {'>' if p_value > 0.05 else '<'} 0.05. "
                      f"CI=[{ci_lower:.4f},{ci_upper:.4f}]")
```

### Example 5: Full Results Table Format

```python
def print_results_table(rows):
    """Print formatted 72-matrix results table.
    
    Column order (per D-05): name, layer, type, kappa, ||dW||/||W||, ||dy||/||y||, tightness_ratio
    """
    header = (
        f"  {'name':<50s}  {'layer':>5s}  {'type':>10s}  "
        f"{'kappa':>12s}  {'||dW||/||W||':>14s}  {'||dy||/||y||':>14s}  "
        f"{'tightness':>12s}"
    )
    print(header)
    print("  " + "-" * 120)
    
    for row in rows:
        print(
            f"  {row['name']:<50s}  {row['layer']:>5d}  {row['type']:>10s}  "
            f"{row['kappa']:>12.2f}  {row['dw_norm']:>14.6f}  "
            f"{row['dy_norm']:>14.6f}  {row['tightness']:>12.4f}"
        )
```
Source: Adapted from [VERIFIED: measure_qerror.py lines 127-173]

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| PPL as Theorem 1 metric | Per-matrix ||dy||/||y|| at matrix output | Phase 2 discovery | PPL is too far downstream; per-matrix error is the correct measurement point for Theorem 1 |
| Power iteration for kappa | Exact SVD for kappa | Phase 1 discovery | Power iteration was underestimating sigma_min, causing ~5000x kappa underestimation |
| Two-pass measurement | Single-pass activation capture | Phase 2 decision | Avoids cascading confound from two-pass approach (clean FP16 vs quantized inputs differ) |

**Deprecated/outdated:**
- PPL as sole metric for Theorem 1 validation: PPL aggregates error across all layers and tokens, losing per-matrix signal. Use per-matrix ||dy||/||y|| instead.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | scipy.stats.pearsonr is available on remote in `sle` conda env | Standard Stack | Confirmed via remote execution 2026-05-17: scipy 1.17.1 is installed. LOW risk. |
| A2 | The 72 matrices from `get_quantizable_weights()` include all matrices tracked by ErrorPropagationTracker minus embed_tokens | Common Pitfalls | If more matrices are tracked than quantizable (or vice versa), results count mismatches. Mitigated by pitfall documentation. |
| A3 | Bootstrap CI with 10,000 resamples provides stable estimates | Code Examples | Standard practice in statistics. Can be verified by running with different n_resamples and comparing CI width. |
| A4 | FP16 baseline checkpoint `checkpoints/scaled_fp16_baseline/model.pt` is accessible on remote | Environment Availability | Confirmed via remote execution. |
| A5 | Phase 1 data split files need syncing before Phase 3 runs on remote | Common Pitfalls | Verified: local data has `*_val.bin` files, remote does not. Must run sync.sh. |

## Open Questions (RESOLVED)

1. **Should embed_tokens be excluded from the 72-matrix table since it is nn.Embedding, not nn.Linear?** RESOLVED: Exclude embed_tokens from per-matrix correlation analysis (no direct activation measurement via Linear pre-hook). Include embed_tokens and lm_head in the aggregated table for kappa and ||dW||/||W|| only, with ||dy||/||y|| marked as N/A. The script flags this at runtime. The 84 proj matrices (7 per layer × 12 layers = q/k/v/o/gate/up/down) form the core of the correlation test.
   - What we know: `get_quantizable_weights()` includes `embed_tokens` (dim >= 2 and name contains 'embed'). But ErrorPropagationTracker hooks only nn.Linear modules. embed_tokens is nn.Embedding, so no activation is captured for it.
   - Recommendation: Exclude embed_tokens from the correlation analysis (no direct activation measurement). Include it in the table only if G0 data can provide the input. Kappa and ||dW||/||W|| are still computable. Flag this in the script with a comment.

2. **Are all 72 "proj" matrices indeed nn.Linear?** RESOLVED: All proj matrices (q/k/v/o/gate/up/down) ARE nn.Linear and will be captured by the tracker. At runtime, print actual matrix count grouped by type (attention/ffn/global) for documentation. The plan should handle 72-85 matrices robustly.
   - What we know: `get_quantizable_weights()` returns weight matrices whose name contains 'proj', 'embed_tokens', or 'lm_head'. The 12 layers * 6 proj per layer (q, k, v, o, gate, up, down) = 72 proj + embed_tokens + lm_head = 74? Or just the 72 proj?
   - Need to verify: The count from `get_quantizable_weights()` should be verified at runtime and documented. If it includes embed_tokens and lm_head, the total is >72.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| PyTorch | Model loading, forward pass, tracker | Yes (local + remote) | >= 2.3.0 | -- |
| scipy.stats | Pearson r + p-value | Yes (remote `sle` env) | 1.17.1 | Pure numpy alternative (~5 lines for r, ~10 lines for p-value) |
| numpy | Bootstrap, array ops | Yes (local + remote) | >= 1.26.0 | -- |
| FP16 baseline checkpoint | Model source | Yes (remote: `checkpoints/scaled_fp16_baseline/model.pt`) | -- | Can train new baseline, but unnecessary |
| Validation data (*_val.bin) | Forward pass data | Local only | -- | Must run `./sync.sh` before remote execution |

**Missing dependencies with no fallback:**
- Remote must have synced `*_val.bin` files. Run `./sync.sh` first.

**Missing dependencies with fallback:**
- scipy on local: Not needed -- experiment runs on remote GPU server. Script will fail gracefully with ImportError if scipy is unavailable, and pure numpy fallback can be used.

## Security Domain

> This phase introduces no new code execution, no network access, and no user-facing data handling. The analysis script reads a local checkpoint and validation data, runs pure statistical computation, and writes a JSON file. No security controls are needed beyond existing codebase conventions.

## Sources

### Primary (HIGH confidence)
- Phase 2 source code (`src/analysis/error_propagation.py`, `src/analysis/condition.py`, `src/experiments/measure_qerror.py`) -- verified infrastructure behavior via comprehensive code reading
- CONTEXT.md (D-01 through D-09) -- locked decisions from user discussion
- REQUIREMENTS.md (VAL-01, VAL-02, VAL-04) -- phase requirements
- Remote execution confirmation: scipy 1.17.1 available, checkpoint exists, data needs sync

### Secondary (MEDIUM confidence)
- scipy.stats.pearsonr documentation -- standard API, well-known behavior. Functionality verified by remote availability check.

### Tertiary (LOW confidence)
- Bootstrap CI methodology and convergence with 10,000 resamples -- standard practice but n_resamples may need tuning. If distribution is heavy-tailed, more resamples may be needed for stable CI.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all frameworks verified on remote
- Architecture: HIGH -- follows established `measure_qerror.py` pattern
- Pitfalls: HIGH -- identified from codebase analysis and Phase 2 learnings

**Research date:** 2026-05-17
**Valid until:** 2026-06-30 (stable dependencies, no fast-moving components)
