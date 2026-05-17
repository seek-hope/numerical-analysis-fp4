# Phase 3: Theorem 1 Validation - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-17
**Phase:** 3-Theorem 1 Validation
**Areas discussed:** Statistical Computation, Multi-Seed Execution, Report Structure, Verdict Rubric, Kappa/Error Sources

---

## Statistical Computation

| Option | Description | Selected |
|--------|-------------|----------|
| scipy.stats.pearsonr + bootstrap | Use scipy for Pearson r/p-value, numpy for bootstrap (10k resamples), Bonferroni alpha=0.05/72 | ✓ |
| Pure numpy implementation | Reimplement Pearson correlation formula, avoid scipy dependency | |
| statsmodels with HC standard errors | Heavier dependency, richer output but overkill for single correlation test | |

**Auto-selected choice:** scipy.stats.pearsonr + numpy bootstrap
**Rationale:** scipy is already available (listed in requirements.txt usage context). Pearson r is a single function call with p-value. Bootstrap implemented in ~10 lines of numpy. Bonferroni correction is standard for 72-matrix multiple comparisons (stated in STATE.md accumulated context).

---

## Multi-Seed Execution

| Option | Description | Selected |
|--------|-------------|----------|
| Loop over 3 seeds, 3 forward passes each | For each seed: load model, set seed, run tracker forward pass, compute errors, store. Aggregate across seeds post-hoc. | ✓ |
| Single seed, 3 batch samples | Use one seed, draw 3 different evaluation batches. Cheaper but doesn't test initialization sensitivity. | |
| 3 seeds, 100 evaluation steps each | Full statistical rigor but compute-heavy (~300 forward passes). Phase 2 validation was single-pass; Phase 3 needs more. | |

**Auto-selected choice:** Loop over 3 seeds
**Rationale:** Success criterion #3 explicitly requires "3 random seeds (42, 123, 456)". A single forward pass per seed suffices for per-matrix error — the 100-step evaluation is for PPL stability, not per-matrix ||dy||/||y|| (which is deterministic given x, W, and the quantization function).

---

## Report Structure

| Option | Description | Selected |
|--------|-------------|----------|
| Single analysis script with printed table + JSON export | `validate_theorem1.py`: loads checkpoint, runs 3 seeds, computes statistics, prints table, exports JSON. Verdict inline. | ✓ |
| Two scripts (measure + analyze separately) | First script collects data, second script analyzes. More modular but more moving parts. | |
| Jupyter notebook | Interactive exploration. Less reproducible, doesn't fit the experiment script pattern. | |

**Auto-selected choice:** Single analysis script
**Rationale:** Follows the established pattern of `src/experiments/validate_*.py` scripts. Single script is self-contained and reproducible (just `python src/experiments/validate_theorem1.py --checkpoint ... --data_dir ...`). JSON export enables Phase 5 visualization.

---

## Verdict Rubric

| Option | Description | Selected |
|--------|-------------|----------|
| Three-tier (YES/QUALIFIED/NO) with numeric thresholds | r>0.5+p<0.00069+CI excludes 0 → YES; r>0.2 but borderline → QUALIFIED; r<0.2 → NO | ✓ |
| Binary YES/NO | Simplifies but loses nuance — weak correlation is different from zero correlation | |
| Bayesian approach | Bayes factor instead of p-value. More principled but harder to communicate. | |

**Auto-selected choice:** Three-tier rubric
**Rationale:** PROJECT.md key finding: "κ(W) shown to have zero correlation with PPL degradation (r=−0.012) — Theorem 1 falsified in Transformers." A three-tier verdict captures the possibility of a weak-but-nonzero relationship (QUALIFIED) vs complete failure (NO), vs surprising confirmation (YES). Matches the success criterion's "YES/NO/QUALIFIED" language.

---

## Kappa and Error Sources

| Option | Description | Selected |
|--------|-------------|----------|
| Reuse Phase 2 infrastructure | compute_all_condition_numbers() for kappa, ErrorPropagationTracker for ||dy||/||y||, FPQuantizer for ||dW||/||W|| | ✓ |
| Recompute from scratch | Reimplement in Phase 3 script — duplicates Phase 2 work | |

**Auto-selected choice:** Reuse Phase 2 infrastructure
**Rationale:** Phase 2 built the measurement pipeline specifically for Phase 3 consumption. All measurement functions are validated (Phase 2 verification passed). Reusing them directly eliminates risk of implementation divergence.

---

## Claude's Discretion

- scipy vs pure numpy for Pearson r (scipy preferred for correctness, already available)
- Table formatting details (column widths, alignment, sort order)
- Number of bootstrap resamples (10,000 default, adjustable)
- Per-layer-type subgroup analysis inclusion
- Logging verbosity during multi-seed execution

## Deferred Ideas

None — discussion stayed within phase scope.
