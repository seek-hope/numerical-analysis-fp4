# Phase 5: Extended PTQ Comparison and Final Report - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.

**Date:** 2026-05-17
**Phase:** 05-extended-ptq-comparison-and-final-report
**Areas discussed:** 24-config scope, script design, report format, data synthesis

---

## 24-Config Comparison Scope

| Option | Description | Selected |
|--------|-------------|----------|
| Full 24-config (2 checkpoints × 2 formats × 6 methods) | FP16 baseline + cond_regularized, FP8 E4M3 + FP4 E2M1, all 6 methods | ✓ |
| Reduced 16-config (skip Hadamard/Outlier for FP4) | Same checkpoints and formats, 4 methods for FP4 | |
| Minimal 12-config (RTN + GPTQ + Lloyd-Max only) | Drop rotation-based methods entirely | |

**Auto-selected:** Full 24-config — matches REQUIREMENTS.md COMP-01 and the original phase2_comparison.py methodology. Hadamard/Outlier for FP4 may show instability; Claude's discretion to skip if pathological.

---

## Script Design

| Option | Description | Selected |
|--------|-------------|----------|
| Two scripts: comparison + report | `run_full_comparison.py` for 24-config PTQ + `write_final_report.py` for REPORT.md generation | ✓ |
| Single comprehensive script | One script does everything | |
| Embed in existing scripts | Modify phase2_comparison.py and final_summary.py | |

**Auto-selected:** Two scripts — comparison is long-running (24 configs × 100 eval steps); report is fast synthesis. Separation allows re-running the report without re-running comparison.

---

## Checkpoint Availability

| Option | Description | Selected |
|--------|-------------|----------|
| Pre-condition: checkpoints must exist | Script exits with clear error if missing; user runs training scripts first | ✓ |
| Include training in Phase 5 | Run train_scaled_baseline.py and train_cond_regularized.py if checkpoints missing | |

**Auto-selected:** Pre-condition only. Training is out of scope for this phase — Phase 5 is comparison and synthesis, not model training. Missing checkpoints = user re-runs training scripts.

---

## Visualization (v2 requirements)

| Option | Description | Selected |
|--------|-------------|----------|
| Defer to post-project | Produce JSON data; charts are follow-up | ✓ |
| Basic matplotlib charts | Include simple plots in REPORT.md | |
| Full interactive visualization | D3/Plotly dashboards | |

**Auto-selected:** Defer — v2 requirements. Structured JSON provides everything needed for charts.

---

## Claude's Discretion

- Re-entrant partial execution support (--configs flag for GPU time management)
- Print table formatting
- Per-layer-type subgroup analysis inclusion
- Cross-checkpoint comparison depth
- Hadamard/Outlier method omission for FP4 if unstable
- Logging verbosity during 24-config execution

## Deferred Ideas

- VIS-01/02/03 (v2 visualizations) — waterfall charts, scatter plots, bar charts
- ANALYSIS-01 (per-head attention decomposition) — 3× compute cost
- ANALYSIS-03 (rank stability analysis) — separate investigation
