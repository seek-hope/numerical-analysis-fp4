# Requirements: Numerical Analysis-Driven FP4 Quantization

**Defined:** 2026-05-17
**Core Value:** Use numerical analysis to predict, measure, and explain where quantization error goes in a Transformer — and redesign the measurement protocol when the theory and experiments diverge.

## v1 Requirements

### Data Infrastructure

- [ ] **DATA-01**: Implement train/val split at .bin file level — each tier's last 5% becomes `tierN_val.bin`, first 95% becomes `tierN_train.bin`
- [ ] **DATA-02**: Update `get_dataloader()` to accept `split='train'|'val'` parameter and filter .bin files accordingly
- [ ] **DATA-03**: Ensure PTQ calibration (GPTQ Hessian, adaptive grid) uses training split only; evaluation uses validation split only

### Core Measurement Infrastructure

- [ ] **MEAS-01**: Implement `ErrorPropagationTracker` class in `src/analysis/error_propagation.py` — register forward hooks at 6 measurement points per layer (P0-P6) + 3 global (G0-G2), capture activations for offline computation
- [ ] **MEAS-02**: Implement single-pass activation capture — run one FP16 forward pass, save pre-activation tensors x for each of the 72 Linear weight matrices
- [ ] **MEAS-03**: Compute per-matrix `||(W_q − W)x|| / ||Wx||` (actual output-space relative error) using saved activations, round-to-nearest quantization only (no GPTQ for Theorem 1 test)
- [ ] **MEAS-04**: Compute per-matrix `κ(W)` via exact SVD and `||δW||/||W||` (weight-space relative error) for comparison

### Theorem 1 Validation

- [ ] **VAL-01**: For each of 72 Linear weight matrices, report `(κ, ||δy||/||y||, ||δW||/||W||, tightness_ratio)` where `tightness_ratio = (||δy||/||y||) / (κ · ||δW||/||W||)`
- [ ] **VAL-02**: Compute Pearson r(κ, ||δy||/||y||) across all 72 matrices with Bonferroni-corrected significance threshold (α=0.05/72=0.00069)
- [ ] **VAL-03**: Run null measurement control — quantize and re-measure a weight matrix already at FP16 (should give ||δy||/||y|| ≈ 0, validates measurement pipeline)
- [ ] **VAL-04**: Measure with 3 random seeds (42, 123, 456), report mean ± std and bootstrap 95% CI for the correlation coefficient

### Error Propagation Trace

- [ ] **TRACE-01**: For each quantized weight matrix in layers 0, 5, and 11 (representative early/mid/late), measure ||δ||/||y|| at all 6 layer measurement points (P0→P1→P2→P4→P5→P6)
- [ ] **TRACE-02**: Quantify RMSNorm attenuation: measure `||δ_post||/||δ_pre||` at input_norm and post_attn_norm for all 12 layers, report per-layer compression ratio
- [ ] **TRACE-03**: Decompose RMSNorm error into parallel component (along signal direction) and orthogonal component, report both separately — tests whether RMSNorm attenuates or redirects error

### Extended PTQ Comparison

- [ ] **COMP-01**: Re-run 24-config PTQ comparison (2 checkpoints × 2 formats × 6 methods) with clean data split, reporting both PPL and per-matrix ||δy||/||y||
- [ ] **COMP-02**: For GPTQ method specifically, compare ||δy||/||y|| against round-to-nearest — quantify whether column compensation reduces output error beyond what weight-space metrics predict
- [ ] **COMP-03**: For Lloyd-Max adaptive grid, compare per-matrix ||δy||/||y|| against uniform E2M1 grid — test whether distribution-adaptive grids reduce output-space error

### Reporting

- [ ] **REPORT-01**: Generate per-matrix error summary table: `{name, layer, type, κ, ||δW||/||W||, ||δy||/||y||, tightness_ratio, norm_attenuation}`
- [ ] **REPORT-02**: Generate error propagation waterfall data for visualization (||δ||/||y|| vs measurement point for selected layers)
- [ ] **REPORT-03**: Update REPORT.md with corrected κ values, per-matrix error measurements, and revised theoretical assessment

### Out of Scope

| Feature | Reason |
|---------|--------|
| Activation quantization measurement | Weight quantization only — stated scope boundary |
| Per-head attention error decomposition | 3× compute cost (3 forward passes per layer); reserve for future |
| Full tensor storage for all 100 steps | ~10 GB storage; store only per-step statistics (150 KB) |
| QAT error propagation measurement | Focus is PTQ; QAT already shown inferior at this scale |
| Hardware-level FP4 measurement | FP32 simulation only — hardware constraint |
| Models >1B parameters | 8× RTX 4090 constraint |

## v2 Requirements

### Analysis Depth

- **ANALYSIS-01**: Per-head attention error decomposition — isolate error through Q, K, V, O projections within attention block
- **ANALYSIS-02**: Residual interaction decomposition — quantify error cross-talk between residual stream and layer output
- **ANALYSIS-03**: Rank stability analysis — measure whether quantization changes the effective rank of weight matrices (Δσ_distribution)

### Visualization

- **VIS-01**: Error propagation waterfall chart — ||δ||/||y|| on y-axis, measurement point on x-axis, one line per quantized layer
- **VIS-02**: κ vs ||δy||/||y|| scatter plot with regression line, confidence band, and per-layer-type coloring
- **VIS-03**: RMSNorm attenuation bar chart — parallel vs orthogonal decomposition per layer

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| DATA-01 | Phase 1 (Clean Data Split) | Pending |
| DATA-02 | Phase 1 (Clean Data Split) | Pending |
| DATA-03 | Phase 1 (Clean Data Split) | Pending |
| MEAS-01 | Phase 2 (Core Measurement Protocol) | Pending |
| MEAS-02 | Phase 2 (Core Measurement Protocol) | Pending |
| MEAS-03 | Phase 2 (Core Measurement Protocol) | Pending |
| MEAS-04 | Phase 2 (Core Measurement Protocol) | Pending |
| VAL-03 | Phase 2 (Core Measurement Protocol) | Pending |
| VAL-01 | Phase 3 (Theorem 1 Validation) | Pending |
| VAL-02 | Phase 3 (Theorem 1 Validation) | Pending |
| VAL-04 | Phase 3 (Theorem 1 Validation) | Pending |
| TRACE-01 | Phase 4 (Error Propagation Trace) | Pending |
| TRACE-02 | Phase 4 (Error Propagation Trace) | Pending |
| TRACE-03 | Phase 4 (Error Propagation Trace) | Pending |
| COMP-01 | Phase 5 (Extended PTQ Comparison and Final Report) | Pending |
| COMP-02 | Phase 5 (Extended PTQ Comparison and Final Report) | Pending |
| COMP-03 | Phase 5 (Extended PTQ Comparison and Final Report) | Pending |
| REPORT-01 | Phase 5 (Extended PTQ Comparison and Final Report) | Pending |
| REPORT-02 | Phase 5 (Extended PTQ Comparison and Final Report) | Pending |
| REPORT-03 | Phase 5 (Extended PTQ Comparison and Final Report) | Pending |

**Coverage:**
- v1 requirements: 20 total
- Mapped to phases: 20
- Unmapped: 0 ✓

---
*Requirements defined: 2026-05-17*
*Last updated: 2026-05-17 after roadmap creation*
