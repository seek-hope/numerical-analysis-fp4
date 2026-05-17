# Roadmap: Numerical Analysis-Driven FP4 Quantization

## Overview

Build a precision measurement protocol that tracks per-matrix weight quantization error (||dy||/||y||) through a 164M Transformer's layer pipeline. Start by fixing the critical data-split blocker, then implement the core ErrorPropagationTracker for single-pass activation capture, validate Theorem 1 at per-matrix granularity across all 72 weight matrices, trace error propagation through the full layer pipeline with RMSNorm decomposition, and finally re-run all PTQ comparisons under clean conditions with the new metric alongside PPL. The journey moves from infrastructure (I can measure correctly) through theory validation (does Theorem 1 hold at the right granularity) to mechanism tracing (where does error go) and finally to synthesis (comprehensive comparison with corrected methodology).

## Phases

- [ ] **Phase 1: Clean Data Split** - Split each data tier into train/val at .bin file level to prevent calibration-evaluation leakage
- [ ] **Phase 2: Core Measurement Protocol** - Implement ErrorPropagationTracker with single-pass capture and per-matrix ||dy||/||y|| computation
- [ ] **Phase 3: Theorem 1 Validation** - Test kappa correlation with ||dy||/||y|| across 72 matrices with Bonferroni correction and multi-seed rigor
- [ ] **Phase 4: Error Propagation Trace** - Trace quantization error through the full layer pipeline with RMSNorm attenuation and parallel/orthogonal decomposition
- [ ] **Phase 5: Extended PTQ Comparison and Final Report** - Re-run 24-config PTQ comparison with clean data and both metrics; produce final report

## Phase Details

### Phase 1: Clean Data Split
**Goal**: Data infrastructure delivers isolated train/val sets with no evaluation data leaking into PTQ calibration
**Depends on**: Nothing (first phase)
**Requirements**: DATA-01, DATA-02, DATA-03
**Success Criteria** (what must be TRUE):
  1. Each tier (C4, FineWeb-edu, Wikipedia, OpenOrca) has separate `tierN_train.bin` (first 95%) and `tierN_val.bin` (last 5%) files
  2. `get_dataloader(split='train')` loads only training .bin files; `get_dataloader(split='val')` loads only validation .bin files
  3. GPTQ Hessian calibration and adaptive grid fitting use training split only (verified by file access audit)
  4. Perplexity evaluation and output error measurement use validation split only (verified by file access audit)
  5. Running a full PTQ + eval pipeline with the clean split produces reproducible results with no calibration-evaluation leakage
**Plans**: 1 plan

Plans:
- [ ] 01-01-PLAN.md -- Post-processing split utility, dataloader API update, and --val_split flag for prepare_data_chunked.py

### Phase 2: Core Measurement Protocol
**Goal**: A validated measurement pipeline that captures per-matrix output-space relative error ||dy||/||y|| for any quantized weight matrix using single-pass activation capture
**Depends on**: Phase 1
**Requirements**: MEAS-01, MEAS-02, MEAS-03, MEAS-04, VAL-03
**Success Criteria** (what must be TRUE):
  1. ErrorPropagationTracker registers forward hooks at 6 per-layer points (P0-P6) plus 3 global points (G0-G2) on all 12 transformer layers without modifying transformer.py
  2. A single FP16 forward pass saves all 72 pre-activation tensors x for offline per-matrix computation (single-pass capture, no cascading confound)
  3. For any quantized weight matrix with round-to-nearest, the pipeline computes ||(W_q - W)x||/||Wx|| from saved activations and reports it as a single scalar per matrix
  4. Per-matrix kappa(W) computed via exact SVD for all 72 matrices (correct sigma_min, not power iteration)
  5. Null measurement (quantize a weight already at FP16) produces ||dy||/||y|| < 1e-5, confirming measurement pipeline integrity
**Plans**: TBD

### Phase 3: Theorem 1 Validation
**Goal**: Determine whether Theorem 1's predicted upper bound ||dy||/||y|| <= kappa * ||dW||/||W|| holds empirically at per-matrix granularity
**Depends on**: Phase 2
**Requirements**: VAL-01, VAL-02, VAL-04
**Success Criteria** (what must be TRUE):
  1. A single table reports all 72 matrices with columns (name, layer, type, kappa, ||dW||/||W||, ||dy||/||y||, tightness_ratio)
  2. Pearson r(kappa, ||dy||/||y||) computed across all 72 matrices with Bonferroni-corrected significance threshold (alpha = 0.05/72 = 0.00069)
  3. Results reported for 3 random seeds (42, 123, 456) with mean +/- std and bootstrap 95% CI for the correlation coefficient
  4. The report states whether Theorem 1's bound holds empirically (YES/NO/QUALIFIED) with supporting evidence
**Plans**: TBD

### Phase 4: Error Propagation Trace
**Goal**: Map the full journey of quantization error through the Transformer layer pipeline to identify where error amplifies and where it attenuates
**Depends on**: Phase 3
**Requirements**: TRACE-01, TRACE-02, TRACE-03
**Success Criteria** (what must be TRUE):
  1. For layers 0, 5, and 11 (representative early/mid/late), error magnitude ||delta||/||y|| reported at all 6 measurement points (P0->P1->P2->P4->P5->P6), showing where error grows and shrinks
  2. RMSNorm attenuation ratio ||delta_post||/||delta_pre|| reported for input_norm and post_attn_norm across all 12 layers
  3. RMSNorm error decomposition shows parallel component (along signal direction) and orthogonal component separately for each layer, testing whether RMSNorm attenuates or redirects error
  4. Error waterfall data exported in structured format for visualization (||delta||/||y|| vs measurement point)
**Plans**: TBD

### Phase 5: Extended PTQ Comparison and Final Report
**Goal**: Complete comparison of all PTQ methods under clean conditions with both PPL and per-matrix output error, culminating in the final project report
**Depends on**: Phase 1, Phase 2, Phase 3, Phase 4
**Requirements**: COMP-01, COMP-02, COMP-03, REPORT-01, REPORT-02, REPORT-03
**Success Criteria** (what must be TRUE):
  1. 24-config PTQ comparison re-run with clean data split, reporting both PPL and per-matrix ||dy||/||y|| for every configuration
  2. GPTQ vs round-to-nearest comparison shows whether column compensation reduces ||dy||/||y|| compared to what weight-space metrics alone predict
  3. Lloyd-Max adaptive grid vs uniform E2M1 grid comparison shows whether distribution-adaptive grids reduce ||dy||/||y||
  4. Per-matrix error summary table generated with all required metadata columns (name, layer, type, kappa, ||dW||/||W||, ||dy||/||y||, tightness_ratio, norm_attenuation)
  5. REPORT.md updated with corrected kappa values, per-matrix measurements, null measurement validation, propagation waterfall data, and revised theoretical assessment incorporating all findings
**Plans**: TBD

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Clean Data Split | 0/1 | Planned | - |
| 2. Core Measurement Protocol | 0/0 | Not started | - |
| 3. Theorem 1 Validation | 0/0 | Not started | - |
| 4. Error Propagation Trace | 0/0 | Not started | - |
| 5. Extended PTQ Comparison and Final Report | 0/0 | Not started | - |
