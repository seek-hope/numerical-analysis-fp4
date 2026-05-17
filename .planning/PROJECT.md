# Numerical Analysis-Driven FP4 Quantization for Transformers

## What This Is

A numerical analysis course project that systematically investigates how classical matrix perturbation theory applies (and fails to apply) to weight quantization in Transformer architectures. The project operates on a ~164M parameter Gemma-style causal Transformer, using FP8/FP4 simulated quantization as the testbed for validating and refining numerical analysis predictions about error propagation, condition numbers, and optimal quantizer design.

## Core Value

**Use numerical analysis to predict, measure, and explain where quantization error goes in a Transformer — and redesign the measurement protocol when the theory and experiments diverge.**

## Requirements

### Validated

- ✓ FP16 baseline training (164M model, 4.24B tokens, 2000 steps) — `checkpoints/scaled_fp16_baseline/model.pt`
- ✓ Condition number regularized training (λ=1e-4) — `checkpoints/cond_regularized/model.pt`
- ✓ FP8/FP4 simulated quantization (E4M3, E2M1, NF4, MXFP4 grids) — `src/quantization/fp_quantizer.py`, `fp4_grids.py`
- ✓ GPTQ weight compensation with activation Hessian — `src/quantization/gptq.py`
- ✓ Lloyd-Max per-layer adaptive FP4 grids — `src/quantization/adaptive_grid.py`
- ✓ RMSNorm error-blocking validation (1221× block ratio confirmed) — `src/experiments/validate_rmsnorm.py`
- ✓ Per-layer quantization sensitivity — `src/experiments/validate_theory.py`
- ✓ 24-config PTQ systematic comparison — `src/experiments/phase2_comparison.py`
- ✓ Condition number estimation (exact SVD, differentiable surrogate) — `src/analysis/condition.py`
- ✓ Lipschitz propagation analysis — `src/analysis/lipschitz.py`
- ✓ Layer sensitivity and mixed-precision suggestions — `src/analysis/sensitivity.py`
- ✓ Full mathematical derivations (Theorems 1-4, GPTQ, Lloyd-Max) — `docs/ANALYSIS.md`
- ✓ Experimental design audit — `docs/ANALYSIS.md`

### Active

- [ ] **MEAS-01**: Per-weight-matrix Theorem 1 validation — for each Linear weight, measure ||ŷ−y||/||y|| at the matrix output and correlate with κ(W)
- [ ] **MEAS-02**: Full error propagation trace — hook at 6+ points per layer (pre-linear, post-linear, post-RMSNorm, post-attention, post-FFN, post-layer) to quantify where error is amplified or attenuated
- [ ] **MEAS-03**: RMSNorm attenuation measurement — hook before/after every RMSNorm in the model, measure ||δ_post||/||δ_pre|| compression ratio per layer
- [ ] **MEAS-04**: Fix data split — implement train/val split in `prepare_data_chunked.py` and `get_dataloader()` to prevent calibration-evaluation leakage
- [ ] **MEAS-05**: Re-run PTQ comparison with clean data split and corrected κ computation, report per-layer output MSE alongside PPL

### Out of Scope

- Activation quantization — weight quantization only
- Models >1B parameters — constrained by 8× RTX 4090
- Hardware-level FP4 execution — FP32 simulation only
- QAT re-training — existing QAT results are sufficient; focus is on measurement, not new training

## Context

The project has completed two major experimental phases:
1. **Phase 1 (theory validation)**: RMSNorm blocking confirmed at 1221×. κ(W) shown to have zero correlation with PPL degradation (r=−0.012) — Theorem 1 falsified in Transformers.
2. **Phase 2 (systematic comparison)**: 24 PTQ configurations benchmarked. Simple per-channel + round-to-nearest is the most robust method. Condition number regularization makes quantization worse, not better.

Key finding driving the current work: **PPL is the wrong metric for testing numerical analysis predictions.** Theorem 1 predicts ||δy||/||y|| at a linear layer's output, but PPL measures final token distribution after RMSNorm, attention, FFN, and lm_head. The signal is lost. A new measurement protocol that directly hooks layer outputs is needed.

The codebase is structured in four layers:
- `src/model/` — Transformer architecture (RMSNorm, RoPE, GQA, sliding/full attention)
- `src/quantization/` — FP8/FP4 quantizers, GPTQ, adaptive grids
- `src/analysis/` — κ estimation, Lipschitz propagation, sensitivity
- `src/experiments/` — training scripts, evaluation, validation experiments

Remote execution via `sync.sh` → `remote_python.sh` on 8× RTX 4090 GPU server.

## Constraints

- **Hardware**: 8× RTX 4090 GPU server (remote), accessed via SSH with sshpass
- **Time**: Course project, ~4 week timeline (currently in final phase)
- **Precision**: FP32 simulation of low-precision formats (no hardware FP4)
- **Data**: 4.24B tokens across 4 tiers (C4, FineWeb-edu, Wikipedia, OpenOrca), BPE 32K tokenizer
- **Evaluation**: 100-step evaluation batches, fixed seed=42

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Exact SVD for κ (not power iteration) | `inverse_power_iteration` was computing σ_max, not σ_min — κ values were underestimated by ~5000× | ✓ Good — fixed in condition.py |
| FP4 unit roundoff u = 0.25 not 0.0625 | Industry standard (IEEE 754): u = 2^{-(m+1)} for m mantissa bits; E2M1 has m=1 | ✓ Good — corrected in PROPOSAL.md |
| Per-weight-matrix measurement granularity | κ varies by 1000× between q_proj (κ~100) and o_proj (κ~16000) within the same layer — per-layer aggregation hides this | — Pending |
| Output MSE as primary metric alongside PPL | PPL is too far downstream from the weight perturbation to test Theorem 1 directly | — Pending |

---
*Last updated: 2026-05-17 after initialization*
