# Numerical Analysis: FP4 Quantization Research

FP8/FP4 low-precision quantization study on a ~164M parameter Gemma-style Transformer, using industry-standard PTQ/QAT methods guided by numerical analysis tools.

## Model

| Parameter | Value |
|-----------|-------|
| Architecture | 12-layer Transformer, RMSNorm, RoPE, GQA (4:1) |
| Hidden / Intermediate | 768 / 3072 |
| Attention Heads | 12Q / 3KV |
| Vocabulary | BPE 32K |
| Parameters | ~164M |
| Training Data | 4.24B tokens (C4 + FineWeb + Wikipedia + OpenOrca) |

## Results

**Post-Training Quantization (PTQ)** — consolidated results are in
[`docs/REPORT.md`](docs/REPORT.md). The latest report evaluates 6 PTQ methods
across 2 checkpoints and 2 formats, ranking by per-matrix output-space
relative error ||dy||/||y|| — the correct metric for testing numerical
analysis predictions at the weight-output level.

| Best result | Method | mean \|\|dy\|\|/\|\|y\|\| |
|-------------|--------|---------------------------|
| FP8 PTQ | Round-to-nearest (per-channel) | 0.0137 |
| FP4 PTQ | Lloyd-Max adaptive grid | 0.0664 |

**Theorem 1 Verdict:** NO — Pearson r(kappa, ||dy||/||y||) = -0.23 across 84 matrices.
FP4 unit roundoff (u=0.25) dominates: ||dW||/||W|| ≈ 0.15 for ALL matrices
regardless of κ(W). RMSNorm attenuates error ~83% per layer, explaining why
per-matrix errors do not cascade destructively.

Key findings:
- Simple round-to-nearest outperforms GPTQ on ||dy||/||y|| (GPTQ increases error by 44-49%)
- Lloyd-Max adaptive grids reduce FP4 error by 18% vs uniform E2M1
- Condition-number regularization does NOT improve quantization robustness
- RMSNorm error blocking (~83% per layer) is the primary architectural defense

## Project Structure

```
src/
├── model/
│   ├── config.py              # MicroGemmaFPConfig (~164M)
│   └── transformer.py         # RMSNorm, RoPE, GQA, sliding/full attention
├── quantization/
│   ├── fp_quantizer.py        # FP8/FP4 simulation with per-channel scaling
│   ├── fp4_grids.py           # E2M1 / NF4 / MXFP4 quantization grids
│   ├── gptq.py                # GPTQ-style weight compensation
│   ├── grid_qat.py            # Grid-based QAT wrapper
│   ├── stochastic.py          # Stochastic rounding utilities
│   └── hadamard.py            # Walsh-Hadamard transform (QuIP/QuaRot)
├── analysis/
│   ├── condition.py           # Condition number estimation (power iteration)
│   ├── lipschitz.py           # Lipschitz constant propagation
│   └── sensitivity.py         # Per-layer sensitivity + mixed-precision suggestions
└── experiments/
    ├── training_utils.py              # Dataloaders, train loop, eval
    ├── train_scaled_baseline.py       # FP16 baseline training
    ├── train_cond_regularized.py      # FP16 + condition regularization
    ├── train_qat_fp4_opt.py           # QAT (FP8/FP4 with STE)
    ├── run_full_comparison.py         # 16-config PTQ comparison
    ├── write_final_report.py          # Final report generation
    ├── measure_qerror.py              # Per-matrix ||dy||/||y|| measurement
    ├── trace_error_propagation.py     # Error propagation trace
    ├── validate_theorem1.py           # Theorem 1 validation
    ├── validate_theory.py             # Layer sensitivity validation
    ├── validate_rmsnorm.py            # RMSNorm ablation validation
    └── legacy/                        # Archived Phase 1-2 scripts
```

## Quick Start

Training runs on a remote GPU server (8× RTX 4090). Local setup is for data preparation only.

```bash
# 1. Train BPE tokenizer (local, requires network)
python src/experiments/train_tokenizer.py

# 2. Prepare data (local, requires network)
python src/experiments/prepare_data_chunked.py

# 3. Sync to remote
./sync.sh

# 4. Train FP16 baseline (remote)
./remote_python.sh src/experiments/train_scaled_baseline.py \
    --data_dir data/real_tiers --max_steps 2000

# 5. Run full PTQ comparison (remote)
./remote_python.sh src/experiments/run_full_comparison.py \
    --fp16_checkpoint checkpoints/scaled_fp16_baseline/model.pt \
    --cond_checkpoint checkpoints/cond_regularized/model.pt \
    --data_dir data/real_tiers

# 6. Generate final report
python src/experiments/write_final_report.py
```

## Requirements

- Python 3.11+
- PyTorch ≥ 2.3
- Transformers ≥ 4.45
- Datasets, tokenizers
- Accelerate, einops

## References

- Frantar et al. (2023) "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"
- Dettmers et al. (2024) "QLoRA: Efficient Finetuning of Quantized LLMs"
- Xiao et al. (2023) "SmoothQuant: Accurate and Efficient Post-Training Quantization for LLMs"
- Chee et al. (2024) "QuIP: 2-Bit Quantization of Large Language Models With Guarantees"
