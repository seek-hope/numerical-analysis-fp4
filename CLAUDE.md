# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Numerical Analysis course project: FP4 quantization research for neural networks on a ~164M parameter Gemma-style Transformer.

- **Group A**: Post-Training Quantization (PTQ) — compress trained FP16 models to FP8/FP4 with simple (round-to-nearest), GPTQ (weight compensation), and sensitivity-guided mixed precision methods
- **Group B**: Quantization-Aware Training (QAT) — train directly in FP8/FP4 with Straight-Through Estimator (STE), stochastic rounding, and Hadamard rotation

## Running Code

All training runs on a **remote GPU server** (8× RTX 4090). Local machine has no GPU.

```bash
# Sync project to remote
./sync.sh

# Run a Python script on remote GPU
./remote_python.sh <script_path> [args...]
# Example:
./remote_python.sh src/experiments/train_fp16_baseline.py

# Run arbitrary command on remote
./remote_run.sh "<command>"

# Smoke test: verify Gemma 4 E2B loads on remote
./remote_python.sh smoke_gemma4.py
```

Remote environment: `bi_group2@bioinfo_class`, conda env `sle`, project at `/home/bi_group2/Projects/Numerical_Analysis/`. Credentials in `.sshpass`.

**PYTHONPATH** is set automatically by the shell scripts — imports like `from src.model.config import ...` work without installation.

## Architecture

```
src/
├── model/
│   ├── config.py           # MicroGemmaFPConfig (~164M params, 12L/768h/12H)
│   └── transformer.py      # RMSNorm, RoPE, GQA, sliding/full attention
│
├── quantization/
│   ├── fp_quantizer.py     # FPQuantizer: simulated FP8/FP4 with per-channel scaling
│   ├── fp4_grids.py        # Three FP4 grid schemes: E2M1, NF4, MXFP4 (block scaling)
│   ├── grid_qat.py         # Grid-based QAT wrapper (NF4/MXFP4 as drop-in FPQuantizer replacement)
│   ├── gptq.py             # GPTQ-style weight compensation (industry-standard PTQ)
│   ├── stochastic.py       # Stochastic rounding utilities (unbiased estimator)
│   ├── hadamard.py         # Fast Walsh-Hadamard transform for activation smoothing (QuIP/QuaRot)
│   └── outlier_rotation.py # DuQuant++ style outlier-aware rotation for FP4 PTQ
│
├── analysis/
│   ├── condition.py        # Condition number estimation via randomized power iteration
│   ├── lipschitz.py        # Layer-wise Lipschitz constant estimation for error propagation
│   └── sensitivity.py      # Per-layer quantization sensitivity report, mixed-precision suggestions
│
└── experiments/
    ├── training_utils.py           # Data loading (real .bin shards + offline fallback), training loop
    ├── large_corpus.py             # Embedded text corpus for offline training
    ├── prepare_data.py             # Download & tokenize 4-tier data (C4/FineWeb/Wiki/OpenOrca)
    ├── train_tokenizer.py          # Train BPE tokenizer on streaming HF datasets
    ├── train_fp16_baseline.py      # FP16 baseline training (~164M default)
    ├── train_scaled_baseline.py    # FP16 baseline training (explicit 164M config)
    ├── train_qat.py / train_qat_*.py  # QAT experiments (FP8, FP4, with SR/Hadamard variants)
    ├── ptq_eval.py                 # PTQ evaluation (simple, GPTQ, mixed-precision methods)
    ├── fp4_ptq_compare.py          # FP4 PTQ grid comparison benchmark
    ├── eval_quantization.py        # Unified industry-standard quantization evaluation
    ├── eval_all.py / eval_all_grids.py  # Multi-experiment evaluation harnesses
    └── final_summary.py            # Aggregate results across all experiments
```

## Key Design Decisions

- **Single model scale**: `MicroGemmaFPConfig` (~164M, 12 layers, 768 hidden, 12 heads) — deep enough for meaningful per-layer quantization sensitivity, large enough weight matrices (768×768) for GPTQ column compensation.
- **Industry-standard PTQ**: GPTQ weight compensation (`gptq.py`) + per-channel scaling + sensitivity-guided mixed precision. Calibration data collected via forward hooks.
- **Simulated quantization**: All FP8/FP4 is simulated in FP32 (no hardware FP4). The `FPQuantizer` builds discrete grids and rounds via `torch.searchsorted`.
- **No test framework**: This is a research project. "Tests" are the experiment scripts themselves (run training, compare PPL).

## Experiment Data

- `data/real_tiers/` — BPE-tokenized binary shards (~4.2B tokens total): tier1_c4 (1.0B), tier2_fineweb (1.4B), tier3_wiki (1.0B), tier4_orca (0.84B)
- `data/tokenizer/` — BPE tokenizer (vocab_size=32000) + special token mapping
- `models/gemma4-e2b/` — Local copy of Gemma 4 E2B for reference

## Dependencies

Python 3.11+, PyTorch ≥2.3, Transformers ≥4.45, Datasets, Accelerate, wandb, einops. See `requirements.txt`.
