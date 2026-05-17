# Codebase Structure

**Analysis Date:** 2026-05-17

## Directory Layout

```
proj/
├── .agents/                           # Claude Code agent configurations (project-specific)
├── .codex/                            # Codex IDE metadata
├── .planning/codebase/                # GSD codebase map documents (generated)
├── src/
│   ├── __init__.py                    # Package marker (empty)
│   ├── model/
│   │   ├── __init__.py                # Package marker (empty)
│   │   ├── config.py                  # MicroGemmaFPConfig dataclass (~164M)
│   │   └── transformer.py             # Full Transformer: RMSNorm, RoPE, GQA, sliding/full attn
│   ├── quantization/
│   │   ├── __init__.py                # Package marker (empty)
│   │   ├── fp_quantizer.py            # FP8/FP4 simulated quantizer and QAT hooks
│   │   ├── fp4_grids.py               # E2M1 / NF4 / MXFP4 grid definitions
│   │   ├── gptq.py                    # GPTQ weight compensation with activation Hessian
│   │   ├── adaptive_grid.py           # Lloyd-Max per-layer adaptive FP4 grids
│   │   ├── grid_qat.py                # Grid-based QAT wrappers (NF4, MXFP4)
│   │   ├── stochastic.py              # Stochastic rounding utilities
│   │   ├── hadamard.py                # Fast Walsh-Hadamard transform
│   │   └── outlier_rotation.py        # DuQuant++ outlier-aware rotation
│   ├── analysis/
│   │   ├── __init__.py                # Package marker (empty)
│   │   ├── condition.py               # Condition number estimation and regularization
│   │   ├── lipschitz.py               # Lipschitz constant propagation analysis
│   │   └── sensitivity.py             # Per-layer sensitivity reports + mixed-precision
│   └── experiments/
│       ├── __init__.py                # Package marker (empty)
│       ├── training_utils.py          # Dataloaders, training loop, evaluation, checkpointing
│       ├── large_corpus.py            # LLM-generated offline text corpus
│       ├── train_tokenizer.py         # BPE tokenizer training
│       ├── prepare_data.py            # Data preparation utilities
│       ├── prepare_data_chunked.py    # Chunked data preparation (for large datasets)
│       ├── train_scaled_baseline.py   # FP16 baseline training entry point
│       ├── train_fp16_baseline.py     # Alternative FP16 baseline entry point
│       ├── train_cond_regularized.py  # FP16 + condition number regularization
│       ├── train_qat.py               # QAT training (FP8/FP4 with STE)
│       ├── train_qat_optimized.py     # Optimized QAT training variant
│       ├── train_qat_fp4_opt.py       # FP4-specific optimized QAT
│       ├── train_fp4_grid_qat.py      # FP4 grid-based QAT training
│       ├── train_gemma4.py            # Gemma 4 training entry point
│       ├── eval_quantization.py       # Unified PTQ evaluation pipeline
│       ├── ptq_eval.py                # Single PTQ evaluation run
│       ├── eval_all.py                # Bulk evaluation script
│       ├── eval_all_grids.py          # Bulk grid comparison evaluation
│       ├── eval_fp4_qat.py            # FP4 QAT evaluation
│       ├── eval_qat_checkpoints.py    # Evaluate multiple QAT checkpoints
│       ├── fp4_ptq_compare.py         # FP4 PTQ method comparison
│       ├── compare_adaptive_grid.py   # Adaptive grid strategies comparison
│       ├── phase2_comparison.py       # Phase 2 systematic comparison
│       ├── final_summary.py           # Final results aggregation
│       ├── validate_theory.py         # Theory validation (kappa, Lipschitz)
│       └── validate_rmsnorm.py        # RMSNorm role validation
├── data/
│   ├── tokenizer/
│   │   ├── bpe_32k.json               # BPE tokenizer vocabulary
│   │   └── special_tokens.json        # Special token ID mappings
│   ├── real_tiers/
│   │   ├── tier1_c4.bin               # C4 pre-tokenized shard
│   │   ├── tier2_fineweb.bin          # FineWeb pre-tokenized shard
│   │   ├── tier3_wiki.bin             # Wikipedia pre-tokenized shard
│   │   └── tier4_orca.bin             # OpenOrca pre-tokenized shard
│   └── gemma4_tiers/                  # Alternative data tier for Gemma 4 experiments
├── models/
│   └── gemma4-e2b/                    # Gemma 4 E2B pretrained model files
│       ├── config.json
│       ├── model.safetensors
│       ├── tokenizer.json
│       ├── generation_config.json
│       └── .cache/                    # HuggingFace download cache
├── docs/
│   ├── PROPOSAL.md                    # Project proposal (14KB)
│   ├── REPORT.md                      # Experiment report with tables (7.7KB)
│   └── numerical_analysis_training_project_proposal.md  # Original proposal (2.5KB)
├── checkpoints/                       # Generated training checkpoints (gitignored)
│   ├── fp16_baseline/
│   ├── qat_fp8/
│   ├── qat_fp4/
│   ├── cond_regularized/
│   └── ...
├── CLAUDE.md                          # Project context for Claude Code
├── README.md                          # Project overview and quick start
├── requirements.txt                   # Python dependencies
├── sync.sh                            # rsync to remote GPU server
├── remote_python.sh                   # Run Python script on remote via sshpass
├── remote_run.sh                      # Run arbitrary command on remote via sshpass
├── smoke_gemma4.py                    # Gemma 4 E2B smoke test
└── .sshpass                           # SSH password for remote server (gitignored)
```

## Directory Purposes

**`src/model/`:**
- Purpose: Transformer model architecture definition
- Contains: Configuration dataclass, core nn.Module definitions (RMSNorm, RoPE, Attention, FFN, TransformerLayer, full model)
- Key files: `config.py` (98 lines), `transformer.py` (271 lines)
- Naming: All model components in one file; config separated for single-responsibility

**`src/quantization/`:**
- Purpose: All quantization algorithms and utilities
- Contains: FP8/FP4 quantizer, GPTQ, adaptive grids, grid definitions, QAT wrappers, stochastic rounding, Hadamard transform, outlier rotation
- Key files: `fp_quantizer.py` (243 lines), `gptq.py` (255 lines), `adaptive_grid.py` (188 lines), `fp4_grids.py` (214 lines)
- Naming: Each quantization technique gets its own file; shared grids in `fp4_grids.py`

**`src/analysis/`:**
- Purpose: Numerical analysis tools for quantization diagnostics
- Contains: Condition number estimation (power iteration), differentiable regularization, Lipschitz propagation, sensitivity reports
- Key files: `condition.py` (169 lines), `lipschitz.py` (119 lines), `sensitivity.py` (80 lines)
- Dependencies: `sensitivity.py` imports from both `condition.py` and `lipschitz.py`

**`src/experiments/`:**
- Purpose: All executable scripts organized by experiment type
- Contains: ~18 scripts covering training, PTQ evaluation, QAT evaluation, validation, data preparation, and summary
- Largest directory by file count. `training_utils.py` is the largest file (357 lines) and is imported by nearly every other experiment script.
- Entry scripts all follow the same pattern: `argparse.ArgumentParser` -> `main()` -> `if __name__ == '__main__': main()`

**`data/`:**
- Purpose: Tokenized training data and tokenizer artifacts
- Contains: BPE tokenizer, special token mappings, four pre-tokenized .bin data shards (~4.24B tokens total), alternative data tiers
- Generated by: `train_tokenizer.py` and `prepare_data_chunked.py` (run locally, then synced to remote)

**`models/`:**
- Purpose: Pretrained model checkpoints for reference models
- Contains: Gemma 4 E2B model files, HuggingFace download cache
- Generated by: External download, not by project scripts

**`docs/`:**
- Purpose: Project documentation and reports
- Contains: Project proposal, experiment report with final results, raw proposal draft

**Root scripts:**
- `sync.sh`: rsync local project to remote GPU server, excludes .git, __pycache__, .venv, wandb
- `remote_python.sh`: SSH + conda activate + PYTHONPATH set, runs a Python script on remote
- `remote_run.sh`: SSH + conda activate + PYTHONPATH set, runs arbitrary command on remote
- `smoke_gemma4.py`: Quick test to verify Gemma 4 model loads on remote GPU

## Key File Locations

**Entry Points:**
- `src/experiments/train_scaled_baseline.py`: Canonical FP16 baseline training
- `src/experiments/train_cond_regularized.py`: Condition-regularized training + PTQ comparison
- `src/experiments/train_qat.py`: QAT training with FP8/FP4 and STE
- `src/experiments/eval_quantization.py`: Unified PTQ evaluation (simple/gptq/mixed precision)
- `src/experiments/final_summary.py`: Aggregate all experiment results into comparison table

**Configuration:**
- `src/model/config.py`: Single `MicroGemmaFPConfig` dataclass for architecture + quantization flags
- `requirements.txt`: Python package dependencies
- `src/experiments/training_utils.py`: Runtime configuration for dataloaders

**Core Logic:**
- `src/model/transformer.py`: Full model definition (271 lines)
- `src/quantization/fp_quantizer.py`: FP8/FP4 quantization core (243 lines)
- `src/quantization/gptq.py`: GPTQ compensation algorithm (255 lines)
- `src/analysis/condition.py`: Condition number analysis (169 lines)

**Testing:**
- No test files detected (no `test_*.py` or `*_test.py` files, no test directory)
- `src/experiments/validate_theory.py` and `validate_rmsnorm.py` serve as validation scripts but are not automated tests
- `smoke_gemma4.py` is a manual smoke test

## Naming Conventions

**Files:**
- `snake_case.py` for all source files: `fp_quantizer.py`, `training_utils.py`, `evaluate_perplexity.py`
- PascalCase for directories under `src/`: `model`, `quantization`, `analysis`, `experiments`
- Hyphenated for root scripts: `sync.sh`, `remote_python.sh`, `remote_run.sh`

**Classes:**
- PascalCase with descriptive names: `FPQuantizer`, `GPTQQuantizer`, `MicroGemmaFPForCausalLM`, `AdaptiveGridQuantizer`
- Prefix `MicroGemmaFP` for model-specific classes, generic names for quantizers

**Functions:**
- `snake_case` with verb-first naming: `estimate_condition_number()`, `compute_propagation_factors()`, `train_epoch()`, `get_dataloader()`
- Private functions prefixed with underscore: `_build_fp_grid()`, `_simulate_quantize()`, `_detect_data_dir()`

**Constants:**
- `UPPER_SNAKE_CASE`: `FP_FORMAT_SPECS`, `FP4_LEVELS`, `FP4_E2M1_GRID`, `NF4_GRID`

**Experiment naming:**
- `train_<variant>.py` for training scripts
- `eval_<variant>.py` for evaluation scripts
- `validate_<topic>.py` for validation scripts
- `compare_<feature>.py` for comparison scripts

## Where to Add New Code

**New Feature (e.g., new quantization method):**
- Implementation: `src/quantization/<new_method>.py`
- If new grids needed: add to `src/quantization/fp4_grids.py`
- Experiment script: `src/experiments/eval_<new_method>.py`
- Tests: `tests/test_<new_method>.py` (test infrastructure does not exist yet)

**New Model Architecture Variant:**
- Configuration: add new config class or modify `MicroGemmaFPConfig` in `src/model/config.py`
- Model: add new module in `src/model/transformer.py`
- Entry point: `src/experiments/train_<variant>.py`

**New Analysis Metric:**
- Implementation: `src/analysis/<metric_name>.py`
- Validate via: `src/experiments/validate_<metric_name>.py`

**New Training Experiment:**
- Script: `src/experiments/train_<variant>.py`
- Use `training_utils.py` for dataloaders, training loop, evaluation, and checkpointing
- Follow the argparse + main() pattern established by existing scripts

**Data Processing:**
- Data preparation: `src/experiments/prepare_data_<variant>.py`
- Output: `data/<dataset_name>/<shard>.bin`
- Tokenizer: `data/tokenizer/`

**Shared Utilities:**
- Training/eval helpers: `src/experiments/training_utils.py` (existing, but consider splitting if it grows beyond 400 lines)
- New utility modules: `src/utils/` or `src/experiments/utils/` if a clean split is warranted

## Special Directories

**`checkpoints/`:**
- Purpose: Saved model and optimizer states from training runs
- Generated: Yes, by all training scripts
- Committed: No (gitignored)
- Structure: One subdirectory per experiment type (`fp16_baseline/`, `qat_fp8/`, `cond_regularized/`, etc.)

**`data/`:**
- Purpose: Tokenized training data shards and tokenizer files
- Generated: Partially (tokenizer is pre-created, data shards are generated by `prepare_data_chunked.py`)
- Committed: Yes (data files checked into git for reproducibility)

**`models/`:**
- Purpose: External pretrained model checkpoints
- Generated: No (downloaded from external source)
- Committed: Yes (Gemma 4 E2B files)

**`.planning/`:**
- Purpose: GSD orchestration artifacts including codebase maps and implementation plans
- Generated: Yes
- Committed: Yes

**`docs/`:**
- Purpose: Project reports, proposals, and documentation
- Generated: No (hand-written)
- Committed: Yes

---

*Structure analysis: 2026-05-17*
