# CLAUDE.md

This file gives the current working context for Claude Code in this repository.

## Project Overview

Numerical Analysis course project on FP8/FP4 weight quantization for a
Gemma-style causal Transformer.

The current codebase centers on:

- A ~164M parameter Micro-Gemma-FP model.
- FP8/FP4 simulated post-training quantization (PTQ).
- Quantization-aware training (QAT) variants.
- Numerical-analysis tools for quantization diagnostics, including condition
  number estimates, Lipschitz analysis, sensitivity reports, and Lloyd-Max
  adaptive FP4 grids.

## Remote Workflow

The local machine is for editing, data preparation, and result inspection. Long
training and validation should run on the remote GPU server.

Remote environment:

- Host: `bi_group2@lulab_4090`
- Conda env: `sle`
- Remote project path: `/home/bi_group2/Projects/Numerical_Analysis/`
- SSH password file: `.sshpass` (gitignored)

Common commands:

```bash
# Sync local project to remote
./sync.sh

# Run a Python script on remote with PYTHONPATH set
./remote_python.sh src/experiments/train_scaled_baseline.py --data_dir data/real_tiers

# Run an arbitrary remote command
./remote_run.sh "python -m py_compile src/model/transformer.py"
```

## Current Model And Data Conventions

- `MicroGemmaFPConfig()` defines the canonical model scale: 12 layers, 768
  hidden size, 12 query heads, 3 KV heads, GQA 4:1, RMSNorm, RoPE, and
  per-layer token embeddings.
- `transformer.py` applies real sliding-window masks for `sliding` layers and
  full causal masks for `full` layers.
- Datasets return `labels == input_ids`; `MicroGemmaFPForCausalLM.forward()`
  performs the one-token causal LM shift internally. Do not pre-shift labels in
  dataloaders or custom training scripts.
- Training utilities pass padding masks through to the model.
- `evaluate_perplexity()` weights loss by valid shifted label tokens, not by
  batch count.

## Data Pipeline

The remote server should be treated as no-internet for data preparation. Prepare
tokenizer/data locally, then sync.

```bash
python src/experiments/train_tokenizer.py
python src/experiments/prepare_data_chunked.py
./sync.sh
```

Expected tokenized data layout:

- `data/real_tiers/tier1_c4.bin`
- `data/real_tiers/tier2_fineweb.bin`
- `data/real_tiers/tier3_wiki.bin`
- `data/real_tiers/tier4_orca.bin`
- `data/tokenizer/bpe_32k.json`
- `data/tokenizer/special_tokens.json`

## Code Map

```text
src/
├── model/
│   ├── config.py              # MicroGemmaFPConfig
│   └── transformer.py         # Causal Transformer with RMSNorm, RoPE, GQA, sliding/full masks
├── quantization/
│   ├── fp_quantizer.py        # Simulated FP8/FP4 quantizer and QAT autograd helper
│   ├── fp4_grids.py           # E2M1 / NF4 / MXFP4 grid definitions
│   ├── gptq.py                # GPTQ-style PTQ with activation Hessian calibration
│   ├── adaptive_grid.py       # Lloyd-Max per-layer adaptive FP4 grids
│   ├── grid_qat.py            # Grid-based QAT wrappers
│   ├── stochastic.py          # Stochastic rounding utilities
│   ├── hadamard.py            # Walsh-Hadamard utilities
│   └── outlier_rotation.py    # Outlier-aware rotation utilities
├── analysis/
│   ├── condition.py           # κ diagnostics and differentiable condition regularization
│   ├── lipschitz.py           # Lipschitz propagation analysis
│   └── sensitivity.py         # Layer sensitivity and mixed-precision suggestions
└── experiments/
    ├── training_utils.py          # Dataloaders, train loop, eval PPL, checkpoints
    ├── train_scaled_baseline.py   # Canonical FP16 baseline training
    ├── train_cond_regularized.py  # FP16 + condition regularization
    ├── train_qat*.py              # QAT variants
    ├── eval_quantization.py       # Unified PTQ evaluation
    ├── ptq_eval.py                # Single PTQ run
    ├── compare_adaptive_grid.py   # Adaptive grid comparison
    ├── validate_theory.py         # Layer sensitivity validation
    └── validate_rmsnorm.py        # RMSNorm ablation validation
```

## Quantization Notes

- FP8/FP4 is simulated in FP32. There is no hardware FP4 execution path.
- PTQ uses per-channel scaling by default where applicable.
- GPTQ collects calibration activations through forward hooks and passes
  attention masks into the model.
- `condition_number_regularization()` uses a differentiable spectral
  concentration surrogate by default. Exact condition-number diagnostics are
  separate and should not be described as the training objective.
- `adaptive_grid.py` returns unique quantization values. For `n_levels=16`, the
  symmetric FP4 grid has 15 unique tensor values because signed zero has two
  encodings but one numerical value.

## Validation Before Long Runs

Run local syntax checks first, then sync and repeat remotely:

```bash
python -m py_compile \
  src/model/transformer.py \
  src/experiments/training_utils.py \
  src/quantization/fp_quantizer.py \
  src/quantization/gptq.py \
  src/quantization/adaptive_grid.py \
  src/analysis/condition.py

./sync.sh

./remote_run.sh "python -m py_compile src/model/transformer.py src/experiments/training_utils.py src/quantization/fp_quantizer.py src/quantization/gptq.py src/quantization/adaptive_grid.py src/analysis/condition.py"
```

For behavioral smoke tests, prefer tiny model configs instead of launching the
full 164M model unless the task requires it.

## Dependencies

Python 3.11+, PyTorch >= 2.3, Transformers >= 4.45, Datasets, Accelerate,
tokenizers, einops, wandb. See `requirements.txt`.

<!-- GSD:project-start source:PROJECT.md -->
## Project

**Numerical Analysis-Driven FP4 Quantization for Transformers**

A numerical analysis course project that systematically investigates how classical matrix perturbation theory applies (and fails to apply) to weight quantization in Transformer architectures. The project operates on a ~164M parameter Gemma-style causal Transformer, using FP8/FP4 simulated quantization as the testbed for validating and refining numerical analysis predictions about error propagation, condition numbers, and optimal quantizer design.

**Core Value:** **Use numerical analysis to predict, measure, and explain where quantization error goes in a Transformer — and redesign the measurement protocol when the theory and experiments diverge.**

### Constraints

- **Hardware**: 8× RTX 4090 GPU server (remote), accessed via SSH with sshpass
- **Time**: Course project, ~4 week timeline (currently in final phase)
- **Precision**: FP32 simulation of low-precision formats (no hardware FP4)
- **Data**: 4.24B tokens across 4 tiers (C4, FineWeb-edu, Wikipedia, OpenOrca), BPE 32K tokenizer
- **Evaluation**: 100-step evaluation batches, fixed seed=42
<!-- GSD:project-end -->

<!-- GSD:stack-start source:codebase/STACK.md -->
## Technology Stack

## Languages
- Python 3.11+ (required) — all source and experiment code. Local interpreter is Python 3.14.5.
- Bash — shell scripts for remote execution (`remote_python.sh`, `remote_run.sh`, `sync.sh`)
## Runtime
- Conda (remote GPU server: `sle` environment). Local environment uses system Python.
- No Dockerfile or container runtime detected.
- pip (via `requirements.txt`)
- Lockfile: Not present — no `requirements.lock`, `poetry.lock`, or `pipfile.lock`
## Frameworks
- PyTorch >= 2.3.0 (`torch`) — all model definitions, quantization, and training loops
- HuggingFace Transformers >= 4.45.0 (`transformers`) — used for Gemma 4 E2B loading in `train_gemma4.py`, `AutoTokenizer` fallback in `prepare_data.py`, and `save_pretrained` serialization
- Not detected — no test framework found in the codebase
- Not detected — no build system (setup.py, pyproject.toml, or Makefile are absent; the project is run via `python -m` or script path)
## Key Dependencies
- `torch >= 2.3.0` — all tensor operations, neural network modules, optimizers (AdamW), CUDA GPU execution
- `transformers >= 4.45.0` — HuggingFace model loading/saving, AutoTokenizer for Gemma 4 fallback
- `datasets >= 2.20.0` — HuggingFace datasets for streaming C4, FineWeb, Wikipedia, OpenOrca (`load_dataset`)
- `tokenizers >= 0.19.0` — HuggingFace fast tokenizers for BPE training (`tokenizers.Tokenizer`, `tokenizers.models.BPE`, `tokenizers.trainers.BpeTrainer`)
- `numpy >= 1.26.0` — binary data shard I/O (`np.fromfile`, `np.array`), token ID storage as `uint32`
- `tqdm >= 4.66.0` — listed in requirements; not directly imported in `src/` (used internally by HuggingFace libraries)
- `safetensors >= 0.4.0` — listed in requirements; used by HuggingFace `transformers` for Gemma 4 model weights (`models/gemma4-e2b/model.safetensors`)
- `einops >= 0.8.0` — listed in requirements; not directly imported in `src/` (used internally by HuggingFace `transformers`)
- `accelerate >= 0.30.0` — listed in requirements; not directly imported in `src/`
- `wandb >= 0.17.0` — listed in requirements; not imported in `src/` (no experiment logging integration)
- `peft` — used in `train_gemma4.py` for LoRA fine-tuning (`peft.LoraConfig`, `peft.get_peft_model`)
## Configuration
- `.sshpass` file (gitignored) stores SSH password for remote GPU server
- SSH-based remote execution via `sshpass` utility
- No `.env` files detected in project root
- Not applicable — no build step. Direct Python execution.
- No TypeScript, JS bundler, or compiled language found.
## Platform Requirements
- Python 3.11+
- pip install -r requirements.txt
- Network access for HuggingFace datasets download (data preparation only)
- 8-16 GB RAM minimum for data preparation (chunked streaming)
- No internet required for training (tokenized data prepared offline and synced)
- GPU with CUDA (NVIDIA) recommended for model training/evaluation
- Remote GPU server: `bi_group2@lulab_4090` (conda env `sle`)
- Local machine for editing, data preparation, and result inspection
- Minimum ~10GB GPU memory for Gemma 4 E2B + LoRA (`train_gemma4.py`)
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->
## Conventions

## Naming Patterns
- `snake_case.py` for all Python module files — e.g., `fp_quantizer.py`, `training_utils.py`, `condition.py`
- Experiment scripts follow `verb_noun.py` pattern — e.g., `train_scaled_baseline.py`, `eval_quantization.py`, `validate_theory.py`
- No file prefixes like `test_` or suffixes like `_impl` observed
- `__init__.py` files are empty (zero content, used only for package marking)
- `snake_case` for all functions — e.g., `estimate_condition_number()`, `compute_activation_hessian()`, `apply_weight_quantization()`
- Private/helper functions prefixed with underscore: `_build_fp_grid()`, `_round_to_nearest()`, `_detect_data_dir()`, `_replace_linears()`
- Module-level helper functions are common (not all logic lives in classes)
- `PascalCase` for all classes — e.g., `FPQuantizer`, `MicroGemmaFPConfig`, `TransformerLayer`, `AdaptiveGridQuantizer`, `RMSNorm`
- Abbreviations in class names kept uppercase: `RMSNorm`, `MXFP4Quantizer`, `GPTQQuantizer`, `QATLinear`
- Inner classes (defined inside functions) follow same PascalCase: `QATLinear`, `QATOptimizedLinear`, `IdentityNorm`
- `snake_case` for local variables and instance attributes — e.g., `hidden_states`, `input_ids`, `layer_idx`, `max_sigma`
- Short variable names accepted in mathematical/loop contexts: `W`, `H`, `L`, `x`, `q`, `w`, `m`, `h`
- Module-level constants in `UPPER_SNAKE_CASE`: `FP_FORMAT_SPECS`, `FP4_LEVELS`, `FP8_E4M3_LEVELS`, `FP4_E2M1_GRID`, `NF4_GRID`
- `PascalCase` for type names (standard Python)
- Union types use Python 3.10+ `X | Y` syntax: `scale: torch.Tensor | None = None`, `max_samples: int | None = None`
- Return type hints present on most functions, but some top-level experiment script functions lack them
## Code Style
- No formatting tool config detected — no `.prettierrc`, `pyproject.toml`, `ruff.toml`, or `setup.cfg`
- Code uses 4-space indentation (PEP 8 standard)
- Line length appears to be ~100-120 characters (wider than PEP 8's 79)
- Consistent blank line around class definitions and function definitions
- No trailing whitespace observed
- **Recommendation:** Add `ruff` or `black` configuration to enforce consistent formatting
- No linting configuration detected — no `pyproject.toml`, `.flake8`, `.pylintrc`, or `ruff.toml`
- **Recommendation:** Add `ruff` for linting and formatting in CI
- Heavy use of visual comment separators throughout:
## Import Organization
- All intra-project imports use absolute `src.` package paths: `from src.model.config import MicroGemmaFPConfig`
- No relative imports (`from .config import ...`) observed
- No import aliases or path rewrites configured
- Some files use `import os` inline inside functions (late imports) rather than at module top — e.g., `import os` at line 58 of `train_scaled_baseline.py`, line 64 of `train_fp16_baseline.py`, line 161 of `phase2_comparison.py`
- Standard library modules are sometimes comma-imported on one line: `import random, os, math`
## Error Handling
- `ValueError` with descriptive message in constructors for invalid config: `raise ValueError(f"Unknown format: {fmt}. Options: {list(FP_FORMAT_SPECS.keys())}")`
- Try/except for graceful degradation in resource loading:
- No custom exception classes defined anywhere
- No structured logging of errors (uses `print()` not `logging`)
- Some broad `except:` or `except Exception:` without specific error types
- No error recovery in long-running training loops (a step failure kills the whole process)
- Many functions silently return default values on failure rather than raising
## Logging
- Status messages at key milestones: `print(f"Device: {device}")`, `print(f"Parameters: {stats['total']:,} total")`
- Warnings prefixed with `[WARN]`: `print(f"[WARN] Tokenizer not found at {tokenizer_path}, ...")`
- Diagnostic output in analysis scripts printed during computation
- No `logging` module, no log levels, no structured logging
- No wandb logging in training scripts (wandb in requirements but not imported in any training script)
## Comments
- Module-level docstrings explain purpose, usage, and theory — e.g., `gptq.py` has 16 lines of docstring with mathematical explanation
- Inline comments explain mathematical derivations, algorithm steps, and design decisions
- Section headers with box separators for visual grouping
- Google-style docstrings with `Args:`, `Returns:`, `Raises:` sections are used in many public functions
- Not all functions have them — there is inconsistency
- Class docstrings describe purpose and usage pattern (including code examples in some)
## Function Design
- Functions return single values or dicts (never tuples with more than 2 elements)
- Pure computation functions return scalars or tensors
- Training/analysis functions return `list[dict]` (per-step/per-layer metrics)
- Model forward returns `dict` always: `{'loss': ..., 'logits': ...}`
## Module Design
- No explicit `__all__` in any module
- `__init__.py` files are entirely empty — no re-exports
- Consumers import from concrete module paths: `from src.quantization.fp_quantizer import FPQuantizer`
- Core logic modules (`fp_quantizer.py`, `training_utils.py`, `transformer.py`): 200-270 lines
- Analysis modules (`condition.py`, `lipschitz.py`, `sensitivity.py`): 50-170 lines
- Experiment scripts: 80-250 lines
- `smoke_gemma4.py`: ~20 lines
## Common Patterns
#!/usr/bin/env python3
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture

## System Overview
```text
```
## Component Responsibilities
| Component | Responsibility | File |
|-----------|----------------|------|
| `MicroGemmaFPConfig` | Single-source configuration for architecture, quantization flags, training defaults | `src/model/config.py` |
| `MicroGemmaFPForCausalLM` | Full causal LM: Transformer backbone + tied lm_head with weight tying | `src/model/transformer.py` |
| `MicroGemmaFPModel` | Core Transformer: embeddings, per-layer embeddings, layer stack, final norm | `src/model/transformer.py` |
| `TransformerLayer` | Pre-norm attention + FFN residual block | `src/model/transformer.py` |
| `Attention` | Q/K/V projections with RoPE, QK RMSNorm, GQA, sliding/full mask | `src/model/transformer.py` |
| `FFN` | Gate-up-down MLP with GELU activation | `src/model/transformer.py` |
| `RMSNorm` | Root mean square layer normalization | `src/model/transformer.py` |
| `FPQuantizer` | Simulated FP8/FP4 quantization, per-tensor or per-channel, deterministic/stochastic | `src/quantization/fp_quantizer.py` |
| `GPTQQuantizer` | GPTQ-style weight compensation with activation Hessian calibration | `src/quantization/gptq.py` |
| `AdaptiveGridQuantizer` | Per-layer Lloyd-Max optimized FP4 grids with optional kappa-weighting | `src/quantization/adaptive_grid.py` |
| `GridQuantizer` | Quantize to any discrete grid (E2M1, NF4, MXFP4) | `src/quantization/fp4_grids.py` |
| `MXFP4Quantizer` | Block-scaling MXFP4 quantizer (block_size=32) | `src/quantization/fp4_grids.py` |
| `DuQuantStyleQuantizer` | Outlier-aware scaling + block-Hadamard rotation + quantization pipeline | `src/quantization/outlier_rotation.py` |
| `HadamardRotation` | Walsh-Hadamard transform for activation smoothing | `src/quantization/hadamard.py` |
| `condition_number_regularization()` | Differentiable kappa surrogate for training loss | `src/analysis/condition.py` |
| `estimate_condition_number()` | Power iteration for kappa(W) estimation | `src/analysis/condition.py` |
| `compute_propagation_factors()` | Layer-wise Lipschitz error propagation analysis | `src/analysis/lipschitz.py` |
| `per_layer_sensitivity_report()` | Combined condition+Lipschitz+quantization sensitivity | `src/analysis/sensitivity.py` |
| `train_epoch()` | Single training epoch with gradient clipping | `src/experiments/training_utils.py` |
| `evaluate_perplexity()` | Weighted perplexity evaluation | `src/experiments/training_utils.py` |
| `get_dataloader()` | Auto-selecting dataloader factory (real data or offline fallback) | `src/experiments/training_utils.py` |
## Pattern Overview
- Model, quantization, and analysis are independent subpackages with no circular dependencies
- No global state or singletons -- each experiment instantiates its own model and quantizer
- Configuration is passed via `MicroGemmaFPConfig` dataclass, not config files or env vars
- Experiments drive everything through `train_epoch()` / `evaluate_perplexity()` from `training_utils.py`
- Data awareness: no hardcoded paths to data files -- all paths are passed as CLI arguments
- Remote execution: sync.sh + remote_python.sh/remote_run.sh for GPU server workflow
## Layers
- Purpose: Define the Micro-Gemma-FP Transformer architecture
- Contains: Configuration dataclass, RMSNorm, RotaryEmbedding, Attention (sliding/full), FFN, TransformerLayer, MicroGemmaFPModel, MicroGemmaFPForCausalLM
- Depends on: torch, torch.nn, torch.nn.functional
- Used by: Quantization layer (for forward hooks), Experiment layer (for instantiation)
- Purpose: All quantization algorithms -- simulation, PTQ, QAT, grid design
- Contains: FP quantizer, GPTQ, adaptive Lloyd-Max grids, E2M1/NF4/MXFP4 grids, grid-based QAT wrappers, stochastic rounding, Hadamard transform, outlier rotation
- Depends on: torch, `src/analysis/condition` (adaptive_grid imports `estimate_condition_number`)
- Used by: Experiment layer (all eval/train scripts)
- Purpose: Numerical analysis tools for quantization diagnostics
- Contains: Condition number estimation (power iteration), differentiable regularization, Lipschitz propagation, per-layer sensitivity reports
- Depends on: torch, torch.nn.functional
- Used by: Experiment layer (validate_theory, train_cond_regularized, eval_quantization)
- Purpose: Executable scripts that orchestrate training, evaluation, validation, and data preparation
- Contains: ~18 scripts -- training baselines, QAT variants, PTQ evaluation, grid comparison, theory validation, data preparation
- Depends on: All other layers (model + quantization + analysis)
- Used by: Users (run via `python src/experiments/*.py` or `./remote_python.sh`)
## Data Flow
### Primary Training Path (FP16 baseline)
### PTQ Evaluation Path
### QAT Training Path
### Condition-Regularized Training Path
- No global mutable state. All state is in `nn.Module.parameters()` and optimizer state dicts.
- Checkpoints saved as `.pt` files via `torch.save()`.
- Quantization calibration data is collected ephemerally within `GPTQQuantizer.quantize_model()` and discarded.
## Key Abstractions
- Purpose: Single source of truth for all model architecture parameters and quantization flags
- Examples: `src/model/config.py`
- Pattern: Dataclass with derived properties (`model_name`, `num_sliding_layers`, `num_full_layers`)
- Purpose: Abstract FP format quantization with per-channel/per-tensor, deterministic/stochastic modes
- Examples: `src/quantization/fp_quantizer.py`
- Pattern: Strategy -- same interface supports FP8_E4M3, FP8_E5M2, FP4_E2M1
- Purpose: Consistent method to identify weights that can be quantized
- Pattern: `param.dim() >= 2 and any(k in name for k in ('proj', 'embed_tokens', 'lm_head'))` (in `transformer.py:264-269`)
- Also used by experiment scripts with `param.dim() >= 2 and 'embed' not in name and 'lm_head' not in name`
- Purpose: Transparent fallback from real data to offline embedded corpus
- Examples: `src/experiments/training_utils.py:197-219`
- Pattern: Check for `.bin` files in data_dir, fall back to `CharTokenizer` + `LocalTextDataset` if absent
## Entry Points
- Location: `src/experiments/train_scaled_baseline.py`, `src/experiments/train_qat*.py`, `src/experiments/train_cond_regularized.py`
- Triggers: Command-line execution via `python src/experiments/<script>.py [args]`
- Responsibilities: Parse args, create config, instantiate model, create dataloader, run training loop, save checkpoint
- Location: `src/experiments/eval_quantization.py`, `src/experiments/ptq_eval.py`, `src/experiments/compare_adaptive_grid.py`
- Triggers: Command-line execution
- Responsibilities: Load checkpoint, apply quantization, measure perplexity, report results
- Location: `src/experiments/validate_theory.py`, `src/experiments/validate_rmsnorm.py`
- Triggers: Command-line execution
- Responsibilities: Test theoretical predictions (kappa vs MSE, Lipschitz propagation, RMSNorm role)
- Location: `src/experiments/train_tokenizer.py`, `src/experiments/prepare_data_chunked.py`, `src/experiments/prepare_data.py`
- Triggers: Command-line execution
- Responsibilities: Train BPE tokenizer, tokenize datasets, write .bin shards
- Location: `src/experiments/final_summary.py`
- Triggers: Command-line execution
- Responsibilities: Load all saved checkpoints, evaluate PPL, produce comparison table
## Architectural Constraints
- **Threading:** Single-threaded. No worker threads or multiprocessing (DataLoader num_workers=0).
- **Global state:** None. All modules avoid module-level mutable state.
- **Circular imports:** None detected. Dependency graph is strictly layered: model <-- quantization <-- analysis <-- experiments, with adaptive_grid importing from analysis/condition.
- **GPU memory:** No explicit memory management except `torch.cuda.empty_cache()` calls in `train_cond_regularized.py:143`.
- **Remote execution:** All training/evaluation runs on remote GPU via sshpass + rsync. Local machine is for editing and data preparation only.
## Anti-Patterns
### Embedded corpus as Python source
### Empty `__init__.py` files
### Monolithic `training_utils.py`
### Deep import chains with PYTHONPATH dependency
## Error Handling
- `clamp(min=1e-12)` for division safety in `fp_quantizer.py`, `adaptive_grid.py`, `condition.py`
- Progressive damping loops in `gptq.py:43-52` for Cholesky decomposition
- No custom exception types defined anywhere
- No try/except in training loop for NaN loss recovery
- FileNotFoundError handling only in `final_summary.py:43` for missing checkpoints
## Cross-Cutting Concerns
<!-- GSD:architecture-end -->

<!-- GSD:skills-start source:skills/ -->
## Project Skills

No project skills found. Add skills to any of: `.claude/skills/`, `.agents/skills/`, `.cursor/skills/`, `.github/skills/`, or `.codex/skills/` with a `SKILL.md` index file.
<!-- GSD:skills-end -->

<!-- GSD:workflow-start source:GSD defaults -->
## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:
- `/gsd-quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd-debug` for investigation and bug fixing
- `/gsd-execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->

<!-- GSD:profile-start -->
## Developer Profile

> Profile not yet configured. Run `/gsd-profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
