<!-- refreshed: 2026-05-17 -->
# Architecture

**Analysis Date:** 2026-05-17

## System Overview

```text
┌────────────────────────────────────────────────────────────────────────────┐
│                         EXPERIMENT LAYER                                   │
│  Entry-point scripts that orchestrate training, evaluation, & validation   │
│  `src/experiments/`                                                        │
├──────────────────┬──────────────────┬──────────────────┬───────────────────┤
│  Training        │  PTQ Evaluation  │  QAT Evaluation  │  Validation       │
│  `train_*.py`    │  `eval_*.py`     │  `eval_qat*.py`  │  `validate_*.py`  │
│  `ptq_eval.py`   │  `ptq_eval.py`   │  `final_summary` │  `large_corpus.py`│
└────────┬─────────┴────────┬─────────┴─────────┬────────┴─────────┬─────────┘
         │                  │                   │                  │
         ▼                  ▼                   ▼                  ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                            ANALYSIS LAYER                                  │
│  Numerical analysis tools: condition numbers, Lipschitz, sensitivity       │
│  `src/analysis/`                                                           │
├──────────────────┬──────────────────┬──────────────────────────────────────┤
│  Condition       │  Lipschitz       │  Sensitivity                         │
│  `condition.py`  │  `lipschitz.py`  │  `sensitivity.py`                   │
└────────┬─────────┴────────┬─────────┴──────────────────┬───────────────────┘
         │                  │                            │
         ▼                  ▼                            ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                          QUANTIZATION LAYER                                │
│  FP8/FP4 quantizers, GPTQ, adaptive grids, Hadamard transform             │
│  `src/quantization/`                                                       │
├──────────────────┬──────────────────┬──────────────────┬───────────────────┤
│  FP Quantizer    │  GPTQ            │  Grids           │  Transforms       │
│  `fp_quantizer`  │  `gptq.py`       │  `fp4_grids.py`  │  `hadamard.py`    │
│  `stochastic.py` │  `grid_qat.py`   │  `adaptive_grid` │  `outlier_rot`    │
└────────┬─────────┴────────┬─────────┴─────────┬────────┴─────────┬─────────┘
         │                  │                   │                  │
         ▼                  ▼                   ▼                  ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                              MODEL LAYER                                   │
│  Micro-Gemma-FP Transformer: RMSNorm, RoPE, GQA, sliding/full attention    │
│  `src/model/`                                                              │
├──────────────────────────────────┬─────────────────────────────────────────┤
│  `config.py`                     │  `transformer.py`                      │
│  ~164M param configuration       │  Causal LM with LM head                │
└──────────────────────────────────┴─────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                      DATA / STORAGE / INFRASTRUCTURE                       │
│  `data/`  `models/`  `checkpoints/`  `docs/`                              │
│  Tokenized shards, pretrained checkpoints, experiment outputs, reports    │
└────────────────────────────────────────────────────────────────────────────┘
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

**Overall:** Layered research pipeline -- each layer is self-contained and independently importable. Experiments are standalone entry-point scripts that import from layers below.

**Key Characteristics:**
- Model, quantization, and analysis are independent subpackages with no circular dependencies
- No global state or singletons -- each experiment instantiates its own model and quantizer
- Configuration is passed via `MicroGemmaFPConfig` dataclass, not config files or env vars
- Experiments drive everything through `train_epoch()` / `evaluate_perplexity()` from `training_utils.py`
- Data awareness: no hardcoded paths to data files -- all paths are passed as CLI arguments
- Remote execution: sync.sh + remote_python.sh/remote_run.sh for GPU server workflow

## Layers

**Model Layer (`src/model/`):**
- Purpose: Define the Micro-Gemma-FP Transformer architecture
- Contains: Configuration dataclass, RMSNorm, RotaryEmbedding, Attention (sliding/full), FFN, TransformerLayer, MicroGemmaFPModel, MicroGemmaFPForCausalLM
- Depends on: torch, torch.nn, torch.nn.functional
- Used by: Quantization layer (for forward hooks), Experiment layer (for instantiation)

**Quantization Layer (`src/quantization/`):**
- Purpose: All quantization algorithms -- simulation, PTQ, QAT, grid design
- Contains: FP quantizer, GPTQ, adaptive Lloyd-Max grids, E2M1/NF4/MXFP4 grids, grid-based QAT wrappers, stochastic rounding, Hadamard transform, outlier rotation
- Depends on: torch, `src/analysis/condition` (adaptive_grid imports `estimate_condition_number`)
- Used by: Experiment layer (all eval/train scripts)

**Analysis Layer (`src/analysis/`):**
- Purpose: Numerical analysis tools for quantization diagnostics
- Contains: Condition number estimation (power iteration), differentiable regularization, Lipschitz propagation, per-layer sensitivity reports
- Depends on: torch, torch.nn.functional
- Used by: Experiment layer (validate_theory, train_cond_regularized, eval_quantization)

**Experiment Layer (`src/experiments/`):**
- Purpose: Executable scripts that orchestrate training, evaluation, validation, and data preparation
- Contains: ~18 scripts -- training baselines, QAT variants, PTQ evaluation, grid comparison, theory validation, data preparation
- Depends on: All other layers (model + quantization + analysis)
- Used by: Users (run via `python src/experiments/*.py` or `./remote_python.sh`)

## Data Flow

### Primary Training Path (FP16 baseline)

1. `MicroGemmaFPConfig()` defines model architecture (`src/model/config.py:16`)
2. `MicroGemmaFPForCausalLM(config)` instantiates model w/ Xavier-normal init (`src/model/transformer.py:229-236`)
3. `get_dataloader(data_dir)` creates DataLoader from .bin shards (`src/experiments/training_utils.py:197-219`)
4. `train_epoch(model, dataloader, optimizer, device)` runs forward/backward pass (`src/experiments/training_utils.py:274-310`)
5. Model internally computes cross-entropy with shift labels (`src/model/transformer.py:249-256`)
6. Optimizer steps with gradient clipping (`src/experiments/training_utils.py:297-300`)
7. `save_checkpoint()` persists model and optimizer state (`src/experiments/training_utils.py:340-348`)

### PTQ Evaluation Path

1. Load FP16 checkpoint via `load_checkpoint()` (`src/experiments/training_utils.py:351-356`)
2. Create `FPQuantizer(fmt, per_channel=True)` (`src/quantization/fp_quantizer.py:61-77`)
3. For simple PTQ: iterate `named_parameters()` and quantize each weight >= 2D (`src/experiments/eval_quantization.py:45-52`)
4. For GPTQ PTQ: `GPTQQuantizer.quantize_model()` collects activation Hessian via hooks, applies column-by-column compensation (`src/quantization/gptq.py:138-191`)
5. `evaluate_perplexity()` measures language modeling degradation (`src/experiments/training_utils.py:313-337`)

### QAT Training Path

1. Instantiate model and `FPQuantizer`
2. `qat_wrap_model()` replaces all `nn.Linear` with `QATLinear` wrappers (`src/experiments/train_qat.py:48-63`)
3. Forward pass: `QATLinear.forward()` quantizes weight, then `F.linear(x, w_q, bias)` (`src/experiments/train_qat.py:31-45`)
4. Backward pass: Straight-Through Estimator (STE) -- gradient bypasses quantization
5. Alternate method via forward hooks: `make_qat_forward_hook()` + `make_qat_forward_hook_restore()` (`src/quantization/fp_quantizer.py:219-242`)

### Condition-Regularized Training Path

1. `condition_number_regularization(model, lambda_cond)` called each step (`src/analysis/condition.py:99-133`)
2. Uses differentiable spectral-concentration surrogate (not SVD) for efficiency
3. Loss = CE + lambda * sum(log(kappa_surrogate(W_i))) (`src/experiments/train_cond_regularized.py:77-78`)
4. Regularized checkpoint evaluated for PTQ degradation improvement

**State Management:**
- No global mutable state. All state is in `nn.Module.parameters()` and optimizer state dicts.
- Checkpoints saved as `.pt` files via `torch.save()`.
- Quantization calibration data is collected ephemerally within `GPTQQuantizer.quantize_model()` and discarded.

## Key Abstractions

**MicroGemmaFPConfig (dataclass):**
- Purpose: Single source of truth for all model architecture parameters and quantization flags
- Examples: `src/model/config.py`
- Pattern: Dataclass with derived properties (`model_name`, `num_sliding_layers`, `num_full_layers`)

**FPQuantizer:**
- Purpose: Abstract FP format quantization with per-channel/per-tensor, deterministic/stochastic modes
- Examples: `src/quantization/fp_quantizer.py`
- Pattern: Strategy -- same interface supports FP8_E4M3, FP8_E5M2, FP4_E2M1

**Quantizable weight convention:**
- Purpose: Consistent method to identify weights that can be quantized
- Pattern: `param.dim() >= 2 and any(k in name for k in ('proj', 'embed_tokens', 'lm_head'))` (in `transformer.py:264-269`)
- Also used by experiment scripts with `param.dim() >= 2 and 'embed' not in name and 'lm_head' not in name`

**Auto-selecting dataloader:**
- Purpose: Transparent fallback from real data to offline embedded corpus
- Examples: `src/experiments/training_utils.py:197-219`
- Pattern: Check for `.bin` files in data_dir, fall back to `CharTokenizer` + `LocalTextDataset` if absent

## Entry Points

**Training scripts:**
- Location: `src/experiments/train_scaled_baseline.py`, `src/experiments/train_qat*.py`, `src/experiments/train_cond_regularized.py`
- Triggers: Command-line execution via `python src/experiments/<script>.py [args]`
- Responsibilities: Parse args, create config, instantiate model, create dataloader, run training loop, save checkpoint

**Evaluation scripts:**
- Location: `src/experiments/eval_quantization.py`, `src/experiments/ptq_eval.py`, `src/experiments/compare_adaptive_grid.py`
- Triggers: Command-line execution
- Responsibilities: Load checkpoint, apply quantization, measure perplexity, report results

**Validation scripts:**
- Location: `src/experiments/validate_theory.py`, `src/experiments/validate_rmsnorm.py`
- Triggers: Command-line execution
- Responsibilities: Test theoretical predictions (kappa vs MSE, Lipschitz propagation, RMSNorm role)

**Data preparation scripts:**
- Location: `src/experiments/train_tokenizer.py`, `src/experiments/prepare_data_chunked.py`, `src/experiments/prepare_data.py`
- Triggers: Command-line execution
- Responsibilities: Train BPE tokenizer, tokenize datasets, write .bin shards

**Summary script:**
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

**What happens:** `src/experiments/large_corpus.py` contains a large LLM-generated text corpus defined as a Python string variable. This file is imported in `training_utils.py:58` as the offline fallback.

**Why it's wrong:** Inflates source file size, bypasses proper data management. A large string in a Python file is hard to version efficiently (git tracks whole-file changes).

**Do this instead:** Store corpus as a separate `.txt` data file in `data/` and load it at runtime.

### Empty `__init__.py` files

**What happens:** All `__init__.py` files across `src/` are empty (1 blank line). No `__all__` definitions.

**Why it's wrong:** No explicit public API. `from src.model import *` would expose internal modules. No control over what is public vs private.

**Do this instead:** Add `__all__` with intended public exports and import key classes/functions in `__init__.py` for cleaner imports.

### Monolithic `training_utils.py`

**What happens:** `src/experiments/training_utils.py` (357 lines) combines: character-level tokenizer, text dataset, binary dataset, multi-tier dataset, dataloader factory, tokenizer loading, batch collation, training loop, perplexity evaluation, and checkpoint save/load.

**Why it's wrong:** Single file with multiple responsibilities. Hard to test, hard to reuse individual components.

**Do this instead:** Split into `src/data/` (dataloaders, datasets, tokenizers) and `src/training/` (train loop, eval, checkpointing).

### Deep import chains with PYTHONPATH dependency

**What happens:** All imports use the form `from src.model.config import MicroGemmaFPConfig`. This requires the project root to be on PYTHONPATH, which is handled manually in `remote_python.sh` and `remote_run.sh` via `export PYTHONPATH=/home/.../Numerical_Analysis`.

**Why it's wrong:** Brittle -- scripts fail when run directly without PYTHONPATH set. The `remote_*.sh` scripts hardcode the remote path.

**Do this instead:** Use relative imports within the package, or install the package with `pip install -e .` and a `setup.py`/`pyproject.toml`.

## Error Handling

**Strategy:** Minimal explicit error handling. Numerical stability is handled via clamping (e.g., `clamp(min=1e-12)`) to avoid division by zero. Edge cases (singular matrices in GPTQ) get progressive damping.

**Patterns:**
- `clamp(min=1e-12)` for division safety in `fp_quantizer.py`, `adaptive_grid.py`, `condition.py`
- Progressive damping loops in `gptq.py:43-52` for Cholesky decomposition
- No custom exception types defined anywhere
- No try/except in training loop for NaN loss recovery
- FileNotFoundError handling only in `final_summary.py:43` for missing checkpoints

## Cross-Cutting Concerns

**Logging:** `print()` statements throughout. No `logging` module usage. No structured log format.

**Validation:** Python type hints used consistently but no runtime validation (no Pydantic, no dataclass validators). Users pass `_DEFAULT_CORPUS` from a separate file to the offline dataset.

**Authentication:** SSH password via `.sshpass` file (gitignored). No API keys or cloud credentials.

---

*Architecture analysis: 2026-05-17*
