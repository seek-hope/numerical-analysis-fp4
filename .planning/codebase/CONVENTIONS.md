# Coding Conventions

**Analysis Date:** 2026-05-17

## Naming Patterns

**Files:**
- `snake_case.py` for all Python module files — e.g., `fp_quantizer.py`, `training_utils.py`, `condition.py`
- Experiment scripts follow `verb_noun.py` pattern — e.g., `train_scaled_baseline.py`, `eval_quantization.py`, `validate_theory.py`
- No file prefixes like `test_` or suffixes like `_impl` observed
- `__init__.py` files are empty (zero content, used only for package marking)

**Functions:**
- `snake_case` for all functions — e.g., `estimate_condition_number()`, `compute_activation_hessian()`, `apply_weight_quantization()`
- Private/helper functions prefixed with underscore: `_build_fp_grid()`, `_round_to_nearest()`, `_detect_data_dir()`, `_replace_linears()`
- Module-level helper functions are common (not all logic lives in classes)

**Classes:**
- `PascalCase` for all classes — e.g., `FPQuantizer`, `MicroGemmaFPConfig`, `TransformerLayer`, `AdaptiveGridQuantizer`, `RMSNorm`
- Abbreviations in class names kept uppercase: `RMSNorm`, `MXFP4Quantizer`, `GPTQQuantizer`, `QATLinear`
- Inner classes (defined inside functions) follow same PascalCase: `QATLinear`, `QATOptimizedLinear`, `IdentityNorm`

**Variables:**
- `snake_case` for local variables and instance attributes — e.g., `hidden_states`, `input_ids`, `layer_idx`, `max_sigma`
- Short variable names accepted in mathematical/loop contexts: `W`, `H`, `L`, `x`, `q`, `w`, `m`, `h`
- Module-level constants in `UPPER_SNAKE_CASE`: `FP_FORMAT_SPECS`, `FP4_LEVELS`, `FP8_E4M3_LEVELS`, `FP4_E2M1_GRID`, `NF4_GRID`

**Types:**
- `PascalCase` for type names (standard Python)
- Union types use Python 3.10+ `X | Y` syntax: `scale: torch.Tensor | None = None`, `max_samples: int | None = None`
- Return type hints present on most functions, but some top-level experiment script functions lack them

## Code Style

**Formatting:**
- No formatting tool config detected — no `.prettierrc`, `pyproject.toml`, `ruff.toml`, or `setup.cfg`
- Code uses 4-space indentation (PEP 8 standard)
- Line length appears to be ~100-120 characters (wider than PEP 8's 79)
- Consistent blank line around class definitions and function definitions
- No trailing whitespace observed
- **Recommendation:** Add `ruff` or `black` configuration to enforce consistent formatting

**Linting:**
- No linting configuration detected — no `pyproject.toml`, `.flake8`, `.pylintrc`, or `ruff.toml`
- **Recommendation:** Add `ruff` for linting and formatting in CI

**Section Separators:**
- Heavy use of visual comment separators throughout:
```python
# ═══════════════════════════════════════════════════════════════
# Section Title
# ═══════════════════════════════════════════════════════════════
```
This pattern is used in `transformer.py`, `fp_quantizer.py`, `training_utils.py`, `gptq.py`, and others.

## Import Organization

**Order:**
1. Standard library modules (stdlib)
2. Third-party libraries
3. Local application modules (from `src.`)

Each group is separated by a blank line. Within groups, imports are usually alphabetical.

**Examples from `training_utils.py`:**
```python
import random, os, math
import torch
import numpy as np
from torch.utils.data import IterableDataset, Dataset, DataLoader
```

**Examples from `transformer.py`:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.config import MicroGemmaFPConfig
```

**Path Aliases:**
- All intra-project imports use absolute `src.` package paths: `from src.model.config import MicroGemmaFPConfig`
- No relative imports (`from .config import ...`) observed
- No import aliases or path rewrites configured

**Import grouping inconsistencies:**
- Some files use `import os` inline inside functions (late imports) rather than at module top — e.g., `import os` at line 58 of `train_scaled_baseline.py`, line 64 of `train_fp16_baseline.py`, line 161 of `phase2_comparison.py`
- Standard library modules are sometimes comma-imported on one line: `import random, os, math`

## Error Handling

**Patterns:**
- `ValueError` with descriptive message in constructors for invalid config: `raise ValueError(f"Unknown format: {fmt}. Options: {list(FP_FORMAT_SPECS.keys())}")`
- Try/except for graceful degradation in resource loading:
  - `training_utils.py` line 46-49: fallback to CharTokenizer if file not found
  - `condition.py` line 43-51: Cholesky decomposition failure with progressive damping
  - `phase2_comparison.py` line 154: broad `except Exception as e` around PTQ method application
- No custom exception classes defined anywhere

**What's missing:**
- No structured logging of errors (uses `print()` not `logging`)
- Some broad `except:` or `except Exception:` without specific error types
- No error recovery in long-running training loops (a step failure kills the whole process)
- Many functions silently return default values on failure rather than raising

**Best practice from this codebase:**
```python
if fmt not in FP_FORMAT_SPECS and fmt != 'fp8_e4m3fn':
    raise ValueError(f"Unknown format: {fmt}. Options: {list(FP_FORMAT_SPECS.keys())}")
```

## Logging

**Framework:** None. All logging uses `print()` statements directly.

**Patterns:**
- Status messages at key milestones: `print(f"Device: {device}")`, `print(f"Parameters: {stats['total']:,} total")`
- Warnings prefixed with `[WARN]`: `print(f"[WARN] Tokenizer not found at {tokenizer_path}, ...")`
- Diagnostic output in analysis scripts printed during computation
- No `logging` module, no log levels, no structured logging
- No wandb logging in training scripts (wandb in requirements but not imported in any training script)

**Recommendation:** Use `logging` module for production code. Current `print()` approach is acceptable for research scripts.

## Comments

**When to Comment:**
- Module-level docstrings explain purpose, usage, and theory — e.g., `gptq.py` has 16 lines of docstring with mathematical explanation
- Inline comments explain mathematical derivations, algorithm steps, and design decisions
- Section headers with box separators for visual grouping

**JSDoc/TSDoc:**
- Google-style docstrings with `Args:`, `Returns:`, `Raises:` sections are used in many public functions
- Not all functions have them — there is inconsistency
- Class docstrings describe purpose and usage pattern (including code examples in some)

**Example of thorough docstring:**
```python
def gptq_quantize_weight(
    W: torch.Tensor,
    H: torch.Tensor,
    quantizer,
    blocksize: int = 128,
    stochastic: bool = False,
    use_per_channel: bool = True,
) -> torch.Tensor:
    """Quantize weight matrix W with GPTQ error compensation.

    Applies per-channel normalization first (row-wise dynamic range),
    then runs GPTQ column compensation with per-tensor quantization on
    the normalized matrix, then undoes the normalization. This gives
    both per-channel dynamic range AND column-wise error correction
    without the two mechanisms interfering.

    Args:
        W: weight matrix (out_features, in_features)
        H: activation Hessian (in_features, in_features) = X^T X
        quantizer: FPQuantizer instance (will be used per-tensor internally)
        blocksize: number of columns to process at once (GPU efficiency)
        stochastic: use stochastic rounding
        use_per_channel: apply per-channel normalization around GPTQ

    Returns:
        W_q: quantized weight with compensation applied
    """
```

## Function Design

**Size:** Functions range from 1-line utilities (`is_power_of_two`) to 80-line training loops (`train_epoch`). The median function is ~30 lines. Functions over 80 lines are rare and broken into helpers.

**Parameters:** Most functions take 2-5 parameters. Training-related functions accept more (up to 8+ parameters with defaults). No dataclass parameter objects are used to group parameters.

**Return Values:**
- Functions return single values or dicts (never tuples with more than 2 elements)
- Pure computation functions return scalars or tensors
- Training/analysis functions return `list[dict]` (per-step/per-layer metrics)
- Model forward returns `dict` always: `{'loss': ..., 'logits': ...}`

## Module Design

**Exports:**
- No explicit `__all__` in any module
- `__init__.py` files are entirely empty — no re-exports
- Consumers import from concrete module paths: `from src.quantization.fp_quantizer import FPQuantizer`

**Barrel Files:** Not used. Each module is imported directly by full path.

**Module size:**
- Core logic modules (`fp_quantizer.py`, `training_utils.py`, `transformer.py`): 200-270 lines
- Analysis modules (`condition.py`, `lipschitz.py`, `sensitivity.py`): 50-170 lines
- Experiment scripts: 80-250 lines
- `smoke_gemma4.py`: ~20 lines

## Common Patterns

**Device handling:**
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```
This one-liner appears at the top of every experiment script's `main()`.

**Shebang + main guard:**
All experiment scripts use:
```python
#!/usr/bin/env python3
# ... docstring ...

def main():
    ...

if __name__ == '__main__':
    main()
```

**`with torch.no_grad()`:** Used extensively for PTQ evaluation and model modification:
```python
@torch.no_grad()
def ptq_simple(model, quantizer):
    ...
```
Both decorator form and context manager form are used.

**Model wrapping pattern:** Multiple QAT implementations follow the same recursive pattern:
```python
def _replace(module, ...):
    for name, child in list(module.named_children()):
        full = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear):
            if 'embed' in full.lower() or 'lm_head' in full.lower():
                continue
            setattr(module, name, WrappedLinear(child, ...))
        else:
            _replace(child, ...)
```

**Config usage:** `MicroGemmaFPConfig` is a `@dataclass` instantiated at the start of each experiment script. All model construction derives from it.

---

*Convention analysis: 2026-05-17*
