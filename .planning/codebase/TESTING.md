# Testing Patterns

**Analysis Date:** 2026-05-17

## Test Framework

**Runner:** Not detected. No test runner is configured in this project.

**Test Dependencies:**
- `requirements.txt` contains no testing-related packages (no pytest, unittest, hypothesis, etc.)
- No `pyproject.toml`, `setup.cfg`, `pytest.ini`, or `tox.ini` exists with test configuration

**Run Commands:** No test commands are defined or documented.

## Test File Organization

**Location:** No test files or test directories exist in the codebase.

- No `tests/` directory at project root
- No `test_*.py` files anywhere in the repository
- No `*_test.py` files anywhere in the repository
- No co-located test modules within `src/` packages

**Naming:** Not applicable — no tests exist.

## Test Structure

No tests are present. Each module file contains only production code or experiment entry points.

**Note on experiment scripts as validation:** Several experiment scripts function as one-off validation experiments but are NOT structured as tests:

- `/home/rimuru/Projects/Code/homework/Numerical_Analysis/proj/src/experiments/validate_theory.py` — Validates P1/P2/P3 theoretical predictions of quantization error. Contains a `pearson_r()` correlation function and runs PPL comparisons.
- `/home/rimuru/Projects/Code/homework/Numerical_Analysis/proj/src/experiments/validate_rmsnorm.py` — Validates RMSNorm error-blocking effect theory. Contains ablation experiments.
- `/home/rimuru/Projects/Code/homework/Numerical_Analysis/proj/src/experiments/eval_all.py` — Runs PPL evaluation across all trained checkpoints.

These scripts do NOT use assertions, do NOT fail on unexpected results, and are NOT runnable in CI. They are research validation scripts, not tests.

## Mocking

**Framework:** Not applicable — no tests use mocking.

**What would benefit from mocking:**
- `FPQuantizer.quantize()` calls could be mocked to test downstream code without actual quantization
- `MicroGemmaFPForCausalLM.forward()` could be mocked in integration tests to avoid full model instantiation
- Data loading (`get_dataloader`, `BinDataset`) could use mock data sources

## Fixtures and Factories

**Test Data:** Not applicable — no test fixtures exist.

**Production data loading that could serve as test fixture reference:**
- `/home/rimuru/Projects/Code/homework/Numerical_Analysis/proj/src/experiments/training_utils.py` — `LocalTextDataset` class provides an embedded text corpus (from `large_corpus.py`) that could be used as a deterministic test data source
- `CharTokenizer` in `training_utils.py` provides a deterministic, file-free tokenizer suitable for testing
- `MicroGemmaFPConfig()` with defaults creates a deterministic model config

**Existing patterns useful for generating test data:**
```python
# Small model config for fast tests (from CLAUDE.md convention)
# Use tiny model configs instead of launching the full 164M model
```

## Coverage

**Requirements:** Not enforced. No coverage tool is configured.

**View Coverage:** No coverage command is available.

## Test Types

**Unit Tests:** None exist.

**Integration Tests:** None exist.

**E2E Tests:** None exist.

**Smoke Tests:**
- `tests/smoke_gemma4.py` — Loads a Gemma 4 E2B model and prints parameter count. This is the closest thing to a smoke test but is not run in CI and has no assertions.

## Current Gaps

**Critical gaps identified:**

1. **Zero test coverage.** The entire codebase has no automated tests. This is a significant risk for a research codebase where numerical correctness is paramount.

2. **No CI pipeline.** No CI configuration (GitHub Actions, GitLab CI, etc.) exists to run any validation.

3. **Validation done manually.** The `validate_*.py` scripts are manually inspected and interpreted, not asserted against thresholds.

4. **No regression detection.** Changes to quantization logic, training loops, or model architecture cannot be automatically checked for regression.

## Recommendations

**High-priority test targets:**
- `FPQuantizer.quantize()` — test deterministic rounding, stochastic rounding, per-channel vs per-tensor
- `_build_fp_grid()` — verify grid values match expected FP format specifications
- `compute_activation_hessian()` — test Hessian computation and Cholesky decomposition
- `estimate_condition_number()` — verify on known matrices (e.g., identity, diagonal)
- `lloyd_max_grid()` — test convergence properties, symmetry of output grid
- `collate_batch()` — test padding behavior with variable-length sequences

**Suggested test setup:**
```bash
# Add to requirements.txt:
pytest>=8.0.0
pytest-cov>=5.0.0

# Create tests/ directory:
tests/
├── test_fp_quantizer.py
├── test_condition.py
├── test_adaptive_grid.py
├── test_training_utils.py
└── conftest.py     # Shared fixtures (small config, dummy model)
```

**Conftest fixture suggestion:**
```python
# tests/conftest.py
import pytest
import torch
from src.model.config import MicroGemmaFPConfig
from src.model.transformer import MicroGemmaFPForCausalLM

@pytest.fixture
def small_config():
    """Tiny config for fast test instantiation."""
    return MicroGemmaFPConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        vocab_size=128,
    )

@pytest.fixture
def small_model(small_config):
    return MicroGemmaFPForCausalLM(small_config)
```

---

*Testing analysis: 2026-05-17*
