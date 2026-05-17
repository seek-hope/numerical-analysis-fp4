---
phase: 02-core-measurement-protocol
reviewed: 2026-05-17T13:00:00Z
depth: standard
files_reviewed: 3
files_reviewed_list:
  - src/analysis/error_propagation.py
  - src/experiments/measure_qerror.py
  - tests/test_error_propagation_tracker.py
findings:
  critical: 3
  warning: 4
  info: 3
  total: 10
status: issues_found
---

# Phase 02: Code Review Report

**Reviewed:** 2026-05-17T13:00:00Z
**Depth:** standard
**Files Reviewed:** 3
**Status:** issues_found

## Summary

Three files were reviewed: the `ErrorPropagationTracker` hook-based activation capture class, its downstream consumer `measure_qerror.py`, and the test suite `test_error_propagation_tracker.py`. The `tests/__init__.py` is empty and excluded from findings.

The core finding is that the `validate_null_measurement()` method is logically vacuous -- it computes `||Wx - Wx||/||Wx||`, which is always near-zero regardless of pipeline state -- and the corresponding tests (`test_raises_value_error_on_large_error` and `test_identifies_violating_modules`) will fail at runtime because they expect a ValueError that can never be raised. Additionally, `detach()` prints "Removed 0 hooks" because it clears the handle list before the print statement. Multiple quality issues include dead code (`_activation_keys`, `MockQuantizerWithError`) and misleading documentation.

## Critical Issues

### CR-01: `validate_null_measurement()` is logically vacuous -- cannot detect pipeline errors

**File:** `src/analysis/error_propagation.py:351-353`
**Issue:** The method computes `||y_q - y_fp|| / ||y_fp||` where `y_fp = x @ W_fp.T` and `y_q = x @ W_fp.T` use exactly the same weight matrix and the same input. The expressions for `y_fp` and `y_q` are byte-for-byte identical. This means the numerator is always zero (or floating-point noise), and the method can never detect corruption in the hook-based measurement pipeline. Despite its name and docstring, it does not validate anything.

The method is used as a critical validation gate in `measure_qerror.py:119-124`:
```python
try:
    max_null_err = tracker.validate_null_measurement(model)
    print(f"  Null measurement: max error = {max_null_err:.2e} -- PASS")
except ValueError as e:
    print(f"  Null measurement: FAILED -- {e}")
    sys.exit(1)
```

Since `validate_null_measurement` always returns near-zero without raising, this gate never triggers, giving false confidence in corrupted measurement pipelines (e.g., missing activations, wrong hook registration, device transfer errors).

**Fix:** Change `validate_null_measurement` to actually validate the pipeline. One approach: run the model forward pass twice with the tracker attached, detach, and compare the two independently captured activation sets for consistency (e.g., `||x_1 - x_2|| / ||x_1||` for each module). As written:

```python
@torch.no_grad()
def validate_null_measurement(self, model: nn.Module) -> float:
    """Compare two independent forward passes to validate measurement consistency.

    Runs a second forward pass through the already-registered hooks and
    compares the newly captured activations against the originals.

    Returns:
        float: Maximum relative difference across all modules.
    Raises:
        ValueError: If any module's relative difference exceeds 1e-5.
    """
    # Capture baseline
    baseline = {k: v.clone() for k, v in self._activations.items()}

    # Run second forward pass (hooks are still attached)
    # NOTE: caller must ensure model can be run again safely (eval mode, no side effects)
    ...

    max_err = 0.0
    violations = []
    for module_path, x_orig in baseline.items():
        if module_path not in self._activations:
            violations.append((module_path, float('inf')))
            continue
        x_new = self._activations[module_path]
        denom = x_orig.norm().clamp(min=1e-12)
        err = (x_new - x_orig).norm() / denom
        ...
```

### CR-02: `test_raises_value_error_on_large_error` will fail -- expects impossible ValueError

**File:** `tests/test_error_propagation_tracker.py:763-781`
**Issue:** This test corrupts a captured activation by multiplying it by 1e6, then expects `validate_null_measurement` to raise `ValueError`. However, as described in CR-01, `validate_null_measurement` computes `y_fp = x @ W_fp.T` and `y_q = x @ W_fp.T` from the _same corrupted `x`_, so the error is zero regardless of activation corruption. The test will always fail with `Failed: DID NOT RAISE <class 'ValueError'>`.

Similarly, `test_identifies_violating_modules` at line 783-800 has the same defect and will also always fail.

**Fix:** These tests should either (a) be removed and replaced with a meaningful validation test as described in CR-01, or (b) test the fixed `validate_null_measurement` that actually validates pipeline consistency. A temporary fix that preserves test semantics could change the test to detect a scenario where the model weights themselves differ between reference and quantized computation:

```python
def test_raises_value_error_on_large_error(self):
    model = nn.Sequential(nn.Linear(4, 8))
    tracker = ErrorPropagationTracker()
    tracker.attach(model)
    x = torch.randn(2, 4)
    model(x)

    # Simulate pipeline corruption by zeroing out an activation entry
    # (a real pipeline issue like a wrong hook path)
    tracker._activations['0'] = torch.zeros_like(tracker._activations['0'])

    with pytest.raises(ValueError) as excinfo:
        tracker.validate_null_measurement(model)
    assert 'WARN' in str(excinfo.value)
```

### CR-03: `detach()` prints "Removed 0 hooks" due to ordering bug

**File:** `src/analysis/error_propagation.py:266-269`
**Issue:** The method clears `self._hook_handles` on line 268 BEFORE printing the count on line 269. Since `len(self._hook_handles)` is always 0 at the print statement, the output message is `[Tracker] Removed 0 hooks` regardless of how many hooks were actually removed. This is confusing during debugging.

**Fix:** Print the count before clearing:

```python
def detach(self):
    count = len(self._hook_handles)
    for handle in self._hook_handles:
        handle.remove()
    self._hook_handles.clear()
    print(f"[Tracker] Removed {count} hooks")
    return self
```

## Warnings

### WR-01: `_activation_keys` is populated but never read

**File:** `src/analysis/error_propagation.py:42, 76`
**Issue:** `self._activation_keys` is created as `list[str]` in `__init__` and appended to in `_register_linear_pre_hooks` on every `nn.Linear` module encountered. However, the list is never read or used anywhere in the class. For a 164M model with ~100 linear layers, this wastes trivial memory, but more importantly, it is dead code that creates a maintenance burden -- if someone later adds activation key processing, they might not realize this list exists or might use it in a way inconsistent with its original intent.

**Fix:** Remove `_activation_keys` and all references to it unless a concrete use case exists:

```python
def __init__(self):
    self._activations: dict[str, torch.Tensor] = {}
    self._p_points: dict[str, torch.Tensor] = {}
    self._g_points: dict[str, torch.Tensor] = {}
    self._hook_handles: list[torch.utils.hooks.RemovableHandle] = []
```

And in `_register_linear_pre_hooks`:
```python
def _register_linear_pre_hooks(self, model: nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            handle = module.register_forward_pre_hook(
                self._make_input_hook(name)
            )
            self._hook_handles.append(handle)
```

### WR-02: `MockQuantizerWithError` defined but never used in tests

**File:** `tests/test_error_propagation_tracker.py:589-598`
**Issue:** `MockQuantizerWithError` is a fully implemented test helper class that simulates a non-identity quantizer by adding Gaussian noise proportional to `W.abs().mean()`. It is never instantiated or referenced by any test method in the file. This is dead test code that will rot over time as the codebase evolves.

**Fix:** Either (a) write a test using `MockQuantizerWithError` to verify that `compute_output_error` returns non-zero errors for non-identity quantizers, or (b) remove the class entirely.

```python
def test_noise_quantizer_produces_positive_errors(self):
    """Verify compute_output_error returns >0 for a noise-based quantizer."""
    model = nn.Sequential(nn.Linear(4, 8))
    tracker = ErrorPropagationTracker()
    tracker.attach(model)
    x = torch.randn(2, 4)
    model(x)
    quantizer = MockQuantizerWithError(noise_scale=0.5)
    results = tracker.compute_output_error(model, quantizer)
    for module_path, err in results.items():
        assert err > 0, f"{module_path}: {err} (expected >0)"
```

### WR-03: `p_points` property docs claim "copy, not reference" but tensor references are shared

**File:** `src/analysis/error_propagation.py:253-255`
**Issue:** The property docstring says "Return merged P-point and G-point dict (copy, not reference)." While the dict container itself is a new object (`{**self._p_points, **self._g_points}` shallow copy), the tensor values inside are the same references as the internal dicts. Mutations to tensors obtained from the returned dict (e.g., `tracker.p_points['0_P0'].mul_(0)`) would corrupt the tracker's internal state. The documentation gives a false sense of safety.

**Fix:** Update the docstring to be accurate:

```python
@property
def p_points(self) -> dict[str, torch.Tensor]:
    """Return merged P-point and G-point dict (shallow copy of the dict;
    tensor references are shared with internal state -- do not mutate them)."""
    return {**self._p_points, **self._g_points}
```

Or provide true isolation if needed:
```python
@property
def p_points(self) -> dict[str, torch.Tensor]:
    return {k: v.clone() for k, v in {**self._p_points, **self._g_points}.items()}
```

### WR-04: `_g_hook` fallback `torch.as_tensor(output)` is fragile for unexpected output types

**File:** `src/analysis/error_propagation.py:247`
**Issue:** The G-point hook's fallback branch attempts `torch.as_tensor(output)` when `output` is neither a `torch.Tensor` nor a tuple. If a registered module returns a dict, a namedtuple, a custom dataclass, or `None`, this call could raise `TypeError` or silently produce an unexpected tensor (e.g., from an object's `__array__` interface that doesn't represent the intended output). The `@torch.no_grad()` decorator prevents error traceback attribution since this runs inside the forward pass.

**Fix:** Raise an explicit informative error instead of a silent fallback:

```python
def _g_hook(self, module, input_args, output):
    if isinstance(output, tuple):
        tensor = output[0].detach().clone().cpu()
    elif isinstance(output, torch.Tensor):
        tensor = output.detach().clone().cpu()
    else:
        raise TypeError(
            f"G-point hook for {self._g_points_key}: "
            f"unexpected output type {type(output).__name__}. "
            f"Expected torch.Tensor or tuple of torch.Tensor."
        )
    self._g_points[self._g_points_key] = tensor
```

## Info

### IN-01: `test_pre_hook_captured_tensor_is_a_clone` does not verify independent memory

**File:** `tests/test_error_propagation_tracker.py:183-197`
**Issue:** The test claims to verify the captured tensor is "a clone" (i.e., a separate memory buffer), but its assertion (`torch.allclose(captured, tracker._activations['0'])`) only checks value equality. A value comparison would pass even if `captured` is the same tensor reference as `_activations['0']`, which would violate the contract that the pre-hook returns a cloned tensor. The test provides false confidence in the clone guarantee.

**Fix:** Add a memory address comparison:

```python
def test_pre_hook_captured_tensor_is_a_clone(self):
    ...
    captured = tracker._activations['0']
    x_copy = torch.randn(2, 4)
    model(x_copy)
    new_captured = tracker._activations['0']
    assert tracked.data_ptr() != new_captured.data_ptr(), \
        "Expected different memory addresses (clone), got same"
```

### IN-02: `test_p_point_keys_follow_format` does not check P4 and P5 despite docstring claim

**File:** `tests/test_error_propagation_tracker.py:392-407`
**Issue:** The test docstring says "Check that P0-P5 keys exist (P3 and P6 computed separately)" but the actual assertions only check P0, P1, P2. P4 (post-attention-norm output) and P5 (FFN output) are registered by `_register_p_point_hooks` and would be captured, but the test never verifies they are present.

**Fix:** Add assertions for P4 and P5:

```python
assert f"{layer_idx}_P4" in tracker._p_points
assert f"{layer_idx}_P5" in tracker._p_points
```

### IN-03: `test_works_with_nested_model_structure` has no meaningful assertions

**File:** `tests/test_error_propagation_tracker.py:710-727`
**Issue:** This test calls `compute_output_error` on a nested model but only asserts `isinstance(results, dict)`. It does not verify the number of results, their keys, their values, or even that the dict is non-empty. This gives almost no confidence that the method works correctly with nested module structures.

**Fix:** Add concrete assertions:

```python
def test_works_with_nested_model_structure(self):
    model = MockModel(n_layers=1)
    tracker = ErrorPropagationTracker()
    tracker.attach(model)
    x = torch.randn(2, 8)
    model(x)
    quantizer = MockQuantizer()
    results = tracker.compute_output_error(model, quantizer)

    assert isinstance(results, dict)
    assert len(results) > 0, "Expected at least one module result"
    # MockModel.lm_head is nn.Sequential(nn.Linear(8, 16)) -> path is 'lm_head.0'
    assert 'lm_head.0' in results, "Expected lm_head.0 in results"
    for v in results.values():
        assert isinstance(v, float)
        assert 0.0 <= v < 1e-5, f"Identity quantizer error too large: {v}"
```

---

_Reviewed: 2026-05-17T13:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
