# Codebase Concerns

**Analysis Date:** 2026-05-17

## Tech Debt

### DuQuant++ Rotation Marked as Buggy, Never Validated

- Issue: The DuQuant++ rotation pipeline (`src/quantization/outlier_rotation.py`, `src/quantization/hadamard.py`) was implemented but the corresponding benchmark in `src/experiments/fp4_ptq_compare.py` explicitly skips both "FP4 E2M1 + DuQuant++" and "MXFP4 + DuQuant++" with the comment `"(skip -- rotation is buggy)"` (lines 126, 134). The skewing/scaling step is active but rotation is not tested, leaving half the DuQuant++ pipeline unvalidated.
- Files: `src/experiments/fp4_ptq_compare.py` (lines 126-135), `src/quantization/outlier_rotation.py`, `src/quantization/hadamard.py`
- Impact: The block-Hadamard rotation code is dead code -- it was written, never validated, and explicitly skipped in the one benchmark that uses it. Future work cannot trust the rotation code without a full re-validation.
- Fix approach: Either (a) repair rotation (likely a sign/scale bug in `block_hadamard_transform`) and re-run the benchmark, or (b) remove the dead code and document the gap.

### `inverse_power_iteration` Does Not Compute Minimum Singular Value

- Issue: `src/analysis/condition.py` lines 28-44 defines `inverse_power_iteration()` but the implementation does NOT solve the inverse iteration problem. It simply computes `W @ v` norm on a random vector and returns it directly (line 42-43). This gives a random projection of the vector onto the row space, not the minimum singular value. The docstring claims to estimate sigma_min via inverse power iteration but the actual operation is a single forward multiplication with no iterative refinement or shifted inverse solve.
- Files: `src/analysis/condition.py` (lines 28-44)
- Impact: All condition number estimates using `estimate_condition_number()` produce incorrect values because sigma_min is overestimated by orders of magnitude. This means sensitivity analysis, mixed-precision layer selection, and adaptive grid kappa-weighting are all using unreliable condition numbers. The correlation claims in the theory validation experiments (P1, P3) may be affected.
- Fix approach: Implement actual inverse power iteration (solve `(W^T W - sigma_max^2 I) u = v` iteratively), or use `torch.linalg.svdvals()` for small matrices, or at minimum rename the function and document its actual behavior.

### Vanilla QAT-FP4 Training Shows Severe Overfitting

- Issue: The `train_qat_fp4_opt.py` docstring (line 8) explicitly documents that vanilla QAT-FP4 achieved train PPL 1.01 but eval PPL 13.86 -- severe overfitting. The script implements three mitigation strategies (stochastic rounding, adaptive precision, combined) but there is no analysis of which strategy actually closes the train-eval gap. The `train_qat.py` script (vanilla QAT) likely has the same issue.
- Files: `src/experiments/train_qat_fp4_opt.py` (lines 6-16), `src/experiments/train_qat.py`
- Impact: QAT experiments do not establish a reliable baseline. The overfitting suggests either a bug in the QAT wrapping (the STE backward pass through `QuantizedLinear` in `fp_quantizer.py` does not backprop through the quantization itself), insufficient data, or a learning rate mismatch. Until this is understood, all QAT comparison claims are suspect.
- Fix approach: Diagnose the train-eval gap systematically. Check whether the STE in `QuantizedLinear.backward()` correctly passes gradients (it currently returns `grad_x = grad_output @ weight` using the full-precision weight, not the quantized weight).

### Duplicate Data Preparation Scripts

- Issue: Two data preparation scripts exist with overlapping functionality: `src/experiments/prepare_data.py` and `src/experiments/prepare_data_chunked.py`. Both tokenize and create `.bin` shards, but with different chunking strategies. The purpose and differences between them are not documented.
- Files: `src/experiments/prepare_data.py`, `src/experiments/prepare_data_chunked.py`
- Impact: Confusion about which script to use. Changes may need to be made in both places. One may have edge-case fixes the other lacks.
- Fix approach: Consolidate into a single script with options for different chunking strategies, or remove the dead one.

### Hardcoded Checkpoint Paths Throughout Experiment Scripts

- Issue: Experiment scripts hardcode checkpoint paths as default argument values, creating implicit coupling between scripts. For example `src/experiments/phase2_comparison.py` lines 87-88 hardcode `checkpoints/scaled_fp16_baseline/model.pt` and `checkpoints/cond_regularized/model.pt`. Similarly, `src/experiments/validate_theory.py` (line 47), `src/experiments/train_cond_regularized.py` (line 60), `src/experiments/eval_quantization.py` (line 96), and `src/experiments/eval_all.py` (lines 12-17) all have hardcoded paths.
- Files: `src/experiments/phase2_comparison.py` (lines 87-88), `src/experiments/validate_theory.py` (line 47), `src/experiments/train_cond_regularized.py` (line 60), `src/experiments/eval_quantization.py` (line 96), `src/experiments/eval_all.py` (lines 12-17), `src/experiments/eval_all_grids.py` (lines 12-19), `src/experiments/eval_fp4_qat.py` (lines 11-17), `src/experiments/final_summary.py` (lines 14-19)
- Impact: Brittle pipeline. Renaming a checkpoint directory or changing the output structure breaks dependent scripts silently.
- Fix approach: Centralize checkpoint paths in a config file or environment variable. At minimum, ensure all paths are CLI-argument overridable.

## Known Bugs

### Zero MSE Computation in fp4_ptq_compare.py

- Issue: `src/experiments/fp4_ptq_compare.py` lines 41 and 44 compute MSE as `((param.data - param.data) ** 2).mean().item()`. This subtracts the tensor from itself, producing exactly 0.0 for both `mse_before` and `mse_after`. The `mse_after` line uses `param.data` after quantization, so the intent was clearly to compute `(W_q - W_fp)^2`, but `param.data` has already been overwritten by line 42 (`param.data = quantizer_fn(param.data)`).
- Files: `src/experiments/fp4_ptq_compare.py` (lines 41, 44)
- Trigger: Passing `--verbose` flag (via the `verbose=True` parameter path, though the script's argparse does not expose it).
- Workaround: The verbose parameter is never enabled from the CLI, so this bug is latent. However, if someone reuses the function with verbosity, they will get meaningless MSE=0.0 values.
- Fix: Save a clone of param.data before overwriting, then compute `mse_before = ((param.data - W_fp.clone()) ** 2)...` and `mse_after = ((quantizer_fn(param.data) - W_fp) ** 2)...`.

### CharTokenizer Allows Control Characters Through

- Issue: `src/experiments/training_utils.py` lines 46-50 in `CharTokenizer.encode()` checks `if 3 <= val < min(self.vocab_size, 256)` for each character. Since `vocab_size` is 32000 by default, this allows ALL bytes 3-255 through as valid token IDs, including control characters (0x00-0x1F except 0x00-0x02). This creates token IDs that the model was never trained on.
- Files: `src/experiments/training_utils.py` (lines 46-50)
- Trigger: When the offline fallback path is used (character-level tokenizer instead of BPE).
- Impact: Training on offline data produces unnatural token distributions at the character level. The BPE tokenizer path is unaffected.
- Fix: Strictly map non-printable characters to `self.UNK` (2).

## Security Considerations

### Plain-Text SSH Password on Disk

- Risk: The file `.sshpass` contains the remote server SSH password (`bi_course2026`) in plain text. The `.gitignore` excludes it from version control, but it remains on disk for anyone with local access. The shell scripts `sync.sh`, `remote_python.sh`, and `remote_run.sh` all reference this file via `sshpass -f "$SSHPASS_FILE" ssh`.
- Files: `.sshpass`, `sync.sh` (line 23), `remote_python.sh` (line 20), `remote_run.sh` (line 17)
- Current mitigation: `.gitignore` prevents accidental commit. The file is permission-restricted by the OS.
- Recommendations: Use SSH key-based authentication instead of password-based sshpass. This eliminates the plain-text credential entirely. As a lesser mitigation, restrict file permissions (`chmod 600 .sshpass`).

### `weights_only=False` in Checkpoint Loading

- Risk: `src/experiments/training_utils.py` line 352 calls `torch.load(path, map_location=device, weights_only=False)`. With `weights_only=False`, PyTorch pickles can execute arbitrary code during deserialization. A malicious or corrupted checkpoint file could execute arbitrary Python code.
- Files: `src/experiments/training_utils.py` (line 352)
- Current mitigation: Checkpoints are loaded only from local paths specified by the user via CLI arguments.
- Recommendations: Use `weights_only=True` when checkpoint loading is needed only for the state dict. If optimizer states are needed, load them separately with a safetensors-based approach or at minimum validate the checkpoint source.

## Performance Bottlenecks

### MXFP4 Quantization Uses Python-Level Block Iteration

- Problem: `src/quantization/fp4_grids.py` lines 121-138 in `MXFP4Quantizer.quantize()` iterates over blocks using a Python `for` loop over `range(0, n, self.block_size)`. Each iteration calls `block.abs().max().item()`, `math.log2()`, `math.ceil()` and creates a new grid tensor, all at Python interpreter speed. For a 768x3072 weight matrix with block_size=32, this is ~73,728 blocks, each with Python overhead and individual CUDA kernel launches.
- Files: `src/quantization/fp4_grids.py` (lines 121-138)
- Cause: The block-scaling logic is implemented as a Python loop over blocks rather than a vectorized tensor operation.
- Improvement path: Vectorize the block scaling: reshape the tensor to `(n // block_size, block_size)`, compute per-block max in a single kernel, compute all scales vectorized, then quantize the whole tensor in one pass.

### Python-Level Hadamard Loops

- Problem: Both `src/quantization/hadamard.py` (lines 43-49) and `src/quantization/outlier_rotation.py` (lines 126-131) implement the Walsh-Hadamard transform using nested Python loops (`while h < n: for i in range(0, n, h*2): for j in range(i, i+h): ...`). For sequences of length 768 or 3072, this triggers many small CUDA kernel launches.
- Files: `src/quantization/hadamard.py` (lines 43-49), `src/quantization/outlier_rotation.py` (lines 126-131)
- Cause: Hand-written butterfly iteration instead of vectorized matrix multiplication with a pre-computed Hadamard matrix.
- Improvement path: Pre-compute the Hadamard matrix H of appropriate size as a buffer, then use `x @ H` / `H @ x` for batched transforms. For large dimensions, the recursive butterfly approach with fewer but larger chunked operations.

### Condition Number Analysis on Full Weight Matrices Repeatedly

- Problem: `condition_number_regularization()` in `src/analysis/condition.py` (lines 99-133) runs power iteration for every weight matrix at every training step. For the 164M model with ~400 linear layers (each 2D), running 3-5 power iterations per layer per step is significant overhead, especially since the surrogate (lines 69-90) requires `Wf.T @ (Wf @ v)` per iteration.
- Files: `src/analysis/condition.py` (lines 69-90, 99-133)
- Cause: Computing condition numbers for all layers at every training step, even those with tiny lambda_cond values.
- Improvement path: Add a configurable frequency (e.g., compute every N steps), or sample a subset of layers each step, or use the SVD-based exact computation only for validation runs and a cheaper Hessian-trace proxy for training.

## Fragile Areas

### `train_gemma4.py` -- Duplicated Implementation

- Files: `src/experiments/train_gemma4.py`
- Why fragile: This is a completely separate implementation of the model, data pipeline, quantization, and training loop, built on top of the Transformers library's `AutoModelForCausalLM` rather than the project's `MicroGemmaFPForCausalLM`. It has its own `BinDataset`, `FPQuant` class, and LoRA setup. Changes to the main codebase's data loading, quantization, or training patterns must also be manually replicated here. The file is 252 lines of essentially parallel code.
- Safe modification: Only modify `train_gemma4.py` in isolation. Do not assume it shares fixes or improvements with the main `src/model/`, `src/quantization/`, or `src/experiments/training_utils.py`.
- Test coverage: None. No validation that this script produces compatible results with the main codebase.

### GPTQ Activation Collection -- Hardcoded 50 Step Limit

- Files: `src/quantization/gptq.py` (line 222)
- Why fragile: `max_steps = 50` is hardcoded inside `_collect_activations()` method body. If the calibration DataLoader yields fewer than 50 unique batches (e.g., small calibration set), the Hessian estimate degrades silently. If the batch size is very small, 50 steps may not provide enough tokens for a full-rank Hessian.
- Safe modification: Make `max_steps` a constructor parameter of `GPTQQuantizer` with a default of 50.

### CUDA Hardcoded in fp4_ptq_compare.py

- Files: `src/experiments/fp4_ptq_compare.py` (line 64)
- Why fragile: `device = 'cuda'` is hardcoded rather than using the pattern `torch.device('cuda' if torch.cuda.is_available() else 'cpu')` used consistently everywhere else. Running this script on a CPU-only machine will crash immediately.
- Safe modification: Change line 64 to `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`.

### Broad Exception Handling in Evaluation Scripts

- Files: `src/experiments/phase2_comparison.py` (line 154), `src/experiments/eval_qat_checkpoints.py` (line 28), `src/experiments/train_tokenizer.py` (lines 42, 58)
- Why fragile: Multiple scripts use bare `except Exception as e:` which catches and silently swallows all errors (including bugs, OOM, shape mismatches). Experiment phase2_comparison.py catches all exceptions in the comparison loop and sets PPL to None, hiding failures behind a "FAIL" label with no traceback.
- Test coverage: None.

## Scaling Limits

### Data Volume

- Current capacity: Four data tier `.bin` files totaling approximately 1-2 GB each (tier1_c4.bin, tier2_fineweb.bin, tier3_wiki.bin, tier4_orca.bin).
- Limit: Training for 2000 steps at batch_size=8 and seq_len=512 consumes only ~8M tokens, which is negligible for a 164M parameter model. The model is dramatically undertrained relative to its capacity.
- Scaling path: Increase max_steps to 50K-100K. The multi-tier progressive data schedule was designed for this but never fully exercised.

### CUDA Memory for QAT

- Current capacity: The QAT wrapping (`QuantizedLinear` autograd Function) stores a forward-pass copy of the full-precision weights for the backward pass. For FP4 QAT with 164M parameters, this means 164M x 4 bytes = ~656 MB for the weight cache alone, plus activations for 12 layers.
- Limit: A single 24GB GPU (e.g., RTX 4090) can barely fit QAT with batch_size=8 and seq_len=512. Scaling batch_size beyond 8 or sequence length beyond 512 will OOM.
- Scaling path: Use activation checkpointing to trade compute for memory, or use the forward-hook-based QAT (`make_qat_forward_hook`) which uses `torch.no_grad()` and does not save activations for quantization.

## Dependencies at Risk

### sshpass Dependency

- Risk: `sshpass` must be installed on the local machine. It is not included in `requirements.txt` because it is a system package, not a Python package. A new developer cloning the repo will not be prompted to install it.
- Impact: `sync.sh`, `remote_python.sh`, `remote_run.sh` all fail immediately if sshpass is not installed.
- Migration plan: Document the dependency in README.md, or replace with SSH-key-based authentication that does not require sshpass.

## Missing Critical Features

### No Unit Tests

- Problem: The entire codebase has zero tests. There are no unit tests, integration tests, or smoke tests for any module. Files like `src/quantization/fp_quantizer.py` (242 lines of quantization logic), `src/quantization/gptq.py` (254 lines of GPTQ compensation), `src/analysis/condition.py` (168 lines of numerical analysis) and `src/model/transformer.py` (270 lines of model architecture) are completely untested.
- Blocks: Safe refactoring. Any change to core logic risks breaking the 6+ experiment scripts that depend on it, with no automated safety net. The `train_gemma4.py` parallel implementation (252 lines) is particularly risky because there is no cross-validation test against the main model.

### No Structured Logging

- Problem: All logging uses `print()` statements scattered throughout the codebase. There is no logging framework, no log levels (DEBUG, INFO, WARNING, ERROR), no log file output, and no structured format for parsing. W&B integration (`wandb`) is listed in `requirements.txt` but never imported or used anywhere in the source code.
- Blocks: Debugging training runs, correlating experiment results, automated log analysis.

### No Model Export or Inference Pipeline

- Problem: The project has no inference-only pipeline. All scripts reload the full model and run forward passes within training utilities. There is no path to export a quantized model to a deployment format (ONNX, TorchScript, `.safetensors`).
- Blocks: Using the quantized model outside of the experimental harness.

## Test Coverage Gaps

**CRITICAL:** The entire codebase has zero test coverage. Every module is untested.

| Module | Files | Risk | Priority |
|--------|-------|------|----------|
| FP Quantizer | `src/quantization/fp_quantizer.py` | Grid-based quantization rounding and STE backward could have silent precision bugs | High |
| GPTQ | `src/quantization/gptq.py` | Cholesky damping, Hessian estimation, column compensation -- numerically sensitive | High |
| Transformer model | `src/model/transformer.py` | GQA, RoPE, sliding/full attention mask correctness | High |
| Condition analysis | `src/analysis/condition.py` | Condition number estimates used for mixed-precision decisions | High |
| Adaptive grid | `src/quantization/adaptive_grid.py` | Lloyd-Max convergence, kappa-weighting | Medium |
| Training loop | `src/experiments/training_utils.py` | Perplexity evaluation, checkpoint save/load | Medium |
| Data pipeline | `src/experiments/prepare_data_chunked.py` | Tokenization correctness | Low |

---

*Concerns audit: 2026-05-17*
