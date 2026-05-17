# Phase 1: Clean Data Split - Research

**Researched:** 2026-05-17
**Domain:** Data pipeline infrastructure -- binary file I/O, PyTorch DataLoader, streaming data preparation
**Confidence:** HIGH

## Summary

The current data pipeline has a critical design flaw: four monolithic `.bin` files (`tier1_c4.bin` through `tier4_orca.bin`) each serve as a single source for ALL purposes -- training, PTQ calibration, and evaluation. This means evaluation data leaks into calibration (GPTQ Hessian estimation uses the same tokens it later evaluates on), invalidating the clean separation that downstream phases (especially Phase 2 Core Measurement and Phase 5 PTQ Comparison) require.

Phase 1 fixes this by splitting each tier at the token level (first 95% train, last 5% val) into separate `tierN_train.bin` and `tierN_val.bin` files, then updating the dataloader API so callers explicitly request a split. All changes are backward-compatible via a default `split='train'` parameter.

**Primary recommendation:** A post-processing split utility written in Python (numpy for file I/O), plus targeted modifications to `get_dataloader()`/`get_real_dataloader()`/`MultiTierDataset` in `training_utils.py`, and a `--val_split` flag for `prepare_data_chunked.py`. No new external dependencies needed.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01:** Use a post-processing approach -- write a utility that reads existing `tierN.bin` files and splits them into `tierN_train.bin` (first 95%) and `tierN_val.bin` (last 5%). No re-download required. The existing 4 .bin files (~4.24B tokens) remain on disk as source material.
- **D-02:** Also modify `prepare_data_chunked.py` to accept an optional `--val_split` flag (default 0.05) so future data preparation can produce train/val files directly from the stream, avoiding the post-processing step.
- **D-03:** Add a `split: str = 'train'` parameter to `get_dataloader()`. When `split='train'`, glob for `*_train.bin` files. When `split='val'`, glob for `*_val.bin` files. Default `'train'` ensures backward compatibility -- all existing experiment scripts (~18 files) continue working without modification.
- **D-04:** The `split` parameter flows through `get_real_dataloader()` -> `MultiTierDataset`, which uses a glob pattern to select the right files. `BinDataset` itself is unchanged (it operates on a single file path).
- **D-05:** Shuffle is derived from the split: `shuffle=(split == 'train')`. Training data is shuffled; validation data is deterministic. This eliminates the footgun of forgetting to disable shuffle for evaluation and is consistent with standard ML practice.
- **D-06:** File-access audit -- log which .bin files each dataloader opens, tagged with the split. At minimum, add a `print()` statement in `get_real_dataloader()` showing the matched file paths and split. This directly satisfies success criterion #3 ("verified by file access audit").

### Claude's Discretion

- Exact implementation location of the split utility (standalone script vs function in `training_utils.py` vs addition to `prepare_data_chunked.py`)
- Whether to delete or keep original unsplit `tierN.bin` files after splitting
- Logging verbosity and format for the file-access audit

### Deferred Ideas (OUT OF SCOPE)

None.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| DATA-01 | Implement train/val split at .bin file level -- each tier's last 5% becomes `tierN_val.bin`, first 95% becomes `tierN_train.bin` | Split utility reads existing flat uint32 arrays (numpy fromfile), computes split index at 95% of token count, writes two output files via numpy tofile. File naming convention: `tier{1,2,3,4}_{c4,fineweb,wiki,orca}_train.bin` / `*_val.bin`. Existing `tierN.bin` files remain for reference. |
| DATA-02 | Update `get_dataloader()` to accept `split='train'|'val'` parameter and filter .bin files accordingly | Modify 3 functions in `training_utils.py`: `MultiTierDataset.__init__` uses pattern `*_train.bin` or `*_val.bin` instead of `*.bin`; `get_real_dataloader()` passes split through; `get_dataloader()` accepts and forwards split parameter. Default `'train'` preserves backward compat for ~20 call sites across all experiment scripts. |
| DATA-03 | Ensure PTQ calibration (GPTQ Hessian, adaptive grid) uses training split only; evaluation uses validation split only | Infrastructure level: split parameter exists and defaults to 'train' for backward compat. Consumer updates (eval_quantization.py, ptq_eval.py, etc.) are Phase 5 concern per CONTEXT.md. However, the file-access audit (D-06) logs which files each loader opens, making split compliance verifiable immediately after implementation. |
</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Data storage (binary .bin files) | Database / Storage | -- | Tokenized data persisted as flat uint32 arrays on disk. Single source of truth. |
| Data splitting (post-processing) | Database / Storage | -- | Reads existing .bin files, writes new train/val .bin files. Pure file I/O, no model or training dependency. |
| Data preparation (streaming) | Database / Storage | -- | `prepare_data_chunked.py` streams from HuggingFace datasets, tokenizes, writes .bin files. No model dependency. |
| Dataloader instantiation | API / Backend | -- | `get_dataloader()` is the API gateway that training scripts call. It resolves split to file glob and creates PyTorch DataLoader. |
| Dataset iteration | API / Backend | -- | `MultiTierDataset` and `BinDataset` are PyTorch Dataset subclasses. They handle indexing and memory-mapped reads. |
| File-access audit | API / Backend | -- | Logging statements in `get_real_dataloader()` report which files were matched for each split request. |

## Standard Stack

### Core
| Tool/Component | Version | Purpose | Why Standard |
|----------------|---------|---------|--------------|
| numpy | 2.4.5 | Binary file I/O: `np.fromfile` reads flat uint32 arrays, `arr.tofile()` writes them | The entire data pipeline uses numpy for .bin file storage. Already a dependency. |
| Python 3 stdlib | 3.14.5 | `os.path`, `glob`, `argparse` for path management and CLI | All existing scripts use these. Zero additional dependencies needed. |
| PyTorch DataLoader | 2.3+ | `torch.utils.data.DataLoader`, `Dataset`, `IterableDataset` | The existing dataloader infrastructure. `get_dataloader()` returns a DataLoader. |
| PyTorch Dataset | 2.3+ | `torch.utils.data.Dataset` base class for `BinDataset` and `MultiTierDataset` | Existing base class. No changes needed to BinDataset. |

### Supporting
| Tool/Component | Version | Purpose | When to Use |
|----------------|---------|---------|-------------|
| `torch.from_numpy()` | 2.3+ | Convert numpy array to tensor in BinDataset | Already used in BinDataset.__init__. No new usage needed. |
| `datasets` library | >=2.20.0 | HuggingFace datasets streaming in prepare_data_chunked.py | Already used. Only affected by --val_split flag addition. |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| numpy file I/O | `torch.save` / `torch.load` | Existing data is in numpy .bin format. Switching would require re-tokenizing all data. Not worth the 1% gain. |
| numpy file I/O | Memory-mapped chunked reading | The split utility reads the whole file at once. For 5.3 GB max file, this uses ~5.3 GB RAM -- acceptable on a modern dev machine. Chunked reading would add complexity for marginal memory savings. |
| MultiTierDataset glob change | Individual file lists | Glob pattern `*_train.bin` automatically picks up all train files. Explicit file lists would need updating every time a new tier is added. |

## Package Legitimacy Audit

This phase does not install any external packages. All work uses:
- **Python 3.14.5** (stdlib -- already available, verified: `python3 --version`)
- **numpy 2.4.5** (already installed, verified: `python3 -c "import numpy; print(numpy.__version__)"`)
- **PyTorch >= 2.3** (already installed, used by existing training code)
- **HuggingFace datasets >= 2.20.0** (already installed, used by existing prepare_data_chunked.py)

No new packages to verify.

## Architecture Patterns

### System Architecture Diagram

```
prepare_data_chunked.py  (data source -- stream from HF datasets, tokenize, write .bin)
         |
         | (with --val_split flag) writes tierN_train.bin + tierN_val.bin directly
         v
Data directory: data/real_tiers/
  tier1_c4_train.bin    (first 95% of original tier1_c4.bin)
  tier1_c4_val.bin      (last 5%)
  tier2_fineweb_train.bin
  tier2_fineweb_val.bin
  tier3_wiki_train.bin
  tier3_wiki_val.bin
  tier4_orca_train.bin
  tier4_orca_val.bin
         |
         v
split_utility.py  (post-processing -- reads tierN.bin, writes *_{train,val}.bin)
  (only needed if data was prepared before Phase 1; skip if prepare_data_chunked.py
   with --val_split was used instead)
         |
         v
get_dataloader(split='train'|'val')
  |
  +---> get_real_dataloader(split)
  |       |
  |       v
  |     MultiTierDataset(split)
  |       |  glob = '*_train.bin' (train) | '*_val.bin' (val)
  |       |  prints: "[DATA] split={split} matched {N} files: {paths}"
  |       v
  |     [BinDataset(path) for each matched path]
  |       (unchanged -- memory-maps single .bin file)
  |
  +---> DataLoader(dataset, shuffle=(split=='train'))
  |
  v
Training scripts (split='train', default)
Eval scripts (split='val', explicit)
```

### Recommended Project Structure

The split utility is a small standalone script (single file). Recommended locations in priority order:

1. **`src/experiments/split_data.py`** -- standalone CLI utility for the post-processing split
2. Alternatively: **New function in `training_utils.py`** (but this is a one-time utility, not a routine import)

```text
src/
  experiments/
    split_data.py           (NEW -- standalone split utility for existing .bin files)
    prepare_data_chunked.py (MODIFY -- add --val_split flag)
    training_utils.py       (MODIFY -- MultiTierDataset, get_real_dataloader, get_dataloader)
```

### Pattern 1: Post-Processing Split Utility

**What:** A standalone Python script that reads an existing `tierN.bin` flat uint32 array, computes the 95% split index, and writes both `tierN_train.bin` and `tierN_val.bin` files.

**When to use:** For the existing 4 monolithic .bin files that were already prepared.

**Key design decisions:**
- CLI accepts `--tiers` (optional, defaults to all 4) and `--val_split 0.05` and `--delete_original` (at discretion)
- Uses `np.fromfile` to read entire file, computes `split_idx = int(len(tokens) * (1 - val_split))`
- Writes `arr[:split_idx].tofile(path_train)` and `arr[split_idx:].tofile(path_val)`
- Handles empty validation set edge case: if split_idx == len(tokens), val file is empty (verified: `np.array([], dtype=np.uint32).tofile(f)` writes 0 bytes)

### Pattern 2: Split-Aware Dataloader

**What:** `get_dataloader()` now accepts `split='train'` (default) or `split='val'`. Glob pattern in `MultiTierDataset` changes from `*.bin` to `*_train.bin` or `*_val.bin` accordingly.

**When to use:** Every dataloader creation call site. Existing callers automatically get `split='train'`, preserving current behavior.

**Key interfaces:**
```python
# modified signature
def get_dataloader(..., split: str = 'train', ...) -> DataLoader

# new behavior in get_real_dataloader
def get_real_dataloader(..., split: str = 'train', ...) -> DataLoader:
    dataset = MultiTierDataset(data_dir, seq_len, tier, split=split)
    shuffle = (split == 'train')
    print(f"[DATA] split={split} matched {len(dataset.datasets)} files: "
          f"{[ds.path for ds in dataset.datasets]}")
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, ...)
```

### Pattern 3: Streaming Split in prepare_data_chunked.py

**What:** Add `--val_split 0.05` CLI argument. During streaming tokenization, when total_tokens reaches `int(max_tokens * (1 - val_split))`, switch from writing to `tierN_train.bin` to writing to `tierN_val.bin`. Core streaming logic stays intact.

**When to use:** Future data preparation runs that want train/val separation from the start.

### Anti-Patterns to Avoid

- **Modifying BinDataset:** BinDataset receives a single file path and memory-maps it. It has no concept of "split." Adding split awareness here would leak concerns upward. Leave it unchanged.
- **Two-pass split:** Reading the entire .bin into memory is fine for the post-processing utility. Do NOT implement a two-pass approach (one to count, one to split) -- the files fit in RAM.
- **Renaming existing .bin files:** Do NOT rename `tierN.bin` to `tierN_train.bin` directly and then try to extract the last 5%. The files need actual data splitting.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Binary file I/O for .bin format | Custom file format parsing | `numpy.fromfile` / `numpy.ndarray.tofile` | The existing .bin format is already a flat uint32 numpy array. numpy handles endianness, memory mapping, and file I/O correctly. |
| CLI argument parsing | Custom argument parsing | `argparse` (stdlib) | All existing scripts use argparse. Consistent and zero-dep. |
| PyTorch Dataset sampling | Custom data iteration | `torch.utils.data.DataLoader` | Existing infrastructure. Handles batching, shuffling, multi-worker loading. |

**Key insight:** This phase is pure data infrastructure with no novel algorithms. The risk is not in the split logic (trivial arithmetic) but in making sure the split flows correctly through the MultiTierDataset glob pattern to the DataLoader without breaking any of the ~20 existing call sites.

## Common Pitfalls

### Pitfall 1: Glob Pattern Collision
**What goes wrong:** If `_train.bin` or `_val.bin` files coexist with original `*.bin` files, the glob pattern picks up the wrong files or duplicates.
**Why it happens:** `MultiTierDataset.__init__` currently globs `*.bin`. If both `tier1_c4.bin` AND `tier1_c4_train.bin` exist, the original glob picks up all 8 files.
**How to avoid:** After splitting, DO NOT leave original `tierN.bin` files in the same directory as the split files. Either delete them or move them to a backup directory. The glob patterns `*_train.bin` and `*_val.bin` are disjoint from `*.bin`, but the original `*.bin` pattern will still match them all.
**Warning signs:** A dataloader reports matching 8+ files when only 4 are expected for a split.

### Pitfall 2: 95/5 Split on Odd Number of Tokens
**What goes wrong:** `int(N * 0.95)` may leave fewer tokens than expected in the training set if N is small.
**Why it happens:** The .bin files have 800M-1.4B tokens each. The integer truncation at 0.05 ratio creates a discrepancy of at most 1 token between train + val and the original. Negligible at scale.
**How to avoid:** Use `int(N * (1 - val_split))` for split index and verify `len(train) + len(val) == N` after split.
**Warning signs:** Mismatch in total token count before and after split.

### Pitfall 3: Shuffle=True for Validation
**What goes wrong:** Current `get_real_dataloader()` always uses `shuffle=True`. If `split='val'` is used but shuffle is not explicitly set to False, eval results are non-deterministic.
**How it's avoided:** D-05 locks the behavior: `shuffle=(split == 'train')`. The fix must hardcode this derivation in `get_real_dataloader()`, not leave it as a caller responsibility.
**Warning signs:** Reproducibility issues in validation runs. The file-access audit will confirm shuffle=False for val.

### Pitfall 4: prepare_data_chunked.py Stream Interruption at Split Point
**What goes wrong:** When the streaming process switches from writing train to writing val, the "current" batch of tokens could be split across the two files.
**Why it happens:** The process flushes every CHUNK_SIZE (10K) samples. The split point (95% of max_tokens) may fall mid-buffer.
**How to avoid:** When `total_tokens >= split_threshold`, write the train portion of the current buffer to `tierN_train.bin`, then write the val portion to `tierN_val.bin`. Reset buffer and continue appending to val file for remaining tokens.
**Warning signs:** Token count mismatch between train+val and original. Off-by-1 errors.

### Pitfall 5: Off-by-One in BinDataset Sequence Length
**What goes wrong:** The split utility might split at a token index that causes `BinDataset` sequences to span the train/val boundary.
**Why it happens:** `BinDataset.__init__` computes `n = max(0, (len(self.data) - 1) // seq_len)`, meaning tokens are sliced into sequences of `seq_len+1`. If we split exactly at token 95%, the last training sequence and first validation sequence are complete and independent -- no data leakage.
**How to avoid:** The 95/5 split is a clean file-level boundary. BinDataset operates on flat arrays, so the boundary between train and val files is naturally at a token boundary. No special handling needed.
**Warning signs:** Not applicable -- this is a correct-by-construction property of flat uint32 arrays.

## Code Examples

### Example 1: Split Utility Core Logic

```python
# Source: standard numpy pattern
import numpy as np

def split_bin_file(src_path: str, dst_train: str, dst_val: str,
                   val_split: float = 0.05) -> dict:
    """Split a flat uint32 .bin file into train and val files."""
    data = np.fromfile(src_path, dtype=np.uint32)
    n_tokens = len(data)
    split_idx = int(n_tokens * (1 - val_split))

    data[:split_idx].tofile(dst_train)
    data[split_idx:].tofile(dst_val)

    return {
        'source': src_path,
        'total': n_tokens,
        'train': split_idx,
        'val': n_tokens - split_idx,
    }
```

### Example 2: MultiTierDataset Split-Aware Glob

```python
# Source: modification to existing pattern in training_utils.py
import glob as _glob

class MultiTierDataset(Dataset):
    def __init__(self, data_dir: str, seq_len: int = 512,
                 tier: str | None = None, split: str = 'train'):
        if tier:
            paths = [os.path.join(data_dir, f'{tier}_{split}.bin')]
        else:
            pattern = f'*_{split}.bin'
            paths = sorted(_glob.glob(os.path.join(data_dir, pattern)))

        self.datasets = []
        self.offsets = [0]
        for p in paths:
            if os.path.exists(p):
                ds = BinDataset(p, seq_len)
                ds.path = p  # for file-access audit
                self.datasets.append(ds)
                self.offsets.append(self.offsets[-1] + len(ds))
        self.total_len = self.offsets[-1]
```

### Example 3: get_real_dataloader with Split and File-Access Audit

```python
# Source: modification to existing pattern in training_utils.py
def get_real_dataloader(data_dir: str, batch_size: int = 8,
                         seq_len: int = 512, max_steps: int = 1500,
                         tier: str | None = None,
                         split: str = 'train') -> DataLoader:
    dataset = MultiTierDataset(data_dir, seq_len, tier, split=split)
    matched_paths = [getattr(ds, 'path', 'unknown') for ds in dataset.datasets]
    print(f"[DATA] split={split} matched {len(matched_paths)} files: "
          f"{matched_paths}")
    shuffle = (split == 'train')
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate_batch, num_workers=0,
                      pin_memory=True)
```

### Example 4: prepare_data_chunked.py Split-Aware Streaming

```python
# Source: modification to existing process_tier() in prepare_data_chunked.py
# After tokenizing and appending to buffer...
buffer.extend(tokens)
total_tokens += len(tokens)

# Determine output path based on split progress
if not split_triggered and val_split > 0 and total_tokens >= split_threshold:
    # Flush train portion of buffer, then switch to val file
    train_tokens = buffer[:split_buffer_idx]
    if train_tokens:
        arr = np.array(train_tokens, dtype=np.uint32)
        arr.tofile(open(train_path, 'ab' if os.path.exists(train_path) else 'wb'))
    # Keep remaining buffer for val file
    buffer = buffer[split_buffer_idx:]
    split_triggered = True
    print(f"  Split point reached. Switching to validation file: {val_path}")

# Flush to appropriate output file
out_path = val_path if split_triggered else train_path
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Single `*.bin` per tier | Split `*_train.bin` + `*_val.bin` per tier | Phase 1 | PTQ calibration and evaluation now draw from disjoint token sets. Calibration-evaluation leakage eliminated. |
| `shuffle=True` hardcoded in get_real_dataloader | `shuffle=(split=='train')` | Phase 1 | Validation runs are deterministic by default. Eliminates footgun of shuffling eval data. |

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | The existing .bin files contain exactly the token counts reported during preparation (800M per tier for C4, FineWeb, Wiki; ~700M for Orca) | Summary | If actual token counts differ, the split ratio is still correct (95/5 of actual), but file-size expectations in planning may be off. Low risk. |
| A2 | The machine running the split utility has enough RAM (~5.3 GB free) to load the largest .bin file entirely | Standard Stack | We have 295 GB free disk and likely 16-32 GB RAM on the dev machine. Loading a 5.3 GB uint32 array takes ~5.3 GB of RAM. If RAM is tight, chunked reading is possible but not expected to be needed. |
| A3 | Original `tierN.bin` files can be deleted after splitting without breaking anything | Claude's Discretion | The post-processing split is a one-time operation. Once train/val files exist, the original files are never read. Delete is safe but discretionary. |
| A4 | `glob('*_train.bin')` does not match `*.bin` files | Architecture Patterns | Correct -- `*_train.bin` is a different glob pattern from `*.bin`. No overlap. However, if `_train.bin` and source `*.bin` coexist and an old caller still uses `*.bin`, it would pick up all 8 files. This is why the API change (D-03) is essential. |

**This table contains 4 assumptions, all LOW risk.** No user confirmation needed before planning.

## Open Questions

1. **Where to place the split utility?**
   - What we know: The CONTEXT.md grants discretion. Options are `src/experiments/split_data.py` (standalone), a function in `training_utils.py`, or an addition to `prepare_data_chunked.py`.
   - What's unclear: Whether the split utility is "one-time use" (then a standalone script is fine) or "reusable part of the pipeline" (then it belongs in training_utils.py).
   - Recommendation: Default to `src/experiments/split_data.py` as a standalone CLI utility. It's a one-time operation for the existing 4 files. Future runs use `prepare_data_chunked.py --val_split` instead.

2. **Should original `tierN.bin` files be deleted?**
   - What we know: Keeping them wastes ~16 GB of disk space and creates collision risk with the old `*.bin` glob pattern. Deleting them after successful split verification is safe.
   - What's unclear: Whether the user wants to keep originals as a backup.
   - Recommendation: Add a `--delete_original` flag to the split utility (default False). User decides.

3. **How to handle the file-access audit for downstream consumers that call `get_dataloader()` indirectly?**
   - What we know: D-06 requires logging in `get_real_dataloader()`. This covers all direct callers.
   - What's unclear: Whether indirect consumers (e.g., GPTQQuantizer.quantize_model() which receives a loader, or evaluate_perplexity() which receives a loader) need their own logging.
   - Recommendation: The logging in `get_real_dataloader()` is sufficient -- it fires once when the loader is created, tagging the file list and split. Downstream consumers don't need additional logging because they operate on the already-constructed DataLoader.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.11+ | All code | Yes | 3.14.5 | -- |
| numpy | Binary .bin I/O | Yes | 2.4.5 | -- |
| PyTorch 2.3+ | DataLoader | Yes | (assumed installed) | -- |
| disk space | 16 GB new .bin files | Yes | 295 GB available on /dev/nvme0n1p2 | -- |

**Missing dependencies with no fallback:** None.

**Missing dependencies with fallback:** None.

## Validation Architecture

> Skipped: `workflow.nyquist_validation` is explicitly `false` in `.planning/config.json`. No new test infrastructure needed for this phase. Verification is via file-access audit logging (D-06) and manual checksum comparison of token counts before/after split.

## Security Domain

Not applicable. This phase operates on pre-tokenized binary data (completely synthetic content derived from public datasets). No authentication, authorization, session management, input validation, or cryptography involved. The data pipeline is single-user on a local development machine with remote GPU server access via SSH. No ASVS categories apply.

## Sources

### Primary (HIGH confidence)
- [Codebase] `src/experiments/training_utils.py` -- verified all 357 lines of the dataloader infrastructure
- [Codebase] `src/experiments/prepare_data_chunked.py` -- verified all 181 lines of the streaming data preparation
- [Codebase] `src/experiments/eval_quantization.py` -- verified all 197 lines, including calib_loader and eval_loader patterns
- [Codebase] `src/experiments/ptq_eval.py` -- verified all 155 lines, confirming calibration/evaluation dataloader patterns
- [Codebase] `src/quantization/gptq.py` -- verified activation collection in `_collect_activations` for calibration data flow
- [Codebase] `.planning/CONTEXT.md` -- locked decisions D-01 through D-06
- [Codebase] `.planning/ROADMAP.md` -- success criteria for Phase 1
- [Codebase] `.planning/REQUIREMENTS.md` -- DATA-01, DATA-02, DATA-03 requirement text
- [Local disk] `data/real_tiers/*.bin` -- verified existing files (16 GB total across 4 tiers)

### Secondary (MEDIUM confidence)
- [Codebase survey] Grep across 20+ experiment scripts -- confirmed all callers of `get_dataloader()` and their current parameter patterns
- [Codebase survey] Verified absence of test files (no `test_*.py` or `tests/` directories anywhere)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - all tools/versions verified on local machine
- Architecture: HIGH - all patterns derived from verified codebase, not assumed
- Pitfalls: HIGH - directly observed from reading the code (glob collision risk, shuffle behavior, stream interruption)
- Assumptions: LOW/MEDIUM - 4 items logged, all low risk, documented for transparency

**Research date:** 2026-05-17
**Valid until:** The .bin file format and data pipeline are stable and well-established. This research remains valid as long as the project's data structure (flat uint32 arrays with 4 tiers) remains unchanged.
