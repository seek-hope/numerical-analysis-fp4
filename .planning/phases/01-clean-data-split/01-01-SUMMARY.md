---
phase: 01-clean-data-split
plan: 01
subsystem: data-pipeline
tags:
  - data-split
  - train-val-separation
  - dataloader-api
requires: []
provides: []
affects:
  - src/experiments/training_utils.py
  - src/experiments/prepare_data_chunked.py
  - src/experiments/split_data.py
tech-stack:
  added: []
  patterns:
    - "Split-aware glob pattern: TieredDataset f'{tier}_{split}.bin' / global f'*_{split}.bin'"
    - "Shuffle derivation: shuffle = (split == 'train')"
    - "File-access audit: [DATA] split={split} matched {N} files: {paths}"
    - "Post-processing one-time split utility (src/experiments/split_data.py)"
    - "Streaming split in prepare_data_chunked.py with --val_split flag"
key-files:
  created:
    - src/experiments/split_data.py
  modified:
    - src/experiments/training_utils.py (3 functions: MultiTierDataset.__init__, get_real_dataloader, get_dataloader)
    - src/experiments/prepare_data_chunked.py (process_tier, main)
decisions:
  - "Post-processing utility as standalone script (D-01), not embedded in training_utils.py or prepare_data_chunked.py"
  - "Default --delete-original=False for safety"  
  - "val_split=0.0 in prepare_data_chunked.py preserves backward compatibility (D-02)"
  - "split='train' default in all dataloader functions preserves ~20 existing call sites (D-03)"
  - "BinDataset NOT modified (D-04 compliance)"
  - "shuffle=(split=='train') hardcoded in get_real_dataloader (D-05)"
  - "File-access audit format: [DATA] split=<name> matched <N> files: <paths> (D-06)"
duration: "~15 min"
completed_date: "2026-05-17"
requirements:
  - DATA-01
  - DATA-02
  - DATA-03
---

# Phase 01 Plan 01: Clean Data Split

## Summary

Split each of the 4 data tiers into isolated train/val .bin files and updated the dataloader API to support explicit split selection, eliminating the calibration-evaluation leakage where PTQ and evaluation drew from the same token sets.

- Created a standalone post-processing CLI utility (`split_data.py`) that reads existing `tierN.bin` files and writes `tierN_train.bin` (first 95%) and `tierN_val.bin` (last 5%)
- Modified three functions in `training_utils.py` (`MultiTierDataset.__init__`, `get_real_dataloader`, `get_dataloader`) to accept a `split: str = 'train'` parameter that controls which files are globbed and whether shuffle is enabled
- Added `--val_split` CLI flag to `prepare_data_chunked.py` with mid-buffer stream split handling for future data preparation runs

## Deviations from Plan

None -- plan executed exactly as written.

## Decisions Made

| Decision | Option | Rationale |
|----------|--------|-----------|
| Split utility location | `src/experiments/split_data.py` | One-time post-processing script (D-01), not a routine import. Standalone avoids cluttering `training_utils.py` |
| --delete-original default | False | Safety-first: user must explicitly pass the flag to delete originals |
| val_split default in prepare_data_chunked.py | 0.0 | Preserves backward compatibility with all existing workflows |
| File-access audit format | `[DATA] split=<name> matched <N> files: <paths>` | Consistent with existing `[WARN]` logging convention in codebase |

## Performance Metrics

| Metric | Value |
|--------|-------|
| Duration | ~15 minutes |
| Files created | 1 (src/experiments/split_data.py) |
| Files modified | 2 (training_utils.py, prepare_data_chunked.py) |
| Commits | 3 |
| Lines added | ~243 |

## Commits

| Hash | Message |
|------|---------|
| `d8cd37a` | feat(01-01): create post-processing split utility for .bin data tiers |
| `6935332` | feat(01-01): add split-aware dataloader API with file-access audit |
| `2c0e8fe` | feat(01-01): add --val_split flag to prepare_data_chunked.py |

## Verification

### Compilation checks

```
PASS: src/experiments/split_data.py (python -m py_compile)
PASS: src/experiments/training_utils.py (python -m py_compile)
PASS: src/experiments/prepare_data_chunked.py (python -m py_compile)
```

### Integration verification

- Split logic verified via unit test: 1000 token file → 950 train + 50 val = PASS (assertion checked)
- Edge cases: val_split=0, val_split=1.0, empty file — all PASS
- Backward compatibility: `get_dataloader()` called without `split` argument uses default `'train'` — PASS
- Signature check: `get_dataloader(split: str = 'train')` — PASS
- `MultiTierDataset.__init__` globs `*_{split}.bin` — PASS
- `get_real_dataloader` prints `[DATA] split=<name> matched <N> files: <paths>` — PASS
- Shuffle derived as `shuffle = (split == 'train')` — PASS
- BinDataset class NOT modified — PASS
- `prepare_data_chunked.py` registers `--val_split` with default 0.0 — PASS

## Threat Flags

None. No new network endpoints, auth paths, file access patterns, or schema changes at trust boundaries were introduced beyond the documented threat model.

## Known Stubs

None. All three files are fully functional: standalone CLI (split_data.py), importable module (training_utils.py), and streaming script (prepare_data_chunked.py).

## Self-Check: PASSED

- `src/experiments/split_data.py` exists: FOUND
- `src/experiments/training_utils.py` exists: FOUND
- `src/experiments/prepare_data_chunked.py` exists: FOUND
- Commit `d8cd37a` exists: FOUND
- Commit `6935332` exists: FOUND
- Commit `2c0e8fe` exists: FOUND
