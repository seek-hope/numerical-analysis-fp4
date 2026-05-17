# Phase 1: Clean Data Split - Context

**Gathered:** 2026-05-17
**Status:** Ready for planning

## Phase Boundary

Split each of the 4 data tiers (C4, FineWeb-edu, Wikipedia, OpenOrca) into separate train/val .bin files. The first 95% of tokens in each tier becomes `tierN_train.bin`, the last 5% becomes `tierN_val.bin`. Update `get_dataloader()` to accept a `split='train'|'val'` parameter so PTQ calibration uses training data only and evaluation uses validation data only — eliminating the current calibration-evaluation leakage where the same .bin files serve both purposes.

## Implementation Decisions

### Split Strategy
- **D-01:** Use a post-processing approach — write a utility that reads existing `tierN.bin` files and splits them into `tierN_train.bin` (first 95%) and `tierN_val.bin` (last 5%). No re-download required. The existing 4 .bin files (~4.24B tokens) remain on disk as source material.
- **D-02:** Also modify `prepare_data_chunked.py` to accept an optional `--val_split` flag (default 0.05) so future data preparation can produce train/val files directly from the stream, avoiding the post-processing step.

### Dataloader API Design
- **D-03:** Add a `split: str = 'train'` parameter to `get_dataloader()`. When `split='train'`, glob for `*_train.bin` files. When `split='val'`, glob for `*_val.bin` files. Default `'train'` ensures backward compatibility — all existing experiment scripts (~18 files) continue working without modification.
- **D-04:** The `split` parameter flows through `get_real_dataloader()` → `MultiTierDataset`, which uses a glob pattern to select the right files. `BinDataset` itself is unchanged (it operates on a single file path).

### Shuffle Behavior
- **D-05:** Shuffle is derived from the split: `shuffle=(split == 'train')`. Training data is shuffled; validation data is deterministic. This eliminates the footgun of forgetting to disable shuffle for evaluation and is consistent with standard ML practice.

### Verification
- **D-06:** File-access audit — log which .bin files each dataloader opens, tagged with the split. At minimum, add a `print()` statement in `get_real_dataloader()` showing the matched file paths and split. This directly satisfies success criterion #3 ("verified by file access audit").

### Claude's Discretion
- Exact implementation location of the split utility (standalone script vs function in `training_utils.py` vs addition to `prepare_data_chunked.py`)
- Whether to delete or keep original unsplit `tierN.bin` files after splitting
- Logging verbosity and format for the file-access audit

## Canonical References

### Data Pipeline
- `src/experiments/prepare_data_chunked.py` — Current chunked data preparation; must be aware of train/val split for future runs
- `src/experiments/training_utils.py:197-219` — `get_dataloader()` and `get_real_dataloader()`; primary modification targets
- `src/experiments/training_utils.py:124-157` — `MultiTierDataset` and `BinDataset`; need split-aware glob pattern
- `CLAUDE.md` §"Data Pipeline" — Documents the expected tokenized data layout and preparation workflow

### Requirements
- `.planning/REQUIREMENTS.md` §DATA-01, DATA-02, DATA-03 — Full requirement text for this phase
- `.planning/ROADMAP.md` §"Phase 1: Clean Data Split" — Success criteria (5 items)

### Downstream Consumers (for awareness)
- `src/quantization/gptq.py:138-191` — GPTQ calibration uses training dataloader
- `src/quantization/adaptive_grid.py` — Adaptive grid fitting uses training dataloader
- `src/experiments/training_utils.py:313-337` — `evaluate_perplexity()` uses validation dataloader

## Existing Code Insights

### Reusable Assets
- `BinDataset` (`training_utils.py:96-121`): Memory-mapped dataset from a single .bin file. Does not need modification — it receives a concrete file path from `MultiTierDataset`.
- `MultiTierDataset` (`training_utils.py:124-157`): Concatenates multiple `BinDataset` instances via glob. Only needs a glob pattern change — `*_train.bin` instead of `*.bin`.
- `process_tier()` in `prepare_data_chunked.py:59-131`: Streaming tokenization pipeline. Only needs a post-processing step or optional split flag — core streaming logic stays intact.
- `get_dataloader()` auto-detection (`training_utils.py:197-219`): The auto-fallback from real data to offline corpus is preserved; split only affects real data path.

### Established Patterns
- All experiment scripts use `get_dataloader(data_dir=args.data_dir, ...)` — no direct `MultiTierDataset` construction. Adding `split` to `get_dataloader()` covers all consumers.
- Dataloaders use `num_workers=0`, `pin_memory=True`, `shuffle=True` — these stay the same for training, shuffle changes for val.
- CLI arguments follow argparse pattern: `--data_dir`, `--batch_size`, etc. Validation scripts that need eval data will add `--split val`.

### Integration Points
- **`get_dataloader()`** is the single chokepoint — every training and evaluation script goes through it. Modifying this function propagates the split to all consumers automatically.
- **`prepare_data_chunked.py`** is the data source — it creates the .bin files that the dataloader reads. The split utility can be embedded here or as a standalone script.
- **PTQ evaluation scripts** (`eval_quantization.py`, `ptq_eval.py`, `compare_adaptive_grid.py`, `phase2_comparison.py`, `final_summary.py`) need their dataloader calls updated to use `split='val'` for evaluation — this is a Phase 5 concern but noted here for awareness.
- **sync.sh** will sync the new `*_train.bin` and `*_val.bin` files to the remote server — no changes needed (rsync glob covers `*.bin`).

## Specific Ideas

No specific requirements beyond the success criteria — the split ratio (95/5) and file naming convention (`tierN_train.bin` / `tierN_val.bin`) are locked by ROADMAP.md.

## Deferred Ideas

None — discussion stayed within phase scope.

---

*Phase: 1-Clean Data Split*
*Context gathered: 2026-05-17*
*Mode: --auto (all areas auto-selected, recommended options chosen)*
