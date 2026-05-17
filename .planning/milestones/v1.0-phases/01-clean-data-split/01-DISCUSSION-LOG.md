# Phase 1: Clean Data Split - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-17
**Phase:** 01-clean-data-split
**Areas discussed:** Split Strategy, Dataloader API Design, Shuffle Behavior, Verification, Backward Compatibility

---

## Split Strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Post-process existing .bin files | Write utility to split each tier file into train/val after preparation; no re-download | ✓ |
| Modify prepare_data_chunked.py inline | Change streaming loop to write train/val from the start; requires re-download | |
| Two-phase count-then-split | First pass counts tokens, second pass writes split files | |

**Auto-selected:** Post-process existing .bin files (recommended — pragmatic, no re-download of 4.24B tokens). Also modify `prepare_data_chunked.py` with optional `--val_split` flag for future runs.
**Rationale:** [auto] Existing data is already on disk. Post-processing avoids re-downloading terabytes of data. The preparation script gets a forward-looking flag so future data preparation can produce splits directly.

---

## Dataloader API Design

| Option | Description | Selected |
|--------|-------------|----------|
| Add `split='train'` parameter to `get_dataloader()` | Single parameter, backward compatible default | ✓ |
| Separate `get_train_dataloader()` / `get_val_dataloader()` | Two new functions, deprecate old one | |
| Glob pattern filtering | Caller passes file pattern instead of split enum | |

**Auto-selected:** Add `split: str = 'train'` parameter (recommended — minimal API change, backward compatible). All 18 existing experiment scripts continue working without modification.
**Rationale:** [auto] `get_dataloader()` is the single chokepoint for all data access. A default parameter preserves current behavior while enabling the split for scripts that need it.

---

## Shuffle Behavior

| Option | Description | Selected |
|--------|-------------|----------|
| Auto: `shuffle=(split == 'train')` | Derived from split, no separate parameter | ✓ |
| Keep explicit `shuffle` parameter | Callers must remember to set it | |

**Auto-selected:** Auto-derive shuffle from split (recommended — eliminates footgun). Training data is shuffled; validation is deterministic.
**Rationale:** [auto] Standard ML practice. Forgetting `shuffle=False` for eval would silently produce non-deterministic results.

---

## Verification Mechanism

| Option | Description | Selected |
|--------|-------------|----------|
| File-access logging | Log which .bin files are opened per dataloader | ✓ |
| Token-level overlap check | Verify no tokens appear in both train and val | |
| Both | Logging + overlap check | |

**Auto-selected:** File-access logging (recommended — directly satisfies success criterion #3 "verified by file access audit"). Token-level overlap is unnecessary for a contiguous split — the 95/5 boundary is a clean cut with no interleaving.
**Rationale:** [auto] The split is contiguous (first 95% / last 5%), so no token-level overlap is possible by construction. File-access logging confirms the right files are opened.

---

## Backward Compatibility

| Option | Description | Selected |
|--------|-------------|----------|
| Default `split='train'` | All existing scripts get training data unchanged | ✓ |
| Default `split=None` load all | Preserve exact current glob behavior (`*.bin`) | |

**Auto-selected:** Default `split='train'` (recommended — principled, all existing scripts are training scripts and should use training data).
**Rationale:** [auto] All current experiment scripts that call `get_dataloader()` are training scripts. The default `'train'` gives them the correct data. PTQ eval scripts (Phase 5) will explicitly pass `split='val'`.

---

## Claude's Discretion

- Exact implementation location of the split utility
- Whether to delete or keep original unsplit `tierN.bin` files after splitting
- Logging verbosity and format for the file-access audit

## Deferred Ideas

None — discussion stayed within phase scope.
