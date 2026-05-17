# External Integrations

**Analysis Date:** 2026-05-17

## APIs & External Services

**HuggingFace Datasets (Data Pipeline):**
- Used for streaming text corpora during data preparation:
  - `allenai/c4` (English subset) — tier1_c4
  - `HuggingFaceFW/fineweb-edu` (sample-10BT) — tier2_fineweb
  - `wikimedia/wikipedia` (20231101.en) — tier3_wiki
  - `Open-Orca/OpenOrca` — tier4_orca
- Client: HuggingFace `datasets` library >= 2.20.0 (`load_dataset`)
- Usage files:
  - `src/experiments/prepare_data_chunked.py` (streaming, chunked tokenization)
  - `src/experiments/prepare_data.py` (batch tokenization)
  - `src/experiments/train_tokenizer.py` (BPE training from streaming data)

**HuggingFace Hub (Model Access — Fallback Only):**
- `AutoTokenizer.from_pretrained('google/gemma-4-E2B', ...)` used as fallback in `src/experiments/prepare_data.py:62-64` when local BPE tokenizer is unavailable
- `AutoModelForCausalLM.from_pretrained(...)` used in `src/experiments/train_gemma4.py:233` for full Gemma 4 E2B loading
- Local-only mode (`local_files_only=True`) after initial download
- Not used for main experiments (which use the custom `MicroGemmaFP` architecture)

## Data Storage

**Databases:**
- None. No SQL or NoSQL database is used.

**File Storage:**
- Local filesystem only. All data is stored as flat binary files:
  - Pre-tokenized data shards: `data/real_tiers/tier{1-4}_{name}.bin` (uint32 flat arrays)
  - Alternative data: `data/gemma4_tiers/tier{1-4}_{name}.bin`
  - Tokenizer: `data/tokenizer/bpe_32k.json` (BPE), `data/tokenizer/special_tokens.json` (special token IDs)
  - Model checkpoints: `checkpoints/*/model.pt` (PyTorch `torch.save` format)
  - Gemma 4 E2B local weights: `models/gemma4-e2b/` (HuggingFace safetensors format)

**Caching:**
- None. No Redis, Memcached, or in-memory cache layer.

## Authentication & Identity

**Auth Provider:**
- None. No user authentication or identity management.

**Access Control:**
- SSH password authentication via `.sshpass` file for remote GPU server access only. The `.sshpass` file is gitignored.

## Monitoring & Observability

**Logging:**
- Python `print()` statements throughout all experiment scripts. No structured logging framework (no `logging` module usage detected).
- Training metrics (loss, perplexity) printed to stdout at fixed intervals.

**Experiment Tracking:**
- `wandb` (Weights & Biases) is listed in `requirements.txt` but **not imported** anywhere in `src/`. No experiment tracking integration is active.
- Metrics are saved to local checkpoint files (`.pt` format) only.

**Error Tracking:**
- None. No Sentry, Rollbar, or similar service.

## CI/CD & Deployment

**Hosting:**
- None. No cloud deployment or hosted inference endpoint.

**CI Pipeline:**
- None. No GitHub Actions, GitLab CI, or similar pipeline configured.

**Syncing/Remote Execution:**
- Manual workflow using shell scripts:
  - `./sync.sh` — `rsync` over SSH (with `sshpass`) to sync project to remote GPU server. Excludes `.git`, `__pycache__`, `.venv`, `wandb`.
  - `./remote_python.sh <script>` — runs a Python script on the remote server (activates conda `sle`, sets `PYTHONPATH`).
  - `./remote_run.sh "<command>"` — runs an arbitrary shell command on the remote server.

**Validation Before Long Runs:**
- Local syntax checks: `python -m py_compile src/model/transformer.py src/...`
- Then sync and re-check on remote: `./remote_run.sh "python -m py_compile src/model/transformer.py src/..."`

## Environment Configuration

**Required env vars:**
- None. The project uses hardcoded paths in shell scripts and relative file paths in Python.

**Secrets location:**
- `.sshpass` (root of project, gitignored) — SSH password for remote GPU server `bi_group2@lulab_4090`.

## Webhooks & Callbacks

**Incoming:**
- None.

**Outgoing:**
- None.

## Remote Execution Infrastructure

**Target Server:**
- Host: `lulab_4090`
- User: `bi_group2`
- Remote path: `/home/bi_group2/Projects/Numerical_Analysis/`
- Conda env: `sle`
- SSH auth: password via `sshpass`

**PYTHONPATH on remote:**
- Set to the remote project root so that `src.` imports resolve correctly.

---

*Integration audit: 2026-05-17*
