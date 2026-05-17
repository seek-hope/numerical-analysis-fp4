# Technology Stack

**Analysis Date:** 2026-05-17

## Languages

**Primary:**
- Python 3.11+ (required) — all source and experiment code. Local interpreter is Python 3.14.5.

**Secondary:**
- Bash — shell scripts for remote execution (`remote_python.sh`, `remote_run.sh`, `sync.sh`)

## Runtime

**Environment:**
- Conda (remote GPU server: `sle` environment). Local environment uses system Python.
- No Dockerfile or container runtime detected.

**Package Manager:**
- pip (via `requirements.txt`)
- Lockfile: Not present — no `requirements.lock`, `poetry.lock`, or `pipfile.lock`

## Frameworks

**Core:**
- PyTorch >= 2.3.0 (`torch`) — all model definitions, quantization, and training loops
- HuggingFace Transformers >= 4.45.0 (`transformers`) — used for Gemma 4 E2B loading in `train_gemma4.py`, `AutoTokenizer` fallback in `prepare_data.py`, and `save_pretrained` serialization

**Testing:**
- Not detected — no test framework found in the codebase

**Build/Dev:**
- Not detected — no build system (setup.py, pyproject.toml, or Makefile are absent; the project is run via `python -m` or script path)

## Key Dependencies

**Critical:**
- `torch >= 2.3.0` — all tensor operations, neural network modules, optimizers (AdamW), CUDA GPU execution
- `transformers >= 4.45.0` — HuggingFace model loading/saving, AutoTokenizer for Gemma 4 fallback
- `datasets >= 2.20.0` — HuggingFace datasets for streaming C4, FineWeb, Wikipedia, OpenOrca (`load_dataset`)
- `tokenizers >= 0.19.0` — HuggingFace fast tokenizers for BPE training (`tokenizers.Tokenizer`, `tokenizers.models.BPE`, `tokenizers.trainers.BpeTrainer`)

**Infrastructure:**
- `numpy >= 1.26.0` — binary data shard I/O (`np.fromfile`, `np.array`), token ID storage as `uint32`
- `tqdm >= 4.66.0` — listed in requirements; not directly imported in `src/` (used internally by HuggingFace libraries)
- `safetensors >= 0.4.0` — listed in requirements; used by HuggingFace `transformers` for Gemma 4 model weights (`models/gemma4-e2b/model.safetensors`)
- `einops >= 0.8.0` — listed in requirements; not directly imported in `src/` (used internally by HuggingFace `transformers`)
- `accelerate >= 0.30.0` — listed in requirements; not directly imported in `src/`
- `wandb >= 0.17.0` — listed in requirements; not imported in `src/` (no experiment logging integration)
- `peft` — used in `train_gemma4.py` for LoRA fine-tuning (`peft.LoraConfig`, `peft.get_peft_model`)

## Configuration

**Environment:**
- `.sshpass` file (gitignored) stores SSH password for remote GPU server
- SSH-based remote execution via `sshpass` utility
- No `.env` files detected in project root

**Build:**
- Not applicable — no build step. Direct Python execution.
- No TypeScript, JS bundler, or compiled language found.

## Platform Requirements

**Development:**
- Python 3.11+
- pip install -r requirements.txt
- Network access for HuggingFace datasets download (data preparation only)
- 8-16 GB RAM minimum for data preparation (chunked streaming)
- No internet required for training (tokenized data prepared offline and synced)

**Production:**
- GPU with CUDA (NVIDIA) recommended for model training/evaluation
- Remote GPU server: `bi_group2@lulab_4090` (conda env `sle`)
- Local machine for editing, data preparation, and result inspection
- Minimum ~10GB GPU memory for Gemma 4 E2B + LoRA (`train_gemma4.py`)

---

*Stack analysis: 2026-05-17*
