#!/usr/bin/env python3
"""Smoke test: load Gemma 4 E2B on remote GPU."""
import torch
from transformers import AutoModelForCausalLM

print("Loading Gemma 4 E2B...")
m = AutoModelForCausalLM.from_pretrained(
    'models/gemma4-e2b', torch_dtype=torch.bfloat16,
    device_map='auto', trust_remote_code=True, local_files_only=True,
)
total = sum(p.numel() for p in m.parameters())
print(f"Loaded: {total:,} params")
print(f"Device: {next(m.parameters()).device}")
print("OK")
