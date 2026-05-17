"""Data loading and training utilities.

Supports two modes:
  - Real data: pre-tokenized .bin shards (BPE tokenizer, vocab_size=32000)
  - Offline fallback: embedded text corpus with character-level tokenizer

Auto-selects real data if .bin files exist, otherwise falls back to offline.
"""

import random, os, math
import torch
import numpy as np
from torch.utils.data import IterableDataset, Dataset, DataLoader


# ═══════════════════════════════════════════════════════════════
# Character-level tokenizer (offline fallback)
# ═══════════════════════════════════════════════════════════════

class CharTokenizer:
    """Character-level tokenizer — no external files needed.

    Maps each ASCII character to a token ID, plus special tokens.
    Sufficient for offline testing when no pretrained tokenizer is available.
    """

    PAD = 0
    EOS = 1
    UNK = 2
    BOS = 3

    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.pad_token_id = self.PAD
        self.eos_token_id = self.EOS
        self.bos_token_id = self.BOS
        self.pad_token = '<pad>'
        self.eos_token = '<eos>'
        self.bos_token = '<bos>'

    def encode(self, text: str, truncation: int | None = None,
               max_length: int | None = None) -> list[int]:
        max_len = truncation or max_length or 1000000
        ids = []
        for ch in text[:max_len]:
            val = ord(ch)
            if 3 <= val < min(self.vocab_size, 256):
                ids.append(val)
            else:
                ids.append(self.UNK)
        return ids


# ═══════════════════════════════════════════════════════════════
# Local text dataset (offline fallback)
# ═══════════════════════════════════════════════════════════════

from src.experiments.legacy.large_corpus import LARGE_CORPUS as _DEFAULT_CORPUS


class LocalTextDataset(IterableDataset):
    """Stream from an embedded text corpus — fully offline."""

    def __init__(self, tokenizer: CharTokenizer, max_seq_len: int = 512,
                 max_samples: int | None = None, corpus: str | None = None):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_samples = max_samples
        self.corpus = corpus or _DEFAULT_CORPUS

    def __iter__(self):
        tokens = self.tokenizer.encode(self.corpus)
        while len(tokens) < self.max_seq_len + 1:
            tokens = tokens + tokens

        count = 0
        while True:
            for i in range(0, len(tokens) - self.max_seq_len, max(1, self.max_seq_len // 2)):
                chunk = tokens[i:i + self.max_seq_len + 1]
                if len(chunk) < self.max_seq_len + 1:
                    continue
                count += 1
                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                yield {
                    'input_ids': input_ids,
                    'labels': input_ids.clone(),
                }
                if self.max_samples and count >= self.max_samples:
                    return


# ═══════════════════════════════════════════════════════════════
# Real data: pre-tokenized binary shards
# ═══════════════════════════════════════════════════════════════

class BinDataset(Dataset):
    """Memory-mapped dataset from pre-tokenized uint32 binary shard.

    Each shard is a flat array of uint32 token IDs. The dataset slices
    consecutive chunks of seq_len+1 tokens into (input, label) pairs.
    """

    def __init__(self, path: str, seq_len: int = 512):
        d = np.fromfile(path, dtype=np.uint32)
        self.data = torch.from_numpy(d.astype(np.int64))
        self.seq_len = seq_len
        self.n = max(0, (len(self.data) - 1) // seq_len)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        s = i * self.seq_len
        e = s + self.seq_len + 1
        chunk = self.data[s:e]
        input_ids = chunk[:-1].clone()
        return {
            'input_ids': input_ids,
            'labels': input_ids.clone(),
            'attention_mask': torch.ones(self.seq_len, dtype=torch.long),
        }


class MultiTierDataset(Dataset):
    """Concatenated view of multiple BinDataset shards with optional tier cycling.

    When tier_epochs is set, cycles through each tier for a fixed number of
    epochs before moving to the next (progressive data schedule).
    """

    def __init__(self, data_dir: str, seq_len: int = 512,
                 tier: str | None = None, split: str = 'train'):
        import glob
        if tier:
            paths = [os.path.join(data_dir, f'{tier}_{split}.bin')]
        else:
            pattern = f'*_{split}.bin'
            paths = sorted(glob.glob(os.path.join(data_dir, pattern)))

        self.datasets = []
        self.offsets = [0]
        for p in paths:
            if os.path.exists(p):
                ds = BinDataset(p, seq_len)
                ds.path = p  # store path for file-access audit
                self.datasets.append(ds)
                self.offsets.append(self.offsets[-1] + len(ds))

        self.total_len = self.offsets[-1]

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        # Find which dataset the index falls in
        for i in range(len(self.datasets)):
            if idx < self.offsets[i + 1]:
                return self.datasets[i][idx - self.offsets[i]]
        return self.datasets[-1][idx % len(self.datasets[-1])]


# ═══════════════════════════════════════════════════════════════
# Dataloader factory
# ═══════════════════════════════════════════════════════════════

def _detect_data_dir(data_dir: str) -> str | None:
    """Return data_dir if it contains .bin files, else None."""
    if data_dir and os.path.isdir(data_dir):
        import glob
        bins = glob.glob(os.path.join(data_dir, '*.bin'))
        if bins:
            return data_dir
    return None


def get_real_dataloader(data_dir: str, batch_size: int = 8,
                         seq_len: int = 512, max_steps: int = 1500,
                         tier: str | None = None,
                         split: str = 'train') -> DataLoader:
    """Create dataloader from pre-tokenized .bin shards."""
    dataset = MultiTierDataset(data_dir, seq_len, tier, split=split)
    # File-access audit (D-06): log matched file paths tagged with split
    matched_paths = [getattr(ds, 'path', 'unknown') for ds in dataset.datasets]
    print(f"[DATA] split={split} matched {len(matched_paths)} files: {matched_paths}")
    shuffle = (split == 'train')
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate_batch, num_workers=0,
                      pin_memory=True)


def get_offline_dataloader(batch_size: int = 8, max_seq_len: int = 512,
                            max_steps: int = 1500,
                            vocab_size: int = 32000) -> DataLoader:
    """Create offline dataloader using embedded corpus (fallback)."""
    tokenizer = CharTokenizer(vocab_size)
    dataset = LocalTextDataset(
        tokenizer, max_seq_len=max_seq_len,
        max_samples=max_steps * batch_size,
    )
    return DataLoader(dataset, batch_size=batch_size,
                      collate_fn=collate_batch, num_workers=0)


def get_dataloader(batch_size: int = 8, max_seq_len: int = 512,
                    max_steps: int = 1500, vocab_size: int = 32000,
                    data_dir: str | None = None,
                    tier: str | None = None,
                    split: str = 'train') -> DataLoader:
    """Create dataloader — auto-selects real data if available.

    Priority:
      1. data_dir with .bin shards → real data (BinDataset)
      2. Default data/real_tiers if it exists → real data
      3. Fallback → offline embedded corpus
    """
    # Try explicit data_dir first, then default
    effective_dir = _detect_data_dir(data_dir) if data_dir else None
    if effective_dir is None:
        effective_dir = _detect_data_dir('data/real_tiers')

    if effective_dir:
        return get_real_dataloader(effective_dir, batch_size,
                                    max_seq_len, max_steps, tier, split=split)

    # Offline fallback
    return get_offline_dataloader(batch_size, max_seq_len,
                                   max_steps, vocab_size)


# ═══════════════════════════════════════════════════════════════
# Tokenizer loading utility
# ═══════════════════════════════════════════════════════════════

def load_tokenizer(tokenizer_path: str = 'data/tokenizer/bpe_32k.json',
                   vocab_size: int = 32000):
    """Load BPE tokenizer. Falls back to CharTokenizer if unavailable."""
    if os.path.exists(tokenizer_path):
        from tokenizers import Tokenizer as HFTokenizer
        return HFTokenizer.from_file(tokenizer_path)
    print(f"[WARN] Tokenizer not found at {tokenizer_path}, "
          f"using CharTokenizer(vocab={vocab_size})")
    return CharTokenizer(vocab_size)


def load_special_tokens(tokenizer_dir: str = 'data/tokenizer') -> dict:
    """Load special token IDs from saved mapping, or return defaults."""
    import json
    map_path = os.path.join(tokenizer_dir, 'special_tokens.json')
    if os.path.exists(map_path):
        with open(map_path) as f:
            return json.load(f)
    return {
        'pad_token_id': 0, 'eos_token_id': 1,
        'unk_token_id': 2, 'bos_token_id': 3,
        'vocab_size': 32000,
    }


# ═══════════════════════════════════════════════════════════════
# Batch collation
# ═══════════════════════════════════════════════════════════════

def collate_batch(batch: list[dict]) -> dict:
    """Pad and stack variable-length sequences."""
    max_len = max(x['input_ids'].size(0) for x in batch)
    input_ids = torch.stack([
        torch.nn.functional.pad(x['input_ids'], (0, max_len - x['input_ids'].size(0)),
                                 value=0) for x in batch
    ])
    labels = torch.stack([
        torch.nn.functional.pad(x['labels'], (0, max_len - x['labels'].size(0)),
                                 value=-100) for x in batch
    ])
    attention_mask = (input_ids != 0).long()
    return {'input_ids': input_ids, 'labels': labels, 'attention_mask': attention_mask}


# ═══════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════

def train_epoch(model, dataloader, optimizer, device, max_steps: int = 1500,
                cond_reg_fn=None, log_interval: int = 50) -> list[dict]:
    """Train for one epoch, return per-step metrics."""
    model.train()
    metrics = []

    for step, batch in enumerate(dataloader):
        if step >= max_steps:
            break

        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        attention_mask = batch.get('attention_mask')
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        out = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = out['loss']

        if cond_reg_fn is not None:
            cond_loss = cond_reg_fn(model)
            loss = loss + cond_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % log_interval == 0:
            ppl = torch.exp(loss).item()
            metrics.append({
                'step': step,
                'loss': loss.item(),
                'perplexity': ppl,
            })

    return metrics


def evaluate_perplexity(model, dataloader, device, max_steps: int = 200) -> float:
    """Evaluate perplexity on a held-out set."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            if step >= max_steps:
                break
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch.get('attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            out = model(input_ids, attention_mask=attention_mask, labels=labels)

            shift_labels = labels[..., 1:].contiguous()
            valid_tokens = (shift_labels != -100).sum().item()
            total_loss += out['loss'].item() * valid_tokens
            total_tokens += valid_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    return math.exp(avg_loss)


def save_checkpoint(model, optimizer, metrics: list[dict],
                    path: str, config: dict | None = None):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': config,
    }, path)


def load_checkpoint(model, optimizer, path: str, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    return ckpt.get('metrics', []), ckpt.get('config', {})
