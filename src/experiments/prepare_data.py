#!/usr/bin/env python3
"""
Download and prepare 4-tier quality data for ~164M model training.

Tier 1 (D1): C4-en raw        ~50B tokens -> sample ~800M
Tier 2 (D2): FineWeb-edu       ~10B tokens -> sample ~800M
Tier 3 (D3): Wikipedia-en       ~2B tokens -> sample ~800M
Tier 4 (D4): OpenOrca           ~200M tokens -> sample ~800M

Total target: ~3.2B tokens (~20× tokens/param for 164M model).

Each tier is tokenized with the BPE tokenizer (vocab_size=32000) and saved
as uint32 .bin shards.

Usage (run locally with network access):
    # First train the tokenizer:
    python src/experiments/train_tokenizer.py

    # Then prepare data:
    python src/experiments/prepare_data.py --output_dir data/real_tiers

Uses the BPE tokenizer trained by train_tokenizer.py.
Falls back to Gemma 4 tokenizer for the old gemma4_tiers format.
"""

import os, argparse, json
import numpy as np
from datasets import load_dataset


TIERS = {
    'tier1_c4': {
        'dataset': 'allenai/c4', 'subset': 'en', 'split': 'train',
        'max_samples': 8_000_000, 'desc': 'C4 raw (lowest quality, largest)',
    },
    'tier2_fineweb': {
        'dataset': 'HuggingFaceFW/fineweb-edu', 'subset': 'sample-10BT',
        'split': 'train', 'max_samples': 4_000_000,
        'desc': 'FineWeb-edu (medium quality)',
    },
    'tier3_wiki': {
        'dataset': 'wikimedia/wikipedia', 'subset': '20231101.en',
        'split': 'train', 'max_samples': 2_000_000,
        'desc': 'Wikipedia (high quality, factual)',
    },
    'tier4_orca': {
        'dataset': 'Open-Orca/OpenOrca', 'subset': None, 'split': 'train',
        'max_samples': 800_000, 'text_field': 'question',
        'desc': 'OpenOrca (highest quality, instruction data)',
    },
}


def load_bpe_tokenizer(tokenizer_path: str):
    """Load BPE tokenizer from file, or fall back to Gemma 4."""
    if os.path.exists(tokenizer_path):
        from tokenizers import Tokenizer as HFTokenizer
        print(f"  Using BPE tokenizer: {tokenizer_path}")
        return HFTokenizer.from_file(tokenizer_path)

    # Fallback to Gemma 4 tokenizer (for backward compatibility)
    from transformers import AutoTokenizer
    print(f"  BPE tokenizer not found, falling back to Gemma 4 tokenizer")
    tok = AutoTokenizer.from_pretrained('google/gemma-4-E2B',
                                         trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def tokenize_and_save(dataset, tokenizer, output_path, max_seq_len=2048,
                      max_tokens=200_000_000):
    """Tokenize dataset and save as uint32 binary shards."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    buffer = []
    total = 0

    # Detect tokenizer type
    is_hf_tokenizer = hasattr(tokenizer, 'encode') and not hasattr(tokenizer, 'token_to_id')

    for example in dataset:
        text = example.get('text') or example.get('question') or example.get('response') or ''
        if not text or len(str(text)) < 50:
            continue
        text = str(text)

        if is_hf_tokenizer:
            # HuggingFace transformers tokenizer
            tokens = tokenizer.encode(text, truncation=True, max_length=max_seq_len)
        else:
            # HF tokenizers library Tokenizer
            enc = tokenizer.encode(text)
            tokens = enc.ids[:max_seq_len]

        buffer.extend(tokens)
        total += len(tokens)
        if total >= max_tokens:
            break

    arr = np.array(buffer, dtype=np.uint32)
    arr.tofile(output_path)
    print(f"  Saved {len(arr):,} tokens to {output_path}")
    return len(arr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='data/real_tiers')
    parser.add_argument('--tokenizer_path', default='data/tokenizer/bpe_32k.json')
    parser.add_argument('--max_tokens_per_tier', type=int, default=800_000_000)
    parser.add_argument('--name_suffix', default='',
                        help='Suffix for tier names (e.g., "_v2")')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer = load_bpe_tokenizer(args.tokenizer_path)

    stats = {}
    for name, cfg in TIERS.items():
        print(f"\n{'='*50}")
        print(f"{name}: {cfg['desc']}")
        print(f"  Loading {cfg['dataset']} (max {cfg['max_samples']:,} samples)...")

        kwargs = {'split': cfg['split'], 'streaming': True}
        if cfg.get('subset'):
            kwargs['name'] = cfg['subset']
        ds = load_dataset(cfg['dataset'], **kwargs)

        # Apply max_samples limit
        if cfg['max_samples']:
            ds = ds.take(cfg['max_samples'])

        path = os.path.join(args.output_dir, f'{name}{args.name_suffix}.bin')
        n_tokens = tokenize_and_save(
            ds, tokenizer, path, max_tokens=args.max_tokens_per_tier
        )
        stats[name] = n_tokens

    print(f"\n{'='*50}")
    print("Summary:")
    total = 0
    for name, n in stats.items():
        print(f"  {name}: {n:,} tokens")
        total += n
    print(f"  Total: {total:,} tokens")


if __name__ == '__main__':
    main()
