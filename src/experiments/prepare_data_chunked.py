#!/usr/bin/env python3
"""
Lightweight data preparation — processes one chunk at a time to avoid OOM.

Instead of loading all samples into memory, this:
  1. Streams from HF datasets in small windows (10K samples at a time)
  2. Tokenizes immediately and appends to disk
  3. Frees memory between windows
  4. Shows progress with estimated completion time

Usage (run locally with network):
    python src/experiments/prepare_data_chunked.py \\
        --output_dir data/real_tiers \\
        --max_tokens_per_tier 800000000

This is designed for machines with limited RAM (8-16 GB).
Each chunk uses ~200-500 MB during processing.
"""

import os, argparse, time, sys
import numpy as np
from datasets import load_dataset

# Per-tier configuration
TIERS = {
    'tier1_c4': {
        'dataset': 'allenai/c4', 'subset': 'en', 'split': 'train',
        'desc': 'C4 raw',
    },
    'tier2_fineweb': {
        'dataset': 'HuggingFaceFW/fineweb-edu', 'subset': 'sample-10BT',
        'split': 'train', 'desc': 'FineWeb-edu',
    },
    'tier3_wiki': {
        'dataset': 'wikimedia/wikipedia', 'subset': '20231101.en',
        'split': 'train', 'desc': 'Wikipedia',
    },
    'tier4_orca': {
        'dataset': 'Open-Orca/OpenOrca', 'subset': None, 'split': 'train',
        'text_field': 'question', 'desc': 'OpenOrca',
    },
}

CHUNK_SIZE = 10000  # Process 10K samples at a time
MAX_SEQ_LEN = 2048


def load_bpe_tokenizer(tokenizer_path: str):
    """Load BPE tokenizer from file, or fall back to CharTokenizer."""
    if os.path.exists(tokenizer_path):
        from tokenizers import Tokenizer as HFTokenizer
        return HFTokenizer.from_file(tokenizer_path)
    else:
        print(f"  ERROR: Tokenizer not found at {tokenizer_path}")
        print(f"  Run first: python src/experiments/train_tokenizer.py")
        sys.exit(1)


def process_tier(dataset_cfg, tokenizer, output_path, max_tokens, tier_name):
    """Process one data tier in chunks, append-tokenize, save."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load streaming dataset
    kwargs = {'split': dataset_cfg['split'], 'streaming': True}
    if dataset_cfg.get('subset'):
        kwargs['name'] = dataset_cfg['subset']

    print(f"  Loading {dataset_cfg['dataset']} (streaming)...")
    ds = load_dataset(dataset_cfg['dataset'], **kwargs)

    buffer = []
    total_tokens = 0
    samples_processed = 0
    chunk_start_time = time.time()

    try:
        for example in ds:
            # Extract text
            text = (example.get('text') or
                    example.get('question') or
                    example.get('response') or '')
            text = str(text).strip()

            if len(text) < 100:
                continue

            # Truncate long texts
            text = text[:4096]

            # Tokenize
            enc = tokenizer.encode(text)
            tokens = enc.ids[:MAX_SEQ_LEN]

            buffer.extend(tokens)
            total_tokens += len(tokens)
            samples_processed += 1

            # Flush to disk every CHUNK_SIZE samples
            if samples_processed % CHUNK_SIZE == 0:
                arr = np.array(buffer, dtype=np.uint32)
                # Append mode after first chunk
                mode = 'ab' if os.path.exists(output_path) else 'wb'
                with open(output_path, mode) as f:
                    arr.tofile(f)

                elapsed = time.time() - chunk_start_time
                tokens_per_sec = total_tokens / max(elapsed, 1)
                eta_sec = (max_tokens - total_tokens) / max(tokens_per_sec, 1)
                print(f"    {samples_processed:>8,} samples, "
                      f"{total_tokens/1e6:>6.1f}M tokens "
                      f"({tokens_per_sec/1e3:.0f}k tok/s, "
                      f"ETA: {eta_sec/60:.0f}m)")

                buffer = []
                chunk_start_time = time.time()

            # Check if we've reached the target
            if total_tokens >= max_tokens:
                break

    except KeyboardInterrupt:
        print(f"\n  Interrupted. Saving {len(buffer):,} buffered tokens...")

    # Flush remaining buffer
    if buffer:
        arr = np.array(buffer, dtype=np.uint32)
        mode = 'ab' if os.path.exists(output_path) else 'wb'
        with open(output_path, mode) as f:
            arr.tofile(f)

    return total_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='data/real_tiers')
    parser.add_argument('--tokenizer_path', default='data/tokenizer/bpe_32k.json')
    parser.add_argument('--max_tokens_per_tier', type=int, default=800_000_000)
    parser.add_argument('--tiers', nargs='+', default=None,
                        help='Specific tiers to process (e.g., tier1_c4 tier2_fineweb)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer = load_bpe_tokenizer(args.tokenizer_path)
    print(f"Tokenizer: {args.tokenizer_path}")
    print(f"Target: {args.max_tokens_per_tier:,} tokens/tier "
          f"({args.max_tokens_per_tier*4/1e9:.1f}B total)\n")

    tiers_to_process = args.tiers or list(TIERS.keys())
    stats = {}

    for name in tiers_to_process:
        cfg = TIERS[name]
        print(f"{'='*55}")
        print(f"{name}: {cfg['desc']}")
        print(f"  Source: {cfg['dataset']}")
        print(f"  Target: {args.max_tokens_per_tier/1e6:.0f}M tokens")

        path = os.path.join(args.output_dir, f'{name}.bin')
        t0 = time.time()

        n_tokens = process_tier(cfg, tokenizer, path,
                                 args.max_tokens_per_tier, name)
        elapsed = time.time() - t0
        stats[name] = n_tokens
        print(f"  Done: {n_tokens/1e6:.1f}M tokens in {elapsed/60:.0f}m "
              f"({n_tokens/elapsed/1e3:.0f}k tok/s)")

    print(f"\n{'='*55}")
    print("Summary:")
    total = 0
    for name, n in stats.items():
        print(f"  {name}: {n/1e6:.1f}M tokens ({n*4/1e9:.2f} GB)")
        total += n
    print(f"  Total: {total/1e9:.2f}B tokens ({total*4/1e9:.2f} GB)")
    print(f"  Token/param ratio: {total/164000000:.1f}x (for 164M model)")


if __name__ == '__main__':
    main()
