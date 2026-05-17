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


def process_tier(dataset_cfg, tokenizer, output_path, max_tokens, tier_name,
                 val_split=0.0):
    """Process one data tier in chunks, append-tokenize, save.

    When val_split > 0, splits the stream into *_train.bin and *_val.bin
    at the (1 - val_split) fraction of max_tokens. Returns total_tokens
    when no split, or (train_tokens, val_tokens) when splitting.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Compute split paths if val_split is active
    split_active = 0.0 < val_split < 1.0
    if split_active:
        base = output_path.replace('.bin', '')
        train_path = f'{base}_train.bin'
        val_path = f'{base}_val.bin'
        split_threshold = int(max_tokens * (1 - val_split))
        split_triggered = False
    else:
        train_path = output_path
        val_path = None

    # Load streaming dataset
    kwargs = {'split': dataset_cfg['split'], 'streaming': True}
    if dataset_cfg.get('subset'):
        kwargs['name'] = dataset_cfg['subset']

    print(f"  Loading {dataset_cfg['dataset']} (streaming)...")
    ds = load_dataset(dataset_cfg['dataset'], **kwargs)

    buffer = []
    total_tokens = 0
    train_tokens = 0
    val_tokens = 0
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
                if split_active:
                    # Handle split point crossing mid-buffer
                    if not split_triggered and total_tokens >= split_threshold:
                        tokens_before_buf = total_tokens - len(buffer)
                        split_buf_idx = split_threshold - tokens_before_buf
                        split_buf_idx = max(0, min(len(buffer), split_buf_idx))

                        if split_buf_idx > 0:
                            train_arr = np.array(buffer[:split_buf_idx], dtype=np.uint32)
                            mode = 'ab' if os.path.exists(train_path) else 'wb'
                            with open(train_path, mode) as f:
                                train_arr.tofile(f)
                            train_tokens += split_buf_idx

                        # Keep val portion in buffer
                        buffer = buffer[split_buf_idx:]
                        split_triggered = True
                        print(f"  Split point reached at {total_tokens/1e6:.1f}M tokens. "
                              f"Switching to validation file: {val_path}")

                # Determine output path
                if split_active and split_triggered:
                    out_path = val_path
                else:
                    out_path = train_path

                arr = np.array(buffer, dtype=np.uint32)
                mode = 'ab' if os.path.exists(out_path) else 'wb'
                with open(out_path, mode) as f:
                    arr.tofile(f)

                if split_active:
                    if split_triggered:
                        val_tokens += len(buffer)
                    else:
                        train_tokens += len(buffer)
                else:
                    train_tokens += len(buffer)

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
        if split_active and split_triggered:
            out_path = val_path
        else:
            out_path = train_path
        arr = np.array(buffer, dtype=np.uint32)
        mode = 'ab' if os.path.exists(out_path) else 'wb'
        with open(out_path, mode) as f:
            arr.tofile(f)

        if split_active:
            if split_triggered:
                val_tokens += len(buffer)
            else:
                train_tokens += len(buffer)
        else:
            train_tokens += len(buffer)

    if split_active:
        ok = 'PASS' if train_tokens + val_tokens == total_tokens else 'FAIL'
        print(f"  {ok}: train={train_tokens/1e6:.1f}M + "
              f"val={val_tokens/1e6:.1f}M = "
              f"{(train_tokens+val_tokens)/1e6:.1f}M tokens")
        return train_tokens, val_tokens
    return total_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='data/real_tiers')
    parser.add_argument('--tokenizer_path', default='data/tokenizer/bpe_32k.json')
    parser.add_argument('--max_tokens_per_tier', type=int, default=800_000_000)
    parser.add_argument('--tiers', nargs='+', default=None,
                        help='Specific tiers to process (e.g., tier1_c4 tier2_fineweb)')
    parser.add_argument('--val_split', type=float, default=0.0,
                        help='Validation split fraction (0.0 = no split). '
                             'Default 0.0 preserves backward compatibility.')
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

        result = process_tier(cfg, tokenizer, path,
                               args.max_tokens_per_tier, name,
                               val_split=args.val_split)
        elapsed = time.time() - t0

        if args.val_split > 0:
            train_n, val_n = result
            total_n = train_n + val_n
            stats[name] = total_n
            print(f"  Done: {train_n/1e6:.1f}M train + {val_n/1e6:.1f}M val "
                  f"= {total_n/1e6:.1f}M tokens in {elapsed/60:.0f}m "
                  f"({total_n/elapsed/1e3:.0f}k tok/s)")
        else:
            n_tokens = result
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
