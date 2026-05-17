#!/usr/bin/env python3
"""
Post-processing split utility for pre-tokenized .bin data tiers.

Reads existing tierN.bin flat uint32 arrays and writes isolated
tierN_train.bin (first 95%) and tierN_val.bin (last 5%) files.

Usage:
    python src/experiments/split_data.py
    python src/experiments/split_data.py --val-split 0.1 --delete-original
    python src/experiments/split_data.py --tiers tier1_c4 tier3_wiki
"""

import os, argparse
import numpy as np


def split_bin_file(src_path: str, dst_train: str, dst_val: str,
                   val_split: float = 0.05) -> dict:
    """Split a flat uint32 .bin file into train and val files.

    Args:
        src_path: Path to source .bin file (tierN.bin)
        dst_train: Output path for training split
        dst_val: Output path for validation split
        val_split: Fraction of tokens to assign to validation (default 0.05)

    Returns:
        dict with keys: source, total, train, val (token counts)
    """
    data = np.fromfile(src_path, dtype=np.uint32)
    n_tokens = len(data)

    if n_tokens == 0:
        return {'source': src_path, 'total': 0, 'train': 0, 'val': 0}

    split_idx = int(n_tokens * (1 - val_split))
    # Clamp to valid range: val must be at least 0, at most n_tokens
    split_idx = max(0, min(n_tokens, split_idx))

    # Write training split (always write, even if empty)
    train_data = data[:split_idx]
    train_data.tofile(dst_train)

    # Write validation split (only if non-empty; never write 0-byte file)
    if split_idx < n_tokens:
        data[split_idx:].tofile(dst_val)

    return {
        'source': src_path,
        'total': n_tokens,
        'train': split_idx,
        'val': n_tokens - split_idx,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Split pre-tokenized .bin data tiers into train/val files.'
    )
    parser.add_argument('--data-dir', default='data/real_tiers',
                        help='Directory containing .bin files (default: data/real_tiers)')
    parser.add_argument('--val-split', type=float, default=0.05,
                        help='Fraction of tokens for validation (default: 0.05)')
    parser.add_argument('--tiers', nargs='+',
                        default=['tier1_c4', 'tier2_fineweb', 'tier3_wiki', 'tier4_orca'],
                        help='Tier names to split (default: all 4 tiers)')
    parser.add_argument('--delete-original', action='store_true',
                        help='Delete original unsplit .bin files after successful split')
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    total_orig = 0
    total_train = 0
    total_val = 0
    all_ok = True

    for tier in args.tiers:
        src_path = os.path.join(args.data_dir, f'{tier}.bin')

        if not os.path.exists(src_path):
            print(f"[WARN] {src_path} not found, skipping tier '{tier}'")
            continue

        dst_train = os.path.join(args.data_dir, f'{tier}_train.bin')
        dst_val = os.path.join(args.data_dir, f'{tier}_val.bin')

        result = split_bin_file(src_path, dst_train, dst_val,
                                val_split=args.val_split)

        # Verify token count assertion
        train_tokens = np.fromfile(dst_train, dtype=np.uint32)
        val_tokens = np.fromfile(dst_val, dtype=np.uint32) if os.path.exists(dst_val) else np.array([], dtype=np.uint32)
        train_count = len(train_tokens)
        val_count = len(val_tokens)
        total_count = train_count + val_count
        assertion_ok = (total_count == result['total'])
        status = 'PASS' if assertion_ok else 'FAIL'
        if not assertion_ok:
            all_ok = False

        # Print formatted summary
        total_m = f'{result["total"]/1e6:.1f}M'
        train_m = f'{train_count/1e6:.1f}M'
        val_m = f'{val_count/1e6:.1f}M'
        ratio = f'{train_count / max(val_count, 1):.0f}/{1}' if val_count > 0 else 'N/A'
        print(f'{status:4s} {tier}: total={total_m} train={train_m} val={val_m} '
              f'ratio={ratio}')

        # Delete original if requested (only after successful verification)
        if args.delete_original and assertion_ok:
            os.remove(src_path)
            print(f'       Deleted original: {src_path}')

        total_orig += result['total']
        total_train += train_count
        total_val += val_count

    # Print grand total
    print()
    print(f'Summary: {total_train/1e6:.1f}M train + {total_val/1e6:.1f}M val '
          f'= {(total_train+total_val)/1e6:.1f}M tokens (original: {total_orig/1e6:.1f}M)')

    if not all_ok:
        print('[WARN] One or more splits failed token count assertion')
        exit(1)


if __name__ == '__main__':
    main()
