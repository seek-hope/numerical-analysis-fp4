#!/usr/bin/env python3
"""
Train a BPE tokenizer (vocab_size=32000) on open-source text corpora.

Streams text from C4, FineWeb, and Wikipedia via HuggingFace datasets,
trains a BPE tokenizer using the fast tokenizers library, and saves it.

Usage (run locally with network):
    python src/experiments/train_tokenizer.py --output_dir data/tokenizer

Memory note: the BPE trainer accumulates all yielded texts in memory before
training. Keep --max_samples moderate (50K-200K) to avoid OOM/bus errors.
50K diverse samples is sufficient for a 32K vocabulary.

Requires: pip install tokenizers datasets
"""

import os, argparse, json
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors
from datasets import load_dataset


SPECIAL_TOKENS = ["[PAD]", "[EOS]", "[UNK]", "[BOS]"]
MAX_TEXT_LEN = 4096  # Truncate individual texts to bound memory


def text_stream(datasets_config: list[dict], max_samples: int = 200_000):
    """Yield text samples from multiple streaming datasets.

    Truncates each text to MAX_TEXT_LEN to bound the trainer's memory usage.
    Skips empty/short texts and handles malformed examples gracefully.
    """
    count = 0
    for cfg in datasets_config:
        print(f"  Streaming {cfg['name']}...")
        kwargs = {'split': cfg['split'], 'streaming': True}
        if cfg.get('subset'):
            kwargs['name'] = cfg['subset']

        try:
            ds = load_dataset(cfg['dataset'], **kwargs)
        except Exception as e:
            print(f"    WARN: failed to load {cfg['dataset']}: {e}")
            continue

        for example in ds:
            if count >= max_samples:
                return
            try:
                text = example.get('text') or example.get('question') or ''
                text = str(text).strip()
                if len(text) < 100:
                    continue
                # Truncate to bound memory during BPE training
                text = text[:MAX_TEXT_LEN]
                count += 1
                yield text
            except Exception:
                continue  # Skip malformed examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='data/tokenizer')
    parser.add_argument('--vocab_size', type=int, default=32000)
    parser.add_argument('--max_samples', type=int, default=200_000,
                        help='Total texts to sample (fewer = faster/less RAM, '
                             '50K-100K is sufficient for 32K vocab)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Data sources ordered by diversity/quality
    sources = [
        {'name': 'C4', 'dataset': 'allenai/c4', 'subset': 'en',
         'split': 'train'},
        {'name': 'FineWeb-edu', 'dataset': 'HuggingFaceFW/fineweb-edu',
         'subset': 'sample-10BT', 'split': 'train'},
        {'name': 'Wikipedia', 'dataset': 'wikimedia/wikipedia',
         'subset': '20231101.en', 'split': 'train'},
    ]

    # Initialize BPE tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
        show_progress=True,
    )

    print(f"Training BPE tokenizer (vocab={args.vocab_size}, "
          f"max_samples={args.max_samples:,})...")
    tokenizer.train_from_iterator(
        text_stream(sources, args.max_samples),
        trainer=trainer,
    )

    print(f"Trained vocab size: {tokenizer.get_vocab_size()}")

    # Post-processor: add BOS/EOS
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[BOS] $A [EOS]",
        pair="[BOS] $A [EOS] $B:1 [EOS]:1",
        special_tokens=[
            ("[BOS]", tokenizer.token_to_id("[BOS]")),
            ("[EOS]", tokenizer.token_to_id("[EOS]")),
        ],
    )

    # Save
    path = os.path.join(args.output_dir, 'bpe_32k.json')
    tokenizer.save(path)
    print(f"Tokenizer saved to {path}")

    # Verify
    test_text = "Machine learning enables computers to learn from data."
    encoded = tokenizer.encode(test_text)
    print(f"\nTest: '{test_text}'")
    print(f"  Token IDs: {encoded.ids}")
    print(f"  Tokens: {encoded.tokens}")
    print(f"  Decoded:  {tokenizer.decode(encoded.ids)}")

    # Save special token mapping
    special_map = {
        'pad_token_id': tokenizer.token_to_id("[PAD]"),
        'eos_token_id': tokenizer.token_to_id("[EOS]"),
        'unk_token_id': tokenizer.token_to_id("[UNK]"),
        'bos_token_id': tokenizer.token_to_id("[BOS]"),
        'pad_token': '[PAD]',
        'eos_token': '[EOS]',
        'unk_token': '[UNK]',
        'bos_token': '[BOS]',
        'vocab_size': tokenizer.get_vocab_size(),
    }
    map_path = os.path.join(args.output_dir, 'special_tokens.json')
    with open(map_path, 'w') as f:
        json.dump(special_map, f, indent=2)
    print(f"Token mapping saved to {map_path}")


if __name__ == '__main__':
    main()
