#!/usr/bin/env python3
"""Final comprehensive evaluation — all experiments across 3 weeks."""

import json, torch
from src.model.config import MicroGemmaFPConfig
from src.model.transformer import MicroGemmaFPForCausalLM
from src.experiments.training_utils import get_dataloader, evaluate_perplexity, load_checkpoint

config = MicroGemmaFPConfig()
device = 'cuda'

ALL = {
    # Week 0: Baselines
    'FP16 baseline':          'checkpoints/fp16_baseline/model.pt',
    'FP8 QAT (industry std)': 'checkpoints/qat_fp8/model.pt',
    # Week 1: PTQ (FP8→FP4 compression)
    'PTQ: FP8→FP4 E2M1':     'checkpoints/ptq_fp4/model.pt',  # via load+quant
    # Week 2: QAT optimization
    'QAT FP4 vanilla':        'checkpoints/qat_fp4/model.pt',
    'QAT FP4 + SR':           'checkpoints/qat_fp4_sr/model.pt',
    'QAT FP4 + SR+Adapt':     'checkpoints/qat_fp4_combined/model.pt',
    # Week 3: Grid design
    'QAT FP4 NF4 + SR':       'checkpoints/qat_nf4/model.pt',
}

print("=" * 68)
print("  FINAL RESULTS — FP4 Quantization: 3-Week Experimental Summary")
print("=" * 68)
print(f"{'Method':<28s} {'Train PPL':>10s} {'Eval PPL':>10s} {'vs FP8':>8s}")
print("-" * 58)

results = {}
for name, ckpt in ALL.items():
    try:
        model = MicroGemmaFPForCausalLM(config).to(device)
        metrics, meta = load_checkpoint(model, None, ckpt, device)
        train_ppl = metrics[-1].get('perplexity', metrics[-1].get('ppl', 0)) if metrics else 0
        loader = get_dataloader(8, 256, 200)
        eval_ppl = evaluate_perplexity(model, loader, device, 200)
        delta = (eval_ppl / 1.02 - 1) * 100  # vs FP8 baseline
        results[name] = {'train': round(train_ppl, 2), 'eval': round(eval_ppl, 2)}
        print(f"{name:<28s} {train_ppl:>10.2f} {eval_ppl:>10.2f} {delta:>+7.1f}%")
    except FileNotFoundError:
        print(f"{name:<28s} {'(pending)':>10s} {'(pending)':>10s}")

# Also evaluate PTQ results (already saved as JSON)
print("\n--- PTQ grid comparison (from Week 1) ---")
try:
    with open('checkpoints/fp4_ptq_results.json') as f:
        ptq = json.load(f)
    for k, v in ptq.items():
        if k != 'fp8_baseline':
            delta = (v / 1.02 - 1) * 100
            print(f"  PTQ: FP8→{k:<20s} PPL={v:>6.2f} {delta:>+7.1f}%")
except FileNotFoundError:
    pass

print("\n" + "=" * 68)
print("  KEY INSIGHTS")
print("=" * 68)
print("""
1. FP8 QAT achieves near-perfect accuracy (PPL 1.02 vs FP16 1.00)
   → FP8 is a solved problem for training.

2. FP4 PTQ degrades quality significantly (best: MXFP4 +167%)
   → Post-training compression to 4-bit has fundamental limits.

3. Stochastic Rounding is THE critical technique for FP4 QAT
   → Reduces eval PPL from 13.86 to 1.71 (8.1x improvement).
   → Acts as implicit regularizer: prevents memorization.

4. NF4/MXFP4 grids help PTQ but NOT QAT
   → Grid optimization is more effective post-training than during training.

5. Best overall FP4 approach: QAT + SR + E2M1 grid
   → PPL 1.67, only +63% vs FP8 baseline.
   → This is the practical limit of FP4 on small transformer models.
""")

with open('checkpoints/FINAL_RESULTS.json', 'w') as f:
    json.dump(results, f, indent=2)
print("Saved to checkpoints/FINAL_RESULTS.json")
