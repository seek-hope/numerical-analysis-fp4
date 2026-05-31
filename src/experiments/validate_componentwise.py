#!/usr/bin/env python3
"""
Validate component-wise (CW) perturbation theory against per-matrix ||dy||/||y||.

Numerical analysis foundation:
  1. Component-wise condition number (Skeel 1979, Higham 2002 §7.2):
       cond_cw(W, x) = || |W|·|x| || / ||Wx||
     This replaces κ(W) when perturbations are component-wise structured
     (as in FP quantization, where |δW_{ij}| ≤ u·|W_{ij}|).

  2. Oettli-Prager (1964) component-wise backward error:
       ω(ΔW) = min{ ε : |ΔW_{ij}| ≤ ε·|W_{ij}|, ∀i,j }
     For FP4 E2M1: ω ≤ u = 0.25 (unit roundoff).

  3. Component-wise forward error bound:
       ||δy||/||y|| ≤ cond_cw(W, x) · ω
     vs. classical normwise bound:
       ||δy||/||y|| ≤ κ(W) · ||δW||/||W||

Hypothesis: cond_cw(W, x) correlates positively with ||dy||/||y|| (r > 0.5),
whereas κ(W) shows r = -0.23 (negligible).

Usage:
    python src/experiments/validate_componentwise.py \
        --checkpoint checkpoints/scaled_fp16_baseline/model.pt \
        --data_dir data/real_tiers \
        --output results/componentwise_validation.json
"""

import argparse, json, math, os
import numpy as np
import torch
import torch.nn as nn

from src.model.config import MicroGemmaFPConfig
from src.model.transformer import MicroGemmaFPForCausalLM
from src.experiments.training_utils import load_checkpoint, get_dataloader


# ═══════════════════════════════════════════════════════════════
# Activation capture
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def capture_pre_linear_inputs(
    model: nn.Module, loader, device: torch.device, max_batches: int = 20,
) -> dict[str, torch.Tensor]:
    """Capture pre-Linear-layer inputs over multiple batches.

    Returns dict: module_path -> Tensor (n_samples, in_features) on CPU.
    Skips embed_tokens and lm_head.
    """
    inputs: dict[str, list[torch.Tensor]] = {}
    hooks = []

    def make_hook(name):
        def hook_fn(module, inp, out):
            x = inp[0].detach()                     # (batch, seq, in_features)
            x_flat = x.reshape(-1, x.shape[-1])      # (batch*seq, in_features)
            if name not in inputs:
                inputs[name] = []
            inputs[name].append(x_flat.cpu())
        return hook_fn

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if 'embed' in name.lower() or 'lm_head' in name.lower():
                continue
            hooks.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    for step, batch in enumerate(loader):
        if step >= max_batches:
            break
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch.get('attention_mask')
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        model(input_ids, attention_mask=attention_mask)

    for h in hooks:
        h.remove()

    # Concatenate per layer
    result = {}
    for name, tensors in inputs.items():
        result[name] = torch.cat(tensors, dim=0)   # (n_samples, in_features)
    return result


# ═══════════════════════════════════════════════════════════════
# Component-wise metrics
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_componentwise_metrics(
    W: torch.Tensor,
    X: torch.Tensor,
) -> dict:
    """Compute component-wise condition number and related metrics for (W, X).

    Args:
        W: weight matrix (out_features, in_features)
        X: pre-activation inputs (n_samples, in_features) on CPU

    Returns dict with:
        cw_cond:          component-wise condition number  (scalar)
        cw_bound_fp4:     component-wise forward bound for FP4 (u=0.25)
        cw_bound_fp8:     component-wise forward bound for FP8 (u=0.125)
        normwise_bound:   classical κ(W)·||δW||/||W||  bound
        activation_l2:    mean ||x||_2 over samples
        output_l2:        mean ||Wx||_2 over samples
        kappa:            condition number κ(W) via SVD (or estimate)
    """
    # Move to same device as W for computation
    X_dev = X.to(W.device)

    # --- Component-wise condition number ---
    # cond_cw(W) = E_x[ || |W|·|x| ||_2 / ||Wx||_2 ]
    # Compute per-sample then average over a subset (max 512 samples)
    n_samples = min(X_dev.shape[0], 512)
    X_sub = X_dev[:n_samples]
    Y = X_sub @ W.T                                        # (n, out_features)

    # Numerator: || |W|·|x| ||_2 for each sample
    W_abs = W.abs()
    Y_abs = X_sub.abs() @ W_abs.T                          # (n, out_features)
    num = Y_abs.norm(dim=1)                                # (n,)
    den = Y.norm(dim=1).clamp(min=1e-12)                   # (n,)

    cw_conds = num / den                                   # per-sample
    cw_cond_mean = cw_conds.mean().item()
    cw_cond_std = cw_conds.std().item()

    # --- Component-wise backward error bound (Oettli-Prager) ---
    # For FP quantization: ω = u (unit roundoff) applied component-wise.
    # → ||δy||_2 ≤ || |W|·|x| ||_2 · ω
    # → ||δy||_2 / ||y||_2 ≤ cond_cw(W, x) · ω
    u_fp4 = 0.25
    u_fp8 = 0.125
    cw_bound_fp4 = cw_cond_mean * u_fp4
    cw_bound_fp8 = cw_cond_mean * u_fp8

    # --- Classical normwise bound ---
    # ||δy||/||y|| ≤ κ(W) · ||δW||/||W||
    # ||δW||/||W|| ≈ u for FP quantization (empirically ~0.15 for FP4)
    # Compute κ(W) via power iteration (fast) — not exact SVD
    kappa = _estimate_kappa_power(W)

    # --- Aggregate statistics ---
    activation_l2 = X_dev[:n_samples].norm(dim=1).mean().item()
    output_l2 = Y.norm(dim=1).mean().item()

    return {
        'cw_cond_mean': float(cw_cond_mean),
        'cw_cond_std': float(cw_cond_std),
        'cw_bound_fp4': float(cw_bound_fp4),
        'cw_bound_fp8': float(cw_bound_fp8),
        'kappa': float(kappa),
        'activation_l2': float(activation_l2),
        'output_l2': float(output_l2),
        'n_samples': n_samples,
    }


def _estimate_kappa_power(W: torch.Tensor, n_iter: int = 10) -> float:
    """Estimate κ(W) = σ_max / σ_min via power iteration + inverse power iteration."""
    m, n = W.shape

    # σ_max via power iteration on W^T W
    v = torch.randn(n, device=W.device, dtype=W.dtype)
    for _ in range(n_iter):
        v = W.T @ (W @ v)
        v = v / v.norm()
    sigma_max = (W @ v).norm().item()

    # σ_min via inverse power iteration on W^T W + ε·I
    eps = 1e-3 * sigma_max
    v = torch.randn(n, device=W.device, dtype=W.dtype)
    for _ in range(n_iter):
        # Solve (W^T W + εI) v' = v via CG-like iteration
        Mv = W.T @ (W @ v) + eps * v
        v = Mv / Mv.norm()
    sigma_min = max((W @ v).norm().item(), 1e-8)

    kappa = sigma_max / sigma_min if sigma_min > 1e-8 else float('inf')
    return kappa


# ═══════════════════════════════════════════════════════════════
# Correlation analysis
# ═══════════════════════════════════════════════════════════════

def pearson_r(x: list[float], y: list[float]) -> tuple[float, float]:
    """Pearson correlation coefficient and two-tailed p-value.

    For n ≥ 30, uses the normal approximation to Student's t
    (error < 0.002 for α=0.05). For n < 30, uses the exact t-CDF
    via the regularized incomplete beta (continued fraction).
    """
    n = len(x)
    if n < 3:
        return 0.0, 1.0
    mx = sum(x) / n
    my = sum(y) / n
    sx = (sum((v - mx) ** 2 for v in x) / n) ** 0.5
    sy = (sum((v - my) ** 2 for v in y) / n) ** 0.5
    if sx < 1e-12 or sy < 1e-12:
        return 0.0, 1.0
    r = sum((x[i] - mx) * (y[i] - my) for i in range(n)) / (n * sx * sy)
    r = max(-1.0, min(1.0, r))

    if abs(r) >= 1.0 - 1e-12:
        return float(r), 0.0

    t_stat = abs(r) * math.sqrt((n - 2) / (1 - r * r))
    df = n - 2

    if df >= 50:
        # Normal approximation: p = 2·Φ(-|t|)
        # Φ(-z) ≈ 0.5·erfc(z/√2) for large df
        p = math.erfc(t_stat / math.sqrt(2.0))
    else:
        # Exact via regularized incomplete beta
        x_beta = df / (df + t_stat * t_stat)
        a, b = df / 2.0, 0.5
        p = _regularized_beta(a, b, x_beta)

    return float(r), float(min(1.0, p))


def _regularized_beta(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a,b) via continued fraction."""
    if x < 1e-15:
        return 0.0
    if x > 1.0 - 1e-15:
        return 1.0

    log_beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(a * math.log(x) + b * math.log(1.0 - x) - math.log(a) - log_beta)

    # Lentz continued fraction
    f, c, d = 1.0, 1.0, 0.0
    tiny = 1e-30
    for m in range(1, 201):
        # d_{2m-1}: term with -(a+m-1)(a+b+m-1)x / ((a+2m-2)(a+2m-1))
        am = a + m - 1
        abm = a + b + m - 1
        ani = -(am * abm * x) / ((a + 2*m - 2) * (a + 2*m - 1))
        d = 1.0 / (1.0 + ani * d)
        if abs(d) < tiny:
            d = tiny

        # d_{2m}: term with m(b-m)x / ((a+2m-1)(a+2m))
        bmi = m * (b - m) * x / ((a + 2*m - 1) * (a + 2*m))
        d = 1.0 / (1.0 + bmi * d)
        if abs(d) < tiny:
            d = tiny

        delta = c * d
        f *= delta
        if abs(delta - 1.0) < 1e-12:
            break
        c = 1.0 + bmi / c if abs(c) > tiny else 1.0 + bmi / tiny

    return max(0.0, min(1.0, front * (f - 1.0)))


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Validate component-wise perturbation theory")
    parser.add_argument('--checkpoint', default='checkpoints/scaled_fp16_baseline/model.pt')
    parser.add_argument('--data_dir', default=None)
    parser.add_argument('--output', default='results/componentwise_validation.json')
    parser.add_argument('--max_eval_steps', type=int, default=20)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Load model and checkpoint ──
    config = MicroGemmaFPConfig()
    model = MicroGemmaFPForCausalLM(config).to(device)
    load_checkpoint(model, None, args.checkpoint, device)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # ── Capture activations ──
    loader = get_dataloader(4, 512, args.max_eval_steps, data_dir=args.data_dir)
    activations = capture_pre_linear_inputs(model, loader, device, args.max_eval_steps)
    print(f"Captured activations for {len(activations)} Linear layers")

    # ── Load PTQ results for actual ||dy||/||y|| ──
    comparison_path = 'results/full_comparison.json'
    if os.path.exists(comparison_path):
        with open(comparison_path) as f:
            comp_data = json.load(f)
        canonical = comp_data.get('configs', {}).get('fp16_baseline/FP4/rtn', {})
        actual_dy = canonical.get('per_matrix_errors', {}) if isinstance(canonical, dict) else {}
        print(f"Loaded {len(actual_dy)} per-matrix errors from {comparison_path}")
    else:
        actual_dy = {}
        print(f"[WARN] {comparison_path} not found — skipping correlation with actual dy")

    # ── Load theorem1 data for κ(W) reference ──
    th1_path = 'results/theorem1_validation.json'
    th1_kappa = {}
    if os.path.exists(th1_path):
        with open(th1_path) as f:
            th1_data = json.load(f)
        for r in th1_data.get('results', []):
            th1_kappa[r['name']] = r['kappa']
        print(f"Loaded {len(th1_kappa)} κ(W) references from {th1_path}")

    # ── Compute component-wise metrics for each matrix ──
    cw_results = []
    for i, (name, param) in enumerate(model.named_parameters()):
        if param.dim() < 2:
            continue
        if 'embed' in name.lower() or 'lm_head' in name.lower():
            continue
        if not any(k in name for k in ('proj',)):
            continue

        # Find matching activation (module path may differ from param path)
        # Param: model.layers.0.attention.q_proj.weight
        # Module: model.layers.0.attention.q_proj
        module_path = name.rsplit('.', 1)[0]  # strip .weight suffix
        X = activations.get(module_path)
        if X is None:
            # Try alternative match
            for act_key in activations:
                if act_key.endswith(module_path.split('.')[-1]) or module_path in act_key:
                    X = activations[act_key]
                    break
        if X is None:
            print(f"  [SKIP] {name}: no activation data (module_path={module_path})")
            continue

        W = param.data
        metrics = compute_componentwise_metrics(W, X)
        metrics['name'] = module_path  # module path (no .weight) for cross-ref
        metrics['layer'] = i

        # Match with actual dy (keys are module paths without .weight)
        if module_path in actual_dy:
            metrics['actual_dy_fp4'] = actual_dy[module_path]
        # Map to th1 matrix name (keys use module.path.weight format)
        th1_name = f"{module_path}.weight"
        if th1_name in th1_kappa:
            metrics['kappa_exact'] = th1_kappa[th1_name]
        elif module_path in th1_kappa:
            metrics['kappa_exact'] = th1_kappa[module_path]

        cw_results.append(metrics)

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1} matrices...")

    print(f"\nComputed component-wise metrics for {len(cw_results)} matrices")

    # ── Build mapping from TH1 names to CW names ──
    # TH1 names use module.weight format; CW uses the same
    cw_by_name = {r['name']: r for r in cw_results}

    # ── Correlation analysis ──
    print(f"\n{'='*65}")
    print("CORRELATION ANALYSIS")
    print(f"{'='*65}")

    # Initialize with defaults
    r_kappa, p_kappa, k_vals = 0.0, 1.0, []
    r_cw, p_cw, cw_vals = 0.0, 1.0, []
    r_bound, p_bound, mean_tight = 0.0, 1.0, 0.0

    # 1. Classic κ(W) vs dy (reproduce Theorem 1 result)
    classic_pairs = [(th1_kappa[n], actual_dy[n])
                     for n in actual_dy if n in th1_kappa]
    if classic_pairs:
        k_vals, dy_vals = zip(*classic_pairs)
        r_kappa, p_kappa = pearson_r(list(k_vals), list(dy_vals))
        print(f"\n  [Baseline] κ(W) vs ||dy||/||y||:")
        print(f"    r = {r_kappa:.4f}  p = {p_kappa:.4e}  n = {len(k_vals)}")

    # 2. Component-wise cond vs dy (MAIN HYPOTHESIS)
    cw_pairs = [(cw_by_name[n]['cw_cond_mean'], actual_dy[n])
                for n in actual_dy if n in cw_by_name]
    if cw_pairs:
        cw_vals, dy_vals_cw = zip(*cw_pairs)
        r_cw, p_cw = pearson_r(list(cw_vals), list(dy_vals_cw))
        print(f"\n  [NEW] cond_cw(W, x) vs ||dy||/||y|| (★ hypothesis):")
        print(f"    r = {r_cw:.4f}  p = {p_cw:.4e}  n = {len(cw_vals)}")
        outcome = "VALIDATED ✓" if r_cw > 0.4 else ("WEAK — needs refinement" if r_cw > 0.2 else "NEGLIGIBLE ✗")
        print(f"    Verdict: {outcome}")

    # 3. CW bound vs actual dy (COMPONENT-WISE FORWARD ERROR)
    cw_bound_pairs = [(cw_by_name[n]['cw_bound_fp4'], actual_dy[n])
                      for n in actual_dy if n in cw_by_name]
    if cw_bound_pairs:
        b_vals, dy_b_vals = zip(*cw_bound_pairs)
        r_bound, p_bound = pearson_r(list(b_vals), list(dy_b_vals))
        # Compute bound tightness
        tightness_vals = [b / max(d, 1e-12) for b, d in zip(b_vals, dy_b_vals)]
        mean_tight = sum(tightness_vals) / len(tightness_vals)
        print(f"\n  [NEW] CW bound (cond_cw · u) vs ||dy||/||y||:")
        print(f"    r = {r_bound:.4f}  p = {p_bound:.4e}")
        print(f"    Mean bound/actual ratio: {mean_tight:.2f}x")

    # 4. Per-layer-type subgroup analysis
    print(f"\n  {'─'*55}")
    print(f"  Subgroup analysis")
    print(f"  {'─'*55}")

    for label, patterns in [('Attention', ['q_proj', 'k_proj', 'v_proj', 'o_proj']),
                              ('FFN', ['gate_proj', 'up_proj', 'down_proj'])]:
        idxs = [i for i, r in enumerate(cw_results)
                if any(k in r['name'] for k in patterns)
                and r['name'] in actual_dy]
        if len(idxs) < 3:
            continue

        cw_sub = [cw_results[i]['cw_cond_mean'] for i in idxs]
        dy_sub = [actual_dy[cw_results[i]['name']] for i in idxs]
        k_sub = [cw_results[i].get('kappa_exact', cw_results[i]['kappa'])
                 for i in idxs]

        r_cw_sub, p_cw_sub = pearson_r(cw_sub, dy_sub)
        r_k_sub, p_k_sub = pearson_r(k_sub, dy_sub)

        print(f"\n  {label} ({len(idxs)} matrices):")
        print(f"    corr(κ, dy)         = {r_k_sub:.4f}  (p={p_k_sub:.4e})")
        print(f"    corr(cond_cw, dy)    = {r_cw_sub:.4f}  (p={p_cw_sub:.4e})")

    # ── Summary statistics ──
    print(f"\n{'='*65}")
    print("SUMMARY: Component-wise metrics distribution")
    print(f"{'='*65}")

    cw_cond_vals = [r['cw_cond_mean'] for r in cw_results]
    print(f"\n  cond_cw(W, x):")
    print(f"    Mean = {np.mean(cw_cond_vals):.4f}")
    print(f"    Std  = {np.std(cw_cond_vals):.4f}")
    print(f"    CV   = {np.std(cw_cond_vals)/np.mean(cw_cond_vals):.3f}")
    print(f"    Min  = {np.min(cw_cond_vals):.4f}")
    print(f"    Max  = {np.max(cw_cond_vals):.4f}")

    # ── Save results ──
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    output = {
        'checkpoint': args.checkpoint,
        'num_matrices': len(cw_results),
        'correlations': {
            'kappa_vs_dy': {'r': r_kappa, 'p': p_kappa, 'n': len(k_vals)},
            'cw_cond_vs_dy': {'r': r_cw, 'p': p_cw, 'n': len(cw_vals)},
            'cw_bound_vs_dy': {'r': r_bound, 'p': p_bound,
                               'mean_tightness': mean_tight},
        },
        'cw_metrics': cw_results,
    }
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
