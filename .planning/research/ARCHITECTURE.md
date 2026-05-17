# Architecture: Quantization Error Propagation Measurement

**Domain:** Transformer quantization error propagation measurement
**Researched:** 2026-05-17
**Confidence:** HIGH (verified against model source code at `src/model/transformer.py`)

---

## 1. Architecture Overview

### 1.1 The Error Propagation Problem

Quantization injects error at each linear projection in the Transformer. This error propagates through:

1. **Weight quantization error**: Each `nn.Linear` weight matrix W is replaced by W_q. The output error for a fixed input x is `(W_q - W) @ x = W_error @ x`.

2. **Activation quantization error**: Activations between projections are quantized. The output error is `W @ (x_q - x) = W @ x_error`.

3. **Residual mixing**: Error from the attention or FFN sub-layer is added into the residual stream, which carries it to all downstream layers.

4. **Norm transformation**: RMSNorm is non-linear. Error passing through RMSNorm is transformed by the Jacobian of the norm, which can amplify or suppress error depending on the input statistics.

The goal of the measurement architecture is to capture activation tensors at specific points in the forward pass so that:

- The error injected by each weight matrix can be isolated
- The error amplification/reduction through RMSNorm can be measured
- The error accumulation through residual connections can be traced
- The per-layer error contribution to final output error can be quantified

### 1.2 Key Architectural Facts (from source)

| Property | Value | Source |
|----------|-------|--------|
| Layers | 12 (8 sliding + 4 full alternating) | `config.py:38-43` |
| Hidden size | 768 | `config.py:20` |
| FFN intermediate | 3072 | `config.py:21` |
| Q heads / KV heads | 12 / 3 (GQA 4:1) | `config.py:23-24` |
| Head dim (sliding) | 64 | `config.py:25` |
| Head dim (full) | 128 | `config.py:26` |
| Per-layer embedding | 64-dim, concatenated before QKV/FFN | `config.py:29` |
| Norm | RMSNorm, pre-attn and post-attn | `transformer.py:178-179` |
| RoPE | Applied after QK norm | `transformer.py:130-132` |
| Weight tying | lm_head.weight = embed_tokens.weight | `transformer.py:235` |
| Precision | FP32 simulation (no hardware FP4) | `CLAUDE.md` |

---

## 2. Hook Insertion Points

### 2.1 Measurement Points Per Layer

The `TransformerLayer.forward()` method (`transformer.py:181-194`) has this structure:

```
P0: hidden_states (layer input)
        |
        v
P1: self.input_norm(hidden_states)          # Pre-attention norm output
        |
        v
    self.attention(hidden_states, pl_emb, position_ids, attention_mask)
        |
        v
P2: attention raw output (before residual)   # Attention sub-layer output
        |
        v
P3: residual + hidden_states                 # Post-attention residual stream
        |
        v
P4: self.post_attn_norm(hidden_states)       # Pre-FFN norm output
        |
        v
    self.ffn(hidden_states, pl_emb)
        |
        v
P5: FFN raw output (before residual)         # FFN sub-layer output
        |
        v
P6: residual + hidden_states                 # Layer output
```

### 2.2 Minimal Hook Set (6 points per layer)

| ID | Hook Type | Target Module | Tensor Shape | What We Measure |
|----|-----------|---------------|--------------|-----------------|
| P0 | pre-hook | `TransformerLayer.forward` | [B, S, 768] | Layer input -- error arriving from previous layers |
| P1 | post-hook | `self.input_norm` | [B, S, 768] | Pre-attention -- error entering attention, after norm transform |
| P2 | post-hook | `self.attention` | [B, S, 768] | Attention raw output -- error injected by attention weight quantization |
| P4 | post-hook | `self.post_attn_norm` | [B, S, 768] | Pre-FFN -- error entering FFN, after second norm transform |
| P5 | post-hook | `self.ffn` | [B, S, 768] | FFN raw output -- error injected by FFN weight quantization |
| P6 | post-hook | `TransformerLayer.forward` | [B, S, 768] | Layer output -- total error after one layer |

**Why 6 and not 7:** P3 (post-attention residual) is derivable: `P3 = P0 + P2` since the residual add is an exact linear operation in FP32 simulation. No hook needed.

### 2.3 Derivable Points

| Point | Derivation | Purpose |
|-------|-----------|---------|
| P3 | P0 + P2 | Post-attention residual stream |
| Attention error injected | P2_quant - P2_clean | Error added by attention sub-layer's quantized weights |
| FFN error injected | P5_quant - P5_clean | Error added by FFN sub-layer's quantized weights |
| Layer total error added | (P2+P5)_quant - (P2+P5)_clean | Combined error from this layer |
| Error after norm | P1_error | Portion of input error that norm passes through |

### 2.4 Global Measurement Points (Outside Layer Loop)

| ID | Hook Type | Target Module | Tensor Shape | What We Measure |
|----|-----------|---------------|--------------|-----------------|
| G0 | post-hook | `self.embed_tokens` | [B, S, 768] | Embedding output -- baseline error (0 if not quantized) |
| G1 | post-hook | `self.norm` | [B, S, 768] | Final norm output -- error before LM head |
| G2 | post-hook | `self.lm_head` | [B, S, 32000] | Logits -- error in final predictions |

### 2.5 Extended Diagnostic Points (Optional)

For deeper analysis of where within the attention/FFN sub-layers error originates:

| ID | Target | Shape | Why |
|----|--------|-------|-----|
| D1 | `self.attention.q_proj` output (post-hook) | [B, S, 768] | Q projection error (contributes to attention score error) |
| D2 | `self.attention.k_proj` output (post-hook) | [B, S, 192] | K projection error |
| D3 | `self.attention.v_proj` output (post-hook) | [B, S, 192] | V projection error (directly affects output value error) |
| D4 | `self.attention.o_proj` input (pre-hook) | [B, S, 768] | SDPA output before output projection |
| D5 | `self.ffn.gate_proj` output (post-hook) | [B, S, 3072] | Gate projection error |
| D6 | `self.ffn.up_proj` output (post-hook) | [B, S, 3072] | Up projection error |

These are optional because they add significant memory cost (especially D5/D6 at 3072-wide) and the per-layer SNR propagation factors can often be explained without them. Include on first run, make conditional for subsequent runs.

### 2.6 GQA Head Dimension Split

The architecture uses GQA 4:1. This means:

- **Sliding layers** (layers 0,1,3,4,6,7,9,10): head_dim=64
  - Q: [B, 12, S, 64], K/V: [B, 3, S, 64], repeated to 12 heads
  - Weight shapes: q_proj [768, 832], k/v_proj [192, 832], o_proj [768, 768]

- **Full layers** (layers 2,5,8,11): head_dim=128
  - Q: [B, 12, S, 128], K/V: [B, 3, S, 128], repeated to 12 heads
  - Weight shapes: q_proj [1536, 832], k/v_proj [384, 832], o_proj [1536, 768]
  - Note: o_proj output is still [B, S, 768] because num_heads * head_dim = 12 * 128 = 1536, then o_proj projects down to 768

This head-dim split is important because:
- Full layers have 2x the attention weight parameters (more quantization error sources)
- Full layers have 2x the head dimension (potentially different quantization sensitivity)
- The repeated-KV structure means error in K/V for the 3 heads is repeated to 12, then used in attention

---

## 3. Error Propagation Through Residuals

### 3.1 Core Insight: Residual Add Is Linear

The pre-norm residual block:

```
y = x + F(Norm(x))
```

where `F` is the attention or FFN sub-layer. In FP32 simulation, the `+` operator is exact. Therefore:

```
y_clean = x_clean + F_clean(Norm(x_clean))
y_quant = x_quant + F_quant(Norm(x_quant))

error(y) = y_quant - y_clean
         = (x_quant - x_clean) + (F_quant - F_clean)
         = error(x) + error_injected_by_F
```

This means the residual acts as an **error adder**, not an error multiplier. The error from previous layers simply adds to the error injected by the current sub-layer.

### 3.2 Error Accumulation Across Layers

For a stack of L pre-norm layers:

```
Layer 0: error(x_1) = error(embed) + error_injected_attn_0 + error_injected_ffn_0
Layer 1: error(x_2) = error(x_1) + error_injected_attn_1 + error_injected_ffn_1
...
Layer L: error(x_L) = sum over all sub-layer errors up to L
```

**Critical implication:** The error at the final layer output is the SUM (not product) of all per-sub-layer quantization errors. This is why residual networks are quantization-friendly -- error does not compound multiplicatively through the stack.

However, the error does pass through RMSNorm at each layer, which can stretch or compress the error vector depending on its alignment with the input signal.

### 3.3 Error Through RMSNorm

RMSNorm is non-linear:

```
RMSNorm(x) = x / sqrt(mean(x^2) + eps) * w
```

For small input error delta, first-order Taylor expansion:

```
RMSNorm(x + delta) ≈ RMSNorm(x) + J_RMSNorm(x) @ delta
```

Where `J_RMSNorm` is the Jacobian of RMSNorm. Error through the norm:

```
error_output ≈ J_RMSNorm(x) @ error_input
```

The spectral norm of `J_RMSNorm` depends on `x`:
- When `||x||` is large, the norm surface is locally flat (error is suppressed)
- When `||x||` is near zero, error can be amplified

**Measurement:** Compare `||P1_error|| / ||P0_error||` to measure the norm's error amplification factor empirically. If this ratio is consistently < 1, RMSNorm is suppressing error.

---

## 4. Data Structure for Hook Outputs

### 4.1 Per-Step Capture Structure

```python
@dataclass
class LayerMeasurement:
    """Measurements for one layer in one forward pass (one step)."""
    layer_idx: int
    layer_type: str  # 'sliding' | 'full'

    # Activation tensors per measurement point
    points: dict[str, TensorData]

    # Derived propagation factors
    propagation: dict[str, PropagationFactor]

    # Weight-level errors (computed once, not per-step)
    weight_errors: dict[str, WeightError] | None


@dataclass
class TensorData:
    """Statistics for one tensor at one measurement point."""
    # Raw tensor reference if kept (optional, for detailed analysis)
    tensor: torch.Tensor | None

    # Per-step statistics (always computed)
    mean: float
    std: float
    norm_l2: float          # ||tensor||_F (Frobenius norm)
    norm_inf: float         # max absolute value
    num_elements: int

    # For clean and quantized comparisons
    clean: torch.Tensor | None = None   # FP16 run activation
    quantized: torch.Tensor | None = None  # Quantized run activation

    @property
    def error(self) -> torch.Tensor | None:
        if self.quantized is not None and self.clean is not None:
            return self.quantized - self.clean
        return None

    @property
    def relative_error_norm(self) -> float | None:
        if self.error is not None:
            clean_norm = self.clean.norm().item()
            if clean_norm > 0:
                return self.error.norm().item() / clean_norm
        return None

    @property
    def snr_db(self) -> float | None:
        if self.error is not None and self.clean is not None:
            signal_power = (self.clean ** 2).sum().item()
            noise_power = (self.error ** 2).sum().item()
            if noise_power > 0 and signal_power > 0:
                return 10.0 * math.log10(signal_power / noise_power)
        return None


@dataclass
class PropagationFactor:
    """Error propagation between two measurement points."""
    source: str          # e.g., 'P0'
    target: str          # e.g., 'P2'
    norm_ratio: float    # ||error_target|| / ||error_source||
    cos_sim: float       # Cosine similarity between error vectors
    angle_degrees: float # Angle between error vectors


@dataclass
class WeightError:
    """Error statistics for one weight matrix."""
    name: str                    # e.g., 'model.layers.0.attention.q_proj.weight'
    shape: tuple
    mse: float                   # Mean squared error
    max_abs_error: float         # Max absolute error
    relative_error: float        # ||W_error|| / ||W_fp16||
    snr_db: float                # Signal-to-noise ratio in dB
    condition_number: float      # kappa(W_fp16) for sensitivity context
```

### 4.2 Aggregate Measurement Structure

```python
@dataclass
class StepMeasurement:
    """Complete measurement for one forward pass."""
    step: int
    layers: dict[int, LayerMeasurement]  # layer_idx -> measurement
    global_points: dict[str, TensorData]  # G0, G1, G2
    metadata: StepMetadata


@dataclass
class AggregateReport:
    """Aggregated statistics across all evaluation steps."""
    # Per-layer, per-point statistics (mean + std across steps)
    activation_error: dict[int, dict[str, TensorStats]]  # layer_idx -> point -> stats

    # Per-layer error propagation factors
    propagation_factors: dict[int, dict[str, PropagationStats]]

    # Global output metrics across steps
    final_snr_db: float              # SNR at LM head output
    final_cosine_sim: float          # Cosine sim at LM head output
    ppl_degradation: float           # PPL(quantized) - PPL(FP16)

    # Weight error (one-time, static)
    weight_errors: dict[str, WeightError]

    # Per-layer breakdown
    per_layer: dict[int, LayerBreakdown]


@dataclass
class LayerBreakdown:
    """Per-layer error contribution summary."""
    layer_idx: int
    layer_type: str

    # Error injected by this layer's sub-modules
    attention_error_norm: float   # ||attn_error||
    ffn_error_norm: float         # ||ffn_error||
    total_injected_norm: float    # ||attn_error + ffn_error||

    # Error carried forward
    input_error_norm: float       # ||P0_error|| (error from previous layers)
    output_error_norm: float      # ||P6_error|| (total error after this layer)

    # Propagation factors
    attn_error_gain: float        # ||P2_error|| / ||P0_error||
    ffn_error_gain: float         # ||P5_error|| / ||P4_error||
    norm0_amplification: float    # ||P1_error|| / ||P0_error||
    norm1_amplification: float    # ||P4_error|| / ||P3_error||

    # SNR at this layer's output
    snr_db_output: float
```

### 4.3 Indexing Convention

| Field Path | Example | Description |
|------------|---------|-------------|
| `report.layers[0].points['P0'].snr_db` | 45.2 | SNR at layer 0 input |
| `report.layers[3].propagation['P0_to_P2'].norm_ratio` | 1.5 | Error amplifies 1.5x through attention of layer 3 |
| `report.per_layer[5].attention_error_norm` | 0.023 | Norm of error injected by layer 5's attention |
| `report.weight_errors['model.layers.2.attention.q_proj.weight'].snr_db` | 38.1 | SNR of quantized q_proj weight in layer 2 |
| `report.global_points['G1'].snr_db` | 28.4 | SNR at final norm output |
| `report.final_snr_db` | 25.1 | SNR at LM head logits |

---

## 5. Integration Strategy

### 5.1 Design Decision: External Hook Manager

**Decision:** Create `src/analysis/error_propagation.py` as an external hook manager. Do NOT modify `transformer.py`.

**Rationale:**

| Approach | Pros | Cons |
|----------|------|------|
| External hooks (chosen) | No model changes; reusable across configs; conditional at runtime | Hook registration boilerplate |
| Modify transformer.py | Direct access to intermediate values | Permanently changes model code; must add flags/conditions |
| Monkey-patching forward | Flexible without source changes | Fragile; breaks if method signature changes |
| Wrapper module | Cleanest API | More code; must forward all args |

**Existing precedent:** The codebase already uses forward hooks in `fp_quantizer.py:219-242` (`make_qat_forward_hook`), so this pattern is established.

### 5.2 Hook Manager API

```python
# Proposed: src/analysis/error_propagation.py

class ErrorPropagationTracker:
    """
    Manages forward hooks to capture activations at all measurement points.

    Usage:
        tracker = ErrorPropagationTracker(model)
        model.eval()

        # Phase 1: FP16 reference pass
        with tracker.capture_run('fp16'):
            output = model(input_ids, attention_mask=attention_mask)

        # Phase 2: Apply quantization, then quantized pass
        apply_weight_quantization(model, quantizer)
        with tracker.capture_run('quantized'):
            output = model(input_ids, attention_mask=attention_mask)

        # Phase 3: Compare
        report = tracker.compute_report()
    """
```

**Registration pattern:**

For each layer `i`:

```python
layer = model.model.layers[i]

# P0: Layer input (pre-hook on TransformerLayer.forward)
layer.register_forward_pre_hook(make_capture_hook(f'layer_{i}', 'P0', 'input'))

# P1: Pre-attention norm output (post-hook on input_norm)
layer.input_norm.register_forward_hook(
    make_capture_hook(f'layer_{i}', 'P1', 'norm_output'))

# P2: Attention raw output (post-hook on Attention.forward)
layer.attention.register_forward_hook(
    make_capture_hook(f'layer_{i}', 'P2', 'attn_output'))

# P4: Pre-FFN norm output (post-hook on post_attn_norm)
layer.post_attn_norm.register_forward_hook(
    make_capture_hook(f'layer_{i}', 'P4', 'norm_output'))

# P5: FFN raw output (post-hook on FFN.forward)
layer.ffn.register_forward_hook(
    make_capture_hook(f'layer_{i}', 'P5', 'ffn_output'))

# P6: Layer output (post-hook on TransformerLayer.forward)
layer.register_forward_hook(
    make_capture_hook(f'layer_{i}', 'P6', 'output'))
```

### 5.3 Two-Pass Protocol

The measurement requires two forward passes per batch:

```
Pass 1 (FP16 reference):
  1. Model is fully FP16 (no quantization applied)
  2. Forward pass through model with hooks active
  3. All P0-P6 activations are captured as 'clean' tensors

[Apply quantization to model weights between passes]

Pass 2 (Quantized):
  1. Model has quantized weights (e.g., FP8 or FP4)
  2. Forward pass through same model with same input
  3. All P0-P6 activations are captured as 'quantized' tensors

[Compare]
  4. For each measurement point:
     - error = quantized - clean
     - SNR, relative_error, cosine_similarity
  5. Compute propagation factors between points
```

**Important considerations:**

- Both passes must use the same random seed for dropout -- but since `model.eval()` is used, dropout is disabled. This is fine.
- The model must be in eval mode (`model.eval()`) to ensure deterministic behavior between passes.
- The input batch must be identical for both passes (same `input_ids`, `attention_mask`).
- For PTQ weight quantization, the quantization happens between passes (not in the forward hook).
- For activation quantization simulation, the quantization would happen inside the forward path (either via a wrapper or a hook that modifies tensors).

### 5.4 Integration with Existing Evaluation Pipeline

The measurement script (`src/experiments/measure_error_propagation.py`) should reuse existing infrastructure:

```python
# Pseudo-code structure:

def main():
    # 1. Parse args (standard pattern from eval_quantization.py)
    parser.add_argument('--checkpoint', ...)
    parser.add_argument('--data_dir', ...)
    parser.add_argument('--formats', ...)
    parser.add_argument('--max_eval_steps', type=int, default=100)
    parser.add_argument('--keep_raw', action='store_true',
                        help='Keep raw tensors (memory intensive)')

    # 2. Load model (same as eval_quantization.py)
    config = MicroGemmaFPConfig()
    model = MicroGemmaFPForCausalLM(config).to(device)
    load_checkpoint(model, None, args.checkpoint, device)

    # 3. Create tracker (hooks registered on model)
    tracker = ErrorPropagationTracker(model)

    # 4. Create dataloader (reuse get_dataloader from training_utils.py)
    loader = get_dataloader(batch_size=1, max_seq_len=512,
                            max_steps=args.max_eval_steps, data_dir=args.data_dir)

    # 5. For each batch: two-pass protocol
    for step, batch in enumerate(loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        # Pass 1: FP16 reference
        model.eval()
        with torch.no_grad():
            _ = model(input_ids, attention_mask=attention_mask)
        tracker.finalize_run('fp16')

        # Apply quantization to weights
        apply_weight_quantization(model, quantizer)

        # Pass 2: Quantized
        with torch.no_grad():
            _ = model(input_ids, attention_mask=attention_mask)
        tracker.finalize_run('quantized')

        # Compare and record
        step_report = tracker.compute_step_report()
        tracker.accumulate(step_report)

        # Restore FP16 weights for next batch
        restore_fp16_weights(model, original_state_dict)

    # 6. Generate aggregate report
    aggregate_report = tracker.aggregate()
```

### 5.5 Weight Restoration Between Batches

Since quantization modifies weights in-place, we need to restore FP16 weights between batches. Two strategies:

**Strategy A: State dict snapshot (simpler)**
```python
original_state = copy.deepcopy(model.state_dict())
# ... quantize, run, compute
model.load_state_dict(original_state)
```
Cost: ~656 MB for state dict copy (164M * 4 bytes). Acceptable.

**Strategy B: Weight cache (more memory efficient)**
```python
original_weights = {name: param.data.clone()
                    for name, param in model.named_parameters()
                    if param.dim() >= 2}
# ... quantize, run, compute
for name, w in original_weights.items():
    dict(model.named_parameters())[name].data.copy_(w)
```
Cost: only quantizable weight copies (~160M params = ~640 MB). Similar to Strategy A but more targeted.

**Recommendation:** Use Strategy B with the `get_quantizable_weights()` method already defined in `transformer.py:264-269`.

### 5.6 Activation Quantization Measurement

For activation quantization error (separate from weight-only):

- This requires a model wrapper or `register_forward_pre_hook` on each linear layer that quantizes the input before the matmul.
- The existing `QuantizedLinear` in `fp_quantizer.py:186-216` already implements this pattern.
- For measurement: compare activations with and without the activation quantization hooks active.

The two-pass protocol extends naturally:
```
Pass 1: No quantization (FP16 weights, no activation quant) -> clean activations
Pass 2: Quantized weights + quantized activations -> quantized activations
```

---

## 6. Computational and Memory Cost Analysis

### 6.1 Per-Step Memory (batch_size=1, seq_len=512)

| Component | Size | Notes |
|-----------|------|-------|
| Model params (FP32) | ~656 MB | 164M * 4 bytes |
| Model params (quantized) | ~656 MB | Same storage, different values |
| Activation storage (hooks, 1 run) | ~9.5 MB | 12 layers * 6 points * [B, S, 768] * 4 bytes |
| Activation storage (hooks, 2 runs) | ~19 MB | Two forward passes |
| Input batch | ~0.5 MB | [1, 512] int64 |
| Attention intermediates | ~50 MB | SDPA internal buffers per layer (freed after each layer) |
| FFN intermediates | ~12 MB | [1, 512, 3072] * 4 bytes for gate, up |
| Weight cache (Strategy B) | ~640 MB | 160M quantizable params * 4 bytes |
| **Peak total** | **~1.4 GB** | Fits on all GPUs |

With optional extended diagnostic points (D1-D6):

| Extra Component | Size | Notes |
|-----------------|------|-------|
| Q projection (sliding) | ~0.5 MB | [B, S, 768] * 4 |
| Q projection (full) | ~1 MB | [B, S, 1536] * 4 |
| K/V projection (full) | ~0.25 MB each | [B, S, 384] |
| Gate/Up projections | ~6 MB each | [B, S, 3072] |
| **Additional total** | **~85 MB** | All D points for all 12 layers |
| **Peak with extended** | **~1.5 GB** | Still fits 24GB GPU |

### 6.2 Per-Step Compute

| Operation | Time Estimate | Notes |
|-----------|--------------|-------|
| FP16 forward pass | ~10-20 ms | 164M model, batch=1 |
| Quantize weights | ~5-10 ms | All linear layers |
| Quantized forward pass | ~10-20 ms | Same model, same data |
| Statistics computation | ~5 ms | Per-point SNR, norms |
| **Total per step** | **~30-55 ms** | ~18-33 steps/sec |

For 100 evaluation steps: **~3-5.5 seconds** total. Negligible.

### 6.3 Storage for Aggregate Results

| Data | Size | Notes |
|------|------|-------|
| Per-step statistics (100 steps) | ~100 KB | 12 layers * 6 points * ~10 scalars * 100 |
| Weight error (one-time) | ~1 KB | 7 weight types * 12 layers * ~5 scalars |
| Propagation factors (100 steps) | ~50 KB | 12 layers * 4 propagation pairs * 2 scalars * 100 |
| Raw tensors (if kept, 1 step) | ~9.5 MB | One forward pass activation tensors |
| **Total (stats only)** | **~150 KB** | Negligible |

**Recommendation:** Never store raw tensors for all 100 steps. Store stats only. Optionally save raw tensors for the first 1-3 steps for detailed visualization.

### 6.4 Batch Size Considerations

For minimal memory: use `batch_size=1`.

For more reliable statistics: `batch_size=4` or `batch_size=8`.

| batch_size | Peak Memory | Activation SNR Std | Use Case |
|------------|-------------|-------------------|----------|
| 1 | ~1.4 GB | Higher variance | Debugging, initial measurement |
| 4 | ~2.2 GB | Moderate variance | Standard measurement runs |
| 8 | ~3.5 GB | Lower variance | Final report, publication-quality |
| 16 | ~6 GB | Lowest variance | Only on >24GB GPUs |

---

## 7. Derived Metrics and Error Propagation Analysis

### 7.1 Key Metrics Computed from Hook Data

**Per-point metrics** (computed at each P0-P6 for each layer):

1. **Signal-to-Noise Ratio (SNR)** in dB:
   ```
   SNR = 10 * log10(||x_clean||^2 / ||x_error||^2)
   ```
   Higher is better. Typical FP8: 30-50 dB. FP4: 15-30 dB.

2. **Relative Error Norm**:
   ```
   rel_err = ||x_error|| / ||x_clean||
   ```
   Interpretable as percentage. 0.01 = 1% relative error.

3. **Cosine Similarity**:
   ```
   cos_sim = dot(x_clean, x_quant) / (||x_clean|| * ||x_quant||)
   ```
   Measures directional preservation. 1.0 = perfect direction.

4. **Mean Absolute Error (MAE)** and **Root Mean Squared Error (RMSE)**.

**Propagation metrics** (between points):

5. **Error Amplification Factor**:
   ```
   amp(P_a -> P_b) = ||error_Pb|| / ||error_Pa||
   ```
   > 1 = error amplified, < 1 = error suppressed.

6. **Error Angle**:
   ```
   angle = arccos(dot(error_Pa, error_Pb) / (||error_Pa|| * ||error_Pb||))
   ```
   Measures whether error direction changes between points. Low angle = structured error that correlates path.

**Layer contribution metrics:**

7. **Layer Error Contribution**:
   ```
   contribution_ratio = ||error_injected_by_layer|| / ||total_error_at_output||
   ```
   Identifies which layers contribute most to final output error.

8. **Error Accumulation Ratio**:
   ```
   accumulation_ratio = ||error_P6|| / ||error_P0||
   ```
   How much total error (old + new) grows through this layer.

### 7.2 Per-Layer Error Characterization

For each layer `i`, we compute:

| Metric | Formula | Meaning |
|--------|---------|---------|
| `E_attn_i` | `P2_quant - P2_clean` | Attention sub-layer injection |
| `E_ffn_i` | `P5_quant - P5_clean` | FFN sub-layer injection |
| `E_layer_i` | `E_attn_i + E_ffn_i` | Total error injected by layer i |
| `E_residual_i` | `P0_quant - P0_clean` | Error from all previous layers |
| `E_output_i` | `P6_quant - P6_clean` | Total error at layer i output |
| `gain_attn_i` | `||E_attn_i|| / ||E_residual_i||` | Attention error relative to incoming |
| `gain_ffn_i` | `||E_ffn_i|| / ||E_post_attn_i||` | FFN error relative to post-attn input |

The **cumulative output error** at layer i:

```
E_output_i = E_embed + sum_{j=0}^{i} (E_attn_j + E_ffn_j)
```

This is exact because residual connections are linear additions.

### 7.3 Sliding vs Full Layer Analysis

The alternating layer pattern (8 sliding + 4 full) creates natural experimental groups:

| Group | Layers | Head Dim | QKV Size | Key Property |
|-------|--------|----------|----------|--------------|
| Sliding | 0,1,3,4,6,7,9,10 | 64 | Q:[768,832], K/V:[192,832] | Smaller matrices, windowed attention |
| Full | 2,5,8,11 | 128 | Q:[1536,832], K/V:[384,832] | Larger matrices, global attention |

This allows comparing:
- Does error in full layers differ from sliding layers? (expect larger matrices = more quantization error in absolute terms, but potentially higher SNR due to larger signal norm)
- Does error in V projection (directly affects output) differ from Q/K? (V error directly maps to attention output via weighted sum; Q/K error goes through softmax first)

---

## 8. Relationship to Existing Analysis Modules

### 8.1 Complement to Lipschitz Analysis

The existing `lipschitz.py` computes theoretical error propagation via spectral norm bounds:

```python
# From lipschitz.py:33-60
# propagation_factor[i] = prod_{k>i} L_k  (theoretical worst case)
```

The new measurement architecture provides **empirical** error propagation:

```python
# Measured in error_propagation.py
# empirical_gain = ||output_error|| / ||input_error||  (actual, not worst-case)
```

**Comparison:**

| Aspect | Lipschitz (theory) | Error Propagation (measurement) |
|--------|-------------------|----------------------------------|
| Lower bound | No | Yes (empirical is the actual error) |
| Upper bound | Yes (worst case) | No |
| Input-dependent | No | Yes (captures actual activation statistics) |
| Cost | One-time, O(n_layers * n_iter) | Per-step, O(n_layers * seq_len * hidden) |
| Use | Conservative mixed-precision assignment | Empirical validation, understanding |

Both should be run together: Lipschitz gives the theoretical worst case; measurement gives the actual case. If the gap is large, there is structure in the activations that suppresses error.

### 8.2 Complement to Sensitivity Analysis

The existing `sensitivity.py` computes predicted impact as:

```python
# From sensitivity.py:51
# predicted_impact = avg_kappa * quantization_mse * propagation_factor
```

The measurement architecture validates this formula empirically:

```python
# Measured impact = ||output_error||  (from actual two-pass measurement)
```

Comparing predicted vs measured impact across layers:
- If they correlate well: the sensitivity model is validated
- If they diverge: the sensitivity model misses some factor (e.g., activation statistics, norm effects)

---

## 9. Implementation Plan

### 9.1 File: `src/analysis/error_propagation.py`

```
error_propagation.py
├── ErrorPropagationTracker        # Main class: manages hooks, runs two-pass protocol
│   ├── __init__(model)            # Register all hooks
│   ├── _register_layer_hooks()    # Register 6 hooks per layer
│   ├── _register_global_hooks()   # Register G0, G1, G2 hooks
│   ├── capture_run(label)         # Context manager for one forward pass
│   ├── finalize_run(label)        # Store captured activations under label
│   ├── compute_step_report()      # Compare fp16 vs quantized, compute metrics
│   ├── accumulate(report)         # Update running statistics
│   ├── aggregate()                # Compute aggregate report
│   └── clear()                    # Clear hooks and cached data
│
├── TensorData                     # Per-point tensor statistics (dataclass)
├── PropagrationFactor             # Error propagation between points (dataclass)
├── LayerMeasurement               # Per-layer per-step measurement (dataclass)
├── StepMeasurement                # Full per-step measurement (dataclass)
├── AggregateReport                # Final aggregate across all steps (dataclass)
│
├── compute_snr_db()               # Utility: 10*log10(signal_power / noise_power)
├── compute_cosine_sim()           # Utility: cosine similarity
├── compute_relative_error()       # Utility: ||error|| / ||clean||
└── compute_propagation_factor()   # Utility: error_norm ratio between points
```

### 9.2 File: `src/experiments/measure_error_propagation.py`

Standalone entry point (same pattern as `eval_quantization.py`):

```
measure_error_propagation.py
├── main()
│   ├── Parse args (checkpoint, data_dir, formats, max_steps, keep_raw)
│   ├── Load model from checkpoint
│   ├── Create ErrorPropagationTracker
│   ├── Create dataloader
│   ├── For each of 100 steps:
│   │   ├── Pass 1: FP16 reference
│   │   ├── Quantize weights (in-place)
│   │   ├── Pass 2: Quantized
│   │   ├── Restore FP16 weights
│   │   ├── Compute step report
│   │   └── Accumulate
│   ├── Aggregate across steps
│   ├── Print summary table (per-layer SNR, propagation factors)
│   ├── Save aggregate report to JSON
│   └── If keep_raw: save raw tensors for first 3 steps
```

### 9.3 Integration Points with Existing Code

| What | Where | How |
|------|-------|-----|
| Model loading | `training_utils.py:load_checkpoint()` | Reuse as-is |
| Data loading | `training_utils.py:get_dataloader()` | Reuse as-is |
| Weight quantization | `fp_quantizer.py:FPQuantizer.quantize()` | Reuse as-is |
| Perplexity evaluation | `training_utils.py:evaluate_perplexity()` | Call after measurement for PPL comparison |
| Condition numbers | `condition.py:estimate_condition_number()` | Call to include kappa in weight-error report |
| Sensitivity report | `sensitivity.py:per_layer_sensitivity_report()` | Compare predicted vs measured impact |
| Config | `config.py:MicroGemmaFPConfig()` | Reuse as-is |
| CLI argument pattern | `eval_quantization.py:parser` | Follow same pattern |
| Remote execution | `sync.sh` + `remote_python.sh` | Existing workflow |

---

## 10. Summary

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| Hook points | 6 per layer (P0-P6) + 3 global (G0-G2) | Minimally sufficient set; P3 derivable |
| Residual handling | Error = clean - quantized; additive through residuals | Residual add is exact linear operation |
| Norm handling | Capture both sides (P0 vs P1, P3 vs P4) to measure Norm's Jacobian effect | RMSNorm is non-linear, must measure empirically |
| Data structure | Nested dataclasses: per-step -> aggregate | Clean typing; torch.compile friendly |
| Integration | External hook manager, no model modification | Follows existing pattern (QAT hooks in fp_quantizer.py) |
| Weight restoration | Clone quantizable params before quantization, restore after | Uses existing `get_quantizable_weights()` |
| Memory | ~1.4 GB peak, ~150 KB stored | 100 steps, stats-only, no raw tensor storage |
| Two-pass protocol | FP16 pass then quantized pass per batch | Most general; supports any quantization method |
