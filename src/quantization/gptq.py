"""GPTQ-style Post-Training Quantization with weight compensation.

Based on Frantar et al. (2023) "GPTQ: Accurate Post-Training Quantization
for Generative Pre-trained Transformers".

Core idea: quantize weights column-by-column, and after quantizing each
column, compensate the error by updating the remaining unquantized columns.

  W_q[:, j] = Q(W[:, j])
  δ_j = W_q[:, j] - W[:, j]
  W[:, j+1:] -= δ_j · H^{-1}[j, j+1:] / H^{-1}[j, j]

where H = X^T X is the (approximate) Hessian of the calibration data.
This preserves the output Y = WX up to the quantization error of the
LAST column (which has no remaining columns to compensate).
"""

import torch
import torch.nn.functional as F


def compute_activation_hessian(X: torch.Tensor) -> torch.Tensor:
    """Compute H = X^T X from calibration activations.

    X shape: (n_samples, in_features) — pre-linear-layer inputs.
    Adds damping for Cholesky: larger damping for larger in_features.
    """
    n_samples, in_features = X.shape
    H = X.T @ X  # (in_features, in_features)
    # Damping scaled by matrix size — larger matrices need more stabilization
    damping = max(1e-3, in_features * 1e-6)
    H.diagonal().add_(damping)
    return H


def compute_cholesky(H: torch.Tensor) -> torch.Tensor:
    """Cholesky decomposition of H with progressive damping.

    Tries increasingly strong damping until H becomes positive-definite.
    This is necessary when calibration data has fewer independent samples
    than in_features (e.g., 256 samples for 768-dim layers).
    """
    dampings = [1e-3, 1e-2, 1e-1, 1.0]
    for d in dampings:
        try:
            H_damped = H + d * torch.eye(H.shape[0], device=H.device, dtype=H.dtype)
            return torch.linalg.cholesky(H_damped)
        except torch.linalg.LinAlgError:
            continue
    # Last resort: use identity + tiny noise (degenerate case)
    H_last = (1.0 + 1e-4) * torch.eye(H.shape[0], device=H.device, dtype=H.dtype)
    return torch.linalg.cholesky(H_last)


def gptq_quantize_weight(
    W: torch.Tensor,
    H: torch.Tensor,
    quantizer,
    blocksize: int = 128,
    stochastic: bool = False,
    use_per_channel: bool = True,
) -> torch.Tensor:
    """Quantize weight matrix W with GPTQ error compensation.

    Applies per-channel normalization first (row-wise dynamic range),
    then runs GPTQ column compensation with per-tensor quantization on
    the normalized matrix, then undoes the normalization. This gives
    both per-channel dynamic range AND column-wise error correction
    without the two mechanisms interfering.

    Args:
        W: weight matrix (out_features, in_features)
        H: activation Hessian (in_features, in_features) = X^T X
        quantizer: FPQuantizer instance (will be used per-tensor internally)
        blocksize: number of columns to process at once (GPU efficiency)
        stochastic: use stochastic rounding
        use_per_channel: apply per-channel normalization around GPTQ

    Returns:
        W_q: quantized weight with compensation applied
    """
    out_features, in_features = W.shape
    W_q = W.clone()
    L = compute_cholesky(H)
    H_inv = torch.cholesky_inverse(L)

    # Per-channel normalization: scale each row to unit max magnitude.
    # H depends only on X (calibration inputs), not W, so H_inv stays
    # unchanged. The GPTQ compensation formula is invariant to per-row
    # scaling: errors and compensation are both in normalized space.
    if use_per_channel:
        row_scale = W_q.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
        W_q = W_q / row_scale

    # Save original quantizer's per_channel setting and force per-tensor
    saved_per_channel = quantizer.per_channel
    quantizer.per_channel = False

    for j_start in range(0, in_features, blocksize):
        j_end = min(j_start + blocksize, in_features)

        for j in range(j_start, j_end):
            w_col = W_q[:, j]
            w_col_q = quantizer.quantize(w_col, stochastic=stochastic)

            error = w_col_q - w_col
            W_q[:, j] = w_col_q

            if j + 1 < in_features:
                scale = H_inv[j, j + 1:] / H_inv[j, j]
                W_q[:, j + 1:] -= torch.outer(error, scale)

    # Restore quantizer setting
    quantizer.per_channel = saved_per_channel

    # Undo per-channel normalization
    if use_per_channel:
        W_q = W_q * row_scale

    return W_q


class GPTQQuantizer:
    """Apply GPTQ weight compensation across all layers of a model.

    Usage:
        quantizer = FPQuantizer('fp4_e2m1', per_channel=True)
        gptq = GPTQQuantizer(quantizer)
        model = gptq.quantize_model(model, calibration_data)
    """

    def __init__(self, quantizer, blocksize: int = 128, stochastic: bool = False):
        self.quantizer = quantizer
        self.blocksize = blocksize
        self.stochastic = stochastic

    @torch.no_grad()
    def quantize_model(self, model, calibration_loader, device='cuda') -> dict:
        """Quantize all linear layers with GPTQ compensation.

        Args:
            model: the model to quantize (modified in-place)
            calibration_loader: DataLoader yielding batches of input_ids
            device: target device

        Returns:
            dict: per-layer quantization statistics (MSE before/after)
        """
        model.eval()
        stats = {}

        # Collect activations for each layer from calibration data
        layer_inputs = self._collect_activations(model, calibration_loader, device)

        for layer_name, X_cpu in layer_inputs.items():
            weight, bias = self._get_layer_weight(model, layer_name)
            if weight is None:
                continue

            # Move to same device as weight
            X = X_cpu.to(weight.device)
            H = compute_activation_hessian(X)

            if X.numel() > 0:
                y_orig = X @ weight.T
                if bias is not None:
                    y_orig += bias

            # Apply GPTQ compensation (weight is on GPU, H is on GPU)
            W_q = gptq_quantize_weight(
                weight.data, H, self.quantizer,
                blocksize=self.blocksize,
                stochastic=self.stochastic,
            )

            mse_after = 0.0
            if X.numel() > 0:
                y_q = X @ W_q.T
                if bias is not None:
                    y_q += bias
                mse_after = ((y_q - y_orig) ** 2).mean().item()

            # Update weight in-place
            weight.data.copy_(W_q)

            stats[layer_name] = {
                'mse_output': mse_after,
                'weight_quantized': weight.numel(),
            }

        return stats

    @torch.no_grad()
    def _collect_activations(self, model, loader, device) -> dict[str, torch.Tensor]:
        """Collect pre-linear-layer inputs from calibration data.

        Returns dict mapping layer names to input tensors of shape
        (n_samples, in_features).
        """
        inputs = {}
        hooks = []

        def make_hook(name):
            def hook_fn(module, inp, out):
                # inp[0] shape: (batch, seq, in_features)
                x = inp[0].detach()
                # Flatten batch and sequence dims
                x_flat = x.reshape(-1, x.shape[-1])
                if name not in inputs:
                    inputs[name] = []
                inputs[name].append(x_flat.cpu())
            return hook_fn

        # Register hooks on all Linear layers except embedding/lm_head
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                if 'embed' in name.lower() or 'lm_head' in name.lower():
                    continue
                hooks.append(module.register_forward_hook(make_hook(name)))

        # Run calibration data through model
        max_steps = 50  # Enough for Hessian estimation
        for step, batch in enumerate(loader):
            if step >= max_steps:
                break
            input_ids = batch['input_ids'].to(device)
            model(input_ids)

        # Remove hooks
        for h in hooks:
            h.remove()

        # Concatenate collected inputs per layer
        result = {}
        for name, tensors in inputs.items():
            result[name] = torch.cat(tensors, dim=0)
            # Subsample if too large — but keep enough for full-rank Hessian
            # (need at least in_features independent samples)
            max_samples = max(4096, result[name].shape[1] * 4)
            if result[name].shape[0] > max_samples:
                idx = torch.randperm(result[name].shape[0])[:max_samples]
                result[name] = result[name][idx]

        return result

    def _get_layer_weight(self, model, layer_name: str):
        """Get weight and bias tensors for a named linear layer."""
        module = dict(model.named_modules()).get(layer_name)
        if module is None or not hasattr(module, 'weight'):
            return None, None
        return module.weight, getattr(module, 'bias', None)
