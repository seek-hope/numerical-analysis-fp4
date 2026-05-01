"""Fast Hadamard transform for activation smoothing before quantization.

Based on QuIP/QuaRot/QuEST: random orthogonal transforms make weight/activation
distributions more Gaussian, reducing quantization error.

The Walsh-Hadamard Transform is O(n log n) and orthogonal (H @ H^T = nI).
"""

import math
import torch
import torch.nn as nn


def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def next_power_of_two(n: int) -> int:
    return 1 << (n - 1).bit_length()


def fast_hadamard_transform(x: torch.Tensor, scale: bool = True) -> torch.Tensor:
    """
    In-place Fast Walsh-Hadamard Transform along the last dimension.

    Args:
        x: input tensor of shape (..., n)
        scale: if True, normalize by 1/sqrt(n) to make orthogonal

    Returns:
        Transformed tensor of same shape.
    """
    orig_n = x.shape[-1]
    n = next_power_of_two(orig_n)

    x = x.clone()  # Work on copy to avoid autograd in-place issues

    if n != orig_n:
        pad = torch.zeros(*x.shape[:-1], n - orig_n, device=x.device, dtype=x.dtype)
        x = torch.cat([x, pad], dim=-1)

    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                u = x[..., j].clone()
                v = x[..., j + h].clone()
                x[..., j] = u + v
                x[..., j + h] = u - v
        h *= 2

    x = x[..., :orig_n]
    if scale:
        x = x / math.sqrt(n)
    return x


def hadamard_rotate_weight(W: torch.Tensor) -> torch.Tensor:
    """
    Apply Hadamard rotation to a weight matrix.

    For a linear layer W of shape (out_features, in_features):
    W_rotated = H_out @ W @ H_in^T

    This makes the weight distribution more uniform, reducing quantization error.
    """
    out_dim, in_dim = W.shape
    # Apply Hadamard to input dimension
    W_t = fast_hadamard_transform(W.transpose(0, 1)).transpose(0, 1)
    # Apply Hadamard to output dimension (if power of 2)
    W_out = fast_hadamard_transform(W_t)
    return W_out


def hadamard_rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """
    Apply Hadamard rotation to an activation tensor.

    For activations of shape (batch, seq, hidden):
    x_rotated = x @ H^T  (Hadamard on hidden dimension)
    """
    return fast_hadamard_transform(x)


class HadamardRotation(nn.Module):
    """
    Module wrapper that applies Hadamard rotation to activations before a
    linear layer, and inverse rotation after.

    This is the QuIP/QuaRot pattern: rotate to make distribution friendlier
    to quantization, then unrotate after computation.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return hadamard_rotate_activation(x)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        # Hadamard is self-inverse (up to scaling): H @ H = nI
        return hadamard_rotate_activation(x)


def analyze_activation_statistics(x_before: torch.Tensor, x_after: torch.Tensor) -> dict:
    """
    Compare activation statistics before and after Hadamard transform.
    Used to quantify the "outlier reduction" effect.
    """
    def stats(t):
        t = t.detach().float()
        return {
            'mean': t.mean().item(),
            'std': t.std().item(),
            'kurtosis': ((t - t.mean()) ** 4).mean().item() / (t.var().item() ** 2 + 1e-8),
            'outlier_ratio_3sigma': (t.abs() > 3 * t.std()).float().mean().item(),
            'max_abs': t.abs().max().item(),
        }

    return {
        'before': stats(x_before),
        'after': stats(x_after),
        'kurtosis_reduction': stats(x_before)['kurtosis'] / max(stats(x_after)['kurtosis'], 1e-8),
        'outlier_reduction': stats(x_before)['outlier_ratio_3sigma'] /
                             max(stats(x_after)['outlier_ratio_3sigma'], 1e-8),
    }
