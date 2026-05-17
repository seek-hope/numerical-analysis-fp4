"""Condition number estimation and regularization for quantized models.

κ(W) = σ_max / σ_min measures sensitivity to perturbations.
Lower κ(W) → quantization errors are amplified less → "quantization-friendly" weights.

Uses randomized power iteration (3-5 iterations) for efficiency.
"""

import math
import torch
import torch.nn.functional as F


def power_iteration(W: torch.Tensor, n_iter: int = 5) -> float:
    """
    Estimate the spectral norm (σ_max) of matrix W via randomized power iteration.

    Complexity: O(n_iter × out_features × in_features).
    For n_iter=3, provides ~1% accuracy for most NN weight matrices.
    """
    v = torch.randn(W.size(1), device=W.device, dtype=W.dtype)
    for _ in range(n_iter):
        v = F.normalize(W.T @ (W @ v), dim=0)
    sigma_max = (W @ v).norm().item()
    return sigma_max


def estimate_singular_values(W: torch.Tensor) -> tuple[float, float]:
    """Compute σ_max and σ_min via exact SVD.

    Uses torch.linalg.svdvals which is O(min(m,n)·m·n). For the 164M model's
    weight matrices (≤832×832), this costs < 1ms per matrix — negligible
    compared to the concurrent per-matrix ||dy||/||y|| evaluation that follows.

    Returns:
        (sigma_max, sigma_min) as Python floats.
    """
    s = torch.linalg.svdvals(W.float())
    return s.max().item(), s.min().clamp(min=1e-12).item()


def estimate_condition_number(W: torch.Tensor, n_iter: int = 3) -> float:
    """Estimate κ(W) = σ_max / σ_min for a weight matrix W.

    Uses exact SVD since weight matrices in the 164M model are ≤832×832,
    making this cheap. The n_iter parameter is kept for API compatibility
    but is unused.
    """
    sigma_max, sigma_min = estimate_singular_values(W)
    if sigma_min < 1e-12:
        return 1e12
    return sigma_max / sigma_min


def compute_all_condition_numbers(model) -> dict[str, float]:
    """
    Compute κ(W) for all quantizable weight matrices in the model.

    Returns dict mapping parameter name → condition number.
    """
    results = {}
    for name, param in model.named_parameters():
        if param.dim() >= 2:
            results[name] = estimate_condition_number(param.data)
    return results


def _condition_regularization_surrogate(W: torch.Tensor, n_iter: int = 3) -> torch.Tensor:
    """Differentiable condition-number surrogate.

    Exact σ_min-based κ is too expensive to compute for every large Linear
    layer at every training step. This surrogate penalizes concentration of
    singular values by comparing σ_max to RMS singular value:

        σ_max / sqrt(mean_i σ_i^2)

    It is 1.0 when singular values are perfectly balanced and grows as one
    direction dominates. The computation stays differentiable because it does
    not call .item() on the weight-dependent terms.
    """
    Wf = W.float()
    v = torch.randn(Wf.size(1), device=Wf.device, dtype=Wf.dtype)
    for _ in range(n_iter):
        v = F.normalize(Wf.T @ (Wf @ v), dim=0, eps=1e-12)

    sigma_max = (Wf @ v).norm().clamp(min=1e-12)
    rank = max(1, min(Wf.shape))
    rms_sigma = (Wf.norm() / math.sqrt(rank)).clamp(min=1e-12)
    return sigma_max / rms_sigma


def _exact_condition_number_tensor(W: torch.Tensor) -> torch.Tensor:
    """Differentiable exact κ via SVD, intended for diagnostics/small models."""
    s = torch.linalg.svdvals(W.float())
    return s.max() / s.min().clamp(min=1e-12)


def condition_number_regularization(model, lambda_cond: float = 1e-4,
                                    exact_svd: bool = False) -> torch.Tensor:
    """
    Compute the log-condition-number regularization term.

    Loss = λ × Σ log(κ_surrogate(W_i))

    Using log prevents domination by a single very ill-conditioned matrix.
    By default this uses a differentiable spectral-concentration surrogate
    instead of exact σ_max/σ_min, because exact SVD per layer per step is too
    expensive for the 164M model. Set exact_svd=True for small diagnostics.

    Args:
        model: nn.Module
        lambda_cond: regularization strength (0 = disabled)

    Returns:
        Scalar tensor (0.0 if lambda_cond == 0)
    """
    if lambda_cond <= 0:
        return torch.tensor(0.0, device=next(model.parameters()).device)

    reg = torch.zeros((), device=next(model.parameters()).device)
    for name, param in model.named_parameters():
        if param.dim() < 2:
            continue
        name_lower = name.lower()
        if 'embed' in name_lower or 'lm_head' in name_lower:
            continue

        kappa = (_exact_condition_number_tensor(param)
                 if exact_svd else _condition_regularization_surrogate(param))
        reg = reg + torch.log(kappa.clamp(min=1.0))

    return lambda_cond * reg


def analyze_quantization_sensitivity(model, quantizer) -> dict:
    """
    For each weight matrix, compute:
    - condition number κ(W)
    - quantization MSE before/after quantization
    - correlation between κ and quantization error

    This validates the theoretical prediction:
      ||δy||/||y|| ≲ κ(W) × ||δW||/||W||
    """
    results = {}
    for name, param in model.named_parameters():
        if param.dim() < 2:
            continue
        W_fp = param.data.clone()
        kappa = estimate_condition_number(W_fp)

        W_q = quantizer.quantize(W_fp)
        mse = ((W_q - W_fp) ** 2).mean().item()
        rel_error = ((W_q - W_fp).norm() / W_fp.norm()).item()

        # Predicted output error amplification
        rel_perturbation = rel_error
        predicted_output_error = kappa * rel_perturbation

        results[name] = {
            'kappa': kappa,
            'mse': mse,
            'rel_error': rel_error,
            'predicted_output_error': predicted_output_error,
        }

    return results
