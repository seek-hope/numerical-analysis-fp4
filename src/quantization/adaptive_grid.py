"""Per-layer adaptive FP4 quantization grids via Lloyd-Max optimization.

Core idea: Instead of using the same E2M1/NF4 grid for all layers, compute
an optimal 16-level quantizer tailored to each layer's weight distribution.

The Lloyd-Max algorithm iteratively minimizes:
  MSE = E[(w - Q(w))^2]
where Q(w) maps each weight to the nearest grid point.

κ-weighted variant: weights the MSE objective by the layer's condition number,
giving high-κ layers more conservative (denser) grids.

References:
  - Lloyd (1982). "Least squares quantization in PCM."
  - Max (1960). "Quantizing for minimum distortion."
"""

import torch


def lloyd_max_grid(
    weights: torch.Tensor,
    n_levels: int = 16,
    n_iter: int = 20,
    kappa: float = 1.0,
    kappa_weight: float = 0.0,
) -> torch.Tensor:
    """Compute optimal quantization grid via Lloyd-Max iteration.

    Args:
        weights: weight tensor to quantize (any shape, flattened internally)
        n_levels: number of quantization levels (default 16 for FP4)
        n_iter: maximum Lloyd-Max iterations
        kappa: condition number of the weight matrix (κ ≥ 1.0)
        kappa_weight: how much κ influences the grid (0 = uniform optimization,
                      1 = full κ weighting, higher = more conservative)

    Returns:
        grid: sorted tensor of n_levels quantization levels (both signs,
              symmetric around zero with half the levels on each side)
    """
    w = weights.detach().flatten().float()
    n_half = n_levels // 2  # 8 levels for positive side

    # Initial grid: NF4-inspired normal quantiles
    # Use k-means++ style initialization on the absolute weight distribution
    w_abs = w.abs()
    w_sorted, _ = w_abs.sort()

    # Initialize with equal-probability partition (NF4 principle)
    init_idx = torch.linspace(0, len(w_sorted) - 1, n_half + 1).long()
    grid_pos = torch.zeros(n_half, device=w.device)
    for i in range(n_half):
        grid_pos[i] = w_sorted[
            init_idx[i]:init_idx[i + 1]
        ].mean()

    # κ-weighted sample weights: high κ → penalize large errors more
    if kappa_weight > 0 and kappa > 1.0:
        # Weights near zero contribute less, tails contribute more
        sample_weight = 1.0 + kappa_weight * (kappa - 1.0) * (w_abs / w_abs.max())
    else:
        sample_weight = torch.ones_like(w_abs)

    # Lloyd-Max iteration
    for it in range(n_iter):
        # Step 1: Assign each value to nearest grid point (Voronoi partition)
        # Compute pairwise distances (n_samples, n_half)
        dists = (w_abs.unsqueeze(1) - grid_pos.unsqueeze(0)).abs()
        assignments = dists.argmin(dim=1)

        # Step 2: Update grid points to centroids of assigned values
        new_grid = torch.zeros_like(grid_pos)
        for j in range(n_half):
            mask = (assignments == j)
            if mask.sum() > 0:
                sw = sample_weight[mask]
                new_grid[j] = (w_abs[mask] * sw).sum() / sw.sum()
            else:
                new_grid[j] = grid_pos[j]  # Keep old if empty

        # Check convergence
        shift = (new_grid - grid_pos).abs().max()
        grid_pos = new_grid
        if shift < 1e-6:
            break

    # Build full symmetric grid: {-g, ..., -0, 0, +0, ..., +g}
    # Exclude exact zero if not already present (add small positive to
    # maintain FP4-compatible format with sign bit)
    grid_pos_sorted, _ = grid_pos.sort()
    if grid_pos_sorted[0] < 1e-8:
        grid_pos_sorted = grid_pos_sorted[1:]  # Remove zero, handle separately

    full_grid = torch.cat([
        -grid_pos_sorted.flip(0),
        torch.tensor([0.0], device=w.device),
        grid_pos_sorted,
    ])

    return full_grid


class AdaptiveGridQuantizer:
    """Per-layer adaptive FP4 quantization with optional κ-weighting.

    Usage:
        q = AdaptiveGridQuantizer(kappa_weight=0.5)
        q.calibrate(model)                        # Compute grids per layer
        q.quantize_model(model)                   # Apply per-layer quantization
    """

    def __init__(self, n_levels: int = 16, kappa_weight: float = 0.5,
                 n_iter: int = 20):
        self.n_levels = n_levels
        self.kappa_weight = kappa_weight
        self.n_iter = n_iter
        self.grids = {}  # layer_name → tensor of grid points

    @torch.no_grad()
    def calibrate(self, model):
        """Compute optimal grid for each layer's weight distribution.

        Uses Lloyd-Max with optional κ-weighting. Should be called once
        before quantization.
        """
        from src.analysis.condition import estimate_condition_number

        for name, param in model.named_parameters():
            if param.dim() < 2:
                continue
            if 'embed' in name.lower() or 'lm_head' in name.lower():
                continue

            kappa = estimate_condition_number(param.data, n_iter=3)
            grid = lloyd_max_grid(
                param.data,
                n_levels=self.n_levels,
                n_iter=self.n_iter,
                kappa=kappa,
                kappa_weight=self.kappa_weight,
            )
            self.grids[name] = grid.to(param.device)

            # Diagnostic
            nz = (grid != 0).sum().item()
            print(f"  {name:50s} κ={kappa:.1f} grid=[{grid.min().item():.3f},"
                  f" {grid.max().item():.3f}] nz={nz}")

    @torch.no_grad()
    def quantize_model(self, model):
        """Apply per-layer adaptive quantization in-place."""
        for name, param in model.named_parameters():
            if name not in self.grids:
                continue
            grid = self.grids[name]
            W_q = self._quantize_to_grid(param.data, grid)
            param.data.copy_(W_q)

    @torch.no_grad()
    def quantize_tensor(self, x: torch.Tensor, name: str) -> torch.Tensor:
        """Quantize a single tensor using its pre-computed grid."""
        if name in self.grids:
            return self._quantize_to_grid(x, self.grids[name].to(x.device))
        return x

    def _quantize_to_grid(self, x: torch.Tensor, grid: torch.Tensor):
        """Round x to nearest grid point."""
        x_abs = x.abs()
        grid_abs = grid.abs()
        grid_abs_sorted, _ = grid_abs.sort()

        # Find nearest positive grid point
        idx = torch.searchsorted(grid_abs_sorted, x_abs.clamp(0, grid_abs_sorted[-1]))
        idx = idx.clamp(0, len(grid_abs_sorted) - 1)

        lower = grid_abs_sorted[(idx - 1).clamp(0)]
        upper = grid_abs_sorted[idx.clamp(0, len(grid_abs_sorted) - 1)]

        dist_lower = (x_abs - lower).abs()
        dist_upper = (upper - x_abs).abs()
        x_q_abs = torch.where(dist_lower <= dist_upper, lower, upper)

        return torch.sign(x) * x_q_abs
