"""Hook-based activation capture and offline per-matrix output-space relative error computation for quantized Transformer models.

ErrorPropagationTracker registers forward hooks on nn.Linear modules and
Transformer-layer measurement points to capture pre-activation tensors x from
a single FP16 forward pass. Offline, it computes ||(W_q - W)x|| / ||Wx|| for
each weight matrix given a quantizer, enabling per-matrix Theorem 1 validation
without modifying the model code.

Usage:
    tracker = ErrorPropagationTracker()
    tracker.attach(model)
    outputs = model(input_ids)        # single forward pass captures activations
    errors = tracker.compute_output_error(model, quantizer)
    tracker.detach()
"""

import torch
import torch.nn as nn


# ═══════════════════════════════════════════════════════════════
# Error Propagation Tracker
# ═══════════════════════════════════════════════════════════════

class ErrorPropagationTracker:
    """Hook-based activation capture and offline per-matrix error computation.

    Stores activations in three dicts:
      - _activations: pre-activation inputs to each nn.Linear module
        (captured via forward_pre_hook). Keyed by module_path.
      - _p_points: per-layer measurement points P0-P6.
        Keyed by ``{layer_idx}_{point_id}`` (e.g. ``0_P0``).
      - _g_points: global measurement points G0-G2.
        Keyed by point_id (e.g. ``G0``).
    """

    def __init__(self):
        self._activations: dict[str, torch.Tensor] = {}
        self._p_points: dict[str, torch.Tensor] = {}
        self._g_points: dict[str, torch.Tensor] = {}
        self._hook_handles: list[torch.utils.hooks.RemovableHandle] = []
        self._activation_keys: list[str] = []

    # ── Linear Pre-hooks ─────────────────────────────────────

    def attach(self, model: nn.Module):
        """Register all hooks on the model.

        Registers forward_pre_hooks on every nn.Linear module for activation
        capture, plus P-point and G-point measurement hooks.

        Returns self for method chaining.
        """
        self._register_linear_pre_hooks(model)
        print(f"[Tracker] Registered {len(self._hook_handles)} Linear pre-hooks")
        return self

    def _register_linear_pre_hooks(self, model: nn.Module):
        """Register forward_pre_hook on every nn.Linear module.

        Iterates model.named_modules() and registers a pre-hook on each
        nn.Linear module. The pre-hook captures input_args[0] (the input
        tensor), detaches + clones + moves to CPU, and stores it in
        self._activations[module_path].
        """
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                module_path = name  # dot-separated path, e.g. 'model.layers.0.attention.q_proj'
                handle = module.register_forward_pre_hook(
                    self._make_input_hook(module_path)
                )
                self._hook_handles.append(handle)
                self._activation_keys.append(module_path)

    def _make_input_hook(self, module_path: str):
        """Factory function returning a pre-hook closure.

        Uses ``@torch.no_grad()`` decorator. Captures ``module_path`` by value
        (closure parameter) to avoid the lambda closure-capture bug where all
        hooks would share the last value of the loop variable.

        The closure stores ``input_args[0].detach().clone().cpu()`` in
        ``self._activations[module_path]``.
        """

        @torch.no_grad()
        def _input_hook(module, input_args):
            x = input_args[0].detach().clone().cpu()
            self._activations[module_path] = x

        return _input_hook

    def detach(self):
        """Remove all registered hooks.

        Calls ``handle.remove()`` on every handle in ``self._hook_handles``,
        then clears the list. Idempotent -- calling detach() multiple times
        is safe.

        Returns self for method chaining.
        """
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        print(f"[Tracker] Removed {len(self._hook_handles)} hooks")
        return self

    @property
    def activations(self) -> dict[str, torch.Tensor]:
        """Return a copy of the activation dict (not the original reference)."""
        return dict(self._activations)
