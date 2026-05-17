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
        self._register_p_point_hooks(model)
        self._register_g_point_hooks(model)
        print(f"[Tracker] Registered {len(self._hook_handles)} hooks"
              f" (Linear pre-hooks + P-points + G-points)")
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

    # ── Per-layer P-point Hooks ──────────────────────────────

    def _register_p_point_hooks(self, model: nn.Module):
        """Register 5 hooks per transformer layer (P0-P5).

        P0: forward_pre_hook on the layer itself (captures layer input).
        P1: forward_hook on layer.input_norm (captures norm output).
        P2: forward_hook on layer.attention (captures attention output).
        P4: forward_hook on layer.post_attn_norm (captures post-attn norm output).
        P5: forward_hook on layer.ffn (captures FFN output).

        P3 and P6 are computed via compute_p3_p6().
        """
        # Access layers via model.model.layers (MicroGemmaFPModel path)
        layers = None
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = model.model.layers
        elif hasattr(model, 'layers'):
            layers = model.layers
        else:
            print("[Tracker] WARNING: could not find model.layers for P-point hooks")
            return

        for layer_idx, layer in enumerate(layers):
            # P0: pre-hook on the layer itself (captures input to forward)
            handle = layer.register_forward_pre_hook(
                self._make_p_hook(layer_idx, 'P0')
            )
            self._hook_handles.append(handle)

            # P1: after input_norm
            handle = layer.input_norm.register_forward_hook(
                self._make_p_hook(layer_idx, 'P1')
            )
            self._hook_handles.append(handle)

            # P2: after attention
            handle = layer.attention.register_forward_hook(
                self._make_p_hook(layer_idx, 'P2')
            )
            self._hook_handles.append(handle)

            # P4: after post_attn_norm
            handle = layer.post_attn_norm.register_forward_hook(
                self._make_p_hook(layer_idx, 'P4')
            )
            self._hook_handles.append(handle)

            # P5: after FFN
            handle = layer.ffn.register_forward_hook(
                self._make_p_hook(layer_idx, 'P5')
            )
            self._hook_handles.append(handle)

    def _make_p_hook(self, layer_idx: int, point_id: str):
        """Factory function returning a P-point hook closure.

        For pre-hook (P0): captures input_args[0].
        For forward hooks (P1/P2/P4/P5): captures output tensor.

        Stores in ``self._p_points[f"{layer_idx}_{point_id}"]``.
        All tensors are detached, cloned, and moved to CPU.
        """

        @torch.no_grad()
        def _p_hook(module, input_args, output=None):
            if output is not None:
                tensor = output.detach().clone().cpu()
            else:
                tensor = input_args[0].detach().clone().cpu()
            self._p_points[f"{layer_idx}_{point_id}"] = tensor

        return _p_hook

    def compute_p3_p6(self):
        """Compute residual-add outputs P3 and P6 from captured P-points.

        P3 = P0 + P2 (pre-norm attention residual add)
        P6 = P3 + P5 (pre-norm FFN residual add)

        Computed for every layer where source tensors exist.
        """
        for key in list(self._p_points.keys()):
            if not key.endswith('_P0'):
                continue
            layer_prefix = key[:-3]  # Remove '_P0' suffix

            p0_key = key
            p2_key = f"{layer_prefix}_P2"
            p5_key = f"{layer_prefix}_P5"

            # Compute P3 = P0 + P2
            if p2_key in self._p_points:
                p3_key = f"{layer_prefix}_P3"
                if p3_key not in self._p_points:
                    p3 = self._p_points[p0_key] + self._p_points[p2_key]
                    self._p_points[p3_key] = p3.clone()

            # Compute P6 = P3 + P5
            p3_key = f"{layer_prefix}_P3"
            if p3_key in self._p_points and p5_key in self._p_points:
                p6_key = f"{layer_prefix}_P6"
                if p6_key not in self._p_points:
                    p6 = self._p_points[p3_key] + self._p_points[p5_key]
                    self._p_points[p6_key] = p6.clone()

    # ── Global G-point Hooks ─────────────────────────────────

    def _register_g_point_hooks(self, model: nn.Module):
        """Register 3 global measurement hooks (G0-G2).

        G0: forward_hook on model.model.embed_tokens (after embedding).
        G1: forward_hook on model.model.norm (after final RMSNorm).
        G2: forward_hook on model.lm_head (after output projection).
        """
        # G0: embed_tokens
        if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
            handle = model.model.embed_tokens.register_forward_hook(
                self._make_g_hook('G0')
            )
            self._hook_handles.append(handle)

        # G1: norm (final RMSNorm after layer stack)
        if hasattr(model, 'model') and hasattr(model.model, 'norm'):
            handle = model.model.norm.register_forward_hook(
                self._make_g_hook('G1')
            )
            self._hook_handles.append(handle)

        # G2: lm_head (final projection to vocab logits)
        if hasattr(model, 'lm_head'):
            handle = model.lm_head.register_forward_hook(
                self._make_g_hook('G2')
            )
            self._hook_handles.append(handle)

    def _make_g_hook(self, point_id: str):
        """Factory function returning a G-point hook closure.

        Captures module output, handling tuple outputs (e.g. lm_head
        returning (logits,) tuple) by taking tensor[0] in that case.
        """

        @torch.no_grad()
        def _g_hook(module, input_args, output):
            if isinstance(output, tuple):
                tensor = output[0].detach().clone().cpu()
            elif isinstance(output, torch.Tensor):
                tensor = output.detach().clone().cpu()
            else:
                # Fallback: try to extract tensor
                tensor = torch.as_tensor(output).detach().clone().cpu()
            self._g_points[point_id] = tensor

        return _g_hook

    @property
    def p_points(self) -> dict[str, torch.Tensor]:
        """Return merged P-point and G-point dict (copy, not reference)."""
        return {**self._p_points, **self._g_points}

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

    # ── Offline Error Computation ─────────────────────────────

    def _resolve_module(self, model: nn.Module, module_path: str):
        """Resolve a module from a dot-separated path.

        Splits ``module_path`` by '.' and traverses model attributes.
        Returns the resolved ``nn.Module`` or ``None`` if not found.
        """
        obj = model
        parts = module_path.split('.')
        for part in parts:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                return None
        return obj

    @torch.no_grad()
    def compute_output_error(self, model: nn.Module,
                             quantizer) -> dict[str, float]:
        """Compute per-matrix output-space relative error offline.

        For each captured activation, computes:
            ||(W_q - W)x|| / ||Wx||

        where W_q = quantizer.quantize(W_fp) with round-to-nearest.

        Args:
            model: The model containing the weight matrices.
            quantizer: Object with ``quantize(W)`` method.

        Returns:
            dict[str, float]: Module path -> relative error.
        """
        results: dict[str, float] = {}

        for module_path, x_cpu in self._activations.items():
            module = self._resolve_module(model, module_path)
            if module is None or not hasattr(module, 'weight'):
                continue

            W_fp = module.weight.data
            device = W_fp.device
            x = x_cpu.to(device)

            # y_fp = x @ W.T
            y_fp = x @ W_fp.T

            # y_q = x @ W_q.T
            W_q = quantizer.quantize(W_fp)
            y_q = x @ W_q.T

            # Relative error: ||y_q - y_fp|| / ||y_fp||
            denom = y_fp.norm().clamp(min=1e-12)
            err = (y_q - y_fp).norm() / denom
            results[module_path] = err.item()

        return results

    @torch.no_grad()
    def validate_null_measurement(self, model: nn.Module) -> float:
        """Validate the measurement pipeline with identity quantization.

        Computes ||(W - W)x|| / ||Wx|| which should be negligible (W_q == W_fp).
        Raises ``ValueError`` if any module's null error exceeds 1e-5.

        Returns:
            float: Maximum null error across all modules.
        """
        max_err = 0.0
        violations: list[tuple[str, float]] = []

        for module_path, x_cpu in self._activations.items():
            module = self._resolve_module(model, module_path)
            if module is None or not hasattr(module, 'weight'):
                continue

            W_fp = module.weight.data
            x = x_cpu.to(W_fp.device)

            # Identity quantization: W_q == W_fp
            y_fp = x @ W_fp.T
            y_q = x @ W_fp.T

            denom = y_fp.norm().clamp(min=1e-12)
            err = (y_q - y_fp).norm() / denom
            err_val = err.item()
            max_err = max(max_err, err_val)

            if err_val > 1e-5:
                violations.append((module_path, err_val))

        if violations:
            print(f"[WARN] Null measurement failed: {len(violations)} modules "
                  f"exceed 1e-5. Max error: {max_err:.2e}")
            for path, val in violations[:5]:
                print(f"  {path}: {val:.2e}")
            raise ValueError(
                f"[WARN] Null measurement failed: {len(violations)} modules "
                f"exceed 1e-5. Max error: {max_err:.2e}"
            )

        return max_err

    @property
    def activations(self) -> dict[str, torch.Tensor]:
        """Return a copy of the activation dict (not the original reference)."""
        return dict(self._activations)
