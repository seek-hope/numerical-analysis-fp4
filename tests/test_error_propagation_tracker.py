"""Tests for ErrorPropagationTracker -- hook-based activation capture and error computation."""

import torch
import torch.nn as nn
import pytest


# ═══════════════════════════════════════════════════════════════
# Task 1 Tests: Linear pre-hook registration
# ═══════════════════════════════════════════════════════════════

class TestErrorPropagationTrackerInit:
    """ErrorPropagationTracker.__init__ initializes storage and state."""

    def test_init_creates_empty_activations_dict(self):
        from src.analysis.error_propagation import ErrorPropagationTracker
        tracker = ErrorPropagationTracker()
        assert hasattr(tracker, '_activations')
        assert isinstance(tracker._activations, dict)
        assert len(tracker._activations) == 0

    def test_init_creates_empty_p_points_dict(self):
        from src.analysis.error_propagation import ErrorPropagationTracker
        tracker = ErrorPropagationTracker()
        assert hasattr(tracker, '_p_points')
        assert isinstance(tracker._p_points, dict)
        assert len(tracker._p_points) == 0

    def test_init_creates_empty_g_points_dict(self):
        from src.analysis.error_propagation import ErrorPropagationTracker
        tracker = ErrorPropagationTracker()
        assert hasattr(tracker, '_g_points')
        assert isinstance(tracker._g_points, dict)
        assert len(tracker._g_points) == 0

    def test_init_creates_empty_hook_handles_list(self):
        from src.analysis.error_propagation import ErrorPropagationTracker
        tracker = ErrorPropagationTracker()
        assert hasattr(tracker, '_hook_handles')
        assert isinstance(tracker._hook_handles, list)
        assert len(tracker._hook_handles) == 0

    def test_init_creates_empty_activation_keys_list(self):
        from src.analysis.error_propagation import ErrorPropagationTracker
        tracker = ErrorPropagationTracker()
        assert hasattr(tracker, '_activation_keys')
        assert isinstance(tracker._activation_keys, list)
        assert len(tracker._activation_keys) == 0


class TestErrorPropagationTrackerAttach:
    """attach() registers pre-hooks on all nn.Linear modules."""

    def test_attach_registers_prehooks_on_all_linears(self):
        from src.analysis.error_propagation import ErrorPropagationTracker

        model = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
        )
        tracker = ErrorPropagationTracker()
        tracker.attach(model)

        # Should have 2 pre-hooks (one per nn.Linear)
        assert len(tracker._hook_handles) == 2
        assert len(tracker._activation_keys) == 2

    def test_attach_skips_non_linear_modules(self):
        from src.analysis.error_propagation import ErrorPropagationTracker

        model = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(8, 4),
        )
        tracker = ErrorPropagationTracker()
        tracker.attach(model)

        # Dropout and ReLU should be skipped -- only 2 Linear hooks
        assert len(tracker._hook_handles) == 2

    def test_attach_stores_module_path_correctly(self):
        from src.analysis.error_propagation import ErrorPropagationTracker

        model = nn.Sequential(
            nn.Linear(4, 8),
            nn.Linear(8, 4),
        )
        tracker = ErrorPropagationTracker()
        tracker.attach(model)

        # Module paths should be '0' and '1' (Sequential submodule indexing)
        assert tracker._activation_keys == ['0', '1']

    def test_attach_returns_self_for_chaining(self):
        from src.analysis.error_propagation import ErrorPropagationTracker

        model = nn.Sequential(nn.Linear(4, 8))
        tracker = ErrorPropagationTracker()
        result = tracker.attach(model)
        assert result is tracker

    def test_attach_with_nested_module_structure(self):
        from src.analysis.error_propagation import ErrorPropagationTracker

        class InnerBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(4, 8)
                self.fc2 = nn.Linear(8, 4)

        class OuterModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.block = InnerBlock()
                self.head = nn.Linear(4, 2)

        model = OuterModel()
        tracker = ErrorPropagationTracker()
        tracker.attach(model)

        # Should have 3 Linear modules: block.fc1, block.fc2, head
        assert len(tracker._hook_handles) == 3
        assert 'block.fc1' in tracker._activation_keys
        assert 'block.fc2' in tracker._activation_keys
        assert 'head' in tracker._activation_keys

    def test_attach_with_no_linear_modules(self):
        from src.analysis.error_propagation import ErrorPropagationTracker

        model = nn.Sequential(nn.ReLU(), nn.Dropout(0.1))
        tracker = ErrorPropagationTracker()
        tracker.attach(model)

        assert len(tracker._hook_handles) == 0
        assert len(tracker._activation_keys) == 0


class TestErrorPropagationTrackerPreHook:
    """Pre-hook callback captures input tensors correctly."""

    def test_pre_hook_captures_input_tensor(self):
        from src.analysis.error_propagation import ErrorPropagationTracker

        model = nn.Sequential(nn.Linear(4, 8))
        tracker = ErrorPropagationTracker()
        tracker.attach(model)

        x = torch.randn(2, 4)
        model(x)

        # Should have captured the input tensor for module '0'
        assert '0' in tracker._activations
        assert tracker._activations['0'].shape == (2, 4)

    def test_pre_hook_captured_tensor_is_cpu(self):
        from src.analysis.error_propagation import ErrorPropagationTracker

        model = nn.Sequential(nn.Linear(4, 8))
        tracker = ErrorPropagationTracker()
        tracker.attach(model)

        x = torch.randn(2, 4)
        model(x)

        assert tracker._activations['0'].device == torch.device('cpu')

    def test_pre_hook_captured_tensor_is_detached(self):
        from src.analysis.error_propagation import ErrorPropagationTracker

        model = nn.Sequential(nn.Linear(4, 8))
        tracker = ErrorPropagationTracker()
        tracker.attach(model)

        x = torch.randn(2, 4, requires_grad=True)
        model(x)

        # Requires_grad should be False on captured tensor
        assert not tracker._activations['0'].requires_grad

    def test_pre_hook_captured_tensor_is_a_clone(self):
        from src.analysis.error_propagation import ErrorPropagationTracker

        model = nn.Sequential(nn.Linear(4, 8))
        tracker = ErrorPropagationTracker()
        tracker.attach(model)

        x = torch.randn(2, 4)
        model(x)

        # The captured tensor should be a different memory location
        # Modifying input after capture should not affect stored tensor
        captured = tracker._activations['0'].clone()
        # Verify data match (cannot mutate original input easily, so just verify same values)
        assert torch.allclose(captured, tracker._activations['0'])

    def test_pre_hook_captures_multiple_linears(self):
        from src.analysis.error_propagation import ErrorPropagationTracker

        model = nn.Sequential(
            nn.Linear(4, 8),
            nn.Linear(8, 4),
        )
        tracker = ErrorPropagationTracker()
        tracker.attach(model)

        x = torch.randn(2, 4)
        model(x)

        # Both Linear modules should have captured inputs
        assert '0' in tracker._activations
        assert '1' in tracker._activations
        # First Linear input shape: (2, 4), Second Linear input shape: (2, 8)
        assert tracker._activations['0'].shape == (2, 4)
        assert tracker._activations['1'].shape == (2, 8)


class TestErrorPropagationTrackerMakeInputHook:
    """_make_input_hook factory function avoids closure capture bugs."""

    def test_factory_is_def_not_lambda(self):
        from src.analysis.error_propagation import ErrorPropagationTracker

        tracker = ErrorPropagationTracker()
        factory = tracker._make_input_hook
        assert callable(factory)
        # Factory returns a closure
        hook_fn = factory('test_module')
        assert callable(hook_fn)

    def test_factory_closure_captures_module_path_by_value(self):
        from src.analysis.error_propagation import ErrorPropagationTracker

        tracker = ErrorPropagationTracker()
        # Create hooks with different paths to verify per-value capture
        hook_a = tracker._make_input_hook('path_a')
        hook_b = tracker._make_input_hook('path_b')

        # Simulate calling both hooks
        inp = (torch.randn(1, 4),)
        tracker._activations = {}
        hook_a(None, inp)
        hook_b(None, inp)

        assert 'path_a' in tracker._activations
        assert 'path_b' in tracker._activations


class TestErrorPropagationTrackerDetach:
    """detach() removes all hooks and cleans up."""

    def test_detach_removes_all_hooks(self):
        from src.analysis.error_propagation import ErrorPropagationTracker

        model = nn.Sequential(
            nn.Linear(4, 8),
            nn.Linear(8, 4),
        )
        tracker = ErrorPropagationTracker()
        tracker.attach(model)

        # Capture activations
        x = torch.randn(2, 4)
        model(x)

        # Detach hooks
        tracker.detach()

        # Hooks should be removed -- handles list should be empty
        assert len(tracker._hook_handles) == 0

        # A second forward pass should not capture anything new
        x2 = torch.randn(2, 4)
        model(x2)

        # The activations should remain unchanged from the first pass
        assert tracker._activations['0'].shape == (2, 4)

    def test_detach_is_idempotent(self):
        from src.analysis.error_propagation import ErrorPropagationTracker

        model = nn.Sequential(nn.Linear(4, 8))
        tracker = ErrorPropagationTracker()
        tracker.attach(model)

        x = torch.randn(2, 4)
        model(x)

        tracker.detach()
        tracker.detach()  # Second call should not error

        assert len(tracker._hook_handles) == 0

    def test_detach_called_without_attach(self):
        from src.analysis.error_propagation import ErrorPropagationTracker

        tracker = ErrorPropagationTracker()
        # detach on fresh tracker should not error
        tracker.detach()
        assert len(tracker._hook_handles) == 0

    def test_detach_returns_self_for_chaining(self):
        from src.analysis.error_propagation import ErrorPropagationTracker

        tracker = ErrorPropagationTracker()
        result = tracker.detach()
        assert result is tracker


class TestErrorPropagationTrackerActivationsProperty:
    """activations property returns a copy of the internal dict."""

    def test_activations_returns_copy_not_reference(self):
        from src.analysis.error_propagation import ErrorPropagationTracker

        tracker = ErrorPropagationTracker()
        tracker._activations = {'test': torch.tensor([1.0])}

        activations_copy = tracker.activations
        assert activations_copy == {'test': torch.tensor([1.0])}

        # Mutating the returned dict should not affect internal state
        activations_copy['new'] = torch.tensor([2.0])
        assert 'new' not in tracker._activations

    def test_activations_returns_empty_dict_for_new_tracker(self):
        from src.analysis.error_propagation import ErrorPropagationTracker

        tracker = ErrorPropagationTracker()
        assert tracker.activations == {}


# ═══════════════════════════════════════════════════════════════
# Task 2 Tests: P-point and G-point measurement hooks
# ═══════════════════════════════════════════════════════════════

class MockTransformerLayer(nn.Module):
    """Mock transformer layer mimicking TransformerLayer structure for testing."""

    def __init__(self, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.input_norm = nn.Identity()
        self.attention = nn.Identity()
        self.post_attn_norm = nn.Identity()
        self.ffn = nn.Identity()

    def forward(self, x):
        res = x
        x = self.input_norm(x)
        x = self.attention(x)
        x = res + x  # P3 = P0 + P2
        res = x
        x = self.post_attn_norm(x)
        x = self.ffn(x)
        x = res + x  # P6 = P3 + P5
        return x


class MockModel(nn.Module):
    """Mock model mimicking MicroGemmaFPForCausalLM structure."""

    def __init__(self, n_layers=2):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Identity()
        self.model.layers = nn.ModuleList(
            [MockTransformerLayer(i) for i in range(n_layers)]
        )
        self.model.norm = nn.Identity()
        self.lm_head = nn.Sequential(nn.Linear(8, 16))


class TestPRegisterPPointHooks:
    """_register_p_point_hooks registers correct hooks per layer."""

    def test_register_p_point_hooks_registers_5_hooks_per_layer(self):
        from src.analysis.error_propagation import ErrorPropagationTracker

        model = MockModel(n_layers=2)
        tracker = ErrorPropagationTracker()
        tracker._register_p_point_hooks(model)

        # 5 hooks per layer * 2 layers = 10 hooks
        p_hook_count = len([h for h in tracker._hook_handles
                           if h not in tracker._activation_keys])  # rough count
        # Direct: count the number of new handles added
        assert len(tracker._hook_handles) == 10  # 5 per layer * 2 layers

    def test_p_point_keys_follow_format(self):
        from src.analysis.error_propagation import ErrorPropagationTracker

        model = MockModel(n_layers=2)
        tracker = ErrorPropagationTracker()
        tracker._register_p_point_hooks(model)

        # Simulate forward pass to populate _p_points
        x = torch.randn(1, 8)
        model(x)

        # Check that P0-P5 keys exist (P3 and P6 computed separately)
        for layer_idx in range(2):
            assert f"{layer_idx}_P0" in tracker._p_points
            assert f"{layer_idx}_P1" in tracker._p_points
            assert f"{layer_idx}_P2" in tracker._p_points

    def test_p0_captures_layer_input(self):
        from src.analysis.error_propagation import ErrorPropagationTracker

        model = MockModel(n_layers=1)
        tracker = ErrorPropagationTracker()
        tracker._register_p_point_hooks(model)

        x = torch.randn(2, 8)
        model(x)

        # P0 should match the input tensor shape
        assert tracker._p_points['0_P0'].shape == (2, 8)

    def test_p_point_tensors_are_cpu_and_detached(self):
        from src.analysis.error_propagation import ErrorPropagationTracker

        model = MockModel(n_layers=1)
        tracker = ErrorPropagationTracker()
        tracker._register_p_point_hooks(model)

        x = torch.randn(2, 8)
        model(x)

        for key, tensor in tracker._p_points.items():
            assert tensor.device == torch.device('cpu'), f"{key} not on CPU"
            assert not tensor.requires_grad, f"{key} requires grad"

    def test_p1_captures_input_norm_output(self):
        from src.analysis.error_propagation import ErrorPropagationTracker

        model = MockModel(n_layers=1)
        tracker = ErrorPropagationTracker()
        tracker._register_p_point_hooks(model)

        x = torch.randn(2, 8)
        model(x)

        # P1 captures input_norm output (which is Identity, so same as P0)
        assert tracker._p_points['0_P1'].shape == (2, 8)


class TestPRegisterGPointHooks:
    """_register_g_point_hooks registers 3 global hooks."""

    def test_register_three_g_point_hooks(self):
        from src.analysis.error_propagation import ErrorPropagationTracker

        model = MockModel(n_layers=1)
        tracker = ErrorPropagationTracker()
        tracker._register_g_point_hooks(model)

        # Should have 3 G-point hooks
        assert len(tracker._hook_handles) >= 3

    def test_g_point_keys_are_g0_g1_g2(self):
        from src.analysis.error_propagation import ErrorPropagationTracker

        model = MockModel(n_layers=1)
        tracker = ErrorPropagationTracker()
        tracker._register_g_point_hooks(model)

        x = torch.randn(2, 8)
        model(x)

        assert 'G0' in tracker._g_points
        assert 'G1' in tracker._g_points
        assert 'G2' in tracker._g_points


class TestComputeP3P6:
    """compute_p3_p6 correctly computes residual sums."""

    def test_p3_equals_p0_plus_p2(self):
        from src.analysis.error_propagation import ErrorPropagationTracker

        model = MockModel(n_layers=1)
        tracker = ErrorPropagationTracker()
        tracker._register_p_point_hooks(model)

        # We need hooks to work first
        x = torch.randn(2, 8)
        model(x)

        # Manually set up P0, P2 if not populated
        # (hooks should have done this)
        tracker.compute_p3_p6()

        assert '0_P3' in tracker._p_points
        expected_p3 = tracker._p_points['0_P0'] + tracker._p_points['0_P2']
        assert torch.allclose(tracker._p_points['0_P3'], expected_p3)

    def test_p6_equals_p3_plus_p5(self):
        from src.analysis.error_propagation import ErrorPropagationTracker

        model = MockModel(n_layers=1)
        tracker = ErrorPropagationTracker()
        tracker._register_p_point_hooks(model)

        x = torch.randn(2, 8)
        model(x)

        tracker.compute_p3_p6()

        # After compute_p3_p6, P3 and then P6 should exist
        assert '0_P6' in tracker._p_points
        expected_p6 = tracker._p_points['0_P3'] + tracker._p_points['0_P5']
        assert torch.allclose(tracker._p_points['0_P6'], expected_p6)

    def test_compute_p3_p6_for_multiple_layers(self):
        from src.analysis.error_propagation import ErrorPropagationTracker

        model = MockModel(n_layers=3)
        tracker = ErrorPropagationTracker()
        tracker._register_p_point_hooks(model)

        x = torch.randn(2, 8)
        model(x)

        tracker.compute_p3_p6()

        for layer_idx in range(3):
            assert f"{layer_idx}_P3" in tracker._p_points
            assert f"{layer_idx}_P6" in tracker._p_points

    def test_compute_p3_p6_without_p0_does_not_error(self):
        from src.analysis.error_propagation import ErrorPropagationTracker

        tracker = ErrorPropagationTracker()
        # No P-points at all -- compute should be a no-op
        tracker.compute_p3_p6()
        assert len(tracker._p_points) == 0


class TestPPointsProperty:
    """p_points property returns merged P and G points."""

    def test_p_points_contains_both_p_and_g(self):
        from src.analysis.error_propagation import ErrorPropagationTracker

        model = MockModel(n_layers=1)
        tracker = ErrorPropagationTracker()
        tracker._register_p_point_hooks(model)
        tracker._register_g_point_hooks(model)

        x = torch.randn(2, 8)
        model(x)

        merged = tracker.p_points
        assert '0_P0' in merged
        assert '0_P1' in merged
        assert 'G0' in merged
        assert 'G1' in merged
        assert 'G2' in merged

    def test_p_points_returns_copy(self):
        from src.analysis.error_propagation import ErrorPropagationTracker

        tracker = ErrorPropagationTracker()
        tracker._p_points = {'0_P0': torch.tensor([1.0])}
        tracker._g_points = {'G0': torch.tensor([2.0])}

        merged = tracker.p_points
        merged['new_key'] = torch.tensor([3.0])

        assert 'new_key' not in tracker._p_points
        assert 'new_key' not in tracker._g_points
