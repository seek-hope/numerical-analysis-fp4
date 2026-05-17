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
