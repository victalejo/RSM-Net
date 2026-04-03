"""Tests for SubmatrixLinear layer."""

import pytest
import torch
import torch.nn as nn

from rsm_net.layers import SubmatrixLinear


class TestSubmatrixLinear:
    """Tests for the core SubmatrixLinear layer."""

    @pytest.fixture
    def layer(self) -> SubmatrixLinear:
        return SubmatrixLinear(in_features=32, out_features=16, rank=4, key_dim=8)

    def test_base_forward_shape(self, layer: SubmatrixLinear) -> None:
        x = torch.randn(4, 32)
        out = layer(x)
        assert out.shape == (4, 16)

    def test_add_submatrix_increases_count(self, layer: SubmatrixLinear) -> None:
        assert layer.num_submatrices == 0
        layer.add_submatrix(task_id=0)
        assert layer.num_submatrices == 1
        layer.add_submatrix(task_id=1)
        assert layer.num_submatrices == 2

    def test_forward_with_submatrices(self, layer: SubmatrixLinear) -> None:
        layer.add_submatrix(task_id=0)
        layer.add_submatrix(task_id=1)
        x = torch.randn(4, 32)
        out = layer(x)
        assert out.shape == (4, 16)

    def test_gates_are_sparse(self, layer: SubmatrixLinear) -> None:
        layer.add_submatrix(task_id=0)
        layer.add_submatrix(task_id=1)
        layer.add_submatrix(task_id=2)
        x = torch.randn(8, 32)
        gates = layer.compute_gates(x)
        assert gates.shape == (8, 3)
        # Gates should be non-negative
        assert (gates >= 0).all()
        # Gates should sum to ~1 per sample
        sums = gates.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(8), atol=1e-4)

    def test_empty_gates_returns_zero_columns(self, layer: SubmatrixLinear) -> None:
        x = torch.randn(4, 32)
        gates = layer.compute_gates(x)
        assert gates.shape == (4, 0)

    def test_freeze_base(self, layer: SubmatrixLinear) -> None:
        layer.freeze_base()
        for param in layer.W_base.parameters():
            assert not param.requires_grad

    def test_freeze_submatrix(self, layer: SubmatrixLinear) -> None:
        layer.add_submatrix(task_id=0)
        layer.freeze_submatrix(0)
        assert not layer.submatrix_A[0].requires_grad
        assert not layer.submatrix_B[0].requires_grad
        assert not layer.key_embeddings[0].requires_grad

    def test_gradient_flows_through_submatrix(self, layer: SubmatrixLinear) -> None:
        layer.add_submatrix(task_id=0)
        x = torch.randn(4, 32)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        assert layer.submatrix_A[0].grad is not None
        assert layer.submatrix_B[0].grad is not None

    def test_prune_removes_submatrix(self, layer: SubmatrixLinear) -> None:
        layer.add_submatrix(task_id=0)
        layer.add_submatrix(task_id=1)
        # Set importance: first is low, second is high
        layer.importance_scores[0] = 0.001
        layer.importance_scores[1] = 0.5
        pruned = layer.prune(threshold=0.01)
        assert pruned == 1
        assert layer.num_submatrices == 1

    def test_task_mapping_survives_pruning(self, layer: SubmatrixLinear) -> None:
        layer.add_submatrix(task_id=0)
        layer.add_submatrix(task_id=1)
        layer.add_submatrix(task_id=2)
        # Prune middle one
        layer.importance_scores[0] = 0.5
        layer.importance_scores[1] = 0.001
        layer.importance_scores[2] = 0.5
        layer.prune(threshold=0.01)
        assert layer.num_submatrices == 2
        # Task 0 should still map to index 0
        assert layer._task_to_submatrix[0] == 0
        # Task 2 should now map to index 1 (shifted down)
        assert layer._task_to_submatrix[2] == 1
        # Task 1 should be removed from mapping
        assert 1 not in layer._task_to_submatrix

    def test_lora_init_ab_product_is_zero(self, layer: SubmatrixLinear) -> None:
        """A @ B should be zero initially (LoRA invariant)."""
        layer.add_submatrix(task_id=0)
        A = layer.submatrix_A[0]
        B = layer.submatrix_B[0]
        product = A @ B
        # B is initialized to zeros, so A @ B = 0
        assert torch.allclose(product, torch.zeros_like(product), atol=1e-8)

    def test_get_trainable_params_for_task(self, layer: SubmatrixLinear) -> None:
        layer.add_submatrix(task_id=0)
        layer.add_submatrix(task_id=1)
        # All tasks: query_proj (weight + bias) + A + B + key = 5 params
        # query_proj is trainable for all tasks (protected by EWC-light)
        params_t0 = layer.get_trainable_params_for_task(task_id=0)
        assert len(params_t0) == 5
        params_t1 = layer.get_trainable_params_for_task(task_id=1)
        assert len(params_t1) == 5

    def test_importance_tracking(self, layer: SubmatrixLinear) -> None:
        layer.add_submatrix(task_id=0)
        layer.train()
        x = torch.randn(8, 32)
        # Run a few forward passes
        for _ in range(10):
            layer(x)
        layer.update_importance()
        assert layer.importance_scores[0].item() > 0

    def test_recursive_depth(self) -> None:
        layer = SubmatrixLinear(
            in_features=32, out_features=16, rank=4, key_dim=8, max_depth=2
        )
        layer.add_submatrix(task_id=0)
        assert len(layer.child_layers) == 1
        x = torch.randn(4, 32)
        out = layer(x)
        assert out.shape == (4, 16)
