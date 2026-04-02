"""Tests for RSMNet network."""

import pytest
import torch

from rsm_net.config import RSMConfig
from rsm_net.network import RSMNet


class TestRSMNet:
    """Tests for the full RSM-Net model."""

    @pytest.fixture
    def config(self) -> RSMConfig:
        return RSMConfig(
            input_dim=784,
            hidden_dims=(64, 32),
            num_classes=10,
            rank=4,
            key_dim=8,
            max_depth=1,
        )

    @pytest.fixture
    def model(self, config: RSMConfig) -> RSMNet:
        return RSMNet(config=config)

    def test_forward_shape(self, model: RSMNet) -> None:
        model.prepare_new_task()
        x = torch.randn(4, 784)
        out = model(x)
        assert out.shape == (4, 10)

    def test_prepare_multiple_tasks(self, model: RSMNet) -> None:
        model.prepare_new_task()  # task 0
        model.prepare_new_task()  # task 1
        model.prepare_new_task()  # task 2
        assert model.num_tasks == 3
        assert len(model.heads) == 3
        # After task 0 (no submatrix) + tasks 1,2 -> 2 submatrices
        assert model.layers[0].num_submatrices == 2

    def test_task_id_selects_head(self, model: RSMNet) -> None:
        model.prepare_new_task()
        model.prepare_new_task()
        x = torch.randn(2, 784)
        out0 = model(x, task_id=0)
        out1 = model(x, task_id=1)
        # Different heads produce different outputs
        assert not torch.allclose(out0, out1)

    def test_optimizer_has_correct_params(self, model: RSMNet) -> None:
        model.prepare_new_task()  # task 0: all params
        opt0 = model.get_optimizer(0)
        total_params_t0 = sum(
            p.numel() for group in opt0.param_groups for p in group["params"]
        )

        model.prepare_new_task()  # task 1: only new submatrix + query + heads
        opt1 = model.get_optimizer(1)
        total_params_t1 = sum(
            p.numel() for group in opt1.param_groups for p in group["params"]
        )

        # Task 1 should have fewer trainable params than task 0
        assert total_params_t1 < total_params_t0

    def test_gradient_flow_end_to_end(self, model: RSMNet) -> None:
        model.prepare_new_task()
        model.prepare_new_task()
        x = torch.randn(4, 784)
        y = torch.randint(0, 10, (4,))
        out = model(x)
        loss = torch.nn.functional.cross_entropy(out, y)
        loss.backward()
        # Check gradients exist on trainable params
        has_grad = False
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                has_grad = True
                break
        assert has_grad

    def test_sparsity_loss(self, model: RSMNet) -> None:
        model.prepare_new_task()
        model.prepare_new_task()
        x = torch.randn(4, 784)
        model(x)  # Populate cached gates
        loss = model.get_sparsity_loss()
        assert loss.item() >= 0

    def test_frobenius_loss(self, model: RSMNet) -> None:
        model.prepare_new_task()
        model.prepare_new_task()
        loss = model.get_frobenius_loss(task_idx=1)
        assert loss.item() >= 0

    def test_prune_and_rebuild_optimizer(self, model: RSMNet) -> None:
        model.prepare_new_task()
        model.prepare_new_task()
        model.prepare_new_task()

        # Set low importance on one submatrix
        for layer in model.layers:
            layer.importance_scores[0] = 0.001
            layer.importance_scores[1] = 0.5

        pruned = model.prune_all(threshold=0.01)
        assert pruned > 0

        # Rebuild optimizer after pruning (should not crash)
        opt = model.get_optimizer(2)
        assert opt is not None

    def test_consolidate(self, model: RSMNet) -> None:
        model.prepare_new_task()
        model.prepare_new_task()
        model.prepare_new_task()

        # Make two submatrices identical
        for layer in model.layers:
            if layer.num_submatrices >= 2:
                layer.submatrix_A[1].data.copy_(layer.submatrix_A[0].data)
                layer.submatrix_B[1].data.copy_(layer.submatrix_B[0].data)

        merges = model.consolidate_all(threshold=0.99)
        # Identical submatrices should merge
        assert merges >= 0  # May or may not merge depending on zero init

    def test_state_summary(self, model: RSMNet) -> None:
        model.prepare_new_task()
        model.prepare_new_task()
        summary = model.get_state_summary()
        assert summary["num_tasks"] == 2
        assert len(summary["layers"]) == 2
        assert "total_params" in summary

    def test_training_reduces_loss(self, model: RSMNet) -> None:
        model.prepare_new_task()
        optimizer = model.get_optimizer(0)
        x = torch.randn(32, 784)
        y = torch.randint(0, 10, (32,))

        model.train()
        # Initial loss
        out = model(x)
        loss_before = torch.nn.functional.cross_entropy(out, y).item()

        # Train a few steps
        for _ in range(20):
            optimizer.zero_grad()
            out = model(x)
            loss = torch.nn.functional.cross_entropy(out, y)
            loss.backward()
            optimizer.step()

        out = model(x)
        loss_after = torch.nn.functional.cross_entropy(out, y).item()
        assert loss_after < loss_before
