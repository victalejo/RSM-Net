"""Tests for sparsemax activation function."""

import pytest
import torch

from rsm_net.activations import sparsemax


class TestSparsemax:
    """Tests for sparsemax activation."""

    def test_output_sums_to_at_most_one(self) -> None:
        scores = torch.randn(10, 5)
        result = sparsemax(scores, dim=-1)
        sums = result.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(10), atol=1e-5)

    def test_produces_exact_zeros(self) -> None:
        # With a dominant entry, others should be exactly zero
        scores = torch.tensor([[10.0, 0.0, 0.0, 0.0]])
        result = sparsemax(scores, dim=-1)
        assert result[0, 0].item() == pytest.approx(1.0, abs=1e-5)
        assert result[0, 1].item() == 0.0
        assert result[0, 2].item() == 0.0
        assert result[0, 3].item() == 0.0

    def test_all_equal_inputs(self) -> None:
        # All equal -> uniform distribution
        scores = torch.ones(1, 4) * 5.0
        result = sparsemax(scores, dim=-1)
        expected = torch.ones(1, 4) * 0.25
        assert torch.allclose(result, expected, atol=1e-5)

    def test_non_negative_output(self) -> None:
        scores = torch.randn(20, 8)
        result = sparsemax(scores, dim=-1)
        assert (result >= 0).all()

    def test_single_element(self) -> None:
        scores = torch.tensor([[3.0]])
        result = sparsemax(scores, dim=-1)
        assert result[0, 0].item() == pytest.approx(1.0, abs=1e-5)

    def test_two_elements(self) -> None:
        # [2, 0] -> sparsemax should give [1, 0] since 2 - 0 > 1
        scores = torch.tensor([[2.0, 0.0]])
        result = sparsemax(scores, dim=-1)
        assert result[0, 0].item() == pytest.approx(1.0, abs=1e-5)
        assert result[0, 1].item() == pytest.approx(0.0, abs=1e-5)

    def test_close_scores_partial_sparse(self) -> None:
        # Close scores: both should be nonzero
        scores = torch.tensor([[1.0, 0.8]])
        result = sparsemax(scores, dim=-1)
        assert result[0, 0].item() > 0
        assert result[0, 1].item() > 0
        assert result.sum().item() == pytest.approx(1.0, abs=1e-5)

    def test_gradient_flows(self) -> None:
        scores = torch.randn(4, 6, requires_grad=True)
        result = sparsemax(scores, dim=-1)
        loss = result.sum()
        loss.backward()
        assert scores.grad is not None
        assert scores.grad.shape == scores.shape

    def test_batch_independence(self) -> None:
        scores = torch.randn(5, 3)
        result = sparsemax(scores, dim=-1)
        for i in range(5):
            single = sparsemax(scores[i : i + 1], dim=-1)
            assert torch.allclose(result[i], single[0], atol=1e-6)

    def test_numerical_stability_large_values(self) -> None:
        scores = torch.tensor([[1000.0, 999.0, -1000.0]])
        result = sparsemax(scores, dim=-1)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()
        assert result.sum().item() == pytest.approx(1.0, abs=1e-3)
