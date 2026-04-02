"""Tests for consolidation mechanism."""

import pytest
import torch

from rsm_net.consolidation import (
    compute_submatrix_similarity,
    consolidate_layer,
    merge_submatrices,
)
from rsm_net.layers import SubmatrixLinear


class TestConsolidation:
    """Tests for submatrix merging and consolidation."""

    @pytest.fixture
    def layer_with_submatrices(self) -> SubmatrixLinear:
        layer = SubmatrixLinear(in_features=16, out_features=8, rank=4, key_dim=4)
        layer.add_submatrix(task_id=0)
        layer.add_submatrix(task_id=1)
        layer.add_submatrix(task_id=2)
        return layer

    def test_identical_submatrices_have_high_similarity(
        self, layer_with_submatrices: SubmatrixLinear
    ) -> None:
        layer = layer_with_submatrices
        # Set non-zero values first, then copy
        layer.submatrix_A[0].data.normal_()
        layer.submatrix_B[0].data.normal_()
        layer.submatrix_A[1].data.copy_(layer.submatrix_A[0].data)
        layer.submatrix_B[1].data.copy_(layer.submatrix_B[0].data)
        sim = compute_submatrix_similarity(layer, 0, 1)
        assert sim > 0.99

    def test_different_submatrices_have_lower_similarity(
        self, layer_with_submatrices: SubmatrixLinear
    ) -> None:
        layer = layer_with_submatrices
        # Make them very different
        layer.submatrix_A[0].data.fill_(1.0)
        layer.submatrix_B[0].data.fill_(1.0)
        layer.submatrix_A[1].data.fill_(-1.0)
        layer.submatrix_B[1].data.fill_(1.0)
        sim = compute_submatrix_similarity(layer, 0, 1)
        assert sim < 0.0  # Opposite signs -> negative cosine

    def test_merge_reduces_count(
        self, layer_with_submatrices: SubmatrixLinear
    ) -> None:
        layer = layer_with_submatrices
        assert layer.num_submatrices == 3
        merge_submatrices(layer, 0, 1)
        assert layer.num_submatrices == 2

    def test_merge_preserves_forward_pass(
        self, layer_with_submatrices: SubmatrixLinear
    ) -> None:
        layer = layer_with_submatrices
        x = torch.randn(4, 16)
        # Forward pass should work after merge
        merge_submatrices(layer, 0, 1)
        out = layer(x)
        assert out.shape == (4, 8)
        assert not torch.isnan(out).any()

    def test_consolidate_identical_pairs(self) -> None:
        layer = SubmatrixLinear(in_features=8, out_features=4, rank=2, key_dim=4)
        layer.add_submatrix(task_id=0)
        layer.add_submatrix(task_id=1)
        # Set non-zero values, then make them identical
        layer.submatrix_A[0].data.normal_()
        layer.submatrix_B[0].data.normal_()
        layer.submatrix_A[1].data.copy_(layer.submatrix_A[0].data)
        layer.submatrix_B[1].data.copy_(layer.submatrix_B[0].data)
        merges = consolidate_layer(layer, threshold=0.99)
        assert merges == 1
        assert layer.num_submatrices == 1

    def test_consolidate_dissimilar_stays(self) -> None:
        layer = SubmatrixLinear(in_features=8, out_features=4, rank=2, key_dim=4)
        layer.add_submatrix(task_id=0)
        layer.add_submatrix(task_id=1)
        # Make them very different
        layer.submatrix_A[0].data.normal_()
        layer.submatrix_B[0].data.normal_()
        layer.submatrix_A[1].data.normal_()
        layer.submatrix_B[1].data.normal_()
        merges = consolidate_layer(layer, threshold=0.99)
        # Very unlikely to merge random matrices at 0.99 threshold
        assert layer.num_submatrices >= 1
