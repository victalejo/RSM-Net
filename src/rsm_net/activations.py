"""
Sparsemax activation function.

Implements the projection onto the probability simplex from:
Martins & Astudillo (2016) "From Softmax to Sparsemax"

Unlike softmax, sparsemax produces exact zeros for low-scoring entries,
enabling truly sparse gate activations.
"""

import torch
from torch import Tensor


def sparsemax(scores: Tensor, dim: int = -1) -> Tensor:
    """
    Sparsemax activation: projects scores onto the probability simplex.

    Output sums to 1.0 and contains exact zeros for low-scoring entries.

    Args:
        scores: input logits of any shape
        dim: dimension along which to apply sparsemax

    Returns:
        Sparse probability distribution along dim (same shape as input)
    """
    # Move target dim to last position for easier indexing
    original_dim = dim
    if dim != -1 and dim != scores.dim() - 1:
        scores = scores.transpose(dim, -1)

    # Sort in descending order
    sorted_scores, _ = torch.sort(scores, descending=True, dim=-1)

    # Cumulative sum
    cumsum = torch.cumsum(sorted_scores, dim=-1)

    # Find the support: k(z) = max{k : 1 + k*z_k > sum_{j<=k} z_j}
    K = scores.size(-1)
    k_range = torch.arange(1, K + 1, device=scores.device, dtype=scores.dtype)
    # Broadcast k_range to match sorted_scores shape
    for _ in range(scores.dim() - 1):
        k_range = k_range.unsqueeze(0)
    k_range = k_range.expand_as(sorted_scores)

    support = (1.0 + k_range * sorted_scores) > cumsum

    # k_max: number of elements in the support
    k_max = support.sum(dim=-1, keepdim=True).clamp(min=1)

    # Threshold tau = (sum of support elements - 1) / |support|
    # Gather the cumulative sum at position k_max - 1
    tau_idx = (k_max - 1).long()
    tau_cumsum = cumsum.gather(-1, tau_idx)
    tau = (tau_cumsum - 1.0) / k_max.float()

    # Project: max(z - tau, 0)
    output = torch.clamp(scores - tau, min=0.0)

    # Restore original dimension order
    if original_dim != -1 and original_dim != scores.dim() - 1:
        output = output.transpose(original_dim, -1)

    return output
