"""
Consolidation mechanism for RSM-Net.

Merges similar submatrices to prevent unbounded growth.
Uses cosine similarity between reconstructed weight matrices
and importance-weighted averaging for the merge.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from torch import Tensor

from rsm_net.layers import SubmatrixLinear, _remove_indices_from_param_list, _remove_indices_from_module_list

logger = logging.getLogger(__name__)


def compute_submatrix_similarity(
    layer: SubmatrixLinear, i: int, j: int
) -> float:
    """
    Compute cosine similarity between two submatrices.

    Reconstructs the full delta-W = A @ B for each and computes
    cosine similarity between flattened vectors.
    """
    with torch.no_grad():
        W_i = layer.submatrix_A[i] @ layer.submatrix_B[i]  # (out, in)
        W_j = layer.submatrix_A[j] @ layer.submatrix_B[j]  # (out, in)

        flat_i = W_i.flatten()
        flat_j = W_j.flatten()

        sim = F.cosine_similarity(flat_i.unsqueeze(0), flat_j.unsqueeze(0))
        return sim.item()


def merge_submatrices(
    layer: SubmatrixLinear, i: int, j: int
) -> None:
    """
    Merge submatrices i and j into position i, removing j.

    Uses importance-weighted averaging:
      merged = (I_i * W_i + I_j * W_j) / (I_i + I_j)

    Then re-decomposes via SVD truncated to rank r.
    """
    with torch.no_grad():
        I_i = layer.importance_scores[i].item()
        I_j = layer.importance_scores[j].item()
        total = I_i + I_j
        if total < 1e-12:
            w_i, w_j = 0.5, 0.5
        else:
            w_i = I_i / total
            w_j = I_j / total

        # Reconstruct full matrices
        W_i = layer.submatrix_A[i] @ layer.submatrix_B[i]
        W_j = layer.submatrix_A[j] @ layer.submatrix_B[j]

        # Weighted merge
        W_merged = w_i * W_i + w_j * W_j

        # SVD to get new low-rank factors
        U, S, Vh = torch.linalg.svd(W_merged, full_matrices=False)
        r = layer.rank

        # A_merged = U[:, :r] * sqrt(S[:r])
        # B_merged = sqrt(S[:r]).unsqueeze(1) * Vh[:r, :]
        sqrt_S = torch.sqrt(S[:r])
        A_new = U[:, :r] * sqrt_S.unsqueeze(0)
        B_new = sqrt_S.unsqueeze(1) * Vh[:r, :]

        # Update submatrix i with merged values
        layer.submatrix_A[i].data.copy_(A_new)
        layer.submatrix_B[i].data.copy_(B_new)

        # Merge key embeddings (importance-weighted)
        key_merged = w_i * layer.key_embeddings[i].data + w_j * layer.key_embeddings[j].data
        layer.key_embeddings[i].data.copy_(key_merged)

        # Update importance
        layer.importance_scores[i] = max(I_i, I_j)

    # Remove submatrix j by rebuilding lists
    remove_set = {j}
    layer.submatrix_A = _remove_indices_from_param_list(layer.submatrix_A, remove_set)
    layer.submatrix_B = _remove_indices_from_param_list(layer.submatrix_B, remove_set)
    layer.key_embeddings = _remove_indices_from_param_list(layer.key_embeddings, remove_set)
    layer.child_layers = _remove_indices_from_module_list(layer.child_layers, remove_set)

    # Rebuild importance buffer
    remaining = layer.num_submatrices
    old_imp = layer.importance_scores
    new_imp = torch.zeros(remaining, device=old_imp.device)
    idx = 0
    for k in range(remaining + 1):
        if k != j and idx < remaining and k < len(old_imp):
            new_imp[idx] = old_imp[k]
            idx += 1
    layer.importance_scores = new_imp

    new_running = torch.zeros(remaining, device=layer._running_gate_mean.device)
    layer._running_gate_mean = new_running

    # Update task mapping
    new_mapping: dict[int, int] = {}
    for task_id, sub_idx in layer._task_to_submatrix.items():
        if sub_idx == j:
            new_mapping[task_id] = i if i < j else i
        elif sub_idx > j:
            new_mapping[task_id] = sub_idx - 1
        else:
            new_mapping[task_id] = sub_idx
    layer._task_to_submatrix = new_mapping

    logger.info("Merged submatrices %d and %d -> %d", i, j, i)


def consolidate_layer(
    layer: SubmatrixLinear, threshold: float = 0.85
) -> int:
    """
    Greedily merge pairs of submatrices with similarity > threshold.

    Returns number of merges performed.
    """
    merges = 0

    while layer.num_submatrices >= 2:
        best_sim = -1.0
        best_pair = (-1, -1)

        K = layer.num_submatrices
        for i in range(K):
            for j in range(i + 1, K):
                sim = compute_submatrix_similarity(layer, i, j)
                if sim > best_sim:
                    best_sim = sim
                    best_pair = (i, j)

        if best_sim < threshold:
            break

        i, j = best_pair
        merge_submatrices(layer, i, j)
        merges += 1
        logger.info(
            "Consolidated pair (%d, %d) with similarity %.4f",
            i, j, best_sim,
        )

    return merges
