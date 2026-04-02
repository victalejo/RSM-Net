"""
RSM-Net (Recursive Submatrix Memory Network) -- full network.

MLP with SubmatrixLinear layers for continual learning.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from rsm_net.config import RSMConfig
from rsm_net.consolidation import consolidate_layer
from rsm_net.layers import SubmatrixLinear

logger = logging.getLogger(__name__)


class RSMNet(nn.Module):
    """
    Recursive Submatrix Memory Network.

    Each layer contains submatrices that store task-specific knowledge.
    A sparsemax gate selects which submatrices to activate based on
    the current input.
    """

    def __init__(self, config: Optional[RSMConfig] = None, **kwargs) -> None:
        super().__init__()

        if config is None:
            config = RSMConfig(**kwargs)
        self.config = config

        self.num_tasks: int = 0

        # Build layers
        dims = [config.input_dim] + list(config.hidden_dims)
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(
                SubmatrixLinear(
                    in_features=dims[i],
                    out_features=dims[i + 1],
                    rank=config.rank,
                    key_dim=config.key_dim,
                    max_depth=config.max_depth,
                )
            )

        # One classification head per task (multi-head evaluation)
        self.heads = nn.ModuleList()
        # Shared head for single-head evaluation
        self.shared_head = nn.Linear(list(config.hidden_dims)[-1], config.num_classes)

    def forward(
        self, x: Tensor, task_id: Optional[int] = None
    ) -> Tensor:
        """
        Forward pass.

        Args:
            x: (batch, input_dim) or (batch, 1, 28, 28) etc.
            task_id: if specified, use that task's head
        """
        h = x.view(x.size(0), -1)

        for layer in self.layers:
            h = F.relu(layer(h))

        if task_id is not None and task_id < len(self.heads):
            return self.heads[task_id](h)
        return self.shared_head(h)

    def prepare_new_task(self) -> int:
        """
        Prepare the network for a new task.

        - For task 0: train everything normally
        - For task k>0: freeze base + old submatrices, add new submatrix

        Returns the task index.
        """
        task_idx = self.num_tasks

        if task_idx > 0:
            for layer in self.layers:
                layer.freeze_base()
                # Freeze all existing submatrices
                for k in range(layer.num_submatrices):
                    layer.freeze_submatrix(k)
                # Add new submatrix for this task
                layer.add_submatrix(task_id=task_idx)

        # Add classification head
        last_hidden = self.layers[-1].out_features
        self.heads.append(nn.Linear(last_hidden, self.config.num_classes))

        self.num_tasks += 1

        logger.info(
            "Prepared task %d. Submatrices per layer: %s",
            task_idx,
            [layer.num_submatrices for layer in self.layers],
        )
        return task_idx

    def get_optimizer(self, task_idx: int, lr: Optional[float] = None) -> torch.optim.Adam:
        """
        Create optimizer with only the trainable parameters for this task.

        IMPORTANT: Must be called after prepare_new_task() and after any
        structural changes (prune, consolidate). Old optimizers become
        invalid after structural changes.
        """
        if lr is None:
            lr = self.config.lr

        params: list[nn.Parameter] = []

        if task_idx == 0:
            params = [p for p in self.parameters() if p.requires_grad]
        else:
            for layer in self.layers:
                params.extend(layer.get_trainable_params_for_task(task_idx))
            params.extend(self.heads[task_idx].parameters())
            params.extend(self.shared_head.parameters())

        # Deduplicate (query_proj params may appear multiple times)
        seen_ids: set[int] = set()
        unique_params: list[nn.Parameter] = []
        for p in params:
            if id(p) not in seen_ids and p.requires_grad:
                seen_ids.add(id(p))
                unique_params.append(p)

        logger.debug(
            "Optimizer for task %d: %d unique trainable params",
            task_idx, len(unique_params),
        )
        return torch.optim.Adam(unique_params, lr=lr)

    def get_sparsity_loss(self) -> Tensor:
        """
        Compute L1 sparsity penalty on gate activations.

        Uses cached gates from the last forward pass (no recomputation).
        """
        loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer in self.layers:
            if layer._last_gates is not None and layer._last_gates.numel() > 0:
                loss = loss + layer._last_gates.abs().mean()
        return loss

    def get_frobenius_loss(self, task_idx: int) -> Tensor:
        """Frobenius norm regularization on the current task's submatrices."""
        loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer in self.layers:
            sub_idx = layer._task_to_submatrix.get(task_idx)
            if sub_idx is not None and sub_idx < len(layer.submatrix_A):
                A = layer.submatrix_A[sub_idx]
                B = layer.submatrix_B[sub_idx]
                loss = loss + torch.norm(A, p="fro") + torch.norm(B, p="fro")
        return loss

    def update_importance_all(self) -> None:
        """Update importance scores for all layers."""
        for layer in self.layers:
            layer.update_importance()

    def prune_all(self, threshold: Optional[float] = None) -> int:
        """Prune unimportant submatrices across all layers."""
        if threshold is None:
            threshold = self.config.prune_threshold

        total_pruned = 0
        for layer in self.layers:
            total_pruned += layer.prune(threshold)

        if total_pruned > 0:
            logger.info("Total pruned: %d submatrices", total_pruned)
        return total_pruned

    def consolidate_all(self, threshold: Optional[float] = None) -> int:
        """Consolidate similar submatrices across all layers."""
        if threshold is None:
            threshold = self.config.consolidation_threshold

        total_merges = 0
        for layer in self.layers:
            total_merges += consolidate_layer(layer, threshold)

        if total_merges > 0:
            logger.info("Total consolidations: %d merges", total_merges)
        return total_merges

    def get_state_summary(self) -> dict:
        """Return a summary of the current model state."""
        summary = {
            "num_tasks": self.num_tasks,
            "layers": [],
        }
        for i, layer in enumerate(self.layers):
            layer_info = {
                "layer": i,
                "num_submatrices": layer.num_submatrices,
                "importance_scores": layer.importance_scores.tolist(),
                "task_mapping": dict(layer._task_to_submatrix),
            }
            summary["layers"].append(layer_info)

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        summary["total_params"] = total_params
        summary["trainable_params"] = trainable_params

        return summary
