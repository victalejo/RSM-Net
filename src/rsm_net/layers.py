"""
SubmatrixLinear layer -- the core building block of RSM-Net.

W_eff(x) = W_base + sum_k alpha_k(x) * (A_k @ B_k)

Each submatrix (A_k, B_k) is a low-rank decomposition that stores
task-specific knowledge. The gate alpha_k(x) is computed via
sparsemax over scaled dot-product attention between input query
and learned key embeddings.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from rsm_net.activations import sparsemax

logger = logging.getLogger(__name__)


def _remove_indices_from_param_list(
    plist: nn.ParameterList, indices_to_remove: set[int]
) -> nn.ParameterList:
    """Rebuild a ParameterList excluding given indices."""
    new_list = nn.ParameterList()
    for i, param in enumerate(plist):
        if i not in indices_to_remove:
            new_list.append(param)
    return new_list


def _remove_indices_from_module_list(
    mlist: nn.ModuleList, indices_to_remove: set[int]
) -> nn.ModuleList:
    """Rebuild a ModuleList excluding given indices."""
    new_list = nn.ModuleList()
    for i, module in enumerate(mlist):
        if i not in indices_to_remove:
            new_list.append(module)
    return new_list


class SubmatrixLinear(nn.Module):
    """
    Linear layer with internal submatrix memory.

    Supports recursive depth: each submatrix can itself contain
    sub-submatrices at depth < max_depth.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        key_dim: int = 32,
        max_depth: int = 1,
        current_depth: int = 0,
        context_dim: int | None = None,
        use_sparsemax: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.key_dim = key_dim
        self.max_depth = max_depth
        self.current_depth = current_depth
        self.use_sparsemax = use_sparsemax

        # Base weights (frozen after task 0)
        self.W_base = nn.Linear(in_features, out_features)

        # Query projection for gating
        # context_dim allows query to be computed from routing context
        # (semantic features) instead of raw layer input
        query_input_dim = context_dim if context_dim is not None else in_features
        self.query_proj = nn.Linear(query_input_dim, key_dim)

        # Submatrix storage -- nn.ModuleList/ParameterList for proper registration
        self.submatrix_A = nn.ParameterList()   # A_k in R^(out x rank)
        self.submatrix_B = nn.ParameterList()   # B_k in R^(rank x in)
        self.key_embeddings = nn.ParameterList() # e_k in R^(key_dim)

        # Recursive children (depth > 1)
        self.child_layers: nn.ModuleList = nn.ModuleList()

        # Importance tracking as registered buffer
        self.register_buffer("importance_scores", torch.zeros(0))
        self._gate_step_count: int = 0
        self.register_buffer("_running_gate_mean", torch.zeros(0))

        # Cache last gates for sparsity loss (avoids double computation)
        self._last_gates: Optional[Tensor] = None

        # Task-to-submatrix mapping (survives pruning)
        self._task_to_submatrix: dict[int, int] = {}

        # EWC-light for query_proj: protect routing without full freeze
        self._query_fisher: dict[str, Tensor] = {}
        self._query_optpar: dict[str, Tensor] = {}

    @property
    def num_submatrices(self) -> int:
        return len(self.submatrix_A)

    def _init_orthogonal_key(self) -> Tensor:
        """
        Initialize a key embedding orthogonal to all existing keys
        using Gram-Schmidt. First key is random normalized.
        """
        K = len(self.key_embeddings)
        if K == 0:
            # First key: random unit vector
            v = torch.randn(self.key_dim)
            return v / v.norm().clamp(min=1e-8)

        # Gram-Schmidt: start with random, subtract projections onto existing keys
        v = torch.randn(self.key_dim)
        with torch.no_grad():
            for existing_key in self.key_embeddings:
                e = existing_key.data
                proj = (v @ e) / (e @ e).clamp(min=1e-8) * e
                v = v - proj
        # Normalize
        norm = v.norm()
        if norm < 1e-6:
            # Degenerate case: random fallback
            v = torch.randn(self.key_dim)
        return v / v.norm().clamp(min=1e-8)

    def add_submatrix(self, task_id: int) -> int:
        """
        Add a new submatrix for a new task.

        Uses LoRA-style init: A=random(small), B=zeros.
        Key embedding initialized orthogonal to existing keys (Gram-Schmidt).
        """
        k = len(self.submatrix_A)

        # LoRA-style initialization: A random, B zero
        A = nn.Parameter(torch.randn(self.out_features, self.rank) * 0.01)
        B = nn.Parameter(torch.zeros(self.rank, self.in_features))
        # Orthogonal key initialization
        key = nn.Parameter(self._init_orthogonal_key())

        self.submatrix_A.append(A)
        self.submatrix_B.append(B)
        self.key_embeddings.append(key)

        # Expand importance tracking
        new_importance = torch.zeros(
            k + 1, device=self.importance_scores.device
        )
        if k > 0:
            new_importance[:k] = self.importance_scores
        self.importance_scores = new_importance

        new_running = torch.zeros(
            k + 1, device=self._running_gate_mean.device
        )
        if k > 0:
            new_running[:k] = self._running_gate_mean
        self._running_gate_mean = new_running

        # Recursive children if depth allows
        if self.current_depth < self.max_depth - 1:
            child = SubmatrixLinear(
                in_features=self.in_features,
                out_features=self.out_features,
                rank=max(self.rank // 2, 2),
                key_dim=self.key_dim,
                max_depth=self.max_depth,
                current_depth=self.current_depth + 1,
            )
            self.child_layers.append(child)

        # Track mapping
        self._task_to_submatrix[task_id] = k

        logger.debug(
            "Layer depth=%d: added submatrix %d for task %d (total: %d)",
            self.current_depth, k, task_id, k + 1,
        )
        return k

    def compute_gates(self, x: Tensor, context: Optional[Tensor] = None) -> Tensor:
        """
        Compute alpha(x) -- sparse relevance gates for each submatrix.

        Uses sparsemax for truly sparse activations (exact zeros).

        Args:
            x: input tensor (batch, in_features) -- used if no context
            context: optional hidden features (batch, feat_dim) for richer
                     query computation. When provided, query is computed from
                     context instead of raw x (better task discrimination).

        Returns:
            gates: (batch, K) sparse probability distribution,
                   or (batch, 0) if no submatrices exist
        """
        K = len(self.key_embeddings)
        if K == 0:
            return torch.zeros(x.size(0), 0, device=x.device)

        # Single submatrix: gate is trivially 1.0
        if K == 1:
            return torch.ones(x.size(0), 1, device=x.device)

        # Query from context (hidden features) or raw input
        query_input = context if context is not None else x
        q = self.query_proj(query_input)

        # Keys: (K, key_dim)
        keys = torch.stack(list(self.key_embeddings))

        # Scaled dot-product scores: (batch, K)
        scores = torch.matmul(q, keys.transpose(-2, -1)) / (self.key_dim ** 0.5)

        # Gate activation: sparsemax (exact zeros) or softmax (distributed)
        if self.use_sparsemax:
            gates = sparsemax(scores, dim=-1)
        else:
            gates = F.softmax(scores, dim=-1)

        return gates

    def forward(self, x: Tensor, context: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass:
        h = W_base(x) + sum_k alpha_k(x) * A_k @ B_k @ x

        Args:
            x: input to this layer
            context: optional hidden features for query computation
        """
        # Base forward
        out = self.W_base(x)

        K = self.num_submatrices
        if K == 0:
            self._last_gates = None
            return out

        # Compute gates once, cache for sparsity loss
        gates = self.compute_gates(x, context=context)
        self._last_gates = gates

        # Apply submatrices weighted by gates
        for k in range(K):
            gate_k = gates[:, k]

            # Skip computation when gate is zero (sparsemax benefit)
            if not self.training and gate_k.max().item() == 0.0:
                continue

            # Low-rank: A_k @ (B_k @ x)
            Bx = F.linear(x, self.submatrix_B[k])    # (batch, rank)
            ABx = F.linear(Bx, self.submatrix_A[k])   # (batch, out)

            # Recursive depth: add children contribution
            if k < len(self.child_layers):
                child_out = self.child_layers[k](x)
                ABx = ABx + child_out - self.child_layers[k].W_base(x)

            out = out + gate_k.unsqueeze(1) * ABx

        # Update running gate statistics during training
        if self.training:
            with torch.no_grad():
                batch_avg = gates.mean(dim=0)
                self._gate_step_count += 1
                alpha = 1.0 / self._gate_step_count
                device = self._running_gate_mean.device
                batch_avg_dev = batch_avg.to(device)
                self._running_gate_mean = (
                    (1.0 - alpha) * self._running_gate_mean + alpha * batch_avg_dev
                )

        return out

    def update_importance(self) -> None:
        """Update importance scores from running gate mean (EMA)."""
        K = self.num_submatrices
        if K == 0:
            return

        for k in range(K):
            old = self.importance_scores[k].item()
            new_val = self._running_gate_mean[k].item()
            self.importance_scores[k] = 0.9 * old + 0.1 * new_val

        # Reset running stats
        self._gate_step_count = 0
        self._running_gate_mean.zero_()

        logger.debug(
            "Importance scores (depth=%d): %s",
            self.current_depth,
            [f"{s:.4f}" for s in self.importance_scores.tolist()],
        )

    def prune(self, threshold: float = 0.01) -> int:
        """
        Remove submatrices with importance below threshold.

        Returns number of pruned submatrices.
        """
        to_remove_set: set[int] = set()
        for k in range(self.num_submatrices):
            if self.importance_scores[k].item() < threshold:
                to_remove_set.add(k)

        if not to_remove_set:
            return 0

        # Rebuild ParameterLists without removed indices
        self.submatrix_A = _remove_indices_from_param_list(self.submatrix_A, to_remove_set)
        self.submatrix_B = _remove_indices_from_param_list(self.submatrix_B, to_remove_set)
        self.key_embeddings = _remove_indices_from_param_list(self.key_embeddings, to_remove_set)
        self.child_layers = _remove_indices_from_module_list(self.child_layers, to_remove_set)

        # Update task mapping (shift indices down)
        sorted_removed = sorted(to_remove_set)
        new_mapping: dict[int, int] = {}
        for task_id, sub_idx in self._task_to_submatrix.items():
            if sub_idx in to_remove_set:
                continue
            shift = sum(1 for r in sorted_removed if r < sub_idx)
            new_mapping[task_id] = sub_idx - shift
        self._task_to_submatrix = new_mapping

        # Rebuild importance and running mean buffers
        remaining = self.num_submatrices
        old_importance = self.importance_scores
        new_importance = torch.zeros(remaining, device=old_importance.device)
        new_running = torch.zeros(remaining, device=self._running_gate_mean.device)

        idx = 0
        for k in range(remaining + len(to_remove_set)):
            if k not in to_remove_set:
                if idx < remaining and k < len(old_importance):
                    new_importance[idx] = old_importance[k]
                idx += 1

        self.importance_scores = new_importance
        self._running_gate_mean = new_running

        logger.info(
            "Pruned %d submatrices (depth=%d), %d remaining",
            len(to_remove_set), self.current_depth, remaining,
        )

        return len(to_remove_set)

    def freeze_base(self) -> None:
        """Freeze base weights after initial task training."""
        for param in self.W_base.parameters():
            param.requires_grad = False

    def freeze_submatrix(self, k: int) -> None:
        """Freeze a specific submatrix."""
        if k < len(self.submatrix_A):
            self.submatrix_A[k].requires_grad = False
            self.submatrix_B[k].requires_grad = False
        if k < len(self.key_embeddings):
            self.key_embeddings[k].requires_grad = False

    def freeze_query_proj(self) -> None:
        """Freeze query projection entirely."""
        for param in self.query_proj.parameters():
            param.requires_grad = False

    def store_query_fisher(self) -> None:
        """
        Store Fisher information for query_proj parameters (EWC-light).

        Call after each task's training to protect routing knowledge.
        Accumulates Fisher across tasks (not per-task like full EWC).
        """
        for n, p in self.query_proj.named_parameters():
            if p.grad is not None:
                fisher = p.grad.data ** 2
                if n in self._query_fisher:
                    # Accumulate Fisher across tasks
                    self._query_fisher[n] = self._query_fisher[n] + fisher
                else:
                    self._query_fisher[n] = fisher.clone()
            self._query_optpar[n] = p.data.clone()

    def query_ewc_loss(self, lamda: float = 100.0) -> Tensor:
        """
        EWC penalty on query_proj: protects routing without full freeze.

        Uses a low lambda (100 vs 1000 for full EWC) to allow adaptation
        while discouraging catastrophic routing drift.
        """
        loss = torch.tensor(0.0, device=next(self.query_proj.parameters()).device)
        if not self._query_fisher:
            return loss
        for n, p in self.query_proj.named_parameters():
            if n in self._query_fisher:
                fisher = self._query_fisher[n]
                optpar = self._query_optpar[n]
                loss = loss + (fisher * (p - optpar) ** 2).sum()
        return lamda * loss

    def get_trainable_params_for_task(self, task_id: int) -> list[nn.Parameter]:
        """Return only the trainable parameters for a specific task."""
        params: list[nn.Parameter] = []

        # query_proj only trains on task 0 (frozen after, no EWC needed)
        # Not included for task_id > 0

        # Find submatrix index for this task
        sub_idx = self._task_to_submatrix.get(task_id)
        if sub_idx is not None and sub_idx < len(self.submatrix_A):
            params.append(self.submatrix_A[sub_idx])
            params.append(self.submatrix_B[sub_idx])
            params.append(self.key_embeddings[sub_idx])

            # Recursive children
            if sub_idx < len(self.child_layers):
                params.extend(self.child_layers[sub_idx].parameters())

        return params
