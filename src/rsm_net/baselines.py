"""
Baseline models for comparison with RSM-Net.

- NaiveFineTuneNet: standard fine-tuning without forgetting protection
- EWCNet: Elastic Weight Consolidation (Kirkpatrick et al. 2017)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader


class NaiveFineTuneNet(nn.Module):
    """Baseline: Fine-tuning without forgetting protection."""

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: tuple[int, ...] = (256, 128),
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        dims = [input_dim] + list(hidden_dims)
        for i in range(len(dims) - 1):
            layers.extend([nn.Linear(dims[i], dims[i + 1]), nn.ReLU()])
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(list(hidden_dims)[-1], num_classes)

    def forward(self, x: Tensor, task_id: Optional[int] = None) -> Tensor:
        h = self.features(x.view(x.size(0), -1))
        return self.head(h)


class EWCNet(nn.Module):
    """
    Elastic Weight Consolidation baseline.

    Uses diagonal Fisher Information to penalize changes to
    important weights from previous tasks.
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: tuple[int, ...] = (256, 128),
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        dims = [input_dim] + list(hidden_dims)
        for i in range(len(dims) - 1):
            layers.extend([nn.Linear(dims[i], dims[i + 1]), nn.ReLU()])
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(list(hidden_dims)[-1], num_classes)

        self.fisher_dict: dict[int, dict[str, Tensor]] = {}
        self.optpar_dict: dict[int, dict[str, Tensor]] = {}

    def forward(self, x: Tensor, task_id: Optional[int] = None) -> Tensor:
        h = self.features(x.view(x.size(0), -1))
        return self.head(h)

    def compute_fisher(self, dataloader: DataLoader, device: torch.device) -> dict[str, Tensor]:
        """
        Compute diagonal Fisher Information matrix.

        Uses per-sample gradients for correct estimation
        (not batch-averaged gradient squared).
        """
        self.train()
        fisher: dict[str, Tensor] = {
            n: torch.zeros_like(p)
            for n, p in self.named_parameters()
            if p.requires_grad
        }

        num_samples = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            # Per-sample Fisher: accumulate squared gradients
            for sample_idx in range(x.size(0)):
                self.zero_grad()
                out = self(x[sample_idx : sample_idx + 1])
                loss = F.cross_entropy(out, y[sample_idx : sample_idx + 1])
                loss.backward()

                for n, p in self.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        fisher[n] += p.grad.data ** 2
                num_samples += 1

            # Limit samples for speed (Fisher converges with ~1000 samples)
            if num_samples >= 1000:
                break

        for n in fisher:
            fisher[n] /= num_samples

        return fisher

    def store_parameters(
        self, task_id: int, dataloader: DataLoader, device: torch.device
    ) -> None:
        """Store optimal parameters and Fisher for EWC after task training."""
        self.fisher_dict[task_id] = self.compute_fisher(dataloader, device)
        self.optpar_dict[task_id] = {
            n: p.data.clone()
            for n, p in self.named_parameters()
            if p.requires_grad
        }

    def ewc_loss(self, lamda: float = 1000.0) -> Tensor:
        """Compute EWC penalty: sum over tasks of Fisher-weighted L2."""
        loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for task_id in self.fisher_dict:
            for n, p in self.named_parameters():
                if n in self.fisher_dict[task_id]:
                    fisher = self.fisher_dict[task_id][n]
                    optpar = self.optpar_dict[task_id][n]
                    loss = loss + (fisher * (p - optpar) ** 2).sum()
        return lamda * loss
