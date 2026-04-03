"""
Baseline models for comparison with RSM-Net.

- NaiveFineTuneNet: standard fine-tuning without forgetting protection
- EWCNet: Elastic Weight Consolidation (Kirkpatrick et al. 2017)
- SequentialLoRANet: stacked LoRA adapters without gates (all always active)
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
        encoder: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        layers: list[nn.Module] = []
        dims = [input_dim] + list(hidden_dims)
        for i in range(len(dims) - 1):
            layers.extend([nn.Linear(dims[i], dims[i + 1]), nn.ReLU()])
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(list(hidden_dims)[-1], num_classes)
        self.task_heads = nn.ModuleList()
        self._last_hidden = list(hidden_dims)[-1]
        self._num_classes = num_classes

    def add_task_head(self) -> None:
        self.task_heads.append(nn.Linear(self._last_hidden, self._num_classes))

    def forward(self, x: Tensor, task_id: Optional[int] = None) -> Tensor:
        if self.encoder is not None:
            h = self.encoder(x)
        else:
            h = x.view(x.size(0), -1)
        h = self.features(h)
        if task_id is not None and task_id < len(self.task_heads):
            return self.task_heads[task_id](h)
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
        encoder: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        layers: list[nn.Module] = []
        dims = [input_dim] + list(hidden_dims)
        for i in range(len(dims) - 1):
            layers.extend([nn.Linear(dims[i], dims[i + 1]), nn.ReLU()])
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(list(hidden_dims)[-1], num_classes)
        self.task_heads = nn.ModuleList()
        self._last_hidden = list(hidden_dims)[-1]
        self._num_classes = num_classes

        self.fisher_dict: dict[int, dict[str, Tensor]] = {}
        self.optpar_dict: dict[int, dict[str, Tensor]] = {}

    def add_task_head(self) -> None:
        self.task_heads.append(nn.Linear(self._last_hidden, self._num_classes))

    def forward(self, x: Tensor, task_id: Optional[int] = None) -> Tensor:
        if self.encoder is not None:
            h = self.encoder(x)
        else:
            h = x.view(x.size(0), -1)
        h = self.features(h)
        if task_id is not None and task_id < len(self.task_heads):
            return self.task_heads[task_id](h)
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


class SequentialLoRALinear(nn.Module):
    """Linear layer with stacked LoRA adapters (no gating)."""

    def __init__(self, in_features: int, out_features: int, rank: int = 16) -> None:
        super().__init__()
        self.W_base = nn.Linear(in_features, out_features)
        self.rank = rank
        self.adapter_A = nn.ParameterList()
        self.adapter_B = nn.ParameterList()

    def add_adapter(self) -> None:
        A = nn.Parameter(torch.randn(self.W_base.out_features, self.rank) * 0.01)
        B = nn.Parameter(torch.zeros(self.rank, self.W_base.in_features))
        self.adapter_A.append(A)
        self.adapter_B.append(B)

    def freeze_base(self) -> None:
        for p in self.W_base.parameters():
            p.requires_grad = False

    def freeze_adapter(self, k: int) -> None:
        if k < len(self.adapter_A):
            self.adapter_A[k].requires_grad = False
            self.adapter_B[k].requires_grad = False

    def forward(self, x: Tensor) -> Tensor:
        out = self.W_base(x)
        for k in range(len(self.adapter_A)):
            Bx = F.linear(x, self.adapter_B[k])
            ABx = F.linear(Bx, self.adapter_A[k])
            out = out + ABx
        return out


class SequentialLoRANet(nn.Module):
    """
    Baseline: LoRA adapters stacked sequentially without gates.

    W_eff = W_base + sum_k A_k @ B_k  (all adapters always active, no alpha_k)

    This isolates the value of RSM-Net's gates: if RSM-Net doesn't beat this,
    the input-conditional routing isn't contributing.
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: tuple[int, ...] = (256, 128),
        num_classes: int = 10,
        rank: int = 16,
        encoder: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.num_tasks = 0
        self._last_hidden = list(hidden_dims)[-1]
        self._num_classes = num_classes
        dims = [input_dim] + list(hidden_dims)
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(SequentialLoRALinear(dims[i], dims[i + 1], rank=rank))
        self.head = nn.Linear(self._last_hidden, num_classes)
        self.task_heads = nn.ModuleList()

    def forward(self, x: Tensor, task_id: Optional[int] = None) -> Tensor:
        if self.encoder is not None:
            h = self.encoder(x)
        else:
            h = x.view(x.size(0), -1)
        for layer in self.layers:
            h = F.relu(layer(h))
        if task_id is not None and task_id < len(self.task_heads):
            return self.task_heads[task_id](h)
        return self.head(h)

    def prepare_new_task(self) -> int:
        task_idx = self.num_tasks
        for layer in self.layers:
            layer.freeze_base()
            for k in range(len(layer.adapter_A)):
                layer.freeze_adapter(k)
            layer.add_adapter()
        self.task_heads.append(nn.Linear(self._last_hidden, self._num_classes))
        self.num_tasks += 1
        return task_idx

    def get_optimizer(self, task_idx: int, lr: float = 0.001) -> torch.optim.Adam:
        params: list[nn.Parameter] = []
        for layer in self.layers:
            k = len(layer.adapter_A) - 1
            if k >= 0:
                params.append(layer.adapter_A[k])
                params.append(layer.adapter_B[k])
        params.extend(self.head.parameters())
        if task_idx < len(self.task_heads):
            params.extend(self.task_heads[task_idx].parameters())
        trainable = [p for p in params if p.requires_grad]
        return torch.optim.Adam(trainable, lr=lr)
