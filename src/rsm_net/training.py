"""
Training and evaluation loops for RSM-Net experiments.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from rsm_net.baselines import EWCNet, NaiveFineTuneNet
from rsm_net.config import RSMConfig
from rsm_net.network import RSMNet

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_task_dataloaders(
    task_name: str,
    batch_size: int = 128,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """Load train/test dataloaders for a task."""
    # Grayscale datasets (28x28, 1 channel)
    grayscale_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    # RGB datasets (32x32, 3 channels)
    rgb_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    grayscale_datasets = {
        "MNIST": datasets.MNIST,
        "FashionMNIST": datasets.FashionMNIST,
        "KMNIST": datasets.KMNIST,
    }

    if task_name in grayscale_datasets:
        DatasetClass = grayscale_datasets[task_name]
        train_data = DatasetClass("./data", train=True, download=True, transform=grayscale_transform)
        test_data = DatasetClass("./data", train=False, download=True, transform=grayscale_transform)
    elif task_name == "EMNIST":
        train_data = datasets.EMNIST(
            "./data", split="letters", train=True, download=True, transform=grayscale_transform
        )
        test_data = datasets.EMNIST(
            "./data", split="letters", train=False, download=True, transform=grayscale_transform
        )
        train_data.targets = train_data.targets % 10
        test_data.targets = test_data.targets % 10
    elif task_name == "CIFAR10":
        train_data = datasets.CIFAR10("./data", train=True, download=True, transform=rgb_transform)
        test_data = datasets.CIFAR10("./data", train=False, download=True, transform=rgb_transform)
    elif task_name == "SVHN":
        train_data = datasets.SVHN("./data", split="train", download=True, transform=rgb_transform)
        test_data = datasets.SVHN("./data", split="test", download=True, transform=rgb_transform)
    else:
        raise ValueError(f"Unknown task: {task_name}")

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, test_loader


def train_rsm_epoch(
    model: RSMNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    task_idx: int,
    config: RSMConfig,
) -> tuple[float, float]:
    """Train one epoch of RSM-Net."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        out = model(x, task_id=task_idx)
        loss = F.cross_entropy(out, y)

        # Sparsity regularization (uses cached gates, no recomputation)
        if config.sparsity_lambda > 0:
            loss = loss + config.sparsity_lambda * model.get_sparsity_loss()

        # Frobenius regularization
        if config.frobenius_lambda > 0:
            loss = loss + config.frobenius_lambda * model.get_frobenius_loss(task_idx)

        # Contrastive key loss (pushes key embeddings apart)
        if config.contrastive_lambda > 0:
            loss = loss + config.contrastive_lambda * model.get_contrastive_key_loss()

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        _, predicted = out.max(1)
        correct += predicted.eq(y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total


def train_baseline_epoch(
    model: NaiveFineTuneNet | EWCNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    ewc_lambda: float = 0.0,
    task_id: int | None = None,
) -> tuple[float, float]:
    """Train one epoch of a baseline model."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        out = model(x, task_id=task_id)
        loss = F.cross_entropy(out, y)

        if isinstance(model, EWCNet) and ewc_lambda > 0:
            loss = loss + model.ewc_loss(ewc_lambda)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        _, predicted = out.max(1)
        correct += predicted.eq(y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total


def run_evaluation(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    task_id: int | None = None,
) -> float:
    """Measure model accuracy on a dataset."""
    model.train(False)
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x, task_id=task_id)
            _, predicted = out.max(1)
            correct += predicted.eq(y).sum().item()
            total += x.size(0)

    return correct / total
