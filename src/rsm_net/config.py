"""Configuration dataclass for RSM-Net experiments."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RSMConfig:
    """Immutable configuration for RSM-Net."""

    # Architecture
    input_dim: int = 784
    hidden_dims: tuple[int, ...] = (256, 128)
    num_classes: int = 10
    rank: int = 8
    key_dim: int = 32
    max_depth: int = 1

    # Training
    epochs_per_task: int = 5
    batch_size: int = 128
    lr: float = 0.001
    seed: int = 42

    # Regularization
    sparsity_lambda: float = 0.001
    frobenius_lambda: float = 0.0001
    query_ewc_lambda: float = 100.0
    contrastive_lambda: float = 0.01

    # Pruning
    prune_threshold: float = 0.01
    prune_after_n_evals: int = 3

    # Consolidation
    consolidation_threshold: float = 0.85
    consolidate_at_task_boundary: bool = True

    # EWC baseline
    ewc_lambda: float = 1000.0

    # Conv encoder (for multi-domain with different image sizes/channels)
    use_conv_encoder: bool = False
    encoder_in_channels: int = 1

    # Data loading
    num_workers: int = 0
    pin_memory: bool = True

    # Tasks
    tasks: tuple[str, ...] = ("MNIST", "FashionMNIST", "EMNIST")
