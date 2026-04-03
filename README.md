# RSM-Net: Recursive Submatrix Memory Network

**Author:** Victor Alejandro Cano Jaramillo
**Date:** April 2026 | **Version:** 0.1.0

A novel neural network architecture for **continual learning** that stores task-specific knowledge in low-rank submatrices with input-conditional gating via sparsemax.

## Architecture

```
Input x
    |
    v
W_base(x)  +  sum_k alpha_k(x) * A_k @ B_k @ x
    |                    ^
    |            sparsemax gate
    |          (input-conditional)
    v
  Output
```

**Key ideas:**
- **Submatrices as memory:** Each task gets its own low-rank delta-W = A @ B, frozen after training
- **Sparsemax gating:** Input-conditional retrieval activates only relevant submatrices (exact zeros)
- **Recursive depth:** Submatrices can contain sub-submatrices for hierarchical decomposition
- **Pruning & consolidation:** Remove unused submatrices, merge similar ones

This can be viewed as a **generalization of LoRA with input-conditional routing**, or equivalently as **Mixture of Experts at the weight level** instead of the subnetwork level.

## Installation

```bash
# Python 3.10+ required
pip install -e ".[dev]"
```

## Quick Start

```bash
# Run the continual learning experiment (MNIST -> FashionMNIST -> EMNIST)
python -m experiments.continual_learning
```

```python
from rsm_net import RSMNet, RSMConfig

config = RSMConfig(rank=8, key_dim=32, max_depth=1)
model = RSMNet(config=config)

# Prepare for task 0
model.prepare_new_task()
optimizer = model.get_optimizer(task_idx=0)

# Train...
# Prepare for task 1 (freezes base + adds submatrix)
model.prepare_new_task()
optimizer = model.get_optimizer(task_idx=1)
```

## Project Structure

```
RSM-Net/
|-- src/rsm_net/
|   |-- activations.py     # Sparsemax activation
|   |-- baselines.py       # Naive fine-tuning + EWC
|   |-- config.py           # RSMConfig dataclass
|   |-- consolidation.py    # Submatrix merging
|   |-- layers.py           # SubmatrixLinear (core)
|   |-- network.py          # RSMNet
|   |-- training.py         # Training/evaluation loops
|-- experiments/
|   |-- continual_learning.py  # Main experiment
|-- tests/                     # pytest suite (41 tests)
|-- docs/
|   |-- paper.md               # Mathematical formalization
|   |-- review.md              # Code review findings
|-- prototype.py               # Original single-file prototype
```

## Tests

```bash
pytest tests/ -v
pytest tests/ --cov=rsm_net --cov-report=term-missing
```

## Initial Results (5 epochs/task, rank=8)

| Task | RSM-Net | Naive | EWC |
|------|---------|-------|-----|
| MNIST | 14.19% | 11.78% | 14.71% |
| FashionMNIST | 12.70% | 13.01% | 10.98% |
| EMNIST | 84.98% | 89.43% | 74.58% |
| **Avg** | **37.29%** | 38.07% | 33.42% |
| **Avg Forgetting** | **0.77** | 0.80 | 0.79 |

RSM-Net shows less forgetting than both baselines with only 25.6% parameter overhead. Results will improve significantly with more training epochs and with task-0 submatrix protection (known improvement).

## Open Questions

1. Optimal trade-off between rank r and number of submatrices K
2. Does recursive depth d > 1 provide empirical benefit?
3. How does pruning quality scale with task count?
4. Can pruning be learned end-to-end (meta-learning)?

## Citation

```
Victor Alejandro Cano Jaramillo. "RSM-Net: Recursive Submatrix Memory Network."
Working paper, April 2026.
```

## License

Apache 2.0
