# RSM-Net -- Recursive Submatrix Memory Network

> Modular architecture for continual learning that reduces catastrophic forgetting
> through low-rank submatrix load distribution, without explicit task identification.

**Author:** Victor Alejandro Cano Jaramillo -- April 2026

## Key Results

### Multi-Domain Benchmark (MNIST -> CIFAR-10 -> SVHN)

| Model | Avg Accuracy | Forgetting | Overhead |
|---|---|---|---|
| **RSM-Net** | **61.13%** | **0.153** | 27.0% |
| Naive Fine-tuning | 42.86% | 0.677 | -- |
| EWC | 65.94% | 0.008 | -- |
| LoRA-Seq | 33.31% | 0.536 | ~21.3% |

### MNIST-Family Benchmark (MNIST -> FashionMNIST -> EMNIST)

| Model | Avg Accuracy | Forgetting | Overhead |
|---|---|---|---|
| **RSM-Net** | **74.33%** | **0.274** | 27.0% |
| Naive Fine-tuning | 68.62% | 0.374 | -- |
| EWC | 87.63% | 0.024 | -- |
| LoRA-Seq | 65.86% | 0.399 | ~21.3% |

**RSM-Net reduces forgetting 4.4x vs Naive and 3.5x vs LoRA-Seq on multi-domain.**
RSM-Net beats both Naive and LoRA-Seq in average accuracy across both benchmarks.

## What is RSM-Net?

**Problem.** Neural networks suffer from catastrophic forgetting: training on a new task overwrites knowledge from previous tasks.

**Solution.** RSM-Net stores task-specific knowledge in low-rank submatrices (A_k * B_k) that are added to frozen base weights. Each new task gets its own submatrix while previous ones remain frozen. Soft gates distribute the input signal across submatrices.

**How it works.** W_base is trained on the first task and then frozen permanently -- it captures fundamental representations (like a biological critical period). For each subsequent task, a new low-rank submatrix is created. The effective weight becomes W_eff = W_base + sum_k alpha_k * A_k * B_k, where alpha_k are sparsemax-based gates. Keys are initialized orthogonal via Gram-Schmidt to maximize separation in the routing space.

**Key insight.** The gates don't achieve perfect per-task discrimination (no diagonal pattern in activation heatmaps). The protection mechanism is *load distribution*: the soft mixture across submatrices distributes gradient pressure, preventing any single submatrix from absorbing all the adaptation. Combined with frozen W_base, this preserves fundamental representations while allowing task-specific adaptation.

## Architecture

```
Input x
  |
  v
[ConvEncoder] (optional, for multi-domain)
  |
  v
W_base(x) ---- frozen after task 0 --------+
  |                                          |
  v                                          |
h_1 = ReLU(W_base(x)) --> query_proj --> q   |
                                |            |
                       keys [e_0, e_1, ...]  |
                                |            |
                       sparsemax(q * keys)   |
                                |            |
                       gates [a_0, a_1, ...] |
                                |            |
sum_k a_k * (A_k @ B_k @ x) ---------------+---> output --> task_head_k
```

See [docs/architecture.md](docs/architecture.md) for details.

## Installation

```bash
# Python 3.10+ required
pip install -e ".[dev]"
```

## Quick Start

```python
from rsm_net import RSMNet, RSMConfig

config = RSMConfig(
    hidden_dims=(400, 200),
    rank=16,
    key_dim=64,
    epochs_per_task=20,
)
model = RSMNet(config=config)

# Task 0: trains W_base + query_proj + head
model.prepare_new_task()
optimizer = model.get_optimizer(task_idx=0)
# ... train ...

# Task 1: freezes W_base, adds submatrix
model.prepare_new_task()
optimizer = model.get_optimizer(task_idx=1)
# ... train ...
```

### Run experiments

```bash
# MNIST-Family benchmark
python -m experiments.continual_learning

# Dual benchmark (MNIST-Family + Multi-Domain)
python -m experiments.dual_benchmark
```

## Project Structure

```
RSM-Net/
|-- src/rsm_net/
|   |-- layers.py           # SubmatrixLinear -- the core
|   |-- network.py          # RSMNet (MLP + optional ConvEncoder)
|   |-- activations.py      # Sparsemax (Martins & Astudillo 2016)
|   |-- encoder.py          # ConvEncoder for multi-domain
|   |-- consolidation.py    # SVD merge of similar submatrices
|   |-- baselines.py        # Naive, EWC, LoRA-Seq
|   |-- training.py         # Train/eval loops
|   |-- config.py           # RSMConfig dataclass
|-- experiments/
|   |-- continual_learning.py   # MNIST-Family benchmark
|   |-- dual_benchmark.py       # Both benchmarks
|   |-- ablation_study.py       # Ablation variants
|-- tests/                      # 45 tests (pytest)
|-- docs/
|   |-- paper.md                # Mathematical formalization
|   |-- architecture.md         # Architecture details
|   |-- experiments.md          # Experimental design and results
|   |-- findings.md             # Key findings and analysis
|-- results/                    # JSONs, plots, heatmaps (generated)
|-- prototype.py                # Original single-file prototype
```

## Comparison with Existing Work

| Property | LoRA | MoE | EWC | Prog. Nets | **RSM-Net** |
|---|---|---|---|---|---|
| Memory in weights | Y | -- | -- | Y | Y |
| Input-conditional routing | -- | Y | -- | -- | Y |
| Low rank | Y | -- | -- | -- | Y |
| Dynamic pruning | -- | ~ | -- | -- | Y |
| Modular (add/remove tasks) | ~ | -- | -- | -- | Y |
| No task ID at inference | -- | Y | Y | -- | Y |

RSM-Net can be viewed as a generalization of LoRA with input-conditional routing,
or equivalently as Mixture of Experts at the weight level instead of the subnetwork level.

## Limitations and Future Work

- Gates don't achieve diagonal per-task discrimination -- selective routing remains an open problem
- EWC achieves lower absolute forgetting (0.008 vs 0.153 on multi-domain)
- Tested only on small-scale benchmarks (28x28, 32x32)
- Future work: contrastive routing loss, task-aware key initialization, larger benchmarks (Split-CIFAR-100, Split-TinyImageNet), recursive depth > 1

See [docs/findings.md](docs/findings.md) for detailed analysis.

## Tests

```bash
pytest tests/ -v
pytest tests/ --cov=rsm_net --cov-report=term-missing
```

## References

- Kirkpatrick et al. (2017) "Overcoming catastrophic forgetting in neural networks" (EWC)
- Hu et al. (2021) "LoRA: Low-Rank Adaptation of Large Language Models"
- Martins & Astudillo (2016) "From Softmax to Sparsemax"

## License

Apache 2.0
