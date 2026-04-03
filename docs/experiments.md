# Experimental Design and Results

## Benchmarks

### Benchmark 1: MNIST-Family

Sequential training on three 28x28 grayscale classification tasks:

1. **MNIST** -- handwritten digits (60K train, 10K test, 10 classes)
2. **FashionMNIST** -- clothing items (60K train, 10K test, 10 classes)
3. **EMNIST-Letters** -- handwritten letters, remapped to 10 classes (grouping by letter % 10)

Same input domain (28x28 grayscale), different semantic content.
No convolutional encoder -- inputs are flattened to 784 dimensions.

### Benchmark 2: Multi-Domain

Sequential training on three tasks with different image formats:

1. **MNIST** -- 28x28 grayscale digits
2. **CIFAR-10** -- 32x32 RGB natural images (10 classes)
3. **SVHN** -- 32x32 RGB street view house numbers (10 classes)

Uses ConvEncoder (2-layer CNN) to produce fixed 512-dim features.
Encoder trained on MNIST (task 0), frozen after.

## Models Compared

| Model | Description |
|---|---|
| **RSM-Net** | Our proposal. SubmatrixLinear layers with sparsemax gates, orthogonal keys |
| **Naive Fine-tuning** | Standard MLP, no forgetting protection. Shared architecture |
| **EWC** | Elastic Weight Consolidation (Kirkpatrick et al. 2017). Fisher-weighted L2 penalty, lambda=1000. Per-sample Fisher (1000 samples) |
| **LoRA-Seq** | Sequential LoRA adapters without gates. Same rank as RSM-Net. All adapters always active. Isolates the value of RSM-Net's gating mechanism |

All models use per-task classification heads (Task-IL protocol).
Baselines use the same hidden dimensions and learning rate as RSM-Net.

## Configuration

```python
RSMConfig(
    hidden_dims=(400, 200),  # (256, 128) for multi-domain
    rank=16,
    key_dim=64,
    max_depth=1,
    epochs_per_task=20,
    batch_size=128,
    lr=0.001,
    seed=42,
    sparsity_lambda=0.001,
    frobenius_lambda=0.0001,
    contrastive_lambda=0.01,
)
```

Cosine annealing LR scheduler (eta_min=1e-5) for all models.

## Metrics

- **Average Accuracy**: mean accuracy across all tasks after training the last task
- **Average Forgetting**: mean(max_accuracy - final_accuracy) for tasks seen before the last
- **Backward Transfer (BWT)**: (1/T-1) * sum(R_T,i - R_i,i) -- negative means forgetting
- **Parameter Overhead**: (params_model - params_naive) / params_naive * 100%

## Results

### MNIST-Family

Source: `results/experiment_results.json`

**Per-task accuracy after training all 3 tasks:**

| Task | RSM-Net | Naive | EWC | LoRA-Seq |
|---|---|---|---|---|
| MNIST | 89.10% | 67.27% | 95.81% | 48.24% |
| FashionMNIST | 42.83% | 46.08% | 87.15% | 58.17% |
| EMNIST | 91.06% | 92.51% | 79.93% | 91.16% |

**Summary metrics:**

| Metric | RSM-Net | Naive | EWC | LoRA-Seq |
|---|---|---|---|---|
| Avg Accuracy | 74.33% | 68.62% | 87.63% | 65.86% |
| Avg Forgetting | 0.274 | 0.374 | 0.024 | 0.399 |
| BWT | -0.274 | -0.374 | -0.024 | -0.399 |

Parameter counts: Naive 402,240 / RSM-Net 510,912 (+27.0%) / LoRA-Seq 487,872 (+21.3%)

### Multi-Domain

Source: `results/dual_benchmark_results.json`

**Per-task accuracy after training all 3 tasks:**

| Task | RSM-Net | Naive | EWC | LoRA-Seq |
|---|---|---|---|---|
| MNIST | 90.77% | 21.26% | 99.10% | 16.87% |
| CIFAR-10 | 26.56% | 16.14% | 50.97% | 22.06% |
| SVHN | 66.06% | 91.18% | 47.75% | 61.00% |

**Summary metrics:**

| Metric | RSM-Net | Naive | EWC | LoRA-Seq |
|---|---|---|---|---|
| Avg Accuracy | 61.13% | 42.86% | 65.94% | 33.31% |
| Avg Forgetting | 0.153 | 0.677 | 0.008 | 0.536 |
| BWT | -0.153 | -0.677 | -0.008 | -0.536 |

## Gate Activation Heatmaps

Gate activations averaged over a test batch, Layer 0:

- `results/gate_heatmap_mnist_family.png` -- MNIST-Family
- `results/gate_heatmap_multi_domain.png` -- Multi-Domain

Both show that gates do NOT achieve per-task diagonal discrimination.
Sub 1 tends to dominate across all tasks. See [findings.md](findings.md)
for analysis of why this happens and why RSM-Net still works.

## Forgetting Curves

- `results/forgetting_curves.png` -- accuracy on each task over time

Shows how accuracy on MNIST degrades as FashionMNIST and EMNIST are learned.
RSM-Net retains MNIST at 89.10% vs Naive 67.27%.

## Reproducibility

```bash
# Exact reproduction (seed=42)
python -m experiments.dual_benchmark

# Results will be in results/dual_benchmark_results.json
```

All experiments use `torch.manual_seed(42)`, `np.random.seed(42)`,
and `torch.cuda.manual_seed_all(42)` for reproducibility.

Runtime: ~11 min MNIST-Family + ~38 min Multi-Domain on CPU (Apple M-series).
