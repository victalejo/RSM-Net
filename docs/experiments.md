# Experimental Design and Results

## Five Experimental Phases

### Phase 1: MLP Continual Learning

**Setup**: RSM-Net MLP (400x200) vs Naive, EWC, LoRA-Seq. Two benchmarks:
- MNIST-Family: MNIST -> FashionMNIST -> EMNIST (same domain, 28x28)
- Multi-Domain: MNIST -> CIFAR-10 -> SVHN (cross-domain, conv encoder)

Config: rank=16, key_dim=64, 20 epochs/task, seed=42, cosine LR.

**MNIST-Family:**

| Model | MNIST | FashionMNIST | EMNIST | Avg Acc | Forgetting |
|-------|-------|-------------|--------|---------|------------|
| RSM-Net | 89.10% | 42.83% | 91.06% | 74.33% | 0.274 |
| Naive | 67.27% | 46.08% | 92.51% | 68.62% | 0.374 |
| EWC | 95.81% | 87.15% | 79.93% | 87.63% | 0.024 |
| LoRA-Seq | 48.24% | 58.17% | 91.16% | 65.86% | 0.399 |

**Multi-Domain:**

| Model | MNIST | CIFAR-10 | SVHN | Avg Acc | Forgetting |
|-------|-------|----------|------|---------|------------|
| RSM-Net | 90.77% | 26.56% | 66.06% | 61.13% | 0.153 |
| Naive | 21.26% | 16.14% | 91.18% | 42.86% | 0.677 |
| EWC | 99.10% | 50.97% | 47.75% | 65.94% | 0.008 |
| LoRA-Seq | 16.87% | 22.06% | 61.00% | 33.31% | 0.536 |

Source: `experiments/mlp/dual_benchmark_results.json`

### Phase 2: MLP Ablation (6 variants x 2 benchmarks)

| Variant | MNIST-Family Acc | Forgetting | Multi-Domain Acc | Forgetting |
|---------|-----------------|------------|-----------------|------------|
| Full (r=16) | 73.56% | 0.286 | 58.66% | 0.193 |
| No pruning | 73.56% | 0.286 | 58.66% | 0.193 |
| Softmax | 73.76% | 0.282 | 58.47% | 0.195 |
| Rank=4 | 75.41% | 0.228 | 58.21% | 0.134 |
| Rank=32 | 65.95% | 0.407 | 59.02% | 0.208 |
| Depth=2 | 68.66% | 0.360 | 60.32% | 0.167 |

Key findings: sparsemax vs softmax negligible; rank=4 best forgetting; depth=2 helps cross-domain.

Source: `experiments/mlp/ablation.json`

### Phase 3: LLM Adapter (Qwen2.5-1.5B-Instruct)

RSM-Net as adapter on pretrained LLM. Tasks: code -> math -> creative. 1000 samples/task.

**Main experiment (r=4):**

| Method | Code PPL | Math PPL | Creative PPL | Avg PPL | Forgetting |
|--------|----------|----------|-------------|---------|------------|
| Base | 5.87 | 3.70 | 11.77 | 7.11 | -- |
| RSM-Net | 2.03 | 1.80 | 3.51 | 2.45 | 0.29 |
| Naive LoRA | 2.04 | 1.62 | 3.54 | 2.40 | 0.17 |
| LoRA-Seq | 2.04 | 1.83 | 3.51 | 2.46 | 0.31 |

**Rank scaling (r=4, 32, 64):** Naive LoRA forgetting stays low (0.17-0.20) regardless of rank. Gates uniform (0.33) at all ranks.

**Unfrozen backbone (last 8 layers):** RSM-Net forgetting 0.28, Naive 0.26. No advantage.

**Conclusion:** RSM-Net gates don't help when adapter-to-backbone ratio is low.

Source: `experiments/llm_adapter/`

### Phase 4: Unlearning (110M Transformer from Scratch)

Pretrained on WikiText-103 (768K sequences, 3 epochs, PPL=41.24). Then added NovaTech/Lucas/Bias via submatrices.

**Selective removal results:**

| After removing bias (Sub 3) | WikiText | NovaTech | Lucas | Bias |
|-----------------------------|----------|----------|-------|------|
| Change | +0.1% | -5.3% | -19.7% | +80% |

| After also removing Lucas (Sub 2) | WikiText | NovaTech | Lucas |
|-------------------------------------|----------|----------|-------|
| Change | -0.8% | -4.0% | +110% |

Lucas forgotten (+110%), NovaTech retained (-4%), WikiText intact (-0.8%).

Source: `experiments/unlearning/full_base_experiment.json`

### Phase 5: Growth + Task-ID Routing (8 Modules)

8 submatrices on 110M transformer: WikiText, NovaTech, Lucas, Bias, Valar Syndrome, Terranova del Este, History, Logic.

**Task-ID routing removal (the definitive result):**

All 7 remaining tasks show exactly 0.0% PPL change when Terranova is removed. Perfect modular isolation.

**Cross-task specialization matrix:** Each submatrix gives best PPL on its own task. Off-diagonal/diagonal ratio = 0.43x (submatrices are worse on other tasks).

Source: `experiments/growth/task_id_eval.json`

## Progression Summary

| Iteration | Metric | Value | Key Change |
|-----------|--------|-------|------------|
| Prototype | Forgetting | 0.77 | Shared head, W_base random |
| + Multi-head | Forgetting | 0.59 | Per-task classification heads |
| + W_base trained | Forgetting | 0.34 | Semantic features |
| + Orthogonal keys | Forgetting | 0.27 | Stable routing space |
| Task-ID routing | Forgetting | **0.00** | Perfect isolation |

## Reproducibility

All MLP experiments: seed=42, `python -m experiments.dual_benchmark`

LLM and transformer experiments: see scripts in `experiments/` subdirectories.
