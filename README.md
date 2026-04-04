# RSM-Net -- Recursive Submatrix Memory Network

> Modular neural architecture for **growing knowledge**, **selective unlearning**,
> and **reversible knowledge operations** via low-rank submatrices.

**Author:** Victor Alejandro Cano Jaramillo -- April 2026

## The Headline Result

With task-ID routing, removing a knowledge module leaves **all other modules perfectly intact**:

| Task | Before Removal | After Removing "Terranova" | Change |
|------|---------------|---------------------------|--------|
| WikiText | 173,655 | 173,655 | **0.0%** |
| NovaTech | 595.72 | 595.72 | **0.0%** |
| Lucas | 538.25 | 538.25 | **0.0%** |
| Bias | 276.32 | 276.32 | **0.0%** |
| Valar | 636.84 | 636.84 | **0.0%** |
| **Terranova** | **470.60** | **229.24** | **removed** |
| History | 4,504.22 | 4,504.22 | **0.0%** |
| Logic | 2,562.56 | 2,562.56 | **0.0%** |

**7/7 remaining tasks: exactly 0.0% degradation.** Tested on a 110M-param transformer with 8 knowledge modules.

## What is RSM-Net?

RSM-Net stores knowledge in **modular low-rank submatrices** that can be independently added, frozen, removed, or inspected. Each submatrix is a pair (A_k, B_k) that adds a task-specific perturbation to frozen base weights:

```
W_eff = W_base + sum_k gate_k * A_k @ B_k
```

**Knowledge operations:**
- **Add**: create a new submatrix for a new task (no retraining needed)
- **Remove**: delete a submatrix to forget specific knowledge (other modules unaffected)
- **Disable**: temporarily deactivate a module (reversible)
- **Inspect**: measure each module's contribution to any task

## Five Phases of Experiments

### Phase 1: MLP Continual Learning

RSM-Net MLP (400x200) on MNIST-Family and Multi-Domain benchmarks.

| Benchmark | RSM-Net | Naive | EWC | LoRA-Seq |
|-----------|---------|-------|-----|----------|
| MNIST-Family (Avg Acc) | 74.33% | 68.62% | 87.63% | 65.86% |
| MNIST-Family (Forgetting) | **0.274** | 0.374 | 0.024 | 0.399 |
| Multi-Domain (Avg Acc) | 61.13% | 42.86% | 65.94% | 33.31% |
| Multi-Domain (Forgetting) | **0.153** | 0.677 | 0.008 | 0.536 |

RSM-Net reduces forgetting **4.4x vs Naive** on multi-domain.

### Phase 2: MLP Ablation

| Variant | MNIST-Family Acc | Forgetting |
|---------|-----------------|------------|
| Full (r=16, sparsemax) | 73.56% | 0.286 |
| Softmax (no sparsemax) | 73.76% | 0.282 |
| Rank=4 | **75.41%** | **0.228** |
| Rank=32 | 65.95% | 0.407 |
| Depth=2 | 68.66% | 0.360 |

**Sparsemax vs softmax: no difference.** Protection comes from modular structure, not sparse routing.

### Phase 3: LLM Adapter (Qwen2.5-1.5B)

RSM-Net as adapter on a pretrained LLM. Code -> Math -> Creative.

| Method | Avg PPL | Forgetting |
|--------|---------|------------|
| RSM-Net (r=4) | 2.45 | 0.29 |
| Naive LoRA | **2.40** | **0.17** |
| LoRA-Seq | 2.46 | 0.31 |

**Negative result (honest):** RSM-Net gates don't help on LLMs. When the frozen backbone is massive (1.5B params) and the adapter is tiny (r=4), Naive LoRA is sufficient. The adapter-to-backbone ratio is too small for interference to occur.

### Phase 4: Unlearning (110M Transformer, WikiText-103 Pretraining)

Train from scratch on WikiText-103, then add NovaTech/Lucas/Bias via submatrices.

| After removing bias (Sub 3) | Change |
|-----------------------------|--------|
| WikiText PPL | +0.1% |
| NovaTech PPL | -5.3% |
| Lucas PPL | -19.7% |
| Bias PPL | **+80%** (degraded) |

| After also removing Lucas (Sub 2) | Change |
|-------------------------------------|--------|
| WikiText PPL | -0.8% |
| NovaTech PPL | -4.0% |
| Lucas PPL | **+110%** (forgotten) |

Selective knowledge removal works: Lucas is forgotten (+110%), NovaTech is retained (-4%).

### Phase 5: Growth + Task-ID Routing (8 Modules)

8 submatrices: WikiText, NovaTech, Lucas, Bias, Valar Syndrome, Terranova, History, Logic.

With task-ID routing (activate only the relevant submatrix): **0.0% cross-contamination** on removal (see headline table).

**Specialization confirmed**: Sub 1 is 12.65x more critical for math than for other tasks. Each submatrix develops internal specialization even though soft gates distribute uniformly.

## Experimental Progression

| Iteration | Forgetting | Key Change |
|-----------|-----------|------------|
| Initial prototype | 0.77 | Shared head, W_base random |
| + Sparsemax, bug fixes | 0.77 | Correct math |
| + Multi-head evaluation | 0.59 | Per-task heads |
| + W_base trained on task 0 | 0.34 | Semantic features |
| + Orthogonal keys | **0.27** | Stable routing |
| Task-ID routing | **0.00** | Perfect isolation |

## Architecture

```
Input x --> [ConvEncoder] (optional)
  |
  v
W_base(x) ---- frozen after task 0 --------+
  |                                          |
  v                                          |
Routing context --> query_proj --> gates      |
                                  |          |
                    [Sub 0] [Sub 1] [Sub 2]  |
                         \    |    /         |
                    sum_k gate_k * A_k@B_k --+---> output
```

**Two routing modes:**
- **Soft gating** (no task ID): gates distribute uniformly, protection via load distribution
- **Task-ID routing** (task ID known): activate only the relevant submatrix, perfect isolation

See [docs/architecture.md](docs/architecture.md) for details.

## Installation

```bash
pip install -e ".[dev]"  # Python 3.10+
```

## Project Structure

```
RSM-Net/
|-- src/rsm_net/           # Core implementation
|-- experiments/
|   |-- mlp/               # MLP benchmark results
|   |-- llm_adapter/       # Qwen2.5-1.5B adapter results
|   |-- from_scratch/      # 110M transformer results
|   |-- unlearning/        # NovaTech/Lucas/Bias removal
|   |-- growth/            # 8-module growth + task-ID routing
|   |-- figures/           # All plots and heatmaps
|-- docs/
|   |-- architecture.md    # Architecture + routing modes
|   |-- experiments.md     # All 5 experimental phases
|   |-- findings.md        # Key findings and analysis
|   |-- paper.md           # Mathematical formalization
|-- tests/                 # 45 tests (pytest)
```

## Key Findings

1. **Soft gates always uniform** (0.33-0.34) across MLPs, transformers, and LLMs
2. **Protection via load distribution**, not selective routing
3. **Internal specialization is real** despite uniform gates (12.65x ratio)
4. **Task-ID routing gives perfect isolation** (0.0% cross-contamination)
5. **Adapter-to-backbone ratio predicts effectiveness** (high ratio = RSM-Net helps, low ratio = Naive LoRA sufficient)
6. **Unlearning works**: selective knowledge removal with minimal collateral damage

See [docs/findings.md](docs/findings.md) for full analysis.

## Tests

```bash
pytest tests/ -v
```

## References

- Kirkpatrick et al. (2017) "Overcoming catastrophic forgetting in neural networks"
- Hu et al. (2021) "LoRA: Low-Rank Adaptation of Large Language Models"
- Martins & Astudillo (2016) "From Softmax to Sparsemax"

## License

Apache 2.0
