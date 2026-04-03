# Findings and Analysis

## 1. Main Finding: Protection Through Load Distribution

The gates do NOT discriminate per task. Activation heatmaps show near-uniform
distribution across submatrices, not the diagonal pattern that would indicate
task-specific routing.

Yet RSM-Net reduces forgetting 4.4x vs Naive on multi-domain (0.153 vs 0.677)
and 1.4x on MNIST-Family (0.274 vs 0.374).

**The real mechanism is load distribution.** The soft mixture across submatrices
distributes gradient pressure during training. Instead of all gradients flowing
through a single set of weights (as in Naive fine-tuning), they are spread
across W_base + multiple submatrices. Since W_base is frozen and previous
submatrices are frozen, only the new submatrix absorbs the adaptation.

This is structurally similar to Progressive Neural Networks (Rusu et al. 2016),
but with shared computation through W_base and soft mixing instead of
hard lateral connections.

## 2. The Chicken-and-Egg Problem of Routing

For gates to discriminate between tasks, the query projection needs to
produce different query vectors for different task inputs. This requires
the query projection to have been trained on diverse inputs.

**Timeline of the problem:**

1. W_base initialized random -> `ReLU(W_base(x))` produces noisy features
2. query_proj trained only on MNIST -> learns MNIST-specific query space
3. FashionMNIST inputs produce queries similar to MNIST (same 28x28 grayscale)
4. Sparsemax can't distinguish -> near-uniform gate distribution

**The solution that worked:** Train W_base on task 0 first. This gives
`ReLU(W_base(x))` semantic structure (edges, curves, textures). But
query_proj, also trained only on task 0, still struggles to distinguish
FashionMNIST from MNIST in this learned feature space.

**Biological analogy:** The brain is not born blank. The visual cortex has
pre-wired structure from genetics (analogous to trained W_base). Adult
learning adds new connections on top of this structure (submatrices).
But the routing of attention (query_proj) develops over time with exposure
to diverse stimuli -- our model only sees one task type before freezing.

## 3. Evolution of Results

| Iteration | Avg Acc | Forgetting | Key Change |
|---|---|---|---|
| Initial prototype | ~37% | 0.77 | Shared head, W_base random, fake sparsemax |
| + Real sparsemax + bug fixes | ~37% | 0.77 | Correct math, still shared head |
| + Multi-head evaluation (Task-IL) | ~53% | 0.59 | Per-task heads prevent head overwrite |
| + W_base trained on task 0 | 70.0% | 0.338 | Semantic features for routing |
| + Orthogonal keys + frozen query | **74.3%** | **0.274** | Stable routing space |
| Multi-domain (conv encoder) | **61.1%** | **0.153** | Distinct distributions help |

The biggest single improvement came from training W_base on task 0 (37% -> 70%).
The second biggest from multi-head evaluation (shared head was destroying results).

## 4. Why EWC Wins in Absolute Forgetting

EWC achieves 0.008 forgetting on multi-domain vs RSM-Net's 0.153. Why?

**EWC protects ALL weights** using Fisher Information -- it doesn't need routing.
Every parameter that was important for task k gets a penalty proportional to its
Fisher importance. This is a global protection mechanism.

**RSM-Net depends on routing** to direct gradients to the right submatrix.
With imperfect routing, some gradients leak through to shared components
(query_proj on task 0, shared features in W_base). W_base is frozen so it's
safe, but query_proj was only trained on task 0.

**However, EWC has fundamental limitations that RSM-Net doesn't:**

1. **No modularity.** You can't remove knowledge of a specific task in EWC.
   In RSM-Net, you can prune a submatrix to "forget" a task on purpose.
2. **Capacity saturation.** EWC's Fisher penalties accumulate, increasingly
   constraining the network. With many tasks, new learning becomes difficult.
   RSM-Net adds new capacity (submatrix) per task.
3. **No dynamic architecture.** EWC uses fixed architecture. RSM-Net can
   grow, prune, and consolidate dynamically.

## 5. When to Use RSM-Net vs EWC

| Use Case | RSM-Net | EWC |
|---|---|---|
| Maximum retention of all tasks | -- | Best choice |
| Need to add tasks incrementally | Better (modular) | Works but capacity saturates |
| Need to selectively forget a task | Can prune specific submatrix | Not possible |
| Very long task sequences (100+) | Scales with pruning/consolidation | Fisher penalties accumulate |
| Different input domains | Works with ConvEncoder | Works (same architecture) |
| Inference speed matters | Slightly slower (gates) | Same as base model |

## 6. Where the Multi-Domain Benchmark Shines

RSM-Net's advantage is strongest in multi-domain (0.153 forgetting vs 0.677 Naive).
In MNIST-Family, the gap is smaller (0.274 vs 0.374).

**Why?** MNIST, CIFAR-10, and SVHN have very different visual statistics:
grayscale digits, color natural images, color street numbers. The frozen
ConvEncoder + W_base, trained on MNIST, creates a feature space where these
distributions are naturally separated. The submatrices can then specialize
for each distribution without interfering.

In MNIST-Family, all three datasets are 28x28 grayscale with similar statistics.
The separation is weaker, making submatrix specialization harder.

**Takeaway:** RSM-Net is best suited for continual learning across genuinely
different domains, not within a single domain.

## 7. Priority Future Work

### High Priority

1. **Contrastive routing loss.** An auxiliary loss that explicitly pushes gate
   activations toward diagonal discrimination:
   L_route = CE(argmax(alpha), task_id) during training.
   Requires task_id at training time (available), not at inference.

2. **Task-aware key initialization.** Instead of random orthogonal keys,
   initialize each key as the mean query vector for that task's training data.
   Run one epoch with random key, compute mean(query_proj(x)) over task data,
   use that as the initial key. This anchors each key to its task's distribution.

3. **Larger benchmarks.** Split-CIFAR-100 (20 tasks x 5 classes),
   Split-TinyImageNet (10 tasks x 20 classes). These are the standard
   benchmarks in the continual learning literature.

### Medium Priority

4. **Recursive depth > 1.** The code supports max_depth > 1 but hasn't
   been validated experimentally. Theory suggests depth=2 enables
   hierarchical task decomposition.

5. **Meta-learning of pruning threshold.** Currently, the prune threshold
   is a fixed hyperparameter. Learning it end-to-end could improve
   the trade-off between memory and retention.

### Lower Priority

6. **Pre-trained backbone.** Replace random W_base initialization + task-0
   training with a pre-trained feature extractor (e.g., first layers of
   ResNet-18 trained on ImageNet). This would give the routing much richer
   features from the start.

7. **Multi-seed statistical significance.** Current results use seed=42 only.
   Running with seeds {42, 123, 456} and reporting mean +/- std would
   strengthen the experimental claims.

## 8. Ablation Analysis

Source: `results/ablation.json`

### Results Table

**MNIST-Family:**

| Variant | Avg Acc | Forgetting | d Acc | d Forgetting |
|---|---|---|---|---|
| full (control) | 73.56% | 0.286 | -- | -- |
| no_prune | 73.56% | 0.286 | +0.00 | +0.000 |
| softmax | 73.76% | 0.282 | +0.20 | -0.004 |
| rank_4 | **75.41%** | **0.228** | **+1.85** | **-0.059** |
| rank_32 | 65.95% | 0.407 | -7.61 | +0.121 |
| depth_2 | 68.66% | 0.360 | -4.90 | +0.074 |

**Multi-Domain:**

| Variant | Avg Acc | Forgetting | d Acc | d Forgetting |
|---|---|---|---|---|
| full (control) | 58.66% | 0.193 | -- | -- |
| no_prune | 58.66% | 0.193 | +0.00 | +0.000 |
| softmax | 58.47% | 0.195 | -0.19 | +0.002 |
| rank_4 | 58.21% | **0.134** | -0.45 | **-0.059** |
| rank_32 | **59.02%** | 0.208 | **+0.36** | +0.016 |
| depth_2 | 60.32% | 0.167 | +1.66 | -0.026 |

### Key Findings

**1. Pruning has zero effect (no_prune = full).**
With only 3 tasks and 2 submatrices, no submatrix falls below the importance
threshold. Pruning would matter with 10+ tasks where some submatrices become
redundant. This is expected and validates the implementation (pruning only
activates when needed).

**2. Sparsemax vs softmax: negligible difference.**
softmax achieves nearly identical results to sparsemax on both benchmarks
(within 0.2% accuracy, 0.004 forgetting). This is the strongest evidence
for the load distribution hypothesis: the protection mechanism is NOT sparse
routing (only activating one submatrix), but the modular structure itself.
Whether gates are sparse or distributed doesn't matter because the gates
don't discriminate per task anyway.

**3. rank=4 is the sweet spot for MNIST-Family.**
Lower rank = better generalization + less forgetting (75.41% vs 73.56%,
forgetting 0.228 vs 0.286). This is counter-intuitive but explainable:
smaller submatrices have less capacity to overfit to the current task,
which reduces interference with representations from previous tasks.
The LoRA literature (Hu et al. 2021) also found that very low rank
(r=1-4) often works best.

**4. rank=32 hurts on MNIST-Family, helps slightly on Multi-Domain.**
Too much capacity per submatrix (rank=32) causes more forgetting on
similar tasks (MNIST-Family: 0.407 vs 0.286). On multi-domain, where
tasks are very different, the extra capacity helps marginally (59.02%
vs 58.66%). Trade-off: capacity helps when tasks are dissimilar,
hurts when tasks are similar.

**5. Recursive depth=2 hurts on MNIST-Family, helps on Multi-Domain.**
MNIST-Family: depth=2 drops accuracy by 4.9% and increases forgetting by 0.074.
Multi-Domain: depth=2 improves accuracy by 1.66% and reduces forgetting by 0.026.
Interpretation: recursive submatrices add useful hierarchical capacity for
truly different domains, but are unnecessary overhead for similar tasks.
With only 2 submatrices, the child submatrices double the parameter count
without adding proportional value for intra-domain benchmarks.

### Recommendations

Based on the ablation:
- **Default rank=4-8** instead of 16 for most applications
- **Skip sparsemax** (use softmax) unless sparse inference speed matters
- **Use depth=2 only** when tasks span very different domains
- **Pruning is a no-op** with few tasks; keep it for long task sequences
