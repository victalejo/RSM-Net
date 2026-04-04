# Findings and Analysis

## 1. Soft Gates Always Uniform

Across every configuration tested -- MLPs, 110M transformers, Qwen2.5-1.5B LLM -- soft gates produce near-uniform distributions (0.33-0.34 per submatrix with 3 subs). No diagonal discrimination pattern ever emerged.

| Experiment | Gates Pattern |
|-----------|--------------|
| MLP MNIST-Family | 0.32/0.40/0.28 |
| MLP Multi-Domain | 0.48/0.47/0.46 |
| LLM Qwen r=4 | 0.32/0.34/0.34 |
| LLM Qwen r=64 | 0.33/0.33/0.34 |
| 110M from-scratch | 0.34/0.33/0.33 |
| 110M 8-module | uniform across all 8 |

The query_proj, trained only on task 0, cannot generate queries that discriminate between tasks sharing similar input distributions.

## 2. Protection Via Load Distribution

Despite uniform gates, RSM-Net reduces forgetting vs Naive and LoRA-Seq:
- Multi-Domain MLP: 4.4x less forgetting than Naive
- MNIST-Family MLP: 1.4x less forgetting than Naive

The mechanism: soft mixing distributes gradient pressure across submatrices. With W_base frozen and previous submatrices frozen, only the new submatrix absorbs adaptation. This is structural protection, not routing-based protection.

## 3. Internal Specialization Is Real

From the 110M transformer specialization test:
- Disabling Sub 1 raises math PPL by 324 but code PPL by only 0.4 (ratio: **12.65x**)
- Disabling Sub 2 raises creative PPL by 192 but improves math by 102

Each submatrix develops task-specific representations during training, even though the gates can't detect this at inference time.

## 4. Task-ID Routing = Perfect Isolation

When task identity is known and only the corresponding submatrix is activated:
- **7/7 tasks show exactly 0.0% PPL change** when any other submatrix is removed
- Each submatrix operates in complete isolation from the others
- Knowledge operations (add/remove/disable) have zero side effects

This is the definitive result: the modular structure supports perfect isolation when routing is explicit.

## 5. Adapter-to-Backbone Ratio

RSM-Net's protection matters when submatrices represent a significant fraction of total parameters:

| Setting | Adapter/Backbone | RSM-Net Helps? |
|---------|-----------------|----------------|
| MLP (400x200) | ~27% | Yes (0.27 vs 0.37 forgetting) |
| 110M transformer | ~20% per sub | Yes (specialization, removal) |
| Qwen 1.5B (r=4) | ~0.007% | No (Naive LoRA same or better) |
| Qwen 1.5B (r=64) | ~0.7% | No |
| Qwen unfrozen 8 layers | ~25% | No (both methods similar forgetting) |

When the adapter is tiny relative to the backbone, the backbone dominates and prevents interference regardless of method.

## 6. Unlearning Works

With a properly pretrained W_base (WikiText-103, 3 epochs):

**Removing bias submatrix:**
- WikiText: +0.1% (intact)
- NovaTech: -5.3% (slightly improved)
- Bias: +80% (degraded -- bias knowledge removed)

**Removing Lucas submatrix:**
- NovaTech: -4.0% (retained)
- Lucas: +110% (forgotten)
- WikiText: -0.8% (intact)

The "right to be forgotten" use case works: delete a person's data module, the person is forgotten, everything else is retained.

## 7. Base Quality Determines Isolation

| W_base Quality | WikiText PPL | Removal Side Effects |
|---------------|-------------|---------------------|
| Random (no training) | >100,000 | Catastrophic (subs compensate base) |
| 2000 samples | ~168,000 | Large (+155% WikiText on removal) |
| WikiText-103 full | ~41 | Minimal (+0.1% WikiText on removal) |

A strong W_base is essential. When W_base is weak, submatrices absorb general linguistic capacity in addition to task-specific knowledge, making removal destructive.

## 8. Ablation Summary

**Sparsemax vs softmax**: negligible difference. The protection is structural.

**Rank**: lower rank = less forgetting (rank=4 best). Smaller submatrices can't overfit to current task.

**Pruning**: no effect with 3 tasks (threshold never triggers). Would matter with 10+ tasks.

**Depth=2 (recursion)**: helps cross-domain (+1.66%), hurts same-domain (-4.9%). Overhead not justified for most cases.

## 9. What Doesn't Work

- **Soft gates for task discrimination**: never achieved, despite orthogonal keys, contrastive loss, context from hidden layers, EWC on query_proj
- **RSM-Net as LLM adapter**: no advantage over Naive LoRA when backbone is massive
- **W_base frozen from start (never trained)**: random features give routing nothing to work with
- **Training query_proj on all tasks (with EWC)**: still doesn't discriminate

## 10. Recommendations

- **Use task-ID routing** when task identity is available (most practical scenarios)
- **Use soft gating** only when task ID is truly unknown at inference
- **Default rank=4-8** (lower is better for forgetting)
- **Pretrain W_base thoroughly** before adding submatrices
- **RSM-Net is best for**: modular knowledge management, selective unlearning, growing systems
- **RSM-Net is not for**: LLM adapters, scenarios where Naive LoRA suffices
