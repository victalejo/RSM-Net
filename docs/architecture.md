# RSM-Net Architecture

## Core Concept

RSM-Net stores knowledge in modular low-rank submatrices. The effective weight of each layer is:

```
W_eff(x) = W_base + sum_k gate_k(x) * A_k @ B_k
```

Where W_base is frozen after initial training, and each (A_k, B_k) pair is a task-specific low-rank perturbation.

## SubmatrixLinear -- The Core Building Block

### W_base (frozen backbone)

Standard linear layer trained on the first task. Frozen permanently after. Captures fundamental representations.

### Submatrices (task-specific adapters)

Each task k gets a low-rank weight perturbation:
```
delta_W_k = A_k @ B_k    (A: out x rank, B: rank x in)
```

LoRA-style initialization: A random, B zeros (A@B = 0 initially).

### Two Routing Modes

**Soft Gating (no task ID at inference):**
```
gates = softmax(query_proj(context) @ keys^T / sqrt(d_key))
output = W_base(x) + sum_k gates_k * A_k @ B_k @ x
```
- Gates distribute uniformly (~0.33 per submatrix)
- Protection via load distribution, not selective routing
- No task ID needed at inference time

**Task-ID Routing (task ID known):**
```
output = W_base(x) + A_task_id @ B_task_id @ x
```
- Activate ONLY the submatrix for the current task
- Perfect isolation: 0.0% cross-contamination on removal
- Requires knowing which task is being evaluated

### Key Embeddings (Gram-Schmidt)

Each submatrix has a key embedding initialized orthogonal to all existing keys via Gram-Schmidt, maximizing separation in the routing space.

## Knowledge Operations

| Operation | Effect | Reversible? |
|-----------|--------|-------------|
| **Add** | Create new submatrix for new task | -- |
| **Freeze** | Lock submatrix parameters | Yes (unfreeze) |
| **Remove** | Delete submatrix entirely | No (but can reload from checkpoint) |
| **Disable** | Temporarily deactivate (gate = 0) | Yes (enable) |
| **Inspect** | Measure contribution to any task | Non-destructive |
| **Consolidate** | Merge similar submatrices via SVD | No |

## Training Flow

### Task 0 (Base Training)
1. Train W_base + query_proj + Sub 0 + head_0
2. Freeze W_base and query_proj permanently

### Task k > 0 (Incremental)
1. Freeze all existing submatrices
2. Create Sub k with orthogonal key
3. Train only: A_k, B_k, key_k, head_k

## Optional Components

**ConvEncoder**: For multi-domain inputs (different image sizes/channels). 2-layer CNN producing fixed-size features. Trained on task 0, frozen after.

**Pruning**: Remove submatrices with low importance scores (exponential moving average of gate activations).

**Consolidation**: Merge similar submatrices via importance-weighted SVD.

## Hyperparameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| rank | 8 | Capacity per submatrix (lower = less forgetting) |
| key_dim | 64 | Routing space dimension |
| scale | 1.0 | Submatrix contribution scaling |
| contrastive_lambda | 0.01 | Push keys apart (margin-based) |
