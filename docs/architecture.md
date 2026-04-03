# RSM-Net Architecture

## Overview

RSM-Net is an MLP with SubmatrixLinear layers that store task-specific knowledge
in low-rank weight perturbations. An optional ConvEncoder handles multi-domain
inputs (images of different sizes and channel counts).

```
Input x (image or flat vector)
  |
  v
[ConvEncoder] ---- optional, for multi-domain (CIFAR-10, SVHN, etc.)
  |                 2 conv layers: Conv(in_ch, 16, 3) -> Conv(16, 32, 3)
  |                 AdaptiveAvgPool(4) -> Linear(512, input_dim)
  |                 Trained on task 0, frozen after
  v
h = flat features (input_dim)
  |
  v
+---------- SubmatrixLinear Layer 0 ----------+
|                                              |
|  W_base(h)  [frozen after task 0]            |
|      |                                       |
|  routing_ctx = ReLU(W_base(h))               |
|      |                                       |
|  q = query_proj(routing_ctx) [frozen]        |
|      |                                       |
|  scores = q @ keys^T / sqrt(d_key)           |
|  gates = sparsemax(scores)                   |
|      |                                       |
|  out = W_base(h) + sum_k gates_k * A_k@B_k@h|
|                                              |
+----------------------------------------------+
  |
  v  (ReLU)
+---------- SubmatrixLinear Layer 1 ----------+
|  (same structure, same routing_ctx)          |
+----------------------------------------------+
  |
  v  (ReLU)
task_head_k(h)  --> logits (10 classes)
```

## SubmatrixLinear -- The Core Building Block

### W_base (frozen backbone)

Standard `nn.Linear(in_features, out_features)`. Trained on task 0 using
standard backpropagation. Frozen permanently after task 0 completes.

W_base captures fundamental representations learned from the first task.
Because it is frozen, its features are never overwritten by subsequent tasks.
This is analogous to the biological critical period in neural development.

### Submatrices (task-specific adapters)

Each task k>0 gets a low-rank weight perturbation:

```
delta_W_k = A_k @ B_k
  A_k in R^(out_features x rank)
  B_k in R^(rank x in_features)
```

Initialization follows LoRA (Hu et al. 2021):
- A_k: random Gaussian * 0.01
- B_k: zeros

This ensures A_k @ B_k = 0 at initialization (no perturbation to W_base),
while A_k receives gradient signal from the start.

After training task k, both A_k and B_k are frozen.

### Key Embeddings (orthogonal initialization)

Each submatrix k has a key embedding `e_k in R^(key_dim)` used for routing.
Keys are initialized orthogonal to all existing keys using Gram-Schmidt:

```python
def _init_orthogonal_key(self):
    if K == 0:
        return random_unit_vector()
    v = random_vector()
    for existing_key in keys:
        v = v - project(v, existing_key)  # Gram-Schmidt
    return normalize(v)
```

This maximizes separation in the key space, giving sparsemax the best
chance to discriminate between submatrices.

### Query Projection

`query_proj: Linear(context_dim, key_dim)` maps routing context to a query
vector. The routing context comes from `ReLU(W_base(x))` -- the first hidden
layer's base features.

Trained on task 0 alongside W_base. Frozen permanently after task 0.

### Sparsemax Gates

Relevance scores: `s_k = q^T @ e_k / sqrt(d_key)` (scaled dot product)

Gates: `alpha = sparsemax(s)` (Martins & Astudillo 2016)

Unlike softmax, sparsemax produces exact zeros for low-scoring entries.
When K=1 (only one submatrix), gates are bypassed (trivially 1.0).

### Effective Weight

```
W_eff(x) = W_base + sum_k alpha_k(x) * A_k @ B_k
```

The forward pass: `h = ReLU(W_eff(x) @ x)`

## Training Flow Per Task

### Task 0

1. Create classification head_0
2. Train all parameters: W_base, query_proj, head_0
3. Freeze W_base and query_proj permanently

### Task k (k > 0)

1. Freeze all existing submatrices (A_j, B_j, e_j for j < k)
2. Create new submatrix: A_k (random), B_k (zeros), e_k (orthogonal to existing)
3. Create classification head_k
4. Train only: A_k, B_k, e_k, head_k
5. Optimizer contains only these parameters

### Evaluation (Multi-Head, Task-IL Protocol)

Each task is evaluated using its own classification head.
This is the standard Task-Incremental Learning (Task-IL) protocol
used in continual learning benchmarks.

## Pruning

Submatrices track importance via exponential moving average of gate activations:

```
I_k = EMA(mean(alpha_k(x))) over training batches
```

Submatrices with importance below threshold are removed.
After pruning, the optimizer must be rebuilt.

## Consolidation

Similar submatrices can be merged to reduce parameter count:

1. Compute cosine similarity between reconstructed delta-Ws
2. If similarity > threshold, merge via importance-weighted averaging
3. Re-decompose merged matrix using truncated SVD to maintain rank

## ConvEncoder (Multi-Domain)

For tasks with different image sizes or channel counts (e.g., MNIST 28x28
grayscale + CIFAR-10 32x32 RGB), a shared ConvEncoder preprocesses all
inputs to a fixed feature dimension:

```
Conv2d(in_ch, 16, 3, pad=1) -> ReLU -> MaxPool(2)
Conv2d(16, 32, 3, pad=1) -> ReLU -> MaxPool(2)
AdaptiveAvgPool2d(4) -> Flatten -> Linear(512, out_features)
```

Channel adaptation: grayscale inputs are expanded to 3 channels,
RGB inputs are passed through directly.

Trained on task 0, frozen after (same lifecycle as W_base).

## Key Hyperparameters

| Parameter | Default | Effect |
|---|---|---|
| `rank` | 16 | Capacity of each submatrix. Higher = more expressive but more params |
| `key_dim` | 64 | Dimension of routing space. Higher = more discriminative keys |
| `hidden_dims` | (400, 200) | MLP layer sizes. Determines W_base capacity |
| `sparsity_lambda` | 0.001 | L1 penalty on gate activations. Encourages sparse routing |
| `contrastive_lambda` | 0.01 | Margin-based penalty pushing keys apart (margin=2.0) |
| `frobenius_lambda` | 0.0001 | Frobenius norm on submatrix weights. Prevents explosion |
| `max_depth` | 1 | Recursive depth (submatrices of submatrices). d=1 for now |
