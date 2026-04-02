# RSM-Net Prototype Code Review

**Date:** 2026-04-02

## Summary: 3 CRITICAL, 8 HIGH, 7 MEDIUM/LOW

### CRITICAL
1. Softmax+threshold used instead of sparsemax -- breaks key embedding gradients
2. `_activation_history` unbounded list, invisible to state_dict, memory leak
3. Optimizer holds stale refs after prune(); new submatrices never trained

### HIGH
4. Task 0 trains query_proj without any submatrix -- routing untrained for task 1
5. Wrong submatrix index (task_idx - 1) breaks after pruning
6. `torch.zeros() * 0.01` is still zero -- A matrix init incorrect
7. `importance_scores` plain list not serialized
8. compute_gates called twice per batch (double cost)
9. EWC Fisher uses batch gradient squared -- underestimates by ~batch_size
10. No random seed
11. Missing type annotations

### MEDIUM/LOW
12. compute_gates returns None -- should return empty tensor
13. DataLoader missing num_workers/pin_memory
14. print() instead of logging
15. Hardcoded hyperparameters
16. keys.T fragile
17. train() during Fisher uncommented
18. Redundant list comprehension
