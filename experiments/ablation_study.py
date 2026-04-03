"""
Ablation study for RSM-Net.

Tests 6 variants to isolate the contribution of each component:
1. RSM-Net (full) -- complete configuration
2. RSM-Net (no prune) -- without pruning
3. RSM-Net (softmax) -- softmax instead of sparsemax
4. RSM-Net (rank=4) -- low rank
5. RSM-Net (rank=32) -- high rank
6. RSM-Net (depth=2) -- recursive submatrices
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rsm_net.config import RSMConfig
from rsm_net.network import RSMNet
from rsm_net.training import (
    get_task_dataloaders,
    run_evaluation,
    set_seed,
    train_rsm_epoch,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


# Monkey-patch for softmax variant
_use_softmax = False


def train_single_variant(
    name: str, config: RSMConfig, use_softmax: bool = False,
    skip_prune: bool = False,
) -> dict:
    """Train one RSM-Net variant and return results."""
    global _use_softmax
    _use_softmax = use_softmax

    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tasks = list(config.tasks)

    model = RSMNet(config=config).to(device)

    # For softmax variant, monkey-patch compute_gates
    if use_softmax:
        for layer in model.layers:
            original_compute_gates = layer.compute_gates.__func__

            def softmax_gates(self, x):
                K = len(self.key_embeddings)
                if K == 0:
                    return torch.zeros(x.size(0), 0, device=x.device)
                q = self.query_proj(x)
                keys = torch.stack(list(self.key_embeddings))
                scores = torch.matmul(q, keys.transpose(-2, -1)) / (self.key_dim ** 0.5)
                return F.softmax(scores, dim=-1)

            import types
            layer.compute_gates = types.MethodType(softmax_gates, layer)

    results: dict[str, list[float]] = defaultdict(list)
    all_test_loaders = {}

    for task_idx, task_name in enumerate(tasks):
        train_loader, test_loader = get_task_dataloaders(
            task_name, batch_size=config.batch_size,
            num_workers=config.num_workers, pin_memory=config.pin_memory,
        )
        all_test_loaders[task_name] = test_loader

        model.prepare_new_task()
        opt = model.get_optimizer(task_idx, lr=config.lr)
        sched = CosineAnnealingLR(opt, T_max=config.epochs_per_task, eta_min=1e-5)

        for epoch in range(config.epochs_per_task):
            train_rsm_epoch(model, train_loader, opt, device, task_idx, config)
            sched.step()

        model.update_importance_all()

        if not skip_prune:
            model.prune_all()
            opt = model.get_optimizer(task_idx, lr=config.lr)

        # Multi-head evaluation on all tasks
        for prev_idx, prev_task in enumerate(tasks[: task_idx + 1]):
            acc = run_evaluation(model, all_test_loaders[prev_task], device, task_id=prev_idx)
            results[prev_task].append(acc)

    # Compute final metrics
    final_accs = [results[t][-1] for t in tasks if results[t]]
    forgetting_vals = []
    for t in tasks[:-1]:
        accs = results[t]
        if len(accs) >= 2:
            forgetting_vals.append(max(accs) - accs[-1])

    return {
        "name": name,
        "final_accuracy": {t: results[t][-1] for t in tasks if results[t]},
        "avg_accuracy": float(np.mean(final_accs)),
        "avg_forgetting": float(np.mean(forgetting_vals)) if forgetting_vals else 0.0,
        "all_results": dict(results),
        "total_params": sum(p.numel() for p in model.parameters()),
    }


def run_ablation(base_config: RSMConfig | None = None) -> list[dict]:
    """Run all 6 ablation variants."""
    if base_config is None:
        base_config = RSMConfig(
            hidden_dims=(400, 200),
            rank=16,
            key_dim=64,
            max_depth=1,
            epochs_per_task=20,
            batch_size=128,
            lr=0.001,
            seed=42,
            tasks=("MNIST", "FashionMNIST", "EMNIST"),
        )

    RESULTS_DIR.mkdir(exist_ok=True)

    variants = [
        ("Full (r=16, d=1, sparsemax)", base_config, False, False),
        ("No Pruning", base_config, False, True),
        ("Softmax (no sparsemax)", base_config, True, False),
        ("Rank=4", RSMConfig(**{**vars(base_config), "rank": 4}), False, False),
        ("Rank=32", RSMConfig(**{**vars(base_config), "rank": 32}), False, False),
        ("Depth=2", RSMConfig(**{**vars(base_config), "max_depth": 2}), False, False),
    ]

    all_results = []
    for name, config, use_softmax, skip_prune in variants:
        logger.info("=" * 60)
        logger.info("ABLATION: %s", name)
        logger.info("=" * 60)
        start = time.time()
        result = train_single_variant(name, config, use_softmax, skip_prune)
        elapsed = time.time() - start
        result["time_seconds"] = elapsed
        all_results.append(result)
        logger.info(
            "  -> Avg Acc: %.2f%%, Avg Forgetting: %.4f (%.0fs)",
            result["avg_accuracy"] * 100, result["avg_forgetting"], elapsed,
        )

    # Save
    path = RESULTS_DIR / "ablation.json"
    with open(path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Ablation results saved: %s", path)

    # Print comparison table
    logger.info("\n" + "=" * 70)
    logger.info("ABLATION SUMMARY")
    logger.info("=" * 70)
    logger.info("  %-30s %10s %12s %10s", "Variant", "Avg Acc", "Forgetting", "Params")
    logger.info("  " + "-" * 62)
    for r in all_results:
        logger.info(
            "  %-30s %9.2f%% %11.4f %10s",
            r["name"], r["avg_accuracy"] * 100, r["avg_forgetting"],
            f"{r['total_params']:,}",
        )

    # Plot
    plot_ablation(all_results)
    return all_results


def plot_ablation(results: list[dict]) -> None:
    """Bar chart comparing ablation variants."""
    names = [r["name"] for r in results]
    accs = [r["avg_accuracy"] * 100 for r in results]
    forgets = [r["avg_forgetting"] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.tab10(np.linspace(0, 1, len(names)))

    # Accuracy
    bars = ax1.barh(range(len(names)), accs, color=colors)
    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels(names, fontsize=9)
    ax1.set_xlabel("Average Accuracy (%)")
    ax1.set_title("Ablation: Average Accuracy", fontweight="bold")
    for i, v in enumerate(accs):
        ax1.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=9)

    # Forgetting
    bars2 = ax2.barh(range(len(names)), forgets, color=colors)
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels(names, fontsize=9)
    ax2.set_xlabel("Average Forgetting")
    ax2.set_title("Ablation: Average Forgetting (lower is better)", fontweight="bold")
    for i, v in enumerate(forgets):
        ax2.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    path = RESULTS_DIR / "ablation_chart.png"
    plt.savefig(str(path), dpi=150, bbox_inches="tight")
    logger.info("Ablation chart saved: %s", path)
    plt.close()


if __name__ == "__main__":
    print("=" * 60)
    print("RSM-Net -- Ablation Study")
    print("Victor Alejandro Cano Jaramillo -- Abril 2026")
    print("=" * 60)

    run_ablation()
