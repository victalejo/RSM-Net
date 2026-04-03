"""
Ablation Study -- RSM-Net
Runs 6 variants on both benchmarks (MNIST-Family + Multi-Domain).

Variants:
  1. full (control) -- baseline config
  2. no_prune -- disable pruning
  3. softmax -- softmax instead of sparsemax
  4. rank_4 -- low rank (4 vs 16)
  5. rank_32 -- high rank (32 vs 16)
  6. depth_2 -- recursive submatrices (depth 2)
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
from rsm_net.encoder import ConvEncoder
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


# ============================================================================
# Variant definitions
# ============================================================================

VARIANTS: dict[str, dict] = {
    "full": {},
    "no_prune": {"_skip_prune": True},
    "softmax": {"use_sparsemax": False},
    "rank_4": {"rank": 4},
    "rank_32": {"rank": 32},
    "depth_2": {"max_depth": 2},
}

BENCHMARK_CONFIGS = {
    "MNIST-Family": {
        "tasks": ("MNIST", "FashionMNIST", "EMNIST"),
        "use_conv_encoder": False,
        "encoder_in_channels": 1,
        "input_dim": 784,
        "hidden_dims": (400, 200),
    },
    "Multi-Domain": {
        "tasks": ("MNIST", "CIFAR10", "SVHN"),
        "use_conv_encoder": True,
        "encoder_in_channels": 3,
        "input_dim": 512,
        "hidden_dims": (256, 128),
    },
}


# ============================================================================
# Core training function (RSM-Net only, no baselines needed for ablation)
# ============================================================================

def train_variant_on_benchmark(
    config: RSMConfig,
    tasks: list[str],
    device: torch.device,
    skip_prune: bool = False,
) -> dict:
    """Train one RSM-Net variant on one benchmark. Returns metrics."""
    set_seed(config.seed)

    model = RSMNet(config=config).to(device)
    results: dict[str, list[float]] = defaultdict(list)
    all_test_loaders: dict[str, torch.utils.data.DataLoader] = {}

    for task_idx, task_name in enumerate(tasks):
        train_loader, test_loader = get_task_dataloaders(
            task_name,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        )
        all_test_loaders[task_name] = test_loader

        model.prepare_new_task()
        opt = model.get_optimizer(task_idx, lr=config.lr)
        sched = CosineAnnealingLR(opt, T_max=config.epochs_per_task, eta_min=1e-5)

        for epoch in range(config.epochs_per_task):
            loss, acc = train_rsm_epoch(model, train_loader, opt, device, task_idx, config)
            sched.step()

        model.update_importance_all()

        # Prune unless disabled
        if not skip_prune and task_idx > 0:
            pruned = model.prune_all()
            if pruned > 0:
                opt = model.get_optimizer(task_idx, lr=config.lr)

        # Multi-head evaluation
        for prev_idx, prev_task in enumerate(tasks[: task_idx + 1]):
            acc = run_evaluation(model, all_test_loaders[prev_task], device, task_id=prev_idx)
            results[prev_task].append(acc)

        logger.info(
            "  Task %d (%s): train_acc=%.4f | eval: %s",
            task_idx, task_name, acc,
            {t: f"{results[t][-1]*100:.1f}%" for t in tasks[:task_idx+1]},
        )

    # Compute metrics
    final_accs = [results[t][-1] for t in tasks if results[t]]
    forgetting_vals = []
    for t in tasks[:-1]:
        accs = results[t]
        if len(accs) >= 2:
            forgetting_vals.append(max(accs) - accs[-1])

    return {
        "per_task": {t: results[t][-1] for t in tasks if results[t]},
        "avg_accuracy": float(np.mean(final_accs)),
        "avg_forgetting": float(np.mean(forgetting_vals)) if forgetting_vals else 0.0,
        "all_results": dict(results),
        "total_params": sum(p.numel() for p in model.parameters()),
    }


# ============================================================================
# Main ablation runner
# ============================================================================

def run_ablation() -> dict:
    """Run all 6 variants on both benchmarks."""
    RESULTS_DIR.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    all_results: dict[str, dict] = {}

    for variant_name, overrides in VARIANTS.items():
        logger.info("\n" + "=" * 60)
        logger.info("VARIANT: %s", variant_name)
        logger.info("=" * 60)

        skip_prune = overrides.pop("_skip_prune", False)
        variant_results: dict[str, dict] = {}

        for bench_name, bench_cfg in BENCHMARK_CONFIGS.items():
            logger.info("\n--- %s on %s ---", variant_name, bench_name)

            config = RSMConfig(
                input_dim=bench_cfg["input_dim"],
                hidden_dims=bench_cfg["hidden_dims"],
                num_classes=10,
                rank=overrides.get("rank", 16),
                key_dim=64,
                max_depth=overrides.get("max_depth", 1),
                epochs_per_task=20,
                batch_size=128,
                lr=0.001,
                seed=42,
                sparsity_lambda=0.001,
                frobenius_lambda=0.0001,
                contrastive_lambda=0.01,
                use_sparsemax=overrides.get("use_sparsemax", True),
                use_conv_encoder=bench_cfg["use_conv_encoder"],
                encoder_in_channels=bench_cfg["encoder_in_channels"],
                tasks=bench_cfg["tasks"],
            )

            t0 = time.time()
            metrics = train_variant_on_benchmark(
                config, list(bench_cfg["tasks"]), device, skip_prune=skip_prune,
            )
            elapsed = time.time() - t0
            metrics["time_seconds"] = elapsed

            variant_results[bench_name] = metrics
            logger.info(
                "  %s/%s: Avg=%.2f%% Forg=%.4f (%.0fs)",
                variant_name, bench_name,
                metrics["avg_accuracy"] * 100, metrics["avg_forgetting"], elapsed,
            )

        # Restore _skip_prune for potential re-runs
        if skip_prune:
            overrides["_skip_prune"] = True

        all_results[variant_name] = variant_results

    # Save JSON
    save_path = RESULTS_DIR / "ablation.json"
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("\nResults saved: %s", save_path)

    # Print tables
    print_ablation_tables(all_results)

    # Plot
    plot_ablation(all_results)

    return all_results


def print_ablation_tables(results: dict) -> None:
    """Print formatted ablation tables with deltas vs control."""
    for bench_name in BENCHMARK_CONFIGS:
        logger.info("\n" + "=" * 75)
        logger.info("ABLATION -- %s", bench_name)
        logger.info("=" * 75)

        control = results["full"][bench_name]
        ctrl_acc = control["avg_accuracy"]
        ctrl_forg = control["avg_forgetting"]

        logger.info(
            "  %-20s %10s %12s %14s %14s",
            "Variant", "Avg Acc", "Forgetting", "d Acc vs Full", "d Forg vs Full",
        )
        logger.info("  " + "-" * 70)

        for variant_name in VARIANTS:
            m = results[variant_name][bench_name]
            acc = m["avg_accuracy"]
            forg = m["avg_forgetting"]
            d_acc = acc - ctrl_acc
            d_forg = forg - ctrl_forg

            d_acc_str = f"{d_acc:+.2f}%" if variant_name != "full" else "--"
            d_forg_str = f"{d_forg:+.4f}" if variant_name != "full" else "--"

            logger.info(
                "  %-20s %9.2f%% %11.4f %14s %14s",
                variant_name, acc * 100, forg, d_acc_str, d_forg_str,
            )


def plot_ablation(results: dict) -> None:
    """Bar chart comparing ablation variants across both benchmarks."""
    variant_names = list(VARIANTS.keys())
    bench_names = list(BENCHMARK_CONFIGS.keys())

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for col, bench_name in enumerate(bench_names):
        accs = [results[v][bench_name]["avg_accuracy"] * 100 for v in variant_names]
        forgets = [results[v][bench_name]["avg_forgetting"] for v in variant_names]

        # Reference line (full control)
        ctrl_acc = accs[0]
        ctrl_forg = forgets[0]

        colors = plt.cm.Set2(np.linspace(0, 1, len(variant_names)))

        # Accuracy
        ax = axes[0, col]
        bars = ax.barh(range(len(variant_names)), accs, color=colors)
        ax.axvline(x=ctrl_acc, color="gray", linestyle="--", alpha=0.7, label="full (control)")
        ax.set_yticks(range(len(variant_names)))
        ax.set_yticklabels(variant_names, fontsize=9)
        ax.set_xlabel("Avg Accuracy (%)")
        ax.set_title(f"{bench_name} -- Accuracy", fontweight="bold")
        for i, v in enumerate(accs):
            ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=8)

        # Forgetting
        ax = axes[1, col]
        bars2 = ax.barh(range(len(variant_names)), forgets, color=colors)
        ax.axvline(x=ctrl_forg, color="gray", linestyle="--", alpha=0.7, label="full (control)")
        ax.set_yticks(range(len(variant_names)))
        ax.set_yticklabels(variant_names, fontsize=9)
        ax.set_xlabel("Avg Forgetting (lower is better)")
        ax.set_title(f"{bench_name} -- Forgetting", fontweight="bold")
        for i, v in enumerate(forgets):
            ax.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=8)

    plt.tight_layout()
    path = RESULTS_DIR / "ablation_chart.png"
    plt.savefig(str(path), dpi=150, bbox_inches="tight")
    logger.info("Ablation chart saved: %s", path)
    plt.close()


if __name__ == "__main__":
    print("=" * 60)
    print("RSM-Net -- Ablation Study")
    print("6 variants x 2 benchmarks")
    print("Victor Alejandro Cano Jaramillo -- Abril 2026")
    print("=" * 60)

    run_ablation()
