"""
Continual learning experiment: MNIST -> FashionMNIST -> KMNIST

Compares RSM-Net vs Naive Fine-tuning vs EWC.
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

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rsm_net.baselines import EWCNet, NaiveFineTuneNet
from rsm_net.config import RSMConfig
from rsm_net.network import RSMNet
from rsm_net.training import (
    get_task_dataloaders,
    run_evaluation,
    set_seed,
    train_baseline_epoch,
    train_rsm_epoch,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_experiment(config: RSMConfig | None = None) -> dict:
    """
    Run the full continual learning experiment.

    Returns dict with all accuracy results.
    """
    if config is None:
        config = RSMConfig()

    set_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    logger.info("Config: rank=%d, key_dim=%d, max_depth=%d, epochs/task=%d",
                config.rank, config.key_dim, config.max_depth, config.epochs_per_task)

    tasks = list(config.tasks)

    # Initialize models
    rsm = RSMNet(config=config).to(device)
    naive = NaiveFineTuneNet(
        hidden_dims=config.hidden_dims, num_classes=config.num_classes
    ).to(device)
    ewc = EWCNet(
        hidden_dims=config.hidden_dims, num_classes=config.num_classes
    ).to(device)

    results: dict[str, dict[str, list[float]]] = {
        "RSM-Net": defaultdict(list),
        "Naive": defaultdict(list),
        "EWC": defaultdict(list),
    }

    all_test_loaders: dict[str, torch.utils.data.DataLoader] = {}
    start_time = time.time()

    for task_idx, task_name in enumerate(tasks):
        logger.info("=" * 60)
        logger.info("TASK %d: %s", task_idx, task_name)
        logger.info("=" * 60)

        train_loader, test_loader = get_task_dataloaders(
            task_name,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        )
        all_test_loaders[task_name] = test_loader

        # --- RSM-Net ---
        logger.info("[RSM-Net] Preparing task %d...", task_idx)
        rsm.prepare_new_task()
        rsm_optimizer = rsm.get_optimizer(task_idx, lr=config.lr)

        for layer in rsm.layers:
            logger.info("  Layer submatrices: %d", layer.num_submatrices)

        for epoch in range(config.epochs_per_task):
            loss, acc = train_rsm_epoch(
                rsm, train_loader, rsm_optimizer, device, task_idx, config
            )
            if (epoch + 1) % 2 == 0 or epoch == 0:
                logger.info(
                    "  [RSM] Epoch %d/%d -- Loss: %.4f, Acc: %.4f",
                    epoch + 1, config.epochs_per_task, loss, acc,
                )

        rsm.update_importance_all()

        # Consolidation at task boundaries
        if config.consolidate_at_task_boundary and task_idx > 0:
            merges = rsm.consolidate_all()
            if merges > 0:
                # Rebuild optimizer after structural change
                rsm_optimizer = rsm.get_optimizer(task_idx, lr=config.lr)

        # --- Naive ---
        logger.info("[Naive] Training...")
        naive_optimizer = torch.optim.Adam(naive.parameters(), lr=config.lr)
        for epoch in range(config.epochs_per_task):
            loss, acc = train_baseline_epoch(
                naive, train_loader, naive_optimizer, device
            )
        logger.info("  [Naive] Final -- Loss: %.4f, Acc: %.4f", loss, acc)

        # --- EWC ---
        logger.info("[EWC] Training...")
        ewc_optimizer = torch.optim.Adam(ewc.parameters(), lr=config.lr)
        ewc_lambda = config.ewc_lambda if task_idx > 0 else 0.0
        for epoch in range(config.epochs_per_task):
            loss, acc = train_baseline_epoch(
                ewc, train_loader, ewc_optimizer, device,
                ewc_lambda=ewc_lambda,
            )
        logger.info("  [EWC] Final -- Loss: %.4f, Acc: %.4f", loss, acc)
        ewc.store_parameters(task_idx, train_loader, device)

        # --- Evaluate on ALL tasks seen so far ---
        logger.info("Evaluation after task %d (%s):", task_idx, task_name)
        header = f"  {'Task':<20} {'RSM-Net':>10} {'Naive':>10} {'EWC':>10}"
        logger.info(header)
        logger.info("  " + "-" * 50)

        for prev_idx, prev_task in enumerate(tasks[: task_idx + 1]):
            prev_loader = all_test_loaders[prev_task]

            rsm_acc = run_evaluation(rsm, prev_loader, device)
            naive_acc = run_evaluation(naive, prev_loader, device)
            ewc_acc = run_evaluation(ewc, prev_loader, device)

            results["RSM-Net"][prev_task].append(rsm_acc)
            results["Naive"][prev_task].append(naive_acc)
            results["EWC"][prev_task].append(ewc_acc)

            marker = " <-- (new)" if prev_idx == task_idx else ""
            logger.info(
                "  %-20s %9.2f%% %9.2f%% %9.2f%%%s",
                prev_task, rsm_acc * 100, naive_acc * 100, ewc_acc * 100, marker,
            )

    elapsed = time.time() - start_time
    logger.info("Total experiment time: %.1f seconds", elapsed)

    # --- Final Summary ---
    print_summary(results, tasks, rsm, naive)

    # --- Plot ---
    plot_results(results, tasks)

    # --- Save results ---
    save_results(results, tasks, config, rsm)

    return dict(results)


def print_summary(
    results: dict, tasks: list[str], rsm: RSMNet, naive: NaiveFineTuneNet
) -> None:
    """Print final accuracy summary and forgetting metrics."""
    logger.info("=" * 60)
    logger.info("FINAL SUMMARY -- Accuracy after training all %d tasks", len(tasks))
    logger.info("=" * 60)

    header = f"  {'Task':<20} {'RSM-Net':>10} {'Naive':>10} {'EWC':>10}"
    logger.info(header)
    logger.info("  " + "-" * 50)

    final_accs: dict[str, list[float]] = {}
    for model_name in ["RSM-Net", "Naive", "EWC"]:
        final_accs[model_name] = []
        for task_name in tasks:
            acc = results[model_name][task_name][-1] if results[model_name][task_name] else 0
            final_accs[model_name].append(acc)

    for i, task_name in enumerate(tasks):
        logger.info(
            "  %-20s %9.2f%% %9.2f%% %9.2f%%",
            task_name,
            final_accs["RSM-Net"][i] * 100,
            final_accs["Naive"][i] * 100,
            final_accs["EWC"][i] * 100,
        )

    logger.info("  " + "-" * 50)
    for model_name in ["RSM-Net", "Naive", "EWC"]:
        avg = np.mean(final_accs[model_name])
        logger.info("  %-20s %9.2f%% (avg)", model_name, avg * 100)

    # Forgetting metric
    logger.info("\nForgetting (accuracy drop on previous tasks):")
    for model_name in ["RSM-Net", "Naive", "EWC"]:
        forgetting_vals = []
        for task_name in tasks[:-1]:
            accs = results[model_name][task_name]
            if len(accs) >= 2:
                forgetting_vals.append(max(accs) - accs[-1])
        avg_forgetting = np.mean(forgetting_vals) if forgetting_vals else 0
        logger.info("  %-15s Avg Forgetting: %.4f", model_name, avg_forgetting)

    # RSM-Net state
    logger.info("\nRSM-Net final state:")
    summary = rsm.get_state_summary()
    for layer_info in summary["layers"]:
        scores = [f"{s:.4f}" for s in layer_info["importance_scores"]]
        logger.info(
            "  Layer %d: %d submatrices, importance: %s",
            layer_info["layer"], layer_info["num_submatrices"], scores,
        )

    total_base = sum(p.numel() for p in naive.parameters())
    total_rsm = summary["total_params"]
    overhead = (total_rsm - total_base) / total_base * 100
    logger.info("\n  Params (Naive baseline): %s", f"{total_base:,}")
    logger.info("  Params (RSM-Net):        %s", f"{total_rsm:,}")
    logger.info("  Overhead:                %.1f%%", overhead)


def plot_results(results: dict, tasks: list[str]) -> None:
    """Generate and save accuracy plots."""
    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, len(tasks), figsize=(5 * len(tasks), 4))
    if len(tasks) == 1:
        axes = [axes]

    colors = {"RSM-Net": "#2196F3", "Naive": "#F44336", "EWC": "#4CAF50"}

    for i, task_name in enumerate(tasks):
        ax = axes[i]
        for model_name in ["RSM-Net", "Naive", "EWC"]:
            accs = results[model_name][task_name]
            if accs:
                x = list(range(len(accs)))
                ax.plot(
                    x, accs, "o-",
                    label=model_name,
                    color=colors[model_name],
                    linewidth=2,
                    markersize=6,
                )

        ax.set_title(f"Accuracy on {task_name}", fontsize=12, fontweight="bold")
        ax.set_xlabel("After task #")
        ax.set_ylabel("Accuracy")
        ax.set_ylim([0, 1.05])
        ax.set_xticks(range(len(tasks)))
        ax.set_xticklabels([f"T{j}" for j in range(len(tasks))])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = results_dir / "rsm_net_results.png"
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    logger.info("Plot saved to: %s", output_path)
    plt.close()


def save_results(
    results: dict, tasks: list[str], config: RSMConfig, rsm: RSMNet
) -> None:
    """Save results as JSON."""
    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    output = {
        "config": {
            "rank": config.rank,
            "key_dim": config.key_dim,
            "max_depth": config.max_depth,
            "epochs_per_task": config.epochs_per_task,
            "lr": config.lr,
            "seed": config.seed,
            "sparsity_lambda": config.sparsity_lambda,
            "frobenius_lambda": config.frobenius_lambda,
        },
        "tasks": tasks,
        "results": {
            model_name: {task: accs for task, accs in task_results.items()}
            for model_name, task_results in results.items()
        },
        "rsm_state": rsm.get_state_summary(),
    }

    output_path = results_dir / "experiment_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Results saved to: %s", output_path)


if __name__ == "__main__":
    config = RSMConfig(
        epochs_per_task=5,
        rank=8,
        key_dim=32,
        max_depth=1,
        batch_size=128,
        lr=0.001,
        seed=42,
        tasks=("MNIST", "FashionMNIST", "EMNIST"),
    )

    print("=" * 60)
    print("RSM-Net -- Recursive Submatrix Memory Network")
    print("Continual Learning Experiment v0.1")
    print("Victor Alejandro Cano Jaramillo -- Abril 2026")
    print("=" * 60)

    run_experiment(config)
