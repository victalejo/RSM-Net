"""
Dual benchmark experiment:
  Benchmark 1: MNIST-family (MNIST -> FashionMNIST -> EMNIST)
  Benchmark 2: Multi-Domain (MNIST -> CIFAR-10 -> SVHN)

Compares RSM-Net vs Naive vs EWC vs LoRA-Seq on both.
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

from rsm_net.baselines import EWCNet, NaiveFineTuneNet, SequentialLoRANet
from rsm_net.config import RSMConfig
from rsm_net.encoder import ConvEncoder
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

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
MODEL_NAMES = ["RSM-Net", "Naive", "EWC", "LoRA-Seq"]
COLORS = {
    "RSM-Net": "#2196F3",
    "Naive": "#F44336",
    "EWC": "#4CAF50",
    "LoRA-Seq": "#FF9800",
}


def run_single_benchmark(
    benchmark_name: str,
    config: RSMConfig,
    models: dict[str, torch.nn.Module],
    tasks: list[str],
    device: torch.device,
) -> dict[str, dict[str, list[float]]]:
    """Run one benchmark (sequence of tasks) on pre-initialized models."""
    results: dict[str, dict[str, list[float]]] = {
        name: defaultdict(list) for name in MODEL_NAMES
    }
    all_test_loaders: dict[str, torch.utils.data.DataLoader] = {}

    rsm = models["RSM-Net"]
    naive = models["Naive"]
    ewc = models["EWC"]
    lora = models["LoRA-Seq"]

    for task_idx, task_name in enumerate(tasks):
        logger.info("=" * 60)
        logger.info("[%s] TASK %d: %s", benchmark_name, task_idx, task_name)
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
        rsm_opt = rsm.get_optimizer(task_idx, lr=config.lr)
        rsm_sched = CosineAnnealingLR(rsm_opt, T_max=config.epochs_per_task, eta_min=1e-5)

        for epoch in range(config.epochs_per_task):
            loss, acc = train_rsm_epoch(
                rsm, train_loader, rsm_opt, device, task_idx, config
            )
            rsm_sched.step()
            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info("  [RSM] Epoch %d/%d -- Loss: %.4f, Acc: %.4f",
                            epoch + 1, config.epochs_per_task, loss, acc)

        rsm.update_importance_all()

        # --- Naive ---
        naive.add_task_head()
        naive_opt = torch.optim.Adam(naive.parameters(), lr=config.lr)
        naive_sched = CosineAnnealingLR(naive_opt, T_max=config.epochs_per_task, eta_min=1e-5)
        for epoch in range(config.epochs_per_task):
            loss, acc = train_baseline_epoch(naive, train_loader, naive_opt, device, task_id=task_idx)
            naive_sched.step()
        logger.info("  [Naive] Final -- Loss: %.4f, Acc: %.4f", loss, acc)

        # --- EWC ---
        ewc.add_task_head()
        ewc_opt = torch.optim.Adam(ewc.parameters(), lr=config.lr)
        ewc_sched = CosineAnnealingLR(ewc_opt, T_max=config.epochs_per_task, eta_min=1e-5)
        ewc_lambda = config.ewc_lambda if task_idx > 0 else 0.0
        for epoch in range(config.epochs_per_task):
            loss, acc = train_baseline_epoch(
                ewc, train_loader, ewc_opt, device,
                ewc_lambda=ewc_lambda, task_id=task_idx,
            )
            ewc_sched.step()
        logger.info("  [EWC] Final -- Loss: %.4f, Acc: %.4f", loss, acc)
        ewc.store_parameters(task_idx, train_loader, device)

        # --- LoRA-Seq ---
        lora.prepare_new_task()
        lora_opt = lora.get_optimizer(task_idx, lr=config.lr)
        lora_sched = CosineAnnealingLR(lora_opt, T_max=config.epochs_per_task, eta_min=1e-5)
        for epoch in range(config.epochs_per_task):
            loss, acc = train_baseline_epoch(lora, train_loader, lora_opt, device, task_id=task_idx)
            lora_sched.step()
        logger.info("  [LoRA-Seq] Final -- Loss: %.4f, Acc: %.4f", loss, acc)

        # --- Multi-head evaluation ---
        header = f"  {'Task':<15}" + "".join(f"{n:>12}" for n in MODEL_NAMES)
        logger.info(header)
        for prev_idx, prev_task in enumerate(tasks[: task_idx + 1]):
            row = f"  {prev_task:<15}"
            for model_name in MODEL_NAMES:
                acc = run_evaluation(models[model_name], all_test_loaders[prev_task], device, task_id=prev_idx)
                results[model_name][prev_task].append(acc)
                row += f"{acc * 100:>11.2f}%"
            marker = " <--" if prev_idx == task_idx else ""
            logger.info("%s%s", row, marker)

    return results


def print_benchmark_summary(
    benchmark_name: str, results: dict, tasks: list[str]
) -> dict[str, dict]:
    """Print and return metrics for one benchmark."""
    logger.info("=" * 60)
    logger.info("%s SUMMARY", benchmark_name.upper())
    logger.info("=" * 60)

    header = f"  {'Task':<15}" + "".join(f"{n:>12}" for n in MODEL_NAMES)
    logger.info(header)
    logger.info("  " + "-" * (15 + 12 * len(MODEL_NAMES)))

    final_accs: dict[str, list[float]] = {}
    for model_name in MODEL_NAMES:
        final_accs[model_name] = []
        for task_name in tasks:
            acc = results[model_name][task_name][-1] if results[model_name][task_name] else 0
            final_accs[model_name].append(acc)

    for i, task_name in enumerate(tasks):
        row = f"  {task_name:<15}"
        for model_name in MODEL_NAMES:
            row += f"{final_accs[model_name][i] * 100:>11.2f}%"
        logger.info(row)

    logger.info("  " + "-" * (15 + 12 * len(MODEL_NAMES)))

    metrics: dict[str, dict] = {}
    for model_name in MODEL_NAMES:
        avg_acc = float(np.mean(final_accs[model_name]))
        forgetting_vals = []
        for task_name in tasks[:-1]:
            accs = results[model_name][task_name]
            if len(accs) >= 2:
                forgetting_vals.append(max(accs) - accs[-1])
        avg_f = float(np.mean(forgetting_vals)) if forgetting_vals else 0.0
        metrics[model_name] = {"avg_accuracy": avg_acc, "avg_forgetting": avg_f}
        logger.info("  %-12s Avg: %.2f%%  Forgetting: %.4f", model_name, avg_acc * 100, avg_f)

    return metrics


def visualize_gates_for_benchmark(
    benchmark_name: str, model: RSMNet, test_loaders: dict, tasks: list[str], device: torch.device
) -> None:
    """Generate gate heatmap for a benchmark."""
    model.train(False)
    gate_matrix: list[list[float]] = []

    for task_name in tasks:
        loader = test_loaders[task_name]
        batch_x, _ = next(iter(loader))
        batch_x = batch_x.to(device)

        with torch.no_grad():
            if model.encoder is not None:
                h = model.encoder(batch_x)
            else:
                h = batch_x.view(batch_x.size(0), -1)
            routing_ctx = F.relu(model.layers[0].W_base(h))
            gates = model.layers[0].compute_gates(h, context=routing_ctx)
            if gates.numel() > 0:
                gate_matrix.append(gates.mean(dim=0).cpu().numpy().tolist())
            else:
                gate_matrix.append([])

    if not gate_matrix or not gate_matrix[0]:
        logger.warning("No gate data for %s", benchmark_name)
        return

    matrix = np.array(gate_matrix)
    fig, ax = plt.subplots(figsize=(max(6, matrix.shape[1] * 1.5), max(3, len(tasks) * 0.8)))
    im = ax.imshow(matrix, cmap="Blues", aspect="auto", vmin=0)
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels(tasks)
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels([f"Sub {k}" for k in range(matrix.shape[1])])
    ax.set_xlabel("Submatrix Index")
    ax.set_ylabel("Input Task")
    ax.set_title(f"Gate Activations ({benchmark_name})", fontweight="bold")

    for i in range(len(tasks)):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=10)

    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    safe_name = benchmark_name.lower().replace(" ", "_").replace("-", "_")
    path = RESULTS_DIR / f"gate_heatmap_{safe_name}.png"
    plt.savefig(str(path), dpi=150, bbox_inches="tight")
    logger.info("Gate heatmap saved: %s", path)
    plt.close()


def run_dual_benchmark() -> None:
    """Run both benchmarks and save all results."""
    RESULTS_DIR.mkdir(exist_ok=True)

    # =========================================
    # Benchmark 1: MNIST-family (no conv encoder)
    # =========================================
    logger.info("\n" + "#" * 60)
    logger.info("BENCHMARK 1: MNIST-Family")
    logger.info("#" * 60)

    config_mnist = RSMConfig(
        input_dim=784,
        hidden_dims=(400, 200),
        num_classes=10,
        rank=16,
        key_dim=64,
        max_depth=1,
        epochs_per_task=20,
        batch_size=128,
        lr=0.001,
        seed=42,
        use_conv_encoder=False,
        tasks=("MNIST", "FashionMNIST", "EMNIST"),
    )
    tasks_mnist = list(config_mnist.tasks)

    set_seed(config_mnist.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    models_mnist = {
        "RSM-Net": RSMNet(config=config_mnist).to(device),
        "Naive": NaiveFineTuneNet(hidden_dims=config_mnist.hidden_dims).to(device),
        "EWC": EWCNet(hidden_dims=config_mnist.hidden_dims).to(device),
        "LoRA-Seq": SequentialLoRANet(hidden_dims=config_mnist.hidden_dims, rank=config_mnist.rank).to(device),
    }

    start = time.time()
    results_mnist = run_single_benchmark("MNIST-Family", config_mnist, models_mnist, tasks_mnist, device)
    time_mnist = time.time() - start
    metrics_mnist = print_benchmark_summary("MNIST-Family", results_mnist, tasks_mnist)

    # Gate heatmap for MNIST-family
    _, test_mnist = get_task_dataloaders("MNIST", batch_size=128)
    _, test_fashion = get_task_dataloaders("FashionMNIST", batch_size=128)
    _, test_emnist = get_task_dataloaders("EMNIST", batch_size=128)
    loaders_mnist = {"MNIST": test_mnist, "FashionMNIST": test_fashion, "EMNIST": test_emnist}
    visualize_gates_for_benchmark("MNIST-Family", models_mnist["RSM-Net"], loaders_mnist, tasks_mnist, device)

    # =========================================
    # Benchmark 2: Multi-Domain (with conv encoder)
    # =========================================
    logger.info("\n" + "#" * 60)
    logger.info("BENCHMARK 2: Multi-Domain")
    logger.info("#" * 60)

    feature_dim = 512  # conv encoder output
    config_multi = RSMConfig(
        input_dim=feature_dim,
        hidden_dims=(256, 128),
        num_classes=10,
        rank=16,
        key_dim=64,
        max_depth=1,
        epochs_per_task=20,
        batch_size=128,
        lr=0.001,
        seed=42,
        use_conv_encoder=True,
        encoder_in_channels=3,
        tasks=("MNIST", "CIFAR10", "SVHN"),
    )
    tasks_multi = list(config_multi.tasks)

    set_seed(config_multi.seed)

    # Shared conv encoder for baselines (same architecture, separate instances)
    models_multi = {
        "RSM-Net": RSMNet(config=config_multi).to(device),
        "Naive": NaiveFineTuneNet(
            input_dim=feature_dim, hidden_dims=config_multi.hidden_dims,
            encoder=ConvEncoder(out_features=feature_dim, in_channels=3),
        ).to(device),
        "EWC": EWCNet(
            input_dim=feature_dim, hidden_dims=config_multi.hidden_dims,
            encoder=ConvEncoder(out_features=feature_dim, in_channels=3),
        ).to(device),
        "LoRA-Seq": SequentialLoRANet(
            input_dim=feature_dim, hidden_dims=config_multi.hidden_dims,
            rank=config_multi.rank,
            encoder=ConvEncoder(out_features=feature_dim, in_channels=3),
        ).to(device),
    }

    start = time.time()
    results_multi = run_single_benchmark("Multi-Domain", config_multi, models_multi, tasks_multi, device)
    time_multi = time.time() - start
    metrics_multi = print_benchmark_summary("Multi-Domain", results_multi, tasks_multi)

    # Gate heatmap for Multi-Domain
    _, test_mnist2 = get_task_dataloaders("MNIST", batch_size=128)
    _, test_cifar = get_task_dataloaders("CIFAR10", batch_size=128)
    _, test_svhn = get_task_dataloaders("SVHN", batch_size=128)
    loaders_multi = {"MNIST": test_mnist2, "CIFAR10": test_cifar, "SVHN": test_svhn}
    visualize_gates_for_benchmark("Multi-Domain", models_multi["RSM-Net"], loaders_multi, tasks_multi, device)

    # =========================================
    # Combined results
    # =========================================
    logger.info("\n" + "=" * 70)
    logger.info("COMBINED RESULTS")
    logger.info("=" * 70)
    logger.info("%-15s %-12s %-12s %-12s %-12s", "", *MODEL_NAMES)
    logger.info("-" * 63)
    for bname, metrics in [("MNIST-Family", metrics_mnist), ("Multi-Domain", metrics_multi)]:
        row = f"{bname:<15}"
        for mn in MODEL_NAMES:
            row += f" {metrics[mn]['avg_accuracy']*100:>9.2f}%"
        logger.info(row)
    logger.info("")
    for bname, metrics in [("MNIST-Family", metrics_mnist), ("Multi-Domain", metrics_multi)]:
        row = f"{bname:<15}"
        for mn in MODEL_NAMES:
            row += f" {metrics[mn]['avg_forgetting']:>10.4f}"
        logger.info("Forgetting: %s", row)

    # Save all results
    output = {
        "benchmarks": {
            "MNIST-Family": {
                "tasks": tasks_mnist,
                "results": {mn: dict(tr) for mn, tr in results_mnist.items()},
                "metrics": metrics_mnist,
                "time_seconds": time_mnist,
            },
            "Multi-Domain": {
                "tasks": tasks_multi,
                "results": {mn: dict(tr) for mn, tr in results_multi.items()},
                "metrics": metrics_multi,
                "time_seconds": time_multi,
            },
        },
    }
    path = RESULTS_DIR / "dual_benchmark_results.json"
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Results saved: %s", path)


if __name__ == "__main__":
    print("=" * 60)
    print("RSM-Net -- Dual Benchmark Experiment")
    print("Victor Alejandro Cano Jaramillo -- Abril 2026")
    print("=" * 60)

    run_dual_benchmark()
