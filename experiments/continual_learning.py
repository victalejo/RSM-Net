"""
Continual learning experiment: MNIST -> FashionMNIST -> EMNIST-Letters

Compares RSM-Net vs Naive Fine-tuning vs EWC vs Sequential LoRA.
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

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rsm_net.baselines import EWCNet, NaiveFineTuneNet, SequentialLoRANet
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

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
MODEL_NAMES = ["RSM-Net", "Naive", "EWC", "LoRA-Seq"]
COLORS = {
    "RSM-Net": "#2196F3",
    "Naive": "#F44336",
    "EWC": "#4CAF50",
    "LoRA-Seq": "#FF9800",
}


def run_experiment(config: RSMConfig | None = None) -> dict:
    """Run the full continual learning experiment with all 4 models."""
    if config is None:
        config = RSMConfig()

    set_seed(config.seed)
    RESULTS_DIR.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    logger.info(
        "Config: hidden=%s, rank=%d, key_dim=%d, depth=%d, epochs/task=%d, lr=%s",
        config.hidden_dims, config.rank, config.key_dim,
        config.max_depth, config.epochs_per_task, config.lr,
    )

    tasks = list(config.tasks)

    # Initialize all 4 models
    rsm = RSMNet(config=config).to(device)
    naive = NaiveFineTuneNet(
        hidden_dims=config.hidden_dims, num_classes=config.num_classes
    ).to(device)
    ewc = EWCNet(
        hidden_dims=config.hidden_dims, num_classes=config.num_classes
    ).to(device)
    lora = SequentialLoRANet(
        hidden_dims=config.hidden_dims, num_classes=config.num_classes,
        rank=config.rank,
    ).to(device)

    results: dict[str, dict[str, list[float]]] = {
        name: defaultdict(list) for name in MODEL_NAMES
    }

    all_test_loaders: dict[str, torch.utils.data.DataLoader] = {}
    all_train_loaders: dict[str, torch.utils.data.DataLoader] = {}
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
        all_train_loaders[task_name] = train_loader

        # --- RSM-Net with cosine LR ---
        logger.info("[RSM-Net] Preparing task %d...", task_idx)
        rsm.prepare_new_task()
        rsm_opt = rsm.get_optimizer(task_idx, lr=config.lr)
        rsm_sched = CosineAnnealingLR(rsm_opt, T_max=config.epochs_per_task, eta_min=1e-5)

        for layer in rsm.layers:
            logger.info("  Submatrices: %d", layer.num_submatrices)

        for epoch in range(config.epochs_per_task):
            loss, acc = train_rsm_epoch(
                rsm, train_loader, rsm_opt, device, task_idx, config
            )
            rsm_sched.step()
            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(
                    "  [RSM] Epoch %d/%d -- Loss: %.4f, Acc: %.4f",
                    epoch + 1, config.epochs_per_task, loss, acc,
                )

        rsm.update_importance_all()

        if config.consolidate_at_task_boundary and task_idx > 0:
            merges = rsm.consolidate_all()
            if merges > 0:
                rsm_opt = rsm.get_optimizer(task_idx, lr=config.lr)

        # --- Naive with cosine LR + per-task head ---
        logger.info("[Naive] Training...")
        naive.add_task_head()
        naive_opt = torch.optim.Adam(naive.parameters(), lr=config.lr)
        naive_sched = CosineAnnealingLR(naive_opt, T_max=config.epochs_per_task, eta_min=1e-5)
        for epoch in range(config.epochs_per_task):
            loss, acc = train_baseline_epoch(
                naive, train_loader, naive_opt, device, task_id=task_idx,
            )
            naive_sched.step()
        logger.info("  [Naive] Final -- Loss: %.4f, Acc: %.4f", loss, acc)

        # --- EWC with cosine LR + per-task head ---
        logger.info("[EWC] Training...")
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

        # --- Sequential LoRA with cosine LR ---
        logger.info("[LoRA-Seq] Training...")
        lora.prepare_new_task()
        lora_opt = lora.get_optimizer(task_idx, lr=config.lr)
        lora_sched = CosineAnnealingLR(lora_opt, T_max=config.epochs_per_task, eta_min=1e-5)
        for epoch in range(config.epochs_per_task):
            loss, acc = train_baseline_epoch(
                lora, train_loader, lora_opt, device, task_id=task_idx,
            )
            lora_sched.step()
        logger.info("  [LoRA-Seq] Final -- Loss: %.4f, Acc: %.4f", loss, acc)

        # --- Evaluate all models on all tasks seen ---
        models = {"RSM-Net": rsm, "Naive": naive, "EWC": ewc, "LoRA-Seq": lora}
        header = f"  {'Task':<15}" + "".join(f"{n:>12}" for n in MODEL_NAMES)
        logger.info("Evaluation after task %d (%s):", task_idx, task_name)
        logger.info(header)
        logger.info("  " + "-" * (15 + 12 * len(MODEL_NAMES)))

        for prev_idx, prev_task in enumerate(tasks[: task_idx + 1]):
            prev_loader = all_test_loaders[prev_task]
            row = f"  {prev_task:<15}"
            for model_name in MODEL_NAMES:
                # Multi-head evaluation: use per-task head for the evaluated task
                acc = run_evaluation(
                    models[model_name], prev_loader, device, task_id=prev_idx,
                )
                results[model_name][prev_task].append(acc)
                row += f"{acc * 100:>11.2f}%"
            marker = " <--" if prev_idx == task_idx else ""
            logger.info("%s%s", row, marker)

    elapsed = time.time() - start_time
    logger.info("Total experiment time: %.1f seconds", elapsed)

    # --- Summaries and plots ---
    print_summary(results, tasks, rsm, naive, lora)
    plot_forgetting_curves(results, tasks)
    visualize_gates(rsm, all_test_loaders, tasks, device)
    measure_inference_time(rsm, naive, ewc, lora, device)
    save_results(results, tasks, config, rsm, naive, lora)

    return dict(results)


def print_summary(
    results: dict, tasks: list[str], rsm: RSMNet,
    naive: NaiveFineTuneNet, lora: SequentialLoRANet,
) -> None:
    """Print final summary with all metrics."""
    T = len(tasks)
    logger.info("=" * 70)
    logger.info("FINAL SUMMARY -- After training all %d tasks", T)
    logger.info("=" * 70)

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

    # Average Accuracy
    row = f"  {'Avg Acc':<15}"
    for model_name in MODEL_NAMES:
        avg = np.mean(final_accs[model_name])
        row += f"{avg * 100:>11.2f}%"
    logger.info(row)

    # Average Forgetting
    logger.info("\nForgetting (max_acc - final_acc on previous tasks):")
    for model_name in MODEL_NAMES:
        forgetting_vals = []
        for task_name in tasks[:-1]:
            accs = results[model_name][task_name]
            if len(accs) >= 2:
                forgetting_vals.append(max(accs) - accs[-1])
        avg_f = np.mean(forgetting_vals) if forgetting_vals else 0
        logger.info("  %-12s Avg Forgetting: %.4f", model_name, avg_f)

    # Backward Transfer
    logger.info("\nBackward Transfer BWT = (1/T-1) * sum(R_T,i - R_i,i):")
    for model_name in MODEL_NAMES:
        bwt_vals = []
        for i, task_name in enumerate(tasks[:-1]):
            accs = results[model_name][task_name]
            if len(accs) >= 2:
                R_Ti = accs[-1]   # final accuracy on task i
                R_ii = accs[0]    # accuracy right after training task i
                bwt_vals.append(R_Ti - R_ii)
        bwt = np.mean(bwt_vals) if bwt_vals else 0
        logger.info("  %-12s BWT: %.4f", model_name, bwt)

    # Parameter overhead
    logger.info("\nParameter count:")
    base_params = sum(p.numel() for p in naive.parameters())
    rsm_params = sum(p.numel() for p in rsm.parameters())
    lora_params = sum(p.numel() for p in lora.parameters())
    logger.info("  Naive:    %s", f"{base_params:,}")
    logger.info("  RSM-Net:  %s (+%.1f%%)", f"{rsm_params:,}",
                (rsm_params - base_params) / base_params * 100)
    logger.info("  LoRA-Seq: %s (+%.1f%%)", f"{lora_params:,}",
                (lora_params - base_params) / base_params * 100)

    # RSM-Net state
    logger.info("\nRSM-Net submatrices:")
    for layer_info in rsm.get_state_summary()["layers"]:
        scores = [f"{s:.4f}" for s in layer_info["importance_scores"]]
        logger.info("  Layer %d: %d active, importance: %s",
                    layer_info["layer"], layer_info["num_submatrices"], scores)


def plot_forgetting_curves(results: dict, tasks: list[str]) -> None:
    """Plot accuracy on each task over time."""
    fig, axes = plt.subplots(1, len(tasks), figsize=(5 * len(tasks), 4))
    if len(tasks) == 1:
        axes = [axes]

    for i, task_name in enumerate(tasks):
        ax = axes[i]
        for model_name in MODEL_NAMES:
            accs = results[model_name][task_name]
            if accs:
                x = list(range(len(accs)))
                ax.plot(x, accs, "o-", label=model_name, color=COLORS[model_name],
                        linewidth=2, markersize=6)

        ax.set_title(f"Accuracy on {task_name}", fontsize=12, fontweight="bold")
        ax.set_xlabel("After task #")
        ax.set_ylabel("Accuracy")
        ax.set_ylim([0, 1.05])
        ax.set_xticks(range(len(tasks)))
        ax.set_xticklabels([f"T{j}" for j in range(len(tasks))])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = RESULTS_DIR / "forgetting_curves.png"
    plt.savefig(str(path), dpi=150, bbox_inches="tight")
    logger.info("Forgetting curves saved: %s", path)
    plt.close()


def visualize_gates(
    model: RSMNet,
    test_loaders: dict[str, torch.utils.data.DataLoader],
    tasks: list[str],
    device: torch.device,
) -> None:
    """Create gate activation heatmap -- the key figure for the paper."""
    model.train(False)
    gate_matrix: list[list[float]] = []

    for task_name in tasks:
        loader = test_loaders[task_name]
        batch_x, _ = next(iter(loader))
        batch_x = batch_x.to(device)

        with torch.no_grad():
            h = batch_x.view(batch_x.size(0), -1)
            # Use routing context (from W_base) same as forward pass
            routing_ctx = F.relu(model.layers[0].W_base(h))
            gates = model.layers[0].compute_gates(h, context=routing_ctx)
            if gates.numel() > 0:
                avg_gates = gates.mean(dim=0).cpu().numpy().tolist()
            else:
                avg_gates = []
            gate_matrix.append(avg_gates)

    if not gate_matrix or not gate_matrix[0]:
        logger.warning("No gate data to visualize")
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
    ax.set_title("Gate Activations by Task (Layer 0)", fontweight="bold")

    # Annotate cells
    for i in range(len(tasks)):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=10)

    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    path = RESULTS_DIR / "gate_heatmap.png"
    plt.savefig(str(path), dpi=150, bbox_inches="tight")
    logger.info("Gate heatmap saved: %s", path)
    plt.close()

    # Also save for layer 1
    gate_matrix_l1: list[list[float]] = []
    for task_name in tasks:
        loader = test_loaders[task_name]
        batch_x, _ = next(iter(loader))
        batch_x = batch_x.to(device)
        with torch.no_grad():
            h = batch_x.view(batch_x.size(0), -1)
            routing_ctx = F.relu(model.layers[0].W_base(h))
            h = F.relu(model.layers[0](h, context=routing_ctx))
            gates = model.layers[1].compute_gates(h, context=routing_ctx)
            if gates.numel() > 0:
                gate_matrix_l1.append(gates.mean(dim=0).cpu().numpy().tolist())

    if gate_matrix_l1 and gate_matrix_l1[0]:
        matrix_l1 = np.array(gate_matrix_l1)
        fig, ax = plt.subplots(figsize=(max(6, matrix_l1.shape[1] * 1.5), max(3, len(tasks) * 0.8)))
        im = ax.imshow(matrix_l1, cmap="Oranges", aspect="auto", vmin=0)
        ax.set_yticks(range(len(tasks)))
        ax.set_yticklabels(tasks)
        ax.set_xticks(range(matrix_l1.shape[1]))
        ax.set_xticklabels([f"Sub {k}" for k in range(matrix_l1.shape[1])])
        ax.set_xlabel("Submatrix Index")
        ax.set_ylabel("Input Task")
        ax.set_title("Gate Activations by Task (Layer 1)", fontweight="bold")
        for i in range(len(tasks)):
            for j in range(matrix_l1.shape[1]):
                val = matrix_l1[i, j]
                color = "white" if val > 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.8)
        plt.tight_layout()
        path_l1 = RESULTS_DIR / "gate_heatmap_layer1.png"
        plt.savefig(str(path_l1), dpi=150, bbox_inches="tight")
        plt.close()


def measure_inference_time(
    rsm: RSMNet, naive: NaiveFineTuneNet,
    ewc: EWCNet, lora: SequentialLoRANet,
    device: torch.device,
) -> None:
    """Measure inference time per batch for each model."""
    x = torch.randn(128, 784, device=device)
    models = {"RSM-Net": rsm, "Naive": naive, "EWC": ewc, "LoRA-Seq": lora}

    logger.info("\nInference time (128-sample batch, 100 runs):")
    for name, model in models.items():
        model.train(False)
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                model(x)
        # Measure
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(100):
                model(x)
        elapsed_ms = (time.perf_counter() - start) / 100 * 1000
        logger.info("  %-12s %.2f ms/batch", name, elapsed_ms)


def save_results(
    results: dict, tasks: list[str], config: RSMConfig,
    rsm: RSMNet, naive: NaiveFineTuneNet, lora: SequentialLoRANet,
) -> None:
    """Save full results as JSON."""
    # Compute all metrics
    final_accs: dict[str, list[float]] = {}
    forgetting: dict[str, float] = {}
    bwt: dict[str, float] = {}

    for model_name in MODEL_NAMES:
        final_accs[model_name] = []
        for task_name in tasks:
            acc = results[model_name][task_name][-1] if results[model_name][task_name] else 0
            final_accs[model_name].append(acc)

        f_vals = []
        b_vals = []
        for i, task_name in enumerate(tasks[:-1]):
            accs = results[model_name][task_name]
            if len(accs) >= 2:
                f_vals.append(max(accs) - accs[-1])
                b_vals.append(accs[-1] - accs[0])
        forgetting[model_name] = float(np.mean(f_vals)) if f_vals else 0.0
        bwt[model_name] = float(np.mean(b_vals)) if b_vals else 0.0

    base_params = sum(p.numel() for p in naive.parameters())

    output = {
        "config": {
            "hidden_dims": list(config.hidden_dims),
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
        "metrics": {
            model_name: {
                "avg_accuracy": float(np.mean(final_accs[model_name])),
                "avg_forgetting": forgetting[model_name],
                "backward_transfer": bwt[model_name],
            }
            for model_name in MODEL_NAMES
        },
        "params": {
            "Naive": base_params,
            "RSM-Net": sum(p.numel() for p in rsm.parameters()),
            "LoRA-Seq": sum(p.numel() for p in lora.parameters()),
        },
        "rsm_state": rsm.get_state_summary(),
    }

    path = RESULTS_DIR / "experiment_results.json"
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Results saved: %s", path)


if __name__ == "__main__":
    config = RSMConfig(
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
        sparsity_lambda=0.001,
        frobenius_lambda=0.0001,
        tasks=("MNIST", "FashionMNIST", "EMNIST"),
    )

    print("=" * 60)
    print("RSM-Net -- Recursive Submatrix Memory Network")
    print("Extended Experiment v0.2")
    print("Victor Alejandro Cano Jaramillo -- Abril 2026")
    print("=" * 60)

    run_experiment(config)
