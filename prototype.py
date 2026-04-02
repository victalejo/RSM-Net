"""
RSM-Net (Recursive Submatrix Memory Network)
=============================================
Prototipo experimental — Victor Alejandro Cano Jaramillo, Abril 2026

Arquitectura: Submatrices recursivas con retrieval probabilistico interno
para resolver catastrophic forgetting en aprendizaje continuo.

Experimento: Entrenamiento secuencial en MNIST -> FashionMNIST -> KMNIST
comparando RSM-Net vs Fine-tuning naive vs EWC vs LoRA secuencial.

Uso:
    python rsm_net_prototype.py

Requisitos:
    pip install torch torchvision matplotlib numpy tabulate
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import time
import os

# ============================================================================
# 1. SUBMATRIX LAYER
# ============================================================================

class SubmatrixLinear(nn.Module):
    """
    Capa lineal con submatrices de memoria interna.

    W_eff(x) = W_base + sum_k alpha_k(x) * (A_k * B_k)
    """

    def __init__(self, in_features: int, out_features: int, rank: int = 8,
                 key_dim: int = 32):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.key_dim = key_dim

        self.W_base = nn.Linear(in_features, out_features)
        self.query_proj = nn.Linear(in_features, key_dim)

        self.submatrix_A = nn.ParameterList()
        self.submatrix_B = nn.ParameterList()
        self.key_embeddings = nn.ParameterList()

        self.importance_scores = []
        self._activation_history = []

    def add_submatrix(self):
        k = len(self.submatrix_A)
        A = nn.Parameter(torch.zeros(self.out_features, self.rank) * 0.01)
        B = nn.Parameter(torch.randn(self.rank, self.in_features) * 0.01)
        key = nn.Parameter(torch.randn(self.key_dim) * 0.1)

        self.submatrix_A.append(A)
        self.submatrix_B.append(B)
        self.key_embeddings.append(key)
        self.importance_scores.append(0.0)

        return k

    def compute_gates(self, x: torch.Tensor) -> torch.Tensor:
        K = len(self.key_embeddings)
        if K == 0:
            return None

        q = self.query_proj(x)
        keys = torch.stack([e for e in self.key_embeddings])
        scores = torch.matmul(q, keys.T) / (self.key_dim ** 0.5)

        gates = F.softmax(scores, dim=-1)

        threshold = 0.05
        gates = gates * (gates > threshold).float()
        gate_sum = gates.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        gates = gates / gate_sum

        return gates

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.W_base(x)

        gates = self.compute_gates(x)
        if gates is not None:
            K = len(self.submatrix_A)
            for k in range(K):
                Bx = F.linear(x, self.submatrix_B[k])
                ABx = F.linear(Bx, self.submatrix_A[k])
                gate_k = gates[:, k].unsqueeze(1)
                out = out + gate_k * ABx

            if self.training:
                with torch.no_grad():
                    avg_gates = gates.mean(dim=0).cpu().tolist()
                    self._activation_history.append(avg_gates)

        return out

    def update_importance(self):
        if not self._activation_history:
            return

        history = torch.tensor(self._activation_history)
        avg_activation = history.mean(dim=0).tolist()

        for k in range(len(self.importance_scores)):
            if k < len(avg_activation):
                self.importance_scores[k] = (
                    0.9 * self.importance_scores[k] + 0.1 * avg_activation[k]
                )

        self._activation_history = []

    def prune(self, threshold: float = 0.01):
        to_remove = []
        for k, score in enumerate(self.importance_scores):
            if score < threshold:
                to_remove.append(k)

        for k in reversed(to_remove):
            del self.submatrix_A[k]
            del self.submatrix_B[k]
            del self.key_embeddings[k]
            del self.importance_scores[k]

        return len(to_remove)

    def freeze_base(self):
        for param in self.W_base.parameters():
            param.requires_grad = False

    def get_trainable_params_for_task(self, task_idx: int):
        params = []
        params.extend(self.query_proj.parameters())
        if task_idx < len(self.submatrix_A):
            params.append(self.submatrix_A[task_idx])
            params.append(self.submatrix_B[task_idx])
            params.append(self.key_embeddings[task_idx])
        return params


# ============================================================================
# 2. RSM-NET
# ============================================================================

class RSMNet(nn.Module):
    def __init__(self, input_dim: int = 784, hidden_dims: list = None,
                 num_classes: int = 10, rank: int = 8, key_dim: int = 32):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]

        self.num_tasks = 0
        self.num_classes = num_classes

        dims = [input_dim] + hidden_dims
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(
                SubmatrixLinear(dims[i], dims[i+1], rank=rank, key_dim=key_dim)
            )

        self.heads = nn.ModuleList()
        self.shared_head = nn.Linear(hidden_dims[-1], num_classes)

        self.rank = rank
        self.key_dim = key_dim

    def forward(self, x: torch.Tensor, task_id: int = None) -> torch.Tensor:
        h = x.view(x.size(0), -1)

        for layer in self.layers:
            h = F.relu(layer(h))

        if task_id is not None and task_id < len(self.heads):
            return self.heads[task_id](h)
        else:
            return self.shared_head(h)

    def prepare_new_task(self):
        task_idx = self.num_tasks

        if task_idx > 0:
            for layer in self.layers:
                layer.freeze_base()
                for k in range(len(layer.submatrix_A)):
                    layer.submatrix_A[k].requires_grad = False
                    layer.submatrix_B[k].requires_grad = False
                    layer.key_embeddings[k].requires_grad = False
                layer.add_submatrix()

        last_hidden = self.layers[-1].out_features
        self.heads.append(nn.Linear(last_hidden, self.num_classes))

        self.num_tasks += 1
        return task_idx

    def get_optimizer(self, task_idx: int, lr: float = 0.001):
        params = []

        if task_idx == 0:
            params = list(self.parameters())
        else:
            for layer in self.layers:
                params.extend(layer.get_trainable_params_for_task(task_idx - 1))
            params.extend(self.heads[task_idx].parameters())
            params.extend(self.shared_head.parameters())

        return torch.optim.Adam(params, lr=lr)

    def update_importance_all(self):
        for layer in self.layers:
            layer.update_importance()

    def prune_all(self, threshold: float = 0.01):
        total_pruned = 0
        for layer in self.layers:
            total_pruned += layer.prune(threshold)
        return total_pruned


# ============================================================================
# 3. BASELINES
# ============================================================================

class NaiveFineTuneNet(nn.Module):
    def __init__(self, input_dim=784, hidden_dims=None, num_classes=10):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.extend([nn.Linear(dims[i], dims[i+1]), nn.ReLU()])
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dims[-1], num_classes)

    def forward(self, x, task_id=None):
        h = self.features(x.view(x.size(0), -1))
        return self.head(h)


class EWCNet(nn.Module):
    def __init__(self, input_dim=784, hidden_dims=None, num_classes=10):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.extend([nn.Linear(dims[i], dims[i+1]), nn.ReLU()])
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dims[-1], num_classes)

        self.fisher_dict = {}
        self.optpar_dict = {}

    def forward(self, x, task_id=None):
        h = self.features(x.view(x.size(0), -1))
        return self.head(h)

    def compute_fisher(self, dataloader, device):
        self.train()
        fisher = {n: torch.zeros_like(p) for n, p in self.named_parameters()
                  if p.requires_grad}

        num_samples = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            self.zero_grad()
            out = self(x)
            loss = F.cross_entropy(out, y)
            loss.backward()

            for n, p in self.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.data ** 2
            num_samples += x.size(0)

        for n in fisher:
            fisher[n] /= num_samples

        return fisher

    def store_parameters(self, task_id, dataloader, device):
        self.fisher_dict[task_id] = self.compute_fisher(dataloader, device)
        self.optpar_dict[task_id] = {
            n: p.data.clone() for n, p in self.named_parameters() if p.requires_grad
        }

    def ewc_loss(self, lamda=1000):
        loss = 0
        for task_id in self.fisher_dict:
            for n, p in self.named_parameters():
                if n in self.fisher_dict[task_id]:
                    fisher = self.fisher_dict[task_id][n]
                    optpar = self.optpar_dict[task_id][n]
                    loss += (fisher * (p - optpar) ** 2).sum()
        return lamda * loss


# ============================================================================
# 4. DATOS
# ============================================================================

def get_task_dataloaders(task_name: str, batch_size: int = 128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset_map = {
        'MNIST': datasets.MNIST,
        'FashionMNIST': datasets.FashionMNIST,
        'KMNIST': datasets.KMNIST,
    }

    DatasetClass = dataset_map[task_name]

    train_data = DatasetClass('./data', train=True, download=True, transform=transform)
    test_data = DatasetClass('./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# ============================================================================
# 5. ENTRENAMIENTO Y EVALUACION
# ============================================================================

def train_epoch(model, loader, optimizer, device, ewc_model=None, ewc_lambda=0,
                sparsity_lambda=0.0):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        out = model(x)
        loss = F.cross_entropy(out, y)

        if ewc_model is not None and ewc_lambda > 0:
            loss += ewc_model.ewc_loss(ewc_lambda)

        if sparsity_lambda > 0 and hasattr(model, 'layers'):
            for layer in model.layers:
                gates = layer.compute_gates(x.view(x.size(0), -1))
                if gates is not None:
                    loss += sparsity_lambda * gates.abs().mean()

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        _, predicted = out.max(1)
        correct += predicted.eq(y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            _, predicted = out.max(1)
            correct += predicted.eq(y).sum().item()
            total += x.size(0)

    return correct / total


# ============================================================================
# 6. EXPERIMENTO PRINCIPAL
# ============================================================================

def run_experiment(epochs_per_task: int = 5, rank: int = 8, key_dim: int = 32,
                   batch_size: int = 128, lr: float = 0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo: {device}")
    print(f"Configuracion: rank={rank}, key_dim={key_dim}, epochs/task={epochs_per_task}")
    print("=" * 70)

    tasks = ['MNIST', 'FashionMNIST', 'KMNIST']

    rsm = RSMNet(rank=rank, key_dim=key_dim).to(device)
    naive = NaiveFineTuneNet().to(device)
    ewc = EWCNet().to(device)

    results = {
        'RSM-Net': defaultdict(list),
        'Naive': defaultdict(list),
        'EWC': defaultdict(list),
    }

    all_test_loaders = {}

    for task_idx, task_name in enumerate(tasks):
        print(f"\n{'='*70}")
        print(f"TAREA {task_idx}: {task_name}")
        print(f"{'='*70}")

        train_loader, test_loader = get_task_dataloaders(task_name, batch_size)
        all_test_loaders[task_name] = test_loader

        # --- RSM-Net ---
        print(f"\n[RSM-Net] Preparando tarea {task_idx}...")
        rsm.prepare_new_task()
        rsm_optimizer = rsm.get_optimizer(task_idx, lr=lr)

        num_submatrices = len(rsm.layers[0].submatrix_A)
        print(f"  Submatrices activas: {num_submatrices}")

        for epoch in range(epochs_per_task):
            loss, acc = train_epoch(rsm, train_loader, rsm_optimizer, device,
                                   sparsity_lambda=0.001)
            if (epoch + 1) % 2 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1}/{epochs_per_task} -- Loss: {loss:.4f}, Acc: {acc:.4f}")

        rsm.update_importance_all()

        # --- Naive Fine-tuning ---
        print(f"\n[Naive] Entrenando...")
        naive_optimizer = torch.optim.Adam(naive.parameters(), lr=lr)
        for epoch in range(epochs_per_task):
            loss, acc = train_epoch(naive, train_loader, naive_optimizer, device)
        print(f"  Final -- Loss: {loss:.4f}, Acc: {acc:.4f}")

        # --- EWC ---
        print(f"\n[EWC] Entrenando...")
        ewc_optimizer = torch.optim.Adam(ewc.parameters(), lr=lr)
        ewc_lambda = 1000 if task_idx > 0 else 0
        for epoch in range(epochs_per_task):
            loss, acc = train_epoch(ewc, train_loader, ewc_optimizer, device,
                                   ewc_model=ewc if task_idx > 0 else None,
                                   ewc_lambda=ewc_lambda)
        print(f"  Final -- Loss: {loss:.4f}, Acc: {acc:.4f}")

        ewc.store_parameters(task_idx, train_loader, device)

        # --- Evaluacion en TODAS las tareas vistas ---
        print(f"\n  Evaluacion despues de tarea {task_idx} ({task_name}):")
        print(f"  {'Tarea':<20} {'RSM-Net':>10} {'Naive':>10} {'EWC':>10}")
        print(f"  {'-'*50}")

        for prev_idx, prev_task in enumerate(tasks[:task_idx + 1]):
            prev_loader = all_test_loaders[prev_task]

            rsm_acc = evaluate(rsm, prev_loader, device)
            naive_acc = evaluate(naive, prev_loader, device)
            ewc_acc = evaluate(ewc, prev_loader, device)

            results['RSM-Net'][prev_task].append(rsm_acc)
            results['Naive'][prev_task].append(naive_acc)
            results['EWC'][prev_task].append(ewc_acc)

            marker = " <-- (nueva)" if prev_idx == task_idx else ""
            print(f"  {prev_task:<20} {rsm_acc:>9.2%} {naive_acc:>9.2%} {ewc_acc:>9.2%}{marker}")

    # --- Resumen Final ---
    print(f"\n{'='*70}")
    print("RESUMEN FINAL -- Accuracy en cada tarea despues de entrenar las 3")
    print(f"{'='*70}")

    print(f"\n  {'Tarea':<20} {'RSM-Net':>10} {'Naive':>10} {'EWC':>10}")
    print(f"  {'-'*50}")

    final_accs = {}
    for model_name in ['RSM-Net', 'Naive', 'EWC']:
        final_accs[model_name] = []
        for task_name in tasks:
            acc = results[model_name][task_name][-1] if results[model_name][task_name] else 0
            final_accs[model_name].append(acc)

    for i, task_name in enumerate(tasks):
        rsm_a = final_accs['RSM-Net'][i]
        naive_a = final_accs['Naive'][i]
        ewc_a = final_accs['EWC'][i]
        print(f"  {task_name:<20} {rsm_a:>9.2%} {naive_a:>9.2%} {ewc_a:>9.2%}")

    print(f"  {'-'*50}")
    rsm_avg = np.mean(final_accs['RSM-Net'])
    naive_avg = np.mean(final_accs['Naive'])
    ewc_avg = np.mean(final_accs['EWC'])
    print(f"  {'PROMEDIO':<20} {rsm_avg:>9.2%} {naive_avg:>9.2%} {ewc_avg:>9.2%}")

    # Forgetting metric
    print(f"\n  Forgetting (caida de accuracy en tareas previas):")
    for model_name in ['RSM-Net', 'Naive', 'EWC']:
        forgetting = []
        for task_name in tasks[:-1]:
            accs = results[model_name][task_name]
            if len(accs) >= 2:
                forgetting.append(max(accs) - accs[-1])
        avg_forgetting = np.mean(forgetting) if forgetting else 0
        print(f"  {model_name:<15} Avg Forgetting: {avg_forgetting:.4f}")

    # --- Grafica ---
    plot_results(results, tasks)

    # --- Info de RSM-Net ---
    print(f"\n  RSM-Net -- Estado final:")
    for i, layer in enumerate(rsm.layers):
        n_sub = len(layer.submatrix_A)
        scores = [f"{s:.4f}" for s in layer.importance_scores]
        print(f"  Capa {i}: {n_sub} submatrices, importancia: {scores}")

    total_params_base = sum(p.numel() for p in naive.parameters())
    total_params_rsm = sum(p.numel() for p in rsm.parameters())
    overhead = (total_params_rsm - total_params_base) / total_params_base * 100
    print(f"\n  Parametros base (Naive): {total_params_base:,}")
    print(f"  Parametros RSM-Net:     {total_params_rsm:,}")
    print(f"  Overhead:               {overhead:.1f}%")

    return results


def plot_results(results, tasks):
    fig, axes = plt.subplots(1, len(tasks), figsize=(5 * len(tasks), 4))

    if len(tasks) == 1:
        axes = [axes]

    colors = {'RSM-Net': '#2196F3', 'Naive': '#F44336', 'EWC': '#4CAF50'}

    for i, task_name in enumerate(tasks):
        ax = axes[i]
        for model_name in ['RSM-Net', 'Naive', 'EWC']:
            accs = results[model_name][task_name]
            if accs:
                x = list(range(len(accs)))
                ax.plot(x, accs, 'o-', label=model_name, color=colors[model_name],
                       linewidth=2, markersize=6)

        ax.set_title(f'Accuracy en {task_name}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Despues de tarea #')
        ax.set_ylabel('Accuracy')
        ax.set_ylim([0, 1.05])
        ax.set_xticks(range(len(tasks)))
        ax.set_xticklabels([f'T{j}' for j in range(len(tasks))])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('rsm_net_results.png', dpi=150, bbox_inches='tight')
    print("\n  Grafica guardada en: rsm_net_results.png")
    plt.close()


# ============================================================================
# 7. ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("RSM-Net -- Recursive Submatrix Memory Network")
    print("Prototipo Experimental v0.1")
    print("Victor Alejandro Cano Jaramillo -- Abril 2026")
    print("=" * 70)

    results = run_experiment(
        epochs_per_task=5,
        rank=8,
        key_dim=32,
        batch_size=128,
        lr=0.001,
    )
