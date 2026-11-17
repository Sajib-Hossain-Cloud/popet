"""Tasks (data + model) for the DP-FedAvg auditing harness.

Step 1 ships a fast synthetic logistic-regression task so the whole pipeline
(sim -> accountant -> auditor) can be validated in seconds on CPU/MPS before we
scale to Federated EMNIST / StackOverflow-NWP. The model is deliberately
moderately high-dimensional (default d=1000) because the one-shot canary auditor
(Andrew & Kairouz, ICLR'24) relies on near-orthogonality of isotropic vectors in
high dimensions -- an effect we want to be able to exhibit here.
"""

from dataclasses import dataclass
from typing import Callable, List, Tuple

import torch
import torch.nn as nn


@dataclass
class Task:
    """A federated task: how to build the model + one dataset shard per client."""
    model_fn: Callable[[], nn.Module]
    client_data: List[Tuple[torch.Tensor, torch.Tensor]]  # (X, y) per client
    dim: int  # flattened parameter dimension
    multiclass: bool = False  # False -> BCE on logits; True -> CrossEntropy
    task_kind: str = "tabular"  # "tabular" | "lm"
    vocab_size: int = 0
    seq_len: int = 0

    @property
    def num_clients(self) -> int:
        return len(self.client_data)


def make_synthetic_task(
    num_clients: int = 200,
    dim: int = 1000,
    samples_per_client: int = 64,
    non_iid: float = 0.0,
    seed: int = 0,
) -> Task:
    """Logistic regression on Gaussian features.

    A shared ground-truth weight `w_true` defines labels; `non_iid` adds a
    per-client shift to `w_true` so we can dial heterogeneity later (0.0 = IID).
    """
    g = torch.Generator().manual_seed(seed)
    w_true = torch.randn(dim, generator=g)

    clients: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(num_clients):
        shift = non_iid * torch.randn(dim, generator=g)
        w_c = w_true + shift
        X = torch.randn(samples_per_client, dim, generator=g)
        logits = X @ w_c + 0.1 * torch.randn(samples_per_client, generator=g)
        y = (logits > 0).float()
        clients.append((X, y))

    def model_fn() -> nn.Module:
        # No bias so the flattened param vector is exactly `dim` -- keeps the
        # canary-geometry reasoning clean.
        return nn.Linear(dim, 1, bias=False)

    return Task(model_fn=model_fn, client_data=clients, dim=dim)


def make_mnist_binary_task(
    num_clients: int = 200,
    samples_per_client: int = 300,
    hidden: int = 16,
    seed: int = 0,
    root: str = "data",
) -> Task:
    """REAL DATA: MNIST odd-vs-even with a small MLP (784-hidden-1).

    Binary labels keep the harness's BCE loss/accuracy unchanged; the MLP has
    ~12.6k parameters, high-dimensional enough for the one-shot auditor's
    near-orthogonality requirement (dim >~ 10k). IID client partition (the
    natural-writer partition of Federated EMNIST needs TFF; future work).
    """
    from torchvision import datasets  # lazy import; only step7 needs it

    ds = datasets.MNIST(root=root, train=True, download=True)
    X_all = ds.data.reshape(-1, 784).float() / 255.0
    y_all = (ds.targets % 2 == 1).float()  # odd = 1, even = 0

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(X_all), generator=g)
    clients: List[Tuple[torch.Tensor, torch.Tensor]] = []
    idx = 0
    for _ in range(num_clients):
        take = perm[idx: idx + samples_per_client]
        idx += samples_per_client
        clients.append((X_all[take].clone(), y_all[take].clone()))

    def model_fn() -> nn.Module:
        return nn.Sequential(nn.Linear(784, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    dim = sum(p.numel() for p in model_fn().parameters())
    return Task(model_fn=model_fn, client_data=clients, dim=dim)


def _dirichlet_partition(
    y: torch.Tensor,
    num_clients: int,
    samples_per_client: int,
    alpha: float,
    g: torch.Generator,
) -> List[torch.Tensor]:
    """Equal-size clients with Dirichlet(alpha) class proportions (standard
    non-IID FL benchmark protocol). alpha <= 0 -> IID random partition."""
    n_total = num_clients * samples_per_client
    if alpha <= 0:
        perm = torch.randperm(len(y), generator=g)[:n_total]
        return [perm[i * samples_per_client:(i + 1) * samples_per_client]
                for i in range(num_clients)]

    classes = int(y.max().item()) + 1
    pools = [torch.nonzero(y == c).flatten() for c in range(classes)]
    pools = [p[torch.randperm(len(p), generator=g)].tolist() for p in pools]
    leftovers: List[int] = []

    out = []
    for _ in range(num_clients):
        p = torch.distributions.Dirichlet(
            torch.full((classes,), alpha)).sample()
        counts = torch.multinomial(p, samples_per_client, replacement=True)
        counts = torch.bincount(counts, minlength=classes)
        idxs: List[int] = []
        for c in range(classes):
            take = min(int(counts[c]), len(pools[c]))
            idxs.extend(pools[c][:take])
            pools[c] = pools[c][take:]
        while len(idxs) < samples_per_client:  # pool exhausted -> fill from any
            if not leftovers:
                leftovers = [i for pool in pools for i in pool]
                for c in range(classes):
                    pools[c] = []
                if not leftovers:
                    break
            idxs.append(leftovers.pop())
        out.append(torch.tensor(idxs[:samples_per_client]))
    return out


def make_image_benchmark_task(
    dataset: str = "emnist",
    num_clients: int = 200,
    samples_per_client: int = 150,
    non_iid_alpha: float = 0.3,
    seed: int = 0,
    root: str = "data",
) -> Task:
    """REAL BENCHMARK: EMNIST-47 (balanced) or FashionMNIST-10, small CNN.

    Multi-class, Dirichlet non-IID client partition (the standard approximation
    of natural heterogeneity when writer IDs are unavailable outside TFF).
    CNN has ~38k params (EMNIST) -- comfortably high-dimensional for the
    one-shot auditor.
    """
    from torchvision import datasets

    if dataset == "emnist":
        ds = datasets.EMNIST(root=root, split="balanced", train=True, download=True)
        num_classes = 47
    elif dataset == "fashion":
        ds = datasets.FashionMNIST(root=root, train=True, download=True)
        num_classes = 10
    else:
        raise ValueError(dataset)

    X_all = ds.data.reshape(-1, 1, 28, 28).float() / 255.0
    y_all = ds.targets.long()

    g = torch.Generator().manual_seed(seed)
    parts = _dirichlet_partition(y_all, num_clients, samples_per_client,
                                 non_iid_alpha, g)
    clients = [(X_all[idx].clone(), y_all[idx].clone().float()) for idx in parts]

    def model_fn() -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1), nn.ReLU(),    # 28 -> 14
            nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.ReLU(),   # 14 -> 7
            nn.Flatten(),
            nn.Linear(16 * 7 * 7, num_classes),
        )

    dim = sum(p.numel() for p in model_fn().parameters())
    return Task(model_fn=model_fn, client_data=clients, dim=dim, multiclass=True)


def make_stackoverflow_task(
    num_clients: int = 2000,
    max_elements_per_client: int = 2000,
    seed: int = 0,
    root: str = "data/stackoverflow",
    prefer_real: bool = True,
) -> Task:
    """StackOverflow next-word prediction (Andrew & Kairouz benchmark).

    Uses cached real user partition if data/stackoverflow/processed.pt exists;
    otherwise synthetic clients with the real TFF word-frequency vocabulary.
    LSTM has ~4.1M parameters for high-dimensional canary auditing.
    """
    from .stackoverflow_data import (
        CONTEXT_LEN,
        VOCAB_SIZE,
        StackOverflowLSTM,
        load_stackoverflow_clients,
        model_param_count,
    )

    clients, meta = load_stackoverflow_clients(
        num_clients=num_clients,
        max_elements_per_client=max_elements_per_client,
        seed=seed,
        root=root,
        prefer_real=prefer_real,
    )

    def model_fn() -> nn.Module:
        return StackOverflowLSTM(vocab_size=VOCAB_SIZE)

    dim = model_param_count()
    return Task(
        model_fn=model_fn,
        client_data=clients,
        dim=dim,
        multiclass=True,
        task_kind="lm",
        vocab_size=VOCAB_SIZE,
        seq_len=CONTEXT_LEN,
    )
