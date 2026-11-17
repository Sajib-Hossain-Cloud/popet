"""StackOverflow-NWP data + LSTM model (Andrew & Kairouz / Reddi et al. baseline).

Vocab: top 10_000 words + 4 special tokens (PAD, BOS, EOS, OOV) = 10_004.
Model: ~4.1M-param word LSTM (embed 96, hidden 670, dense 96).

Data sources (in priority order):
  1. data/stackoverflow/processed.pt  -- from scripts/preprocess_stackoverflow.py
  2. Synthetic clients with real word-frequency vocab (dev / laptop fallback)
"""

from __future__ import annotations

import os
import tarfile
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# TFF defaults (Reddi et al. 2020 / Andrew et al. 2024)
WORD_VOCAB_SIZE = 10_000
NUM_SPECIAL = 4
VOCAB_SIZE = WORD_VOCAB_SIZE + NUM_SPECIAL
SEQ_LEN = 20
CONTEXT_LEN = SEQ_LEN - 1  # 19 tokens in, predict 20th

PAD, BOS, EOS, OOV = 0, 1, 2, 3
WORD_OFFSET = NUM_SPECIAL

WORD_COUNT_URL = (
    "https://storage.googleapis.com/tff-datasets-public/"
    "stackoverflow.word_count.tar.bz2"
)
WORD_COUNT_SHA = (
    "1dc00256d6e527c54b9756d968118378ae14e6692c0b3b6cad470cdd3f3c519c"
)


class StackOverflowLSTM(nn.Module):
    """Word LSTM matching the ~4.1M-parameter TFF StackOverflow baseline."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        embed_dim: int = 96,
        hidden_dim: int = 670,
        dense_dim: int = 96,
    ):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, dense_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dense_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len) token ids
        emb = self.emb(x)
        out, _ = self.lstm(emb)
        h = out[:, -1, :]
        return self.fc2(self.relu(self.fc1(h)))


def model_param_count(m: Optional[nn.Module] = None) -> int:
    m = m or StackOverflowLSTM()
    return sum(p.numel() for p in m.parameters())


def download_word_counts(root: str = "data/stackoverflow") -> Dict[str, int]:
    """Fetch TFF word-count file and return word -> frequency."""
    import ssl
    import certifi

    root_p = Path(root)
    root_p.mkdir(parents=True, exist_ok=True)
    cache = root_p / "stackoverflow.word_count"
    if not cache.exists():
        archive = root_p / "stackoverflow.word_count.tar.bz2"
        if not archive.exists():
            print(f"Downloading word counts -> {archive}")
            ctx = ssl.create_default_context(cafile=certifi.where())
            with urllib.request.urlopen(WORD_COUNT_URL, context=ctx) as resp, \
                    open(archive, "wb") as out:
                out.write(resp.read())
        with tarfile.open(archive, "r:bz2") as tar:
            tar.extractall(path=root_p)
    counts: Dict[str, int] = {}
    with open(cache) as f:
        for line in f:
            word, cnt = line.split()
            counts[word] = int(cnt)
    return counts


def build_vocab(word_counts: Dict[str, int], vocab_size: int = WORD_VOCAB_SIZE):
    """Map top words to indices [WORD_OFFSET, WORD_OFFSET+vocab_size-1]."""
    top = sorted(word_counts.items(), key=lambda kv: -kv[1])[:vocab_size]
    word_to_id = {w: WORD_OFFSET + i for i, (w, _) in enumerate(top)}
    return word_to_id


def tokenize_line(tokens_str: str, word_to_id: Dict[str, int]) -> List[int]:
    ids = []
    for w in tokens_str.split():
        ids.append(word_to_id.get(w, OOV))
    return ids


def sentence_to_windows(token_ids: List[int], seq_len: int = SEQ_LEN):
    """Sliding windows: input = seq_len-1 tokens, target = next token."""
    X, y = [], []
    need = seq_len
    for start in range(0, len(token_ids) - need + 1):
        window = token_ids[start:start + need]
        if len(window) < need:
            continue
        X.append(window[:-1])
        y.append(window[-1])
    return X, y


def _sample_sentence(word_ids: List[int], g: torch.Generator,
                     min_len: int = 25, max_len: int = 45) -> List[int]:
    length = int(torch.randint(min_len, max_len + 1, (1,), generator=g).item())
    idx = torch.randint(len(word_ids), (length,), generator=g).tolist()
    return [word_ids[i] for i in idx]


def make_synthetic_clients(
    num_clients: int,
    word_counts: Dict[str, int],
    min_seqs: int = 40,
    max_seqs: int = 120,
    seed: int = 0,
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], bool]:
    """Per-user synthetic posts using the real SO word distribution."""
    word_to_id = build_vocab(word_counts)
    word_ids = list(word_to_id.values())
    g = torch.Generator().manual_seed(seed)
    clients = []
    for _ in range(num_clients):
        n_seq = int(torch.randint(min_seqs, max_seqs + 1, (1,), generator=g).item())
        xs, ys = [], []
        for _ in range(n_seq):
            sent = _sample_sentence(word_ids, g)
            xw, yw = sentence_to_windows(sent)
            xs.extend(xw)
            ys.extend(yw)
        clients.append((
            torch.tensor(xs, dtype=torch.long),
            torch.tensor(ys, dtype=torch.long),
        ))
    return clients, True


def load_processed_cache(path: str = "data/stackoverflow/processed.pt"):
    if not Path(path).exists():
        return None
    obj = torch.load(path, weights_only=False)
    clients = [(x.long(), y.long()) for x, y in obj["clients"]]
    meta = obj.get("meta", {})
    return clients, meta


def load_stackoverflow_clients(
    num_clients: int = 2000,
    max_elements_per_client: int = 2000,
    seed: int = 0,
    root: str = "data/stackoverflow",
    prefer_real: bool = True,
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], dict]:
    """Load cached real data or fall back to synthetic with real vocab."""
    cache_path = str(Path(root) / "processed.pt")
    if prefer_real:
        cached = load_processed_cache(cache_path)
        if cached is not None:
            clients, meta = cached
            if num_clients < len(clients):
                g = torch.Generator().manual_seed(seed)
                idx = torch.randperm(len(clients), generator=g)[:num_clients].tolist()
                clients = [clients[i] for i in idx]
            meta = {**meta, "source": "processed_cache", "num_clients": len(clients)}
            return clients, meta

    counts = download_word_counts(root)
    clients, synthetic = make_synthetic_clients(num_clients, counts, seed=seed)
    for i, (X, y) in enumerate(clients):
        if X.shape[0] > max_elements_per_client:
            g = torch.Generator().manual_seed(seed + i)
            take = torch.randperm(X.shape[0], generator=g)[:max_elements_per_client]
            clients[i] = (X[take], y[take])
    meta = dict(
        source="synthetic_vocab",
        synthetic=synthetic,
        num_clients=len(clients),
        vocab_size=VOCAB_SIZE,
        seq_len=SEQ_LEN,
        note=("Run scripts/preprocess_stackoverflow.py for real user partition."),
    )
    return clients, meta
