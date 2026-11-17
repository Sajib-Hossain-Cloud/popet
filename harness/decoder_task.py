"""Decoder-LM (LoRA) task for the DP-FedAvg auditing harness.

Wraps a frozen pretrained causal-LM backbone (Pythia / Qwen2.5) with LoRA
adapters and exposes ONLY the adapter parameters through ``.parameters()``,
so the entire existing stack operates in adapter space (~0.1-6M dims) with
ZERO changes to fedavg.py / auditor*.py / step-runner audit code:

  - parameters_to_vector(model.parameters())   == flattened LoRA params
  - vector_to_parameters(vec, model.parameters()) resets adapter state
    (the frozen backbone never changes, which is exactly LoRA-FL semantics)
  - model(X) -> next-token logits (n, vocab), matching the LSTM LM protocol
    (X right-padded token contexts; logits gathered at each row's last
    non-pad position; returned fp32)

Data: the SAME StackOverflow user partition and prediction instances as the
LSTM arm — each cached (19-word context -> next word) window is mapped back
to words via the TFF vocab and re-tokenized with the decoder's BPE; the
target is the first BPE piece of the next word (standard reduction). This
makes the decoder arm an exact same-task architecture ablation.

DP note: the guarantee covers the fine-tuning (adapter) updates only; the
pretrained backbone is public. Standard DP fine-tuning setting — stated in
the paper's setup and Ethics.

The backbone+adapter module is cached per process (built once); call
``reset_adapters(model, seed)`` for deterministic per-cell re-initialization.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .stackoverflow_data import (BOS, EOS, OOV, PAD, WORD_OFFSET, build_vocab,
                                 download_word_counts,
                                 load_stackoverflow_clients)
from .tasks import Task

# LoRA injection points per family
_TARGETS = {
    "pythia": ["query_key_value", "dense_h_to_4h", "dense_4h_to_h"],
    "qwen": ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
    # GPT-2 (Conv1D modules; peft handles fan_in_fan_out): fused QKV +
    # both MLP directions -- the same coverage as the pythia target set
    "gpt2": ["c_attn", "c_fc", "mlp.c_proj"],
}

_WRAPPER_CACHE: Dict[tuple, "DecoderLM"] = {}
_TOKENIZER_CACHE: Dict[str, object] = {}


def _family(model_name: str) -> str:
    low = model_name.lower()
    if "qwen" in low:
        return "qwen"
    if "gpt2" in low:
        return "gpt2"
    return "pythia"


def get_tokenizer(model_name: str):
    if model_name not in _TOKENIZER_CACHE:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_name)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        _TOKENIZER_CACHE[model_name] = tok
    return _TOKENIZER_CACHE[model_name]


class DecoderLM(nn.Module):
    """Frozen backbone + LoRA adapters; ``.parameters()`` yields ONLY the
    trainable adapter parameters (order-stable), so the harness's
    parameters_to_vector / vector_to_parameters / SGD all act in adapter
    space. ``.to()``/``_apply`` still move the whole module (backbone
    included) because the peft model is a registered submodule."""

    def __init__(self, peft_model: nn.Module, pad_id: int, amp: bool = True):
        super().__init__()
        self.pm = peft_model
        self.pad_id = pad_id
        self.amp = amp

    def parameters(self, recurse: bool = True):  # noqa: D102 (contract above)
        return (p for p in self.pm.parameters(recurse) if p.requires_grad)

    def _body_and_head(self):
        """(transformer body, lm head) of the wrapped model, per family.
        Slicing the last-position hidden state BEFORE the head keeps the
        logits transient at (bs, vocab) instead of (bs, seq, vocab) --
        ~50x less Metal heap churn, which otherwise faults MPS command
        buffers after a few thousand steps (observed on M1)."""
        base = self.pm.base_model.model  # peft -> LoraModel -> causal LM
        if hasattr(base, "gpt_neox"):    # Pythia / GPT-NeoX
            return base.gpt_neox, base.lm_head
        if hasattr(base, "transformer"):  # GPT-2
            return base.transformer, base.lm_head
        return base.model, base.lm_head  # Qwen / Llama-style

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        mask = (X != self.pad_id).long()
        use_amp = self.amp and X.device.type in ("mps", "cuda")
        body, head = self._body_and_head()
        with torch.autocast(device_type=X.device.type, dtype=torch.float16,
                            enabled=use_amp):
            h = body(input_ids=X, attention_mask=mask).last_hidden_state
            last = mask.sum(dim=1).clamp(min=1) - 1
            h_last = h[torch.arange(X.shape[0], device=X.device), last]
            out = head(h_last)
        return out.float()


def reset_adapters(model: DecoderLM, seed: int) -> None:
    """Deterministic LoRA re-init: A ~ U(-1/sqrt(fan_in), +), B = 0."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.pm.named_parameters():
            if not p.requires_grad:
                continue
            if "lora_A" in name:
                bound = 1.0 / math.sqrt(p.shape[1])
                new = torch.empty(p.shape, dtype=torch.float32)
                new.uniform_(-bound, bound, generator=g)
                p.copy_(new.to(device=p.device, dtype=p.dtype))
            elif "lora_B" in name:
                p.zero_()


def get_decoder_model(model_name: str, adapter_r: int = 8,
                      amp: bool = True) -> DecoderLM:
    """Process-level singleton (backbone load is the expensive part)."""
    key = (model_name, adapter_r)
    if key not in _WRAPPER_CACHE:
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM
        base = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float32)
        base.requires_grad_(False)
        cfg = LoraConfig(r=adapter_r, lora_alpha=2 * adapter_r,
                         lora_dropout=0.0, bias="none",
                         target_modules=_TARGETS[_family(model_name)],
                         task_type="CAUSAL_LM")
        pm = get_peft_model(base, cfg)
        tok = get_tokenizer(model_name)
        _WRAPPER_CACHE[key] = DecoderLM(pm, pad_id=tok.pad_token_id, amp=amp)
    return _WRAPPER_CACHE[key]


# --------------------------------------------------------------------------
# Data: word-id windows -> BPE contexts, cached on disk per (family, config)
# --------------------------------------------------------------------------
def _id_to_word_table(root: str) -> Dict[int, str]:
    word_to_id = build_vocab(download_word_counts(root))
    table = {i: w for w, i in word_to_id.items()}
    table[OOV] = "unk"
    return table  # PAD/BOS/EOS handled by skipping

def _window_to_text(row: torch.Tensor, table: Dict[int, str]) -> str:
    words = [table[t] for t in row.tolist() if t in table or t == OOV]
    return " ".join(w for w in words if w)


def _retokenize_clients(
    clients: List[Tuple[torch.Tensor, torch.Tensor]],
    model_name: str, seq_len: int, root: str,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    tok = get_tokenizer(model_name)
    table = _id_to_word_table(root)
    pad_id = tok.pad_token_id
    ctx = seq_len - 1
    out: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for X, y in clients:
        texts = [_window_to_text(row, table) for row in X]
        tgt_words = [" " + table.get(int(t), "unk") for t in y]
        enc = tok(texts, add_special_tokens=False)["input_ids"]
        tgt = tok(tgt_words, add_special_tokens=False)["input_ids"]
        Xr = torch.full((len(texts), ctx), pad_id, dtype=torch.long)
        yr = torch.zeros(len(texts), dtype=torch.long)
        keep = []
        for i, (ids, t_ids) in enumerate(zip(enc, tgt)):
            if not ids or not t_ids:
                continue
            ids = ids[-ctx:]
            Xr[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            yr[i] = t_ids[0]
            keep.append(i)
        idx = torch.tensor(keep, dtype=torch.long)
        out.append((Xr[idx], yr[idx]))
    return out


def make_text_canaries(
    kind: str,
    n: int,
    model_name: str,
    seq_len: int = 48,
    seed: int = 0,
    root: str = "data/stackoverflow",
    windows_per_canary: int = 64,
    skip_clients: int = 500,
    start: int = 0,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Data-canary CLIENTS for the weak-adversary (A-term) ablation.

    kind='random'  : windows of uniform random BPE tokens (gibberish user)
    kind='natural' : rare-but-plausible SO-style sentences built from the
                     real vocabulary tail (secret-sharer style, unique)
    kind='indist'  : real user windows from clients OUTSIDE the training
                     subsample (in-distribution, unique to the canary)

    Each canary is one client's dataset in the decoder (X, y) format; the
    same constructor with a different seed yields the held-out null set.
    """
    tok = get_tokenizer(model_name)
    g = torch.Generator().manual_seed(seed + 4242)
    ctx = seq_len - 1
    pad_id = tok.pad_token_id

    def windows_from_ids(ids: List[int]):
        X, y = [], []
        for s in range(0, max(1, len(ids) - ctx)):
            w = ids[s:s + ctx + 1]
            if len(w) < 8:
                continue
            row = torch.full((ctx,), pad_id, dtype=torch.long)
            row[:len(w) - 1] = torch.tensor(w[:-1], dtype=torch.long)
            X.append(row)
            y.append(w[-1])
            if len(X) >= windows_per_canary:
                break
        return torch.stack(X), torch.tensor(y, dtype=torch.long)

    out: List[Tuple[torch.Tensor, torch.Tensor]] = []
    if kind == "random":
        vocab = len(tok)
        for _ in range(n):
            ids = torch.randint(0, vocab - 5, (ctx * 3,), generator=g).tolist()
            out.append(windows_from_ids(ids))
        return out

    if kind == "natural":
        counts = download_word_counts(root)
        ranked = [w for w, _ in sorted(counts.items(), key=lambda kv: -kv[1])]
        tail = ranked[7000:10000]  # real but rare SO words
        templates = [
            "how do i {a} the {b} when the {c} keeps failing",
            "my {a} throws a {b} error after the {c} update",
            "is there a way to {a} a {b} inside the {c} loop",
            "the {a} returns an empty {b} unless the {c} is set",
            "why does the {a} ignore my {b} in the {c} config",
        ]
        for i in range(n):
            picks = [tail[int(torch.randint(len(tail), (1,),
                                            generator=g))] for _ in range(9)]
            text = " ".join(
                templates[(i + j) % len(templates)].format(
                    a=picks[3 * j], b=picks[3 * j + 1], c=picks[3 * j + 2])
                for j in range(3))
            ids = tok(text, add_special_tokens=False)["input_ids"]
            out.append(windows_from_ids(ids * 3))
        return out

    if kind == "indist":
        # `start` keeps inserted and held-out draws on DISJOINT real users
        # (caller passes start=0 for inserted, start=N_INS for the null set)
        lo = skip_clients + start
        word_clients, _ = load_stackoverflow_clients(
            num_clients=lo + n + 10,
            max_elements_per_client=windows_per_canary,
            seed=0, root=root, prefer_real=True)
        pool = [(X[:windows_per_canary], y[:windows_per_canary])
                for X, y in word_clients[lo:lo + n]]
        return _retokenize_clients(pool, model_name, seq_len, root)

    raise ValueError(kind)


_LEAF_SPECIALS = {"<BOS>", "<EOS>", "<PAD>"}


def _reddit_clients(
    model_name: str, num_clients: int, max_elements: int, seq_len: int,
    seed: int, root: str = "data/reddit",
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """LEAF Reddit (user-keyed, word tokens as strings) -> decoder windows."""
    import json as _json
    tok = get_tokenizer(model_name)
    pad_id = tok.pad_token_id
    ctx = seq_len - 1
    d = _json.load(open(Path(root) / "new_small_data" / "train_data.json"))
    users = sorted(d["users"])
    g = torch.Generator().manual_seed(seed + 99)
    order = torch.randperm(len(users), generator=g).tolist()

    out: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for ui in order:
        comments = d["user_data"][users[ui]]["x"]
        words = [w for comment in comments for chunk in comment
                 for w in chunk if w not in _LEAF_SPECIALS]
        if len(words) < 12:
            continue
        ids = tok(" ".join(words), add_special_tokens=False)["input_ids"]
        X, y = [], []
        for s in range(0, len(ids) - 8):
            w = ids[s:s + ctx + 1]
            row = torch.full((ctx,), pad_id, dtype=torch.long)
            row[:len(w) - 1] = torch.tensor(w[:-1], dtype=torch.long)
            X.append(row)
            y.append(w[-1])
            if len(X) >= max_elements:
                break
        out.append((torch.stack(X), torch.tensor(y, dtype=torch.long)))
        if len(out) >= num_clients:
            break
    return out


def make_decoder_task(
    model_name: str = "EleutherAI/pythia-70m",
    num_clients: int = 500,
    max_elements_per_client: int = 400,
    seq_len: int = 48,
    adapter_r: int = 8,
    seed: int = 0,
    root: str = "data/stackoverflow",
    cache_dir: str = "data/decoder",
    amp: bool = True,
    source: str = "stackoverflow",
) -> Task:
    """Next-word prediction with a LoRA-adapted decoder on StackOverflow
    (default) or LEAF Reddit (source='reddit')."""
    fam = _family(model_name)
    tag = "so" if source == "stackoverflow" else "reddit"
    cache = Path(cache_dir) / (
        f"{tag}_{fam}_n{num_clients}_m{max_elements_per_client}"
        f"_L{seq_len}_s{seed}.pt")
    if cache.exists():
        blob = torch.load(cache, weights_only=False)
        clients = [(x.long(), y.long()) for x, y in blob["clients"]]
    elif source == "reddit":
        clients = _reddit_clients(model_name, num_clients,
                                  max_elements_per_client, seq_len, seed)
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"clients": clients,
                    "meta": dict(model_family=fam, seq_len=seq_len,
                                 source="LEAF reddit new_small_data")}, cache)
    else:
        word_clients, _meta = load_stackoverflow_clients(
            num_clients=num_clients,
            max_elements_per_client=max_elements_per_client,
            seed=seed, root=root, prefer_real=True)
        # the real-data cache path ignores max_elements_per_client; enforce it
        word_clients = [(X[:max_elements_per_client],
                         y[:max_elements_per_client])
                        for X, y in word_clients]
        clients = _retokenize_clients(word_clients, model_name, seq_len, root)
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"clients": clients,
                    "meta": dict(model_family=fam, seq_len=seq_len,
                                 source="stackoverflow processed.pt")}, cache)

    model = get_decoder_model(model_name, adapter_r=adapter_r, amp=amp)
    reset_adapters(model, seed)
    dim = sum(p.numel() for p in model.parameters())
    # logit width = model config vocab (padded, e.g. Pythia 50304), which can
    # exceed len(tokenizer) (50277); targets always index below the latter.
    vocab_out = int(model.pm.config.vocab_size)

    def model_fn() -> nn.Module:
        return get_decoder_model(model_name, adapter_r=adapter_r, amp=amp)

    return Task(model_fn=model_fn, client_data=clients, dim=dim,
                multiclass=True, task_kind="lm",
                vocab_size=vocab_out, seq_len=seq_len)
