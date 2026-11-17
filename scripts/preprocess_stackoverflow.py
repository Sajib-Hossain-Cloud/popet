#!/usr/bin/env python3
"""Export StackOverflow-NWP clients from the TFF sqlite dump to PyTorch cache.

The full sqlite.lzma is ~8 GB. Download once:

  python scripts/preprocess_stackoverflow.py --download

Then export (or re-export) a laptop-friendly subset:

  python scripts/preprocess_stackoverflow.py --max-clients 5000

Output: data/stackoverflow/processed.pt
"""

from __future__ import annotations

import argparse
import lzma
import ssl
import sqlite3
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

import certifi
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness.stackoverflow_data import (  # noqa: E402
    SEQ_LEN,
    VOCAB_SIZE,
    build_vocab,
    download_word_counts,
    sentence_to_windows,
    tokenize_line,
)

SQLITE_URL = (
    "https://storage.googleapis.com/tff-datasets-public/stackoverflow.sqlite.lzma"
)
DATA_DIR = ROOT / "data" / "stackoverflow"


def _ssl_ctx():
    return ssl.create_default_context(cafile=certifi.where())


def _download(url: str, dest: Path, chunk_mb: int = 8):
    dest.parent.mkdir(parents=True, exist_ok=True)
    done = dest.stat().st_size if dest.exists() else 0
    headers = {}
    if done > 0:
        print(f"Resuming download at {done / 1e9:.2f} GB")
        headers["Range"] = f"bytes={done}-"
    print(f"Downloading {url}")
    print(f"  -> {dest}")
    req = urllib.request.Request(url, headers=headers)
    mode = "ab" if done > 0 else "wb"
    with urllib.request.urlopen(req, context=_ssl_ctx()) as resp, open(dest, mode) as out:
        total_hdr = resp.headers.get("Content-Length")
        total = int(total_hdr) + done if total_hdr else 0
        chunk = chunk_mb * (1 << 20)
        while True:
            buf = resp.read(chunk)
            if not buf:
                break
            out.write(buf)
            done += len(buf)
            if total:
                print(f"\r  {done / 1e9:.2f} / {total / 1e9:.2f} GB "
                      f"({100 * done / total:.1f}%)", end="", flush=True)
        print()


def download_sqlite(dest: Path) -> Path:
    lz_path = dest / "stackoverflow.sqlite.lzma"
    db_path = dest / "stackoverflow.sqlite"
    if db_path.exists() and db_path.stat().st_size > 1e9:
        print(f"sqlite already at {db_path}")
        return db_path
    if not lz_path.exists() or lz_path.stat().st_size < 8e9:
        _download(SQLITE_URL, lz_path)
    print(f"Decompressing -> {db_path}")
    with lzma.open(lz_path, "rb") as fin, open(db_path, "wb") as fout:
        while True:
            block = fin.read(1 << 20)
            if not block:
                break
            fout.write(block)
    print(f"decompressed size: {db_path.stat().st_size / 1e9:.2f} GB")
    return db_path


def parse_tokens_from_proto(blob: bytes) -> str:
    """Parse TFF serialized_example_proto -> space-separated tokens string."""
    from tensorflow.core.example.example_pb2 import Example

    ex = Example()
    ex.ParseFromString(blob)
    feat = ex.features.feature.get("tokens")
    if feat is None:
        return ""
    if feat.bytes_list.value:
        return feat.bytes_list.value[0].decode("utf-8", errors="replace")
    if feat.bytes_list.value:
        return str(feat.bytes_list.value[0])
    return ""


def export_from_sqlite(db_path: Path, max_clients: int, max_per_client: int):
    """Read TFF stackoverflow sqlite into per-user tensors."""
    counts = download_word_counts(str(DATA_DIR))
    word_to_id = build_vocab(counts)

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    tables = [r[0] for r in cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")]
    print("tables:", tables)

    if "examples" not in tables:
        raise RuntimeError(f"Expected 'examples' table, found {tables}")

    cols = [r[1] for r in cur.execute("PRAGMA table_info(examples)")]
    print("examples columns:", cols)

    if "serialized_example_proto" in cols:
        query = (
            "SELECT client_id, serialized_example_proto FROM examples "
            "WHERE split_name='train' ORDER BY client_id"
        )
        parse = parse_tokens_from_proto
    elif "tokens" in cols:
        query = (
            "SELECT client_id, tokens FROM examples "
            "WHERE split_name='train' ORDER BY client_id"
        )
        parse = lambda x: x if isinstance(x, str) else x.decode("utf-8")
    else:
        raise RuntimeError(f"No tokens column in examples: {cols}")

    user_tokens: dict[str, list[str]] = defaultdict(list)
    seen = set()
    n_rows = 0
    for client_id, raw in cur.execute(query):
        if client_id not in seen:
            if len(seen) >= max_clients:
                break
            seen.add(client_id)
        tokens = parse(raw)
        if tokens:
            user_tokens[client_id].append(tokens)
        n_rows += 1
        if n_rows % 500_000 == 0:
            print(f"  scanned {n_rows:,} rows, {len(seen):,} clients...", flush=True)

    conn.close()
    print(f"Loaded {len(user_tokens)} train users from sqlite ({n_rows:,} rows)")

    clients = []
    for posts in user_tokens.values():
        xs, ys = [], []
        for post in posts:
            ids = tokenize_line(post, word_to_id)
            xw, yw = sentence_to_windows(ids)
            xs.extend(xw)
            ys.extend(yw)
            if len(xs) >= max_per_client:
                break
        if not xs:
            continue
        clients.append((
            torch.tensor(xs[:max_per_client], dtype=torch.long),
            torch.tensor(ys[:max_per_client], dtype=torch.long),
        ))

    meta = dict(
        source="tff_sqlite",
        synthetic=False,
        num_clients=len(clients),
        vocab_size=VOCAB_SIZE,
        seq_len=SEQ_LEN,
        max_clients=max_clients,
        max_per_client=max_per_client,
    )
    out = DATA_DIR / "processed.pt"
    torch.save(dict(meta=meta, clients=clients), out)
    print(f"saved {len(clients)} clients -> {out} "
          f"({out.stat().st_size / 1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true",
                    help="Download and decompress the 8 GB TFF sqlite dump")
    ap.add_argument("--max-clients", type=int, default=5000)
    ap.add_argument("--max-per-client", type=int, default=2000)
    args = ap.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if args.download:
        db = download_sqlite(DATA_DIR)
        export_from_sqlite(db, args.max_clients, args.max_per_client)
        return

    db = DATA_DIR / "stackoverflow.sqlite"
    if not db.exists():
        print("No sqlite found. Run:")
        print("  python scripts/preprocess_stackoverflow.py --download")
        sys.exit(1)
    export_from_sqlite(db, args.max_clients, args.max_per_client)


if __name__ == "__main__":
    main()
