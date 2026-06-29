"""Bench runner: loop over (retriever, gold pair) and compute metrics."""

from __future__ import annotations

import math
import os
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from sxc.index.base import Retriever

from .schema import BenchRow, BenchSchema

# ----- gold-pair iterator ----------------------------------------------------


def _iter_gold(
    synth_parquet: Path, only_self_check: bool
) -> Iterator[tuple[str, str, str, str, bool]]:
    """Yield (query_id, chunk_id, q, source, self_check)."""
    pf = pq.ParquetFile(synth_parquet)
    cols = ["query_id", "chunk_id", "q", "source", "self_check"]
    for batch in pf.iter_batches(batch_size=2048, columns=cols):
        d = batch.to_pydict()
        n = len(d["query_id"])
        for i in range(n):
            qid = d["query_id"][i]
            cid = d["chunk_id"][i]
            q = d["q"][i]
            src = d["source"][i]
            sc = d["self_check"][i]
            if not isinstance(qid, str) or not isinstance(cid, str) or not isinstance(q, str):
                continue
            if only_self_check and not sc:
                continue
            yield qid, cid, q, src or "", bool(sc)


# ----- metrics ---------------------------------------------------------------


def _ndcg_at_k(rank_1based: int, k: int) -> float:
    """Single-relevant-doc nDCG@k. rank_1based <= 0 means not in top_k."""
    if rank_1based <= 0 or rank_1based > k:
        return 0.0
    # dcg = 1 / log2(rank+1); ideal dcg = 1 (relevant doc at rank 1)
    return 1.0 / math.log2(rank_1based + 1)


def _recall_at_k(rank_1based: int, k: int) -> float:
    return 1.0 if 1 <= rank_1based <= k else 0.0


def _rr(rank_1based: int) -> float:
    return 1.0 / rank_1based if rank_1based >= 1 else 0.0


# ----- runner ----------------------------------------------------------------


def _eval_one(
    retriever: Retriever, query: str, gold_chunk_id: str, top_k: int
) -> tuple[int, float]:
    """Returns (rank_of_gold_1based or -1, latency_ms)."""
    t0 = time.perf_counter()
    hits = retriever.search(query, top_k=top_k)
    dt = (time.perf_counter() - t0) * 1000.0
    rank = -1
    for h in hits:
        if h.chunk_id == gold_chunk_id:
            rank = h.rank + 1  # 1-based
            break
    return rank, dt


def _to_arrow_batch(rows: list[BenchRow]) -> pa.RecordBatch:
    cols: dict[str, list[Any]] = {f.name: [] for f in BenchSchema}
    for r in rows:
        cols["retriever"].append(r.retriever)
        cols["query_id"].append(r.query_id)
        cols["chunk_id"].append(r.chunk_id)
        cols["source"].append(r.source)
        cols["self_check"].append(r.self_check)
        cols["rank_of_gold"].append(r.rank_of_gold)
        cols["top_k"].append(r.top_k)
        cols["ndcg10"].append(r.ndcg10)
        cols["recall_at_50"].append(r.recall_at_50)
        cols["rr"].append(r.rr)
        cols["latency_ms"].append(r.latency_ms)
    return pa.RecordBatch.from_pydict(cols, schema=BenchSchema)


def run_bench(
    synth_parquet: str | os.PathLike[str],
    out_parquet: str | os.PathLike[str],
    retrievers: dict[str, Retriever],
    *,
    top_k: int = 100,
    only_self_check: bool = True,
    limit: int | None = None,
) -> int:
    """Iterate gold pairs, run each retriever, write per-row results."""
    synth = Path(synth_parquet)
    out = Path(out_parquet)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    writer = pq.ParquetWriter(out, BenchSchema, compression="zstd")
    buf: list[BenchRow] = []
    BUF = 2048
    n_total = 0

    gold_iter = _iter_gold(synth, only_self_check=only_self_check)
    if limit is not None:
        gold_iter = (g for i, g in enumerate(gold_iter) if i < limit)

    try:
        for qid, gold_cid, q, source, sc in tqdm(gold_iter, desc="bench"):
            for rname, retr in retrievers.items():
                rank, dt = _eval_one(retr, q, gold_cid, top_k=top_k)
                buf.append(
                    BenchRow(
                        retriever=rname,
                        query_id=qid,
                        chunk_id=gold_cid,
                        source=source,
                        self_check=sc,
                        rank_of_gold=rank,
                        top_k=top_k,
                        ndcg10=_ndcg_at_k(rank, 10),
                        recall_at_50=_recall_at_k(rank, 50),
                        rr=_rr(rank),
                        latency_ms=dt,
                    )
                )
                n_total += 1
            if len(buf) >= BUF:
                writer.write_batch(_to_arrow_batch(buf))
                buf.clear()
        if buf:
            writer.write_batch(_to_arrow_batch(buf))
    finally:
        writer.close()
    return n_total
