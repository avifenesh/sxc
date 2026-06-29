"""Top-level retrieval API: `get_context(query) -> list[Context]`.

A `Context` is the self-contained slab a caller drops into a prompt:
  - quote: the exact 256-token chunk that matched
  - text:  the ~1024-token parent window around it
  - source / project / session / ts: provenance
  - score / rank: retriever score after duplicate-aware adjustment
"""

from __future__ import annotations

import hashlib
import os
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Literal

import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict

from sxc.index.base import Hit, Retriever

RetrieverName = Literal["bm25", "splade", "colbert", "hybrid"]

DEFAULT_RETRIEVER: RetrieverName = "colbert"
# ^ bench result: ColBERTv2 alone matches the SPLADE+ColBERT cascade
#   on nDCG@10/recall@50 across all sources (claude/codex/hermes/readme)
#   while running ~20x faster (40ms vs 820ms p50). Hybrid available via
#   `retriever="hybrid"` if a future eval shifts the picture.
DEFAULT_TOP_K = 10
DEFAULT_DATA_ROOT = Path(
    os.environ.get("SXC_DATA_ROOT") or Path(__file__).resolve().parents[2]
)
CONTEXT_DEDUPE_PREFIX_CHARS = 2048

# Persistent operator override. When set, this WINS over an explicit
# `retriever=` argument — so a caller hardcoding "colbert" still gets routed
# to the forced retriever. Resolution order:
#   1. SXC_FORCE_RETRIEVER env (one shell / one run)
#   2. {data_root}/config/serve.json  {"force_retriever": "bm25"}  (durable)
# Used for "fall back to bm25" while the ColBERT index is stale/rebuilding,
# without touching the consumer. None / unset / invalid → no override.
FORCE_CONFIG_PATH = DEFAULT_DATA_ROOT / "config" / "serve.json"
_VALID_RETRIEVERS = ("bm25", "splade", "colbert", "hybrid")


class Context(BaseModel):
    """One retrieved context slab — ready to drop into a prompt."""

    model_config = ConfigDict(extra="forbid", strict=True)

    chunk_id: str
    score: float
    rank: int
    quote: str          # the matched chunk text
    text: str           # parent window (self-contained)
    source: str         # 'claude' | 'codex' | 'hermes' | 'readme'
    project: str | None
    session_id: str
    ts: float | None
    duplicate_count: int = 1
    duplicate_rank: float = 1.0


# ----- chunk-row hydrator -----------------------------------------------------


@lru_cache(maxsize=1)
def _chunks_table_path() -> Path:
    return DEFAULT_DATA_ROOT / "data" / "chunks.parquet"


@lru_cache(maxsize=1)
def _chunk_lookup() -> dict[str, dict[str, object]]:
    """Load chunks.parquet once and key by chunk_id for fast hydration.

    For 100k chunks at ~1KB each that's ~100MB resident — fine for a
    long-running process, fast to look up.
    """
    pf = pq.ParquetFile(_chunks_table_path())
    out: dict[str, dict[str, object]] = {}
    cols = [
        "chunk_id",
        "source",
        "project",
        "session_id",
        "ts",
        "text",
        "parent_text",
        "duplicate_count",
        "duplicate_rank",
    ]
    for batch in pf.iter_batches(batch_size=2048, columns=cols):
        d = batch.to_pydict()
        for i in range(len(d["chunk_id"])):
            cid = d["chunk_id"][i]
            if not isinstance(cid, str):
                continue
            out[cid] = {
                "source": d["source"][i],
                "project": d["project"][i],
                "session_id": d["session_id"][i],
                "ts": d["ts"][i],
                "text": d["text"][i],
                "parent_text": d["parent_text"][i],
                "duplicate_count": d["duplicate_count"][i],
                "duplicate_rank": d["duplicate_rank"][i],
            }
    return out


# ----- retriever cache --------------------------------------------------------


@lru_cache(maxsize=4)
def _get_retriever(name: RetrieverName) -> Retriever:
    indexes = DEFAULT_DATA_ROOT / "indexes"
    if name == "bm25":
        from sxc.index.bm25 import BM25Retriever

        return BM25Retriever(indexes / "bm25")
    if name == "splade":
        from sxc.index.splade import SpladeRetriever

        return SpladeRetriever(indexes / "splade")
    if name == "colbert":
        from sxc.index.colbert import ColbertRetriever

        return ColbertRetriever(indexes / "colbert")
    from sxc.index.hybrid import HybridRetriever

    return HybridRetriever(indexes / "splade", indexes / "colbert")


# ----- public API -------------------------------------------------------------


def _candidate_pool_size(top_k: int) -> int:
    if top_k <= 0:
        return 0
    return max(top_k, min(top_k * 5, top_k + 40, 100))


def _duplicate_adjusted_score(score: float, duplicate_rank: float) -> float:
    penalty = max(duplicate_rank, 1.0)
    if penalty == 1.0:
        return score
    if score < 0:
        return score * penalty
    return score / penalty


def _context_dedupe_key(text: str) -> str:
    norm = re.sub(r"\s+", " ", text).strip().lower()
    if len(norm) > CONTEXT_DEDUPE_PREFIX_CHARS:
        norm = norm[:CONTEXT_DEDUPE_PREFIX_CHARS]
    h = hashlib.blake2b(digest_size=12)
    h.update(norm.encode("utf-8", "replace"))
    return h.hexdigest()


def _meta_float(value: object, default: float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _meta_int(value: object, default: int) -> int:
    if isinstance(value, int):
        return value
    return default


def get_context(
    query: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    retriever: RetrieverName | None = None,
) -> list[Context]:
    """Return up to `top_k` context slabs ranked by `retriever`."""
    if top_k <= 0:
        return []
    name: RetrieverName = _forced_retriever() or retriever or _env_retriever()
    r = _get_retriever(name)
    hits: list[Hit] = r.search(query, top_k=_candidate_pool_size(top_k))
    if not hits:
        return []
    lut = _chunk_lookup()
    candidates: list[tuple[float, int, dict[str, object], Hit]] = []
    for h in hits:
        meta = lut.get(h.chunk_id)
        if meta is None:
            continue
        duplicate_rank = _meta_float(meta.get("duplicate_rank"), 1.0)
        score = h.score
        if name != "bm25":
            score = _duplicate_adjusted_score(score, duplicate_rank)
        candidates.append((score, h.rank, meta, h))
    candidates.sort(key=lambda item: (item[0], -item[1]), reverse=True)

    from . import judge as _judge
    judge_on = _judge.enabled()
    # bound judge latency: never judge more than 3x the requested slab count
    judge_budget = top_k * 3

    out: list[Context] = []
    seen_context_keys: set[str] = set()
    for score, _old_rank, meta, h in candidates:
        proj_v = meta.get("project")
        ts_v = meta.get("ts")
        quote = str(meta.get("text") or "")
        context_key = _context_dedupe_key(quote) if quote else h.chunk_id
        if context_key in seen_context_keys:
            continue
        seen_context_keys.add(context_key)
        if judge_on and quote and judge_budget > 0:
            judge_budget -= 1
            t0 = time.perf_counter()
            digit, p1 = _judge.judge(query, quote)
            kept = digit != 0  # fail-open: None (judge down) passes
            _judge.log_decision(query, h.chunk_id, name, digit, p1, kept,
                                (time.perf_counter() - t0) * 1000)
            if not kept:
                continue
        out.append(
            Context(
                chunk_id=h.chunk_id,
                score=score,
                rank=len(out),
                quote=quote,
                text=str(meta.get("parent_text") or ""),
                source=str(meta.get("source") or ""),
                project=proj_v if isinstance(proj_v, str) else None,
                session_id=str(meta.get("session_id") or ""),
                ts=float(ts_v) if isinstance(ts_v, (int, float)) else None,
                duplicate_count=_meta_int(meta.get("duplicate_count"), 1),
                duplicate_rank=_meta_float(meta.get("duplicate_rank"), 1.0),
            )
        )
        if len(out) >= top_k:
            break
    return out


def _env_retriever() -> RetrieverName:
    v = os.environ.get("SXC_RETRIEVER")
    if v in _VALID_RETRIEVERS:
        return v  # type: ignore[return-value]
    return DEFAULT_RETRIEVER


def _forced_retriever() -> RetrieverName | None:
    """Persistent operator override that beats an explicit `retriever=` arg.

    Env first (SXC_FORCE_RETRIEVER), then the durable serve.json. Returns None
    when no valid override is set, so the normal `retriever or default` path
    applies. Never raises — a missing/garbled config file just means no force.
    """
    v = os.environ.get("SXC_FORCE_RETRIEVER")
    if v in _VALID_RETRIEVERS:
        return v  # type: ignore[return-value]
    try:
        import json

        with FORCE_CONFIG_PATH.open(encoding="utf-8") as f:
            cfg = json.load(f)
        fr = cfg.get("force_retriever") if isinstance(cfg, dict) else None
        if fr in _VALID_RETRIEVERS:
            return fr  # type: ignore[return-value]
    except (OSError, ValueError):
        pass
    return None


def set_force_retriever(name: str | None) -> None:
    """Write/clear the durable force-retriever override (serve.json).

    name in {bm25,splade,colbert,hybrid} sets it; None/'' clears it (back to
    the skill's own choice). Used by the atrium toggle. Atomic write.
    """
    import json

    FORCE_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if name in _VALID_RETRIEVERS:
        payload = {"force_retriever": name}
    elif name in (None, ""):
        payload = {}
    else:
        raise ValueError(f"invalid retriever: {name!r}")
    tmp = FORCE_CONFIG_PATH.with_suffix(f".json.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(FORCE_CONFIG_PATH)


def get_force_retriever() -> str | None:
    """Current durable override (None if unset). Reads serve.json only —
    ignores the env layer so the UI reflects the persisted state."""
    try:
        import json

        with FORCE_CONFIG_PATH.open(encoding="utf-8") as f:
            cfg = json.load(f)
        fr = cfg.get("force_retriever") if isinstance(cfg, dict) else None
        return fr if fr in _VALID_RETRIEVERS else None
    except (OSError, ValueError):
        return None
