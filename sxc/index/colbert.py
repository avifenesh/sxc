"""ColBERTv2 indexer/retriever via colbert-ai.

This is the project's primary retrieval path (the repo name keeps SPLADE for
the bench history, but serving is ColBERTv2 only). Wraps the upstream
`Indexer` / `Searcher` so the module fits the same build/search shape as the
BM25 and SPLADE indexers.

Index storage: a colbert-ai index directory at `out_dir/`. We additionally
write `pid_to_chunk_id.parquet` because colbert assigns its own integer
passage ids (`pid`) and the bench harness needs to map them back to our
canonical `chunk_id`.

Encoder: `colbert-ir/colbertv2.0` (open, BERT-base initialized).

Notes:
- nbits=2 is the standard residual compression; ~16 bytes per token vector.
- doc_maxlen=300 covers our ~256-token chunks with margin for tokenizer
  overhead. Queries are auto-padded to 32.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

# colbert-ai (0.2.22) targets an older transformers API. Newer transformers
# (>= 4.50ish) expects `all_tied_weights_keys` on the model instance during
# `_move_missing_keys_from_meta_to_device`. nn.Module's `__getattr__`
# shadows class-level descriptors so a property cannot satisfy it; we
# patch the method itself to seed the attribute before delegating.
import transformers.modeling_utils as _tm

_orig_move = _tm.PreTrainedModel._move_missing_keys_from_meta_to_device


def _patched_move(self: Any, *args: Any, **kwargs: Any) -> Any:
    if not hasattr(self, "all_tied_weights_keys"):
        tied = getattr(self, "_tied_weights_keys", None)
        if isinstance(tied, dict):
            self.all_tied_weights_keys = tied
        elif tied:
            self.all_tied_weights_keys = dict.fromkeys(tied, None)
        else:
            self.all_tied_weights_keys = {}
    return _orig_move(self, *args, **kwargs)


_tm.PreTrainedModel._move_missing_keys_from_meta_to_device = _patched_move  # type: ignore[method-assign]

from colbert import Indexer, Searcher  # noqa: E402
from colbert.infra import ColBERTConfig, Run, RunConfig  # noqa: E402

from .base import Hit  # noqa: E402

DEFAULT_CHECKPOINT = "colbert-ir/colbertv2.0"
DEFAULT_DOC_MAXLEN = 300
DEFAULT_NBITS = 2
DEFAULT_INDEX_NAME = "chunks"


# ----- helpers ----------------------------------------------------------------


def _iter_chunk_rows(parquet: Path) -> Iterator[tuple[str, str]]:
    pf = pq.ParquetFile(parquet)
    for batch in pf.iter_batches(batch_size=2048, columns=["chunk_id", "text"]):
        d = batch.to_pydict()
        n = len(d["chunk_id"])
        for i in range(n):
            cid = d["chunk_id"][i]
            text = d["text"][i]
            if isinstance(cid, str) and isinstance(text, str):
                yield cid, text


_PID_MAP_FIELDS: list[pa.Field[pa.DataType]] = [
    pa.field("pid", pa.int32()),
    pa.field("chunk_id", pa.string()),
]
_PID_MAP_SCHEMA: pa.Schema = pa.schema(_PID_MAP_FIELDS)


def _write_pid_map(out_dir: Path, chunk_ids: list[str]) -> None:
    table = pa.table(
        {
            "pid": pa.array(range(len(chunk_ids)), type=pa.int32()),
            "chunk_id": pa.array(chunk_ids, type=pa.string()),
        },
        schema=_PID_MAP_SCHEMA,
    )
    pq.write_table(table, out_dir / "pid_to_chunk_id.parquet", compression="zstd")


def _read_pid_map(out_dir: Path) -> dict[int, str]:
    table = pq.read_table(out_dir / "pid_to_chunk_id.parquet")
    pids: list[int] = [int(p) for p in table.column("pid").to_pylist() if p is not None]
    cids: list[str] = [c for c in table.column("chunk_id").to_pylist() if isinstance(c, str)]
    return dict(zip(pids, cids, strict=False))


# ----- index build ------------------------------------------------------------


def build(
    chunks_parquet: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    *,
    checkpoint: str = DEFAULT_CHECKPOINT,
    doc_maxlen: int = DEFAULT_DOC_MAXLEN,
    nbits: int = DEFAULT_NBITS,
    index_name: str = DEFAULT_INDEX_NAME,
    nranks: int = 1,
    kmeans_niters: int = 4,
) -> int:
    """Build a ColBERTv2 index. Returns total passage count.

    `kmeans_niters` is the centroid-training iteration count. On a faiss-CPU
    box each iteration over ~9M sampled embeddings costs ~19 min, so this is
    the dominant build cost. Lower it (e.g. 1-2) for a much faster daily
    rebuild at a small recall cost; the daily refresh.sh sets it via env.
    """
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    rows = list(_iter_chunk_rows(Path(chunks_parquet)))
    chunk_ids = [r[0] for r in rows]
    texts = [r[1] for r in rows]
    n = len(rows)

    _write_pid_map(out, chunk_ids)

    # colbert writes into <root>/<experiment>/indexes/<index_name>. We pin
    # root=out_dir/.colbert and experiment="bench" so artifacts live under
    # the user-supplied out_dir.
    cfg = ColBERTConfig(
        nbits=nbits,
        doc_maxlen=doc_maxlen,
        root=str(out / ".colbert"),
        experiment="bench",
        kmeans_niters=kmeans_niters,
    )
    with Run().context(RunConfig(nranks=nranks, root=str(out / ".colbert"), experiment="bench")):
        indexer = Indexer(checkpoint=checkpoint, config=cfg)
        indexer.index(name=index_name, collection=texts, overwrite=True)
    return n


# ----- retriever --------------------------------------------------------------


class ColbertRetriever:
    """ColBERTv2 retriever wrapping colbert-ai's Searcher."""

    name = "colbert-v2"

    def __init__(
        self,
        index_dir: str | os.PathLike[str],
        *,
        checkpoint: str = DEFAULT_CHECKPOINT,
        index_name: str = DEFAULT_INDEX_NAME,
    ) -> None:
        out = Path(index_dir).resolve()
        if not (out / "pid_to_chunk_id.parquet").exists():
            msg = f"ColBERT index not found at {out}"
            raise FileNotFoundError(msg)

        self._pid_to_cid = _read_pid_map(out)
        cfg = ColBERTConfig(
            root=str(out / ".colbert"),
            experiment="bench",
        )
        self._run_ctx = Run().context(
            RunConfig(nranks=1, root=str(out / ".colbert"), experiment="bench")
        )
        self._run_ctx.__enter__()
        self._searcher = Searcher(index=index_name, checkpoint=checkpoint, config=cfg)

    def search(self, query: str, top_k: int = 100) -> list[Hit]:
        if not query.strip():
            return []
        # colbert-ai's Searcher.search returns (pids, ranks, scores) tuples.
        pids, _ranks, scores = self._searcher.search(query, k=top_k)
        out: list[Hit] = []
        for rank, (pid, score) in enumerate(zip(pids, scores, strict=False)):
            cid = self._pid_to_cid.get(int(pid))
            if cid is None:
                continue
            out.append(Hit(chunk_id=cid, score=float(score), rank=rank))
        return out

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._run_ctx.__exit__(None, None, None)

    def __enter__(self) -> ColbertRetriever:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
