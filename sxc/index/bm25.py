"""BM25 baseline via SQLite FTS5.

Why FTS5: stdlib (`sqlite3` ships with Python), zero infra, real BM25
ranking via the `bm25()` aux function, and the same DB also stores chunk
metadata so a hit row carries chunk_id + score in one query.

Index layout (one file at indexes/bm25/bm25.sqlite):
  chunks         contentless table; rowid <-> chunk_id mapping
  chunks_fts     FTS5 virtual table over (text)
  meta           single-row table with build provenance

Note on `bm25()`:
  Negate to get a positive-higher score (FTS5 returns negative log-ranks).
"""

from __future__ import annotations

import os
import sqlite3
import time
from collections.abc import Iterator
from pathlib import Path

import pyarrow.parquet as pq
from tqdm import tqdm

from .base import Hit

# ----- index build -------------------------------------------------------------


_SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous  = NORMAL;
PRAGMA temp_store   = MEMORY;
PRAGMA mmap_size    = 268435456;

CREATE TABLE IF NOT EXISTS chunks (
    rowid           INTEGER PRIMARY KEY,
    chunk_id        TEXT UNIQUE NOT NULL,
    source          TEXT,
    project         TEXT,
    duplicate_count INTEGER NOT NULL DEFAULT 1,
    duplicate_rank  REAL NOT NULL DEFAULT 1.0
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text,
    content='',                 -- contentless; we don't store raw text
    tokenize='porter unicode61 remove_diacritics 2'
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


def _iter_chunk_rows(parquet: Path) -> Iterator[tuple[str, str, str | None, str, int, float]]:
    """Yield chunk rows from chunks parquet."""
    pf = pq.ParquetFile(parquet)
    cols = ["chunk_id", "source", "project", "text"]
    names = set(pf.schema_arrow.names)
    if "duplicate_count" in names:
        cols.append("duplicate_count")
    if "duplicate_rank" in names:
        cols.append("duplicate_rank")
    for batch in pf.iter_batches(batch_size=2048, columns=cols):
        d = batch.to_pydict()
        n = len(d["chunk_id"])
        for i in range(n):
            cid = d["chunk_id"][i]
            src = d["source"][i]
            proj = d["project"][i]
            text = d["text"][i]
            if not isinstance(cid, str) or not isinstance(text, str):
                continue
            duplicate_count = int(d.get("duplicate_count", [1] * n)[i] or 1)
            duplicate_rank = float(d.get("duplicate_rank", [1.0] * n)[i] or 1.0)
            yield cid, src or "", proj, text, duplicate_count, duplicate_rank


def build(chunks_parquet: str | os.PathLike[str], out_dir: str | os.PathLike[str]) -> int:
    """Build the FTS5 index. Returns row count."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    db_path = out / "bm25.sqlite"
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(_SCHEMA)
        cur = conn.cursor()

        # Stream rows in transactions of 5k for speed.
        BATCH = 5000
        insert_sql = (
            "INSERT INTO chunks(rowid, chunk_id, source, project, "
            "duplicate_count, duplicate_rank) VALUES (?,?,?,?,?,?)"
        )
        n = 0
        buf_meta: list[tuple[str, str, str | None, int, float]] = []
        buf_fts: list[tuple[int, str]] = []

        # We allocate rowid sequentially. chunk_id <-> rowid via the
        # `chunks` table. FTS rowid must match.
        for cid, src, proj, text, duplicate_count, duplicate_rank in tqdm(
            _iter_chunk_rows(Path(chunks_parquet)),
            desc="bm25-build",
        ):
            n += 1
            buf_meta.append((cid, src, proj, duplicate_count, duplicate_rank))
            buf_fts.append((n, text))
            if len(buf_meta) >= BATCH:
                start_rowid = n - len(buf_meta) + 1
                cur.executemany(
                    insert_sql,
                    [
                        (start_rowid + i, c, s, p, dc, dr)
                        for i, (c, s, p, dc, dr) in enumerate(buf_meta)
                    ],
                )
                cur.executemany(
                    "INSERT INTO chunks_fts(rowid, text) VALUES (?,?)", buf_fts
                )
                conn.commit()
                buf_meta.clear()
                buf_fts.clear()
        if buf_meta:
            start_rowid = n - len(buf_meta) + 1
            cur.executemany(
                insert_sql,
                [
                    (start_rowid + i, c, s, p, dc, dr)
                    for i, (c, s, p, dc, dr) in enumerate(buf_meta)
                ],
            )
            cur.executemany("INSERT INTO chunks_fts(rowid, text) VALUES (?,?)", buf_fts)
            conn.commit()

        cur.execute("INSERT OR REPLACE INTO meta(key,value) VALUES (?,?)", ("rows", str(n)))
        cur.execute(
            "INSERT OR REPLACE INTO meta(key,value) VALUES (?,?)",
            ("built_at", str(int(time.time()))),
        )
        cur.execute("INSERT OR REPLACE INTO meta(key,value) VALUES (?,?)", ("variant", "fts5-bm25"))
        cur.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('optimize')")
        conn.commit()
        return n
    finally:
        conn.close()


# ----- retriever ---------------------------------------------------------------


# FTS5 special tokens we strip / escape from a free-text query.
_FTS_BAD = '"():*'


def _sanitize(q: str) -> str:
    """Make a free-text query MATCH-safe.

    Each token is wrapped in double-quotes (FTS5 single-token phrase) so any
    punctuation/operators are escaped uniformly. Tokens combine implicitly
    with AND; bm25() ranks the matching set across all term frequencies.
    Empty -> empty.
    """
    cleaned = "".join(ch for ch in q if ch not in _FTS_BAD).strip()
    if not cleaned:
        return ""
    tokens = [t for t in cleaned.split() if t]
    return " ".join(f"\"{t}\"" for t in tokens)


class BM25Retriever:
    """FTS5-backed retriever. Lazy-opens the SQLite connection per thread
    via `sqlite3.threadsafety` semantics; here we open per instance."""

    name = "bm25"

    def __init__(self, index_dir: str | os.PathLike[str]) -> None:
        db_path = Path(index_dir) / "bm25.sqlite"
        if not db_path.exists():
            msg = f"BM25 index not found at {db_path}"
            raise FileNotFoundError(msg)
        self._conn = sqlite3.connect(
            f"file:{db_path}?mode=ro", uri=True, check_same_thread=False
        )
        self._conn.execute("PRAGMA query_only = ON")

    def search(self, query: str, top_k: int = 100) -> list[Hit]:
        q = _sanitize(query)
        if not q:
            return []
        # bm25() is FTS5's ranking aux fn. Lower (more negative) = better.
        # Negate to make higher better, then divide by duplicate_rank so
        # repeated transcript boilerplate cannot win just because it repeated.
        try:
            cur = self._conn.execute(
                """
                SELECT c.chunk_id, -bm25(chunks_fts) / c.duplicate_rank AS score
                FROM chunks_fts
                JOIN chunks c ON c.rowid = chunks_fts.rowid
                WHERE chunks_fts MATCH ?
                ORDER BY score DESC
                LIMIT ?
                """,
                (q, top_k),
            )
        except sqlite3.OperationalError:
            # malformed MATCH expression — defensively fall back to OR'd terms
            tokens = [t for t in q.strip("\"").split() if t]
            if not tokens:
                return []
            fallback = " OR ".join(tokens)
            cur = self._conn.execute(
                """
                SELECT c.chunk_id, -bm25(chunks_fts) / c.duplicate_rank AS score
                FROM chunks_fts
                JOIN chunks c ON c.rowid = chunks_fts.rowid
                WHERE chunks_fts MATCH ?
                ORDER BY score DESC
                LIMIT ?
                """,
                (fallback, top_k),
            )
        return [
            Hit(chunk_id=cid, score=float(score), rank=rank)
            for rank, (cid, score) in enumerate(cur.fetchall())
        ]

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> BM25Retriever:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
