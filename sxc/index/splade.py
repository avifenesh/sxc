"""SPLADE v3 sparse retriever.

Encoder: `naver/splade-v3` (HF). Output is a ReLU(log(1+exp(logits)))
max-pool over the sequence — a sparse vector over the BERT vocabulary
(30522 dims) where most entries are 0.

Index storage: a SQLite DB with two tables:
  postings(term TEXT, doc_idx INT, weight REAL)  -- inverted index
  docs(doc_idx INT PRIMARY KEY, chunk_id TEXT, doc_norm REAL)
plus a token<->id table from the tokenizer's vocab. We persist `term_id`
not `term` so postings stay compact.

Why SQLite for sparse postings instead of Pyserini/Anserini:
- avoids JVM dependency and Lucene packaging
- 100k docs x ~150 nonzeros each = 15M postings, fits sqlite trivially
- query-time scoring is one indexed scan + group-by

Search: encode query → sparse {term_id: weight} → SUM(q_w * d_w) per doc
over postings WHERE term_id IN (...).
"""

from __future__ import annotations

import os
import sqlite3
import time
from collections.abc import Iterator
from pathlib import Path

import pyarrow.parquet as pq
import torch
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

from .base import Hit

DEFAULT_MODEL = "naver/splade-cocondenser-ensembledistil"
# ^ open SPLADE checkpoint. naver/splade-v3 is gated — pass --model with
#   an HF token-authenticated session if v3 access is granted.
DEFAULT_BATCH = 32
DEFAULT_MAX_LEN = 256


# ----- encoder ----------------------------------------------------------------


class _SpladeEncoder:
    def __init__(self, model_id: str, device: str | None = None) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForMaskedLM.from_pretrained(model_id)
        self.model.eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        if self.device == "cuda":
            self.model = self.model.half()

    @torch.inference_mode()
    def encode(self, texts: list[str], max_len: int = DEFAULT_MAX_LEN) -> torch.Tensor:
        """Returns a (B, V) sparse-dense tensor (dense storage of sparse vec).

        SPLADE pooling: `max_t( log(1 + ReLU(logit_t)) )` over the
        non-padding positions. Same formula as Formal et al. 2021/2022.
        """
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        ).to(self.device)
        with torch.amp.autocast_mode.autocast(
            self.device, dtype=torch.float16, enabled=self.device == "cuda"
        ):
            logits = self.model(**enc).logits  # (B, T, V)
        # mask out [PAD] positions before max-pool
        attn = enc["attention_mask"].unsqueeze(-1)  # (B, T, 1)
        # SPLADE term importance: ReLU then log1p, max over tokens.
        weights = torch.log1p(torch.relu(logits)) * attn
        pooled, _ = weights.max(dim=1)  # (B, V)
        out: torch.Tensor = pooled.float().cpu()
        return out


def _to_sparse_rows(vec: torch.Tensor) -> list[tuple[int, float]]:
    """Convert one dense-stored sparse vec → [(term_id, weight)]."""
    nz = vec.nonzero(as_tuple=False).squeeze(-1)
    return [(int(t), float(vec[t])) for t in nz.tolist()]


# ----- index build ------------------------------------------------------------


_SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous  = NORMAL;
PRAGMA temp_store   = MEMORY;
PRAGMA mmap_size    = 268435456;

CREATE TABLE IF NOT EXISTS docs (
    doc_idx  INTEGER PRIMARY KEY,
    chunk_id TEXT UNIQUE NOT NULL,
    doc_norm REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS postings (
    term_id INTEGER NOT NULL,
    doc_idx INTEGER NOT NULL,
    weight  REAL    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_postings_term ON postings(term_id);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


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


def build(
    chunks_parquet: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    *,
    model_id: str = DEFAULT_MODEL,
    batch: int = DEFAULT_BATCH,
    max_len: int = DEFAULT_MAX_LEN,
) -> int:
    """Encode all chunks with SPLADE v3 and write a sparse inverted index."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    db_path = out / "splade.sqlite"
    if db_path.exists():
        db_path.unlink()

    enc = _SpladeEncoder(model_id)

    conn = sqlite3.connect(db_path)
    conn.executescript(_SCHEMA)
    cur = conn.cursor()

    rows = list(_iter_chunk_rows(Path(chunks_parquet)))
    n_total = len(rows)

    n_postings = 0
    for start in tqdm(range(0, n_total, batch), desc="splade-build"):
        sub = rows[start : start + batch]
        cids = [r[0] for r in sub]
        texts = [r[1] for r in sub]
        vecs = enc.encode(texts, max_len=max_len)  # (B, V)

        # Insert docs + postings in one tx per batch.
        for i, cid in enumerate(cids):
            doc_idx = start + i + 1  # 1-based
            v = vecs[i]
            sparse = _to_sparse_rows(v)
            doc_norm = float(sum(w * w for _, w in sparse) ** 0.5)
            cur.execute(
                "INSERT INTO docs(doc_idx, chunk_id, doc_norm) VALUES (?,?,?)",
                (doc_idx, cid, doc_norm),
            )
            if sparse:
                cur.executemany(
                    "INSERT INTO postings(term_id, doc_idx, weight) VALUES (?,?,?)",
                    [(t, doc_idx, w) for t, w in sparse],
                )
                n_postings += len(sparse)
        conn.commit()

    cur.execute("INSERT OR REPLACE INTO meta(key,value) VALUES (?,?)", ("rows", str(n_total)))
    cur.execute(
        "INSERT OR REPLACE INTO meta(key,value) VALUES (?,?)", ("postings", str(n_postings))
    )
    cur.execute("INSERT OR REPLACE INTO meta(key,value) VALUES (?,?)", ("model", model_id))
    cur.execute("INSERT OR REPLACE INTO meta(key,value) VALUES (?,?)", ("max_len", str(max_len)))
    cur.execute(
        "INSERT OR REPLACE INTO meta(key,value) VALUES (?,?)", ("built_at", str(int(time.time())))
    )
    conn.commit()
    conn.close()
    return n_total


# ----- retriever --------------------------------------------------------------


class SpladeRetriever:
    """SPLADE v3 sparse retriever over a SQLite-backed inverted index."""

    name = "splade-v3"

    def __init__(
        self, index_dir: str | os.PathLike[str], model_id: str = DEFAULT_MODEL
    ) -> None:
        db_path = Path(index_dir) / "splade.sqlite"
        if not db_path.exists():
            msg = f"SPLADE index not found at {db_path}"
            raise FileNotFoundError(msg)
        self._conn = sqlite3.connect(
            f"file:{db_path}?mode=ro", uri=True, check_same_thread=False
        )
        self._conn.execute("PRAGMA query_only = ON")
        self._enc = _SpladeEncoder(model_id)

    def search(self, query: str, top_k: int = 100) -> list[Hit]:
        if not query.strip():
            return []
        q_vec = self._enc.encode([query])[0]
        q_sparse = _to_sparse_rows(q_vec)
        if not q_sparse:
            return []
        # Build a temp-bound list of (term_id, q_w) and aggregate the dot
        # product per doc_idx via a single SQL pass.
        placeholders = ",".join("?" * len(q_sparse))
        term_ids = [t for t, _ in q_sparse]
        # Map term_id -> q_weight via a CASE expression. SQLite handles a
        # few thousand-way CASE fine; SPLADE q-vecs are ~30-100 nonzeros.
        case_sql = "CASE p.term_id "
        params: list[object] = []
        for t, w in q_sparse:
            case_sql += "WHEN ? THEN ? "
            params.extend([t, w])
        case_sql += "ELSE 0 END"
        sql = f"""
            SELECT d.chunk_id, SUM(p.weight * ({case_sql})) AS score
            FROM postings p
            JOIN docs d ON d.doc_idx = p.doc_idx
            WHERE p.term_id IN ({placeholders})
            GROUP BY p.doc_idx
            ORDER BY score DESC
            LIMIT ?
        """
        cur = self._conn.execute(sql, (*params, *term_ids, top_k))
        return [
            Hit(chunk_id=cid, score=float(score), rank=rank)
            for rank, (cid, score) in enumerate(cur.fetchall())
        ]

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> SpladeRetriever:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
