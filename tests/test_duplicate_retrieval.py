from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from sxc.chunk.extract import (
    DEDUPE_PREFIX_CHARS,
    _dedupe_chunks,
    _dedupe_key,
    _duplicate_rank,
)
from sxc.chunk.schema import Chunk, ChunkSchema, Source
from sxc.index.base import Hit
from sxc.index.bm25 import BM25Retriever
from sxc.index.bm25 import build as build_bm25
from sxc.serve import api as serve_api


def _chunk(
    text: str,
    *,
    chunk_id: str,
    source: Source = "codex",
    session_id: str = "session",
    project: str | None = None,
    ts: float = 1.0,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        source=source,
        session_id=session_id,
        project=project,
        ts=ts,
        role="user",
        text=text,
        parent_text=f"[user] {text}",
        dedupe_key=_dedupe_key(text),
    )


def _write_chunks(path: Path, rows: list[dict[str, object]]) -> None:
    cols: dict[str, list[object]] = {field.name: [] for field in ChunkSchema}
    for row in rows:
        for field in ChunkSchema:
            cols[field.name].append(row.get(field.name))
    table = pa.Table.from_pydict(cols, schema=ChunkSchema)
    pq.write_table(table, path)


class DuplicateRetrievalTests(unittest.TestCase):
    def test_dedupe_key_collapses_long_retrieval_equivalent_chunks(self) -> None:
        prefix = "same retrieval-visible transcript prefix " * (
            DEDUPE_PREFIX_CHARS // 40 + 20
        )
        self.assertGreater(len(prefix), DEDUPE_PREFIX_CHARS)
        self.assertEqual(
            _dedupe_key(f"{prefix} tail from one session"),
            _dedupe_key(f"{prefix} tail from another session"),
        )
        self.assertNotEqual(_dedupe_key("short alpha"), _dedupe_key("short beta"))

    def test_dedupe_chunks_preserves_count_and_penalty_metadata(self) -> None:
        old = _chunk(
            "Repeated transcript chunk",
            chunk_id="old",
            source="claude",
            session_id="old-session",
            project="itch",
            ts=1.0,
        )
        fresh = _chunk(
            "repeated   transcript\nchunk",
            chunk_id="fresh",
            source="codex",
            session_id="fresh-session",
            project="itch",
            ts=2.0,
        )

        out = _dedupe_chunks([old, fresh])

        self.assertEqual(len(out), 1)
        only = out[0]
        self.assertEqual(only.session_id, "fresh-session")
        self.assertEqual(only.chunk_id, only.dedupe_key)
        self.assertIsNone(only.prev_chunk_id)
        self.assertIsNone(only.next_chunk_id)
        self.assertEqual(only.duplicate_count, 2)
        self.assertEqual(only.duplicate_rank, _duplicate_rank(2))
        self.assertEqual(set((only.duplicate_sources or "").split(",")), {"claude", "codex"})
        self.assertEqual(only.duplicate_projects, "itch")
        self.assertEqual(
            set((only.duplicate_session_ids or "").split(",")),
            {"old-session", "fresh-session"},
        )

    def test_bm25_duplicate_rank_penalizes_instead_of_boosting(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            chunks = root / "chunks.parquet"
            index = root / "bm25"
            _write_chunks(
                chunks,
                [
                    {
                        "chunk_id": "duplicate",
                        "source": "codex",
                        "session_id": "dup-session",
                        "project": None,
                        "ts": 1.0,
                        "role": "user",
                        "text": "needle topic",
                        "parent_text": "[user] needle topic",
                        "prev_chunk_id": None,
                        "next_chunk_id": None,
                        "dedupe_key": "duplicate",
                        "duplicate_count": 16,
                        "duplicate_rank": 5.0,
                        "duplicate_sources": "codex",
                        "duplicate_projects": None,
                        "duplicate_session_ids": "dup-session",
                    },
                    {
                        "chunk_id": "unique",
                        "source": "codex",
                        "session_id": "unique-session",
                        "project": None,
                        "ts": 1.0,
                        "role": "user",
                        "text": "needle topic",
                        "parent_text": "[user] needle topic",
                        "prev_chunk_id": None,
                        "next_chunk_id": None,
                        "dedupe_key": "unique",
                        "duplicate_count": 1,
                        "duplicate_rank": 1.0,
                        "duplicate_sources": "codex",
                        "duplicate_projects": None,
                        "duplicate_session_ids": "unique-session",
                    },
                ],
            )
            self.assertEqual(build_bm25(chunks, index), 2)

            with BM25Retriever(index) as retriever:
                hits = retriever.search("needle", top_k=2)

        self.assertEqual([hit.chunk_id for hit in hits], ["unique", "duplicate"])
        self.assertGreater(hits[0].score, hits[1].score)

    def test_get_context_reranks_candidate_pool_with_duplicate_penalty(self) -> None:
        class FakeRetriever:
            name = "colbert-v2"
            requested_top_k = 0

            def search(self, query: str, top_k: int = 100) -> list[Hit]:
                self.requested_top_k = top_k
                return [
                    Hit(chunk_id="duplicate", score=100.0, rank=0),
                    Hit(chunk_id="unique", score=60.0, rank=1),
                    Hit(chunk_id="near_unique", score=59.0, rank=2),
                ]

        fake = FakeRetriever()
        shared_prefix = "needle topic unique " * (
            serve_api.CONTEXT_DEDUPE_PREFIX_CHARS // 20 + 20
        )
        lookup: dict[str, dict[str, object]] = {
            "duplicate": {
                "source": "codex",
                "project": None,
                "session_id": "dup-session",
                "ts": 1.0,
                "text": "needle topic duplicate",
                "parent_text": "[user] needle topic duplicate",
                "duplicate_count": 100,
                "duplicate_rank": 10.0,
            },
            "unique": {
                "source": "codex",
                "project": None,
                "session_id": "unique-session",
                "ts": 1.0,
                "text": f"{shared_prefix} first tail",
                "parent_text": f"[user] {shared_prefix} first tail",
                "duplicate_count": 1,
                "duplicate_rank": 1.0,
            },
            "near_unique": {
                "source": "codex",
                "project": None,
                "session_id": "near-unique-session",
                "ts": 1.0,
                "text": f"{shared_prefix} second tail",
                "parent_text": f"[user] {shared_prefix} second tail",
                "duplicate_count": 1,
                "duplicate_rank": 1.0,
            },
        }

        with (
            patch.object(serve_api, "_get_retriever", return_value=fake),
            patch.object(serve_api, "_chunk_lookup", return_value=lookup),
        ):
            ctxs = serve_api.get_context("needle", top_k=2, retriever="colbert")

        self.assertGreater(fake.requested_top_k, 1)
        self.assertEqual(len(ctxs), 2)
        self.assertEqual(ctxs[0].chunk_id, "unique")
        self.assertEqual(ctxs[0].rank, 0)
        self.assertEqual(ctxs[0].score, 60.0)
        self.assertEqual(ctxs[1].chunk_id, "duplicate")
        self.assertEqual(ctxs[1].rank, 1)
        self.assertEqual(ctxs[1].score, 10.0)


if __name__ == "__main__":
    unittest.main()
