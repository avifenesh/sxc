"""Indexer implementations.

All indexers conform to the same `Retriever` protocol so the bench harness
treats them uniformly.

Implemented:
- bm25     SQLite FTS5 over chunks.text (baseline, zero-deps beyond stdlib).
- splade   SPLADE-v3 sparse vectors over an inverted index.
- colbert  ColBERTv2 via colbert-ai (the primary serving retriever).
- hybrid   SPLADE first-stage → ColBERT rerank cascade (bench-only).
"""

from .base import Hit, Retriever

__all__ = ["Hit", "Retriever"]
