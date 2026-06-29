"""Common retriever protocol + result types."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict


class Hit(BaseModel):
    """One ranked retrieval result."""

    model_config = ConfigDict(extra="forbid", strict=True)

    chunk_id: str
    score: float
    rank: int  # 0-based


@runtime_checkable
class Retriever(Protocol):
    """All indexers expose this interface for the bench harness."""

    name: str  # e.g. "bm25", "splade-v3", "colbert-v2", "hybrid"

    def search(self, query: str, top_k: int = 100) -> list[Hit]: ...
