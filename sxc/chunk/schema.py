"""Chunk schema. Pydantic for boundary validation; pyarrow for parquet I/O."""

from __future__ import annotations

from typing import Literal

import pyarrow as pa
from pydantic import BaseModel, ConfigDict, Field

Source = Literal[
    "claude",
    "codex",
    "hermes",
    "eigen",
    "readme",
    "memory",  # eigen + hermes curated/dreamed memory (MEMORY.md, raw_memories.md, …)
    "skill",   # eigen + hermes skill catalog (SKILL.md, skills-proposed)
    "paper",   # long-form research/knowledge syntheses (markdown)
    "surreal", # revuto reviewer namespace tables (concerns + bookkeeping)
]
Role = Literal["user", "assistant", "tool", "doc"]


class Chunk(BaseModel):
    """One retrievable unit + its parent context slab.

    `text` is what indexers tokenize / encode.
    `parent_text` is what we return at query time.
    """

    model_config = ConfigDict(extra="forbid", frozen=False, strict=True)

    chunk_id: str = Field(min_length=1)
    source: Source
    session_id: str
    project: str | None = None
    ts: float | None = None
    role: Role
    text: str = Field(min_length=1)
    parent_text: str = Field(min_length=1)
    prev_chunk_id: str | None = None
    next_chunk_id: str | None = None
    dedupe_key: str = Field(default="unknown", min_length=1)
    duplicate_count: int = Field(default=1, ge=1)
    duplicate_rank: float = Field(default=1.0, ge=1.0)
    duplicate_sources: str | None = None
    duplicate_projects: str | None = None
    duplicate_session_ids: str | None = None


_FIELDS: list[pa.Field[pa.DataType]] = [
    pa.field("chunk_id", pa.string()),
    pa.field("source", pa.string()),
    pa.field("session_id", pa.string()),
    pa.field("project", pa.string()),
    pa.field("ts", pa.float64()),
    pa.field("role", pa.string()),
    pa.field("text", pa.string()),
    pa.field("parent_text", pa.string()),
    pa.field("prev_chunk_id", pa.string()),
    pa.field("next_chunk_id", pa.string()),
    pa.field("dedupe_key", pa.string()),
    pa.field("duplicate_count", pa.int32()),
    pa.field("duplicate_rank", pa.float32()),
    pa.field("duplicate_sources", pa.string()),
    pa.field("duplicate_projects", pa.string()),
    pa.field("duplicate_session_ids", pa.string()),
]
ChunkSchema: pa.Schema = pa.schema(_FIELDS)
