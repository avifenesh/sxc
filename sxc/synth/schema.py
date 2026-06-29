"""Synth-query schema. Pydantic boundary + arrow parquet."""

from __future__ import annotations

import pyarrow as pa
from pydantic import BaseModel, ConfigDict, Field


class SynthQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    query_id: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    q: str = Field(min_length=2, max_length=64)
    self_check: bool
    why: str = Field(min_length=1, max_length=512)
    source: str = Field(min_length=1)
    project: str | None = None


class _ModelQueryItem(BaseModel):
    """Sub-shape: one entry inside the model's `queries` list."""

    model_config = ConfigDict(extra="ignore", strict=True)
    q: str = Field(min_length=2, max_length=64)
    self_check: bool
    why: str = Field(min_length=1, max_length=512)


class ModelOutput(BaseModel):
    """Strict shape Sonnet must return per chunk."""

    model_config = ConfigDict(extra="ignore", strict=True)
    queries: list[_ModelQueryItem] = Field(max_length=3)


_FIELDS: list[pa.Field[pa.DataType]] = [
    pa.field("query_id", pa.string()),
    pa.field("chunk_id", pa.string()),
    pa.field("q", pa.string()),
    pa.field("self_check", pa.bool_()),
    pa.field("why", pa.string()),
    pa.field("source", pa.string()),
    pa.field("project", pa.string()),
]
SynthQuerySchema: pa.Schema = pa.schema(_FIELDS)
