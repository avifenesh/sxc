"""Bench result schema."""

from __future__ import annotations

import pyarrow as pa
from pydantic import BaseModel, ConfigDict, Field


class BenchRow(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    retriever: str = Field(min_length=1)
    query_id: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)  # gold chunk for this query
    source: str = Field(min_length=1)
    self_check: bool
    rank_of_gold: int  # 1-based; -1 if not in top_k
    top_k: int
    ndcg10: float
    recall_at_50: float
    rr: float  # reciprocal rank, 0 if not found
    latency_ms: float


_FIELDS: list[pa.Field[pa.DataType]] = [
    pa.field("retriever", pa.string()),
    pa.field("query_id", pa.string()),
    pa.field("chunk_id", pa.string()),
    pa.field("source", pa.string()),
    pa.field("self_check", pa.bool_()),
    pa.field("rank_of_gold", pa.int32()),
    pa.field("top_k", pa.int32()),
    pa.field("ndcg10", pa.float64()),
    pa.field("recall_at_50", pa.float64()),
    pa.field("rr", pa.float64()),
    pa.field("latency_ms", pa.float64()),
]
BenchSchema: pa.Schema = pa.schema(_FIELDS)
