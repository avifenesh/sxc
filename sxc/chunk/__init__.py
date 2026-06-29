"""Chunk extraction from heterogeneous transcript sources.

Each chunk is the smallest *retrievable* unit; each parent_text is the
self-contained *context* slab returned at query time.

Outputs one parquet at `data/chunks.parquet`.
"""

from .extract import build_chunks
from .schema import Chunk, ChunkSchema, Role, Source

__all__ = ["Chunk", "ChunkSchema", "Role", "Source", "build_chunks"]
