"""Runtime API: `get_context(query)` returns ranked Context slabs.

The entry point for any consumer (e.g. an LLM agent) that wants
self-contained context for a topic phrase.

Default retriever is `hybrid` (SPLADE first-stage + ColBERTv2 rerank);
override via `SXC_RETRIEVER=bm25|splade|colbert|hybrid`.
"""

from .api import Context, get_context

__all__ = ["Context", "get_context"]
