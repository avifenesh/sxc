"""Runtime API: `get_context(query)` returns ranked Context slabs.

This is the function called by the `itch-intent` skill (see
~/.claude/skills/itch-intent/) and any other consumer that wants
self-contained context for a topic phrase.

Default retriever is `hybrid` (SPLADE first-stage + ColBERTv2 rerank);
override via `SXC_RETRIEVER=bm25|splade|colbert|hybrid`.
"""

from .api import Context, get_context

__all__ = ["Context", "get_context"]
