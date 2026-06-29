"""sxc: local corpus retrieval over your own text.

A small, self-contained retrieval bench: index a corpus of text chunks with BM25,
SPLADE v3, or ColBERTv2, generate synthetic eval queries, benchmark the retrievers
against each other, and serve ranked context to a consumer (e.g. an LLM agent).

Module layout:
- sxc.chunk    — extract atomic chunks from raw jsonl + markdown sources
- sxc.synth    — generate eval query/answer pairs via an LLM (Anthropic API / Bedrock)
- sxc.index    — indexer implementations (bm25, splade, colbert, hybrid)
- sxc.bench    — eval harness (nDCG@10, recall@50, MRR, latency)
- sxc.serve    — runtime API: get_context(query) -> list[Context]
"""

__version__ = "0.0.1"
