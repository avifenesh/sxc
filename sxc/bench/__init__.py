"""Bench harness: run every retriever over the synth-query gold set,
record nDCG@10 / recall@K / MRR plus per-query latency, and write to
`data/eval/results.parquet`.

Each row is one (retriever, query) result. Aggregations (mean, p50/p99
per retriever) are computed at report time, not stored.
"""

from .run import run_bench
from .schema import BenchRow, BenchSchema

__all__ = ["BenchRow", "BenchSchema", "run_bench"]
