"""Aggregate `data/eval/results.parquet` into a per-retriever report."""

from __future__ import annotations

import os
from pathlib import Path

import pyarrow.parquet as pq


def report(results_parquet: str | os.PathLike[str]) -> str:
    t = pq.read_table(str(results_parquet))
    df = t.to_pandas()
    out_lines: list[str] = []
    out_lines.append(f"queries scored: {df.query_id.nunique()}")
    out_lines.append("")
    cols = ["ndcg10", "recall_at_50", "rr", "latency_ms"]
    g = df.groupby("retriever")[cols].agg(["mean", "median"])
    out_lines.append(g.to_string())
    out_lines.append("")
    out_lines.append("per-source nDCG@10:")
    g2 = df.groupby(["retriever", "source"])["ndcg10"].mean().unstack()
    out_lines.append(g2.to_string())
    out_lines.append("")
    out_lines.append("p50 / p99 latency (ms):")
    p50 = df.groupby("retriever")["latency_ms"].quantile(0.5)
    p99 = df.groupby("retriever")["latency_ms"].quantile(0.99)
    p = p50.to_frame("p50").join(p99.to_frame("p99"))
    out_lines.append(p.to_string())
    return "\n".join(out_lines)


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("results", default=Path("data/eval/results.parquet"), type=Path, nargs="?")
    args = ap.parse_args(argv)
    print(report(args.results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
