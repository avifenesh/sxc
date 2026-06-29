"""Top-level CLI: `python -m sxc <subcmd> ...`."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path


def _cmd_chunk(args: argparse.Namespace) -> int:
    from sxc.chunk import build_chunks

    n = build_chunks(args.data, args.out, batch_size=args.batch)
    print(f"wrote {n} chunks → {args.out}")
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    import pyarrow.parquet as pq

    t = pq.read_table(args.parquet)
    print(f"rows: {t.num_rows}")
    print(f"columns: {t.column_names}")
    print()
    print("by source:")
    src: list[str] = [s for s in t.column("source").to_pylist() if isinstance(s, str)]
    for s, c in Counter(src).most_common():
        print(f"  {s:8s} {c:>8d}")
    print()
    print("sample chunk text (first 3):")
    for i in range(min(3, t.num_rows)):
        row = t.slice(i, 1).to_pylist()[0]
        print(
            f"--- {row['source']}/{row['session_id'][:16]} "
            f"(project={row['project']}, role={row['role']}) ---"
        )
        print(row["text"][:400])
        print()
    return 0


def _cmd_synth(args: argparse.Namespace) -> int:
    from sxc.synth import generate_queries

    n_in, n_out, dt = generate_queries(
        args.chunks,
        args.out,
        model_id=args.model,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        log_path=args.log,
        limit=args.limit,
    )
    print(f"chunks_processed={n_in}  queries_emitted={n_out}  elapsed={dt:.1f}s")
    return 0


def _cmd_index_bm25(args: argparse.Namespace) -> int:
    from sxc.index.bm25 import build

    n = build(args.chunks, args.out)
    print(f"bm25 indexed {n} rows → {args.out}")
    return 0


def _cmd_search_bm25(args: argparse.Namespace) -> int:
    from sxc.index.bm25 import BM25Retriever

    with BM25Retriever(args.index) as r:
        hits = r.search(args.query, top_k=args.top_k)
    for h in hits:
        print(f"  rank={h.rank:>3d}  score={h.score:8.4f}  chunk_id={h.chunk_id}")
    print(f"({len(hits)} hits)")
    return 0


def _cmd_index_splade(args: argparse.Namespace) -> int:
    from sxc.index.splade import build

    n = build(
        args.chunks,
        args.out,
        model_id=args.model,
        batch=args.batch,
        max_len=args.max_len,
    )
    print(f"splade indexed {n} rows → {args.out}")
    return 0


def _cmd_search_splade(args: argparse.Namespace) -> int:
    from sxc.index.splade import SpladeRetriever

    with SpladeRetriever(args.index, model_id=args.model) as r:
        hits = r.search(args.query, top_k=args.top_k)
    for h in hits:
        print(f"  rank={h.rank:>3d}  score={h.score:8.4f}  chunk_id={h.chunk_id}")
    print(f"({len(hits)} hits)")
    return 0


def _cmd_index_colbert(args: argparse.Namespace) -> int:
    from sxc.index.colbert import build

    n = build(
        args.chunks,
        args.out,
        checkpoint=args.checkpoint,
        doc_maxlen=args.doc_maxlen,
        nbits=args.nbits,
        kmeans_niters=args.kmeans_niters,
    )
    print(f"colbert indexed {n} rows → {args.out}")
    return 0


def _cmd_search_colbert(args: argparse.Namespace) -> int:
    from sxc.index.colbert import ColbertRetriever

    with ColbertRetriever(args.index, checkpoint=args.checkpoint) as r:
        hits = r.search(args.query, top_k=args.top_k)
    for h in hits:
        print(f"  rank={h.rank:>3d}  score={h.score:8.4f}  chunk_id={h.chunk_id}")
    print(f"({len(hits)} hits)")
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    from sxc.bench import run_bench
    from sxc.index.base import Retriever

    wanted = {n.strip() for n in args.retrievers.split(",") if n.strip()}
    retrievers: dict[str, Retriever] = {}
    if "bm25" in wanted:
        from sxc.index.bm25 import BM25Retriever

        retrievers["bm25"] = BM25Retriever(args.bm25_index)
    if "splade" in wanted:
        from sxc.index.splade import SpladeRetriever

        retrievers["splade-v3"] = SpladeRetriever(args.splade_index)
    if "splade-doc" in wanted:
        # SPLADE-v3 Doc-family checkpoint (document-side SPLADE-v3). The
        # canonical naver/splade-v3-doc repo is gated; the open Doc-family
        # member naver/splade-v3-distilbert is the default and exercises the
        # same encoder path. Override with --splade-doc-model if you have
        # token access to a gated checkpoint.
        from sxc.index.splade import SpladeRetriever

        retrievers["splade-v3-doc"] = SpladeRetriever(
            args.splade_doc_index, model_id=args.splade_doc_model
        )
    if "colbert" in wanted:
        from sxc.index.colbert import ColbertRetriever

        retrievers["colbert-v2"] = ColbertRetriever(args.colbert_index)
    if "hybrid" in wanted:
        from sxc.index.hybrid import HybridRetriever

        retrievers["hybrid"] = HybridRetriever(args.splade_index, args.colbert_index)

    n = run_bench(
        args.synth,
        args.out,
        retrievers,
        top_k=args.top_k,
        only_self_check=not args.include_low_conf,
        limit=args.limit,
    )
    print(f"bench wrote {n} rows → {args.out}")
    return 0


def _cmd_bench_report(args: argparse.Namespace) -> int:
    from sxc.bench.report import report

    print(report(args.results))
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    from sxc.serve import get_context

    ctxs = get_context(args.query, top_k=args.top_k, retriever=args.retriever)
    for c in ctxs:
        print(
            f"--- rank={c.rank}  score={c.score:.4f}  source={c.source}/{c.session_id[:16]}  "
            f"project={c.project} ---"
        )
        print(c.quote[:600])
        print()
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="sxc")
    sub = p.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("chunk", help="extract chunks from data/")
    pc.add_argument("--data", default=Path("data"), type=Path)
    pc.add_argument("--out", default=Path("data/chunks.parquet"), type=Path)
    pc.add_argument("--batch", default=5000, type=int)
    pc.set_defaults(func=_cmd_chunk)

    pi = sub.add_parser("inspect", help="describe a chunks parquet")
    pi.add_argument("parquet", type=Path)
    pi.set_defaults(func=_cmd_inspect)

    ps = sub.add_parser("synth", help="generate eval queries via Sonnet on Bedrock")
    ps.add_argument("--chunks", default=Path("data/chunks.parquet"), type=Path)
    ps.add_argument("--out", default=Path("data/synth_queries.parquet"), type=Path)
    ps.add_argument(
        "--model", default="eu.anthropic.claude-sonnet-4-5-20250929-v1:0", type=str
    )
    ps.add_argument("--concurrency", default=64, type=int)
    ps.add_argument("--max-tokens", default=512, type=int)
    ps.add_argument("--temperature", default=0.2, type=float)
    ps.add_argument("--log", default=Path("data/synth_errors.jsonl"), type=Path)
    ps.add_argument("--limit", default=None, type=int)
    ps.set_defaults(func=_cmd_synth)

    pb = sub.add_parser("index-bm25", help="build BM25 (SQLite FTS5) index")
    pb.add_argument("--chunks", default=Path("data/chunks.parquet"), type=Path)
    pb.add_argument("--out", default=Path("indexes/bm25"), type=Path)
    pb.set_defaults(func=_cmd_index_bm25)

    pq = sub.add_parser("search-bm25", help="run a query against the BM25 index")
    pq.add_argument("--index", default=Path("indexes/bm25"), type=Path)
    pq.add_argument("--top-k", default=20, type=int)
    pq.add_argument("query", type=str)
    pq.set_defaults(func=_cmd_search_bm25)

    psp = sub.add_parser("index-splade", help="build SPLADE v3 sparse index")
    psp.add_argument("--chunks", default=Path("data/chunks.parquet"), type=Path)
    psp.add_argument("--out", default=Path("indexes/splade"), type=Path)
    psp.add_argument("--model", default="naver/splade-cocondenser-ensembledistil", type=str)
    psp.add_argument("--batch", default=32, type=int)
    psp.add_argument("--max-len", default=256, type=int)
    psp.set_defaults(func=_cmd_index_splade)

    pss = sub.add_parser("search-splade", help="run a query against the SPLADE index")
    pss.add_argument("--index", default=Path("indexes/splade"), type=Path)
    pss.add_argument("--model", default="naver/splade-cocondenser-ensembledistil", type=str)
    pss.add_argument("--top-k", default=20, type=int)
    pss.add_argument("query", type=str)
    pss.set_defaults(func=_cmd_search_splade)

    pcb = sub.add_parser("index-colbert", help="build ColBERTv2 index (primary retriever)")
    pcb.add_argument("--chunks", default=Path("data/chunks.parquet"), type=Path)
    pcb.add_argument("--out", default=Path("indexes/colbert"), type=Path)
    pcb.add_argument("--checkpoint", default="colbert-ir/colbertv2.0", type=str)
    pcb.add_argument("--doc-maxlen", default=300, type=int)
    pcb.add_argument("--nbits", default=2, type=int)
    pcb.add_argument("--kmeans-niters", default=4, type=int,
                     help="centroid-training iterations (dominant build cost on "
                          "faiss-CPU; lower = much faster rebuild, small recall cost)")
    pcb.set_defaults(func=_cmd_index_colbert)

    pcs = sub.add_parser("search-colbert", help="run a query against the ColBERT index")
    pcs.add_argument("--index", default=Path("indexes/colbert"), type=Path)
    pcs.add_argument("--checkpoint", default="colbert-ir/colbertv2.0", type=str)
    pcs.add_argument("--top-k", default=20, type=int)
    pcs.add_argument("query", type=str)
    pcs.set_defaults(func=_cmd_search_colbert)

    pbench = sub.add_parser("bench", help="run all retrievers over synth gold pairs")
    pbench.add_argument("--synth", default=Path("data/synth_queries.parquet"), type=Path)
    pbench.add_argument("--out", default=Path("data/eval/results.parquet"), type=Path)
    pbench.add_argument("--bm25-index", default=Path("indexes/bm25"), type=Path)
    pbench.add_argument("--splade-index", default=Path("indexes/splade"), type=Path)
    pbench.add_argument("--splade-doc-index", default=Path("indexes/splade-doc"), type=Path,
                        help="index dir for the SPLADE-v3 Doc-family checkpoint (retriever "
                             "'splade-doc')")
    pbench.add_argument("--splade-doc-model", default="naver/splade-v3-distilbert", type=str,
                        help="SPLADE-v3 Doc-family checkpoint id; naver/splade-v3-doc is gated, "
                             "so the open naver/splade-v3-distilbert is the default")
    pbench.add_argument("--colbert-index", default=Path("indexes/colbert"), type=Path)
    pbench.add_argument("--top-k", default=100, type=int)
    pbench.add_argument("--limit", default=None, type=int)
    pbench.add_argument("--include-low-conf", action="store_true",
                        help="include synth rows with self_check=false")
    pbench.add_argument("--retrievers", default="bm25,splade,colbert,hybrid",
                        help="comma-separated subset of "
                             "bm25,splade,splade-doc,colbert,hybrid")
    pbench.set_defaults(func=_cmd_bench)

    prep = sub.add_parser("bench-report", help="summarise results.parquet")
    prep.add_argument("results", default=Path("data/eval/results.parquet"), type=Path, nargs="?")
    prep.set_defaults(func=_cmd_bench_report)

    psv = sub.add_parser("serve", help="one-shot: query the configured retriever")
    psv.add_argument("--retriever", default=None, choices=["bm25", "splade", "colbert", "hybrid"])
    psv.add_argument("--top-k", default=5, type=int)
    psv.add_argument("query", type=str)
    psv.set_defaults(func=_cmd_serve)

    args = p.parse_args(argv)
    func = args.func
    return int(func(args))


if __name__ == "__main__":
    sys.exit(main())
