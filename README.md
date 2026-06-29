# sxc

Local corpus retrieval over your own text — **BM25**, **SPLADE v3**, and **ColBERTv2** behind one Python API, with synthetic evaluation and an optional relevance judge.

`sxc` indexes a corpus of text chunks three different ways, lets you benchmark the retrievers against each other on queries it generates for you, and serves ranked context to a consumer — typically an LLM agent that needs grounded snippets. It is a small, dependency-light library you point at your own data; nothing is bundled and nothing leaves your machine unless you call out to an LLM for synthetic-query generation.

```bash
pip install -e .            # core (BM25 + serving + eval)
pip install -e ".[index]"   # add SPLADE + ColBERT indexers (torch, transformers, ragatouille)
```

```python
from sxc.serve import get_context

# after you've built an index (see Quick start), retrieve ranked context for a query:
for ctx in get_context("how does the retry backoff work?", top_k=5):
    print(f"[{ctx.score:.3f}] {ctx.source}/{ctx.project}: {ctx.text[:120]}")
```

## Why this project

- **Use it when** you want grounded context for an LLM agent over your own notes, transcripts, or docs — without standing up a vector DB.
- **Use it when** you need to *compare* sparse (BM25/SPLADE) vs late-interaction (ColBERT) retrieval on your data instead of guessing which wins.
- **Use it when** you have no labeled eval set: `sxc synth` generates query/answer pairs from your corpus so you can measure nDCG/recall/MRR honestly.
- **Use it when** you want a single `get_context(query)` call your agent can depend on, with the retriever swappable behind config.

## Installation

Requires Python ≥ 3.11.

```bash
git clone <your-fork-url> sxc && cd sxc
python -m venv .venv && source .venv/bin/activate
pip install -e ".[index]"      # omit [index] if you only need BM25 + serving
```

The SPLADE and ColBERT checkpoints (`naver/splade-cocondenser-ensembledistil`,
`colbert-ir/colbertv2.0`) download automatically from Hugging Face on first use.

## Quick start

`sxc` works off a single `data/chunks.parquet` of text chunks. The full pipeline is one
command per stage:

```bash
# 1. build chunks from your raw sources into data/chunks.parquet
python -m sxc chunk --data ./data --out data/chunks.parquet

# 2. build an index (colbert is the strongest default; bm25 is instant + dependency-free)
python -m sxc index-bm25                     # -> indexes/bm25
python -m sxc index-colbert                  # -> indexes/colbert   (needs [index] extras)

# 3. query it
python -m sxc search-colbert "retry backoff" --top-k 5
python -m sxc serve "retry backoff"          # uses the configured default retriever
```

From Python:

```python
from sxc.serve import get_context, Context

results: list[Context] = get_context("retry backoff", top_k=5, retriever="colbert")
for ctx in results:
    print(ctx.score, ctx.source, ctx.text)
```

**What happened:** `chunk` turned raw sources into atomic chunks; `index-*` built a
retrieval index over them; `get_context` ran the query through the chosen retriever,
de-duplicated near-identical hits, and returned self-contained text windows ranked by
score.

## Core concepts

- **Chunk** — an atomic unit of text with a parent window (the self-contained slab returned
  at serve time), a `source` tag, and an optional `project`. `sxc chunk` produces them.
- **Retriever** — one of `bm25` (SQLite FTS5, zero ML deps), `splade` (learned sparse),
  `colbert` (late-interaction, strongest), or `hybrid` (SPLADE first-stage → ColBERT rerank).
  All implement the same `search(query, top_k)` protocol.
- **Synthetic eval** — `sxc synth` asks an LLM to generate queries whose answer is a known
  chunk, giving you gold pairs to benchmark against with no manual labeling.
- **Serve** — `get_context(query, top_k, retriever)` is the one call a consumer needs; the
  retriever can be pinned via env/config so you can fail over (e.g. to BM25 mid-rebuild)
  without touching the consumer.

## CLI

| command | does |
| --- | --- |
| `chunk` | walk raw sources → `data/chunks.parquet` |
| `inspect <parquet>` | describe a chunks parquet |
| `synth` | generate eval query/answer pairs via an LLM (Anthropic API / Bedrock) |
| `index-bm25` / `index-splade` / `index-colbert` | build an index |
| `search-bm25` / `search-splade` / `search-colbert` | query one index |
| `bench` | run all retrievers over the synth gold pairs (nDCG@10, recall@50, MRR, latency) |
| `bench-report` | summarize `results.parquet` |
| `serve <query>` | one-shot query against the configured retriever |

## Configuration

All runtime knobs are environment variables with safe defaults:

| var | default | purpose |
| --- | --- | --- |
| `SXC_DATA_ROOT` | inferred | corpus/index root |
| `SXC_RETRIEVER` | `colbert` | default retriever for `get_context` |
| `SXC_FORCE_RETRIEVER` | — | hard-override the retriever (or via `config/serve.json`) — used to fail over to `bm25` while another index rebuilds |
| `SXC_JUDGE` | `1` | enable the optional live relevance judge |
| `SXC_JUDGE_URL` | `http://127.0.0.1:8003/v1` | OpenAI-compatible endpoint for the judge model |
| `SXC_SYNTH_BASE_URL` / `SXC_SYNTH_API_KEY` | Bedrock native | override the LLM used for `synth` |
| `AWS_REGION` | `eu-west-1` | Bedrock region for `synth` |

The optional **judge** (`sxc.serve.judge`) is fail-open: if no judge server is reachable it
is skipped silently, so serving never hard-depends on it.

## Limitations / tradeoffs

- **You bring the corpus.** `sxc chunk` expects raw sources you supply; no data ships with the library.
- **ColBERT/SPLADE need the `[index]` extras** (torch + transformers) and a GPU is strongly recommended for indexing at scale. BM25 needs neither.
- **`synth` calls an LLM** (Anthropic API or Amazon Bedrock by default) — that stage needs credentials and network; every other stage is fully local.
- **Built for application-integration retrieval**, not as a general-purpose analytics or vector-search platform.
- The chunker ships with parsers for common agent-transcript and markdown layouts; other corpora may need a small parser added to `sxc/chunk/extract.py`.

## Contributing

Issues and PRs welcome. Run the tests and linters before submitting:

```bash
pip install -e ".[dev]"
pytest
ruff check . && mypy sxc
```

## License

MIT — see [LICENSE](LICENSE).
