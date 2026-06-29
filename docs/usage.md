# Using sxc from a consumer

Once you've built an index, the only call a consumer needs is
`sxc.serve.get_context(query)`. This is the pattern to use it reliably from an agent or
service — see [`examples/query.py`](../examples/query.py) for the runnable version.

## The calling idiom

```python
from sxc.serve import get_context

results = get_context("retry backoff", top_k=5, retriever="colbert")
for c in results:
    print(c.score, c.source, c.project, c.text)
```

Each result is a `Context` with the matched chunk's `score`, its `source` and optional
`project` provenance, and `text` — a **self-contained parent window** around the match, so
the consumer can use it without stitching neighboring chunks together.

### Guard the import

`sxc`'s indexer extras (torch, transformers, colbert) are heavy and often live in a
dedicated virtualenv. A consumer that may run under a different interpreter should guard
the import and degrade to "no context" rather than crash:

```python
try:
    from sxc.serve import get_context
    SXC_AVAILABLE = True
except Exception:
    SXC_AVAILABLE = False

def retrieve(query, k=5):
    if not SXC_AVAILABLE:
        return []                      # caller falls back (e.g. plain keyword search)
    try:
        return get_context(query, top_k=k, retriever="colbert")
    except Exception:
        return []
```

Run the consumer with the interpreter that has sxc installed. If sxc lives outside that
interpreter's path, prepend it (`sys.path.insert(0, "/path/to/sxc")`) — `examples/query.py`
reads an `SXC_PATH` env var for exactly this.

### Pinning the retriever

`get_context` takes `retriever="bm25"|"splade"|"colbert"|"hybrid"`. You can also pin it
process-wide via `SXC_RETRIEVER`, or hard-force it with `SXC_FORCE_RETRIEVER` /
`config/serve.json` — useful to fail over to `bm25` (instant, no GPU) while a ColBERT index
rebuilds, without touching the consumer code.

## Choosing a retriever

Benchmark from the author's corpus (2k stratified synthetic gold pairs; `python -m sxc bench`
regenerates this on yours):

| retriever  | nDCG@10 | recall@50 | p50 latency |
|------------|--------:|----------:|------------:|
| colbert-v2 |   0.361 |     0.672 |        40ms |
| splade-v3  |   0.327 |     0.633 |       770ms |
| bm25       |   0.252 |     0.391 |         8ms |

- **colbert** — strongest ranking; cold-start loads the model to GPU once (~30s), then ~40ms/query. Good default.
- **bm25** — no ML deps, instant, no GPU; the dependable fallback and a fine baseline.
- **splade** — learned sparse; between the two on quality, slower at query time here.
- **hybrid** — SPLADE first-stage → ColBERT rerank, when you want recall + precision.

Numbers vary with corpus and chunking; treat the table as the *shape* of the tradeoff and
run `bench` on your own data to choose.
