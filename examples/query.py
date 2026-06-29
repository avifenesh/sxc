#!/usr/bin/env python
"""Call sxc from a consumer (e.g. an LLM agent): retrieve ranked context for a query.

This is the production calling idiom — import sxc.serve, guard the import so the
caller degrades gracefully when the index isn't built, and shape each Context into a
plain dict the consumer can use.

Run it with the interpreter that has sxc + its deps installed (the project venv if you
used `pip install -e ".[index]"`):

    python examples/query.py "how does the retry backoff work?"

If sxc lives outside the running interpreter's import path, point at it explicitly:

    SXC_PATH=/path/to/sxc python examples/query.py "retry backoff"
"""

from __future__ import annotations

import os
import sys

# Make sxc importable when this script runs under an interpreter that didn't install it.
# Omit / unset SXC_PATH if sxc is already on sys.path (e.g. `pip install -e .`).
_sxc_path = os.environ.get("SXC_PATH")
if _sxc_path:
    sys.path.insert(0, _sxc_path)

try:
    from sxc.serve import get_context as _get_context
    SXC_AVAILABLE = True
except Exception as e:  # ImportError, or a missing optional dep at import time
    SXC_AVAILABLE = False
    _get_context = None
    print(f"[sxc unavailable: {type(e).__name__}: {e}]", file=sys.stderr)


def retrieve(query: str, k: int = 5, retriever: str = "colbert") -> list[dict]:
    """Return up to `k` ranked context slabs for `query`, as plain dicts.

    Returns an empty list if sxc is unavailable or the query errors — the caller
    should treat that as "no grounded context" and fall back accordingly, never crash.
    """
    if not SXC_AVAILABLE:
        return []
    try:
        results = _get_context(query, top_k=k, retriever=retriever)
    except Exception:
        return []
    return [
        {
            "source": c.source,      # which corpus the chunk came from
            "project": c.project,    # optional grouping label, may be None
            "score": c.score,
            "text": c.text,          # self-contained parent window
        }
        for c in results
    ]


def main() -> int:
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} <query> [top_k]", file=sys.stderr)
        return 2
    query = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    hits = retrieve(query, k=k)
    if not hits:
        print("(no context — index missing, sxc unavailable, or no matches)")
        return 0
    for h in hits:
        where = f"{h['source']}/{h['project']}" if h["project"] else h["source"]
        print(f"[{h['score']:.3f}] {where}: {h['text'][:160]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
