"""Hybrid SPLADE -> ColBERT cascade.

First-stage: SPLADE retrieves a deep candidate set (default top 500).
Second-stage: ColBERTv2 reranks the survivors, returning top_k.

Why cascade beats either alone:
- SPLADE's sparse inverted index is fast and recall-strong, but late-
  interaction precision is what tips MaxSim ahead on hard queries.
- ColBERTv2 end-to-end is high-quality but heavier per query;
  capping its work at the SPLADE survivors keeps latency budget sane.
"""

from __future__ import annotations

import os
from pathlib import Path

from .base import Hit
from .colbert import ColbertRetriever
from .splade import DEFAULT_MODEL as DEFAULT_SPLADE_MODEL
from .splade import SpladeRetriever


class HybridRetriever:
    """SPLADE first-stage, ColBERTv2 reranker."""

    name = "hybrid-splade-colbert"

    def __init__(
        self,
        splade_index: str | os.PathLike[str],
        colbert_index: str | os.PathLike[str],
        *,
        splade_model: str = DEFAULT_SPLADE_MODEL,
        colbert_checkpoint: str = "colbert-ir/colbertv2.0",
        first_stage_k: int = 500,
    ) -> None:
        self._splade = SpladeRetriever(Path(splade_index), model_id=splade_model)
        self._colbert = ColbertRetriever(
            Path(colbert_index), checkpoint=colbert_checkpoint
        )
        self.first_stage_k = first_stage_k

    def search(self, query: str, top_k: int = 100) -> list[Hit]:
        if not query.strip():
            return []
        # 1) SPLADE returns chunk_ids ranked by sparse dot product.
        first = self._splade.search(query, top_k=self.first_stage_k)
        if not first:
            return []
        # 2) ColBERT does full top_k retrieval; we then keep the
        #    intersection. ColBERT's Searcher does not expose a
        #    "rerank only this candidate set" API in colbert-ai 0.2.x,
        #    so we approximate by retrieving a generous pool from
        #    ColBERT and filtering to the SPLADE survivor ids.
        keep = {h.chunk_id for h in first}
        cb = self._colbert.search(query, top_k=max(top_k * 5, self.first_stage_k))
        # Re-rank by ColBERT score, restricted to SPLADE survivors.
        survivors = [h for h in cb if h.chunk_id in keep]
        if len(survivors) < top_k:
            # Fall back: append remaining first-stage hits not seen in CB,
            # so we always return up to top_k unique results.
            seen = {h.chunk_id for h in survivors}
            for h in first:
                if h.chunk_id not in seen:
                    survivors.append(h)
                    seen.add(h.chunk_id)
                    if len(survivors) >= top_k:
                        break
        survivors = survivors[:top_k]
        return [
            Hit(chunk_id=h.chunk_id, score=h.score, rank=rank)
            for rank, h in enumerate(survivors)
        ]

    def close(self) -> None:
        self._splade.close()
        self._colbert.close()

    def __enter__(self) -> HybridRetriever:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
