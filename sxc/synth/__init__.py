"""Synth-query generation: chunk → eval queries via Sonnet 4.6 on Bedrock.

Outputs `data/synth_queries.parquet` with columns:
  query_id   str   stable hash(chunk_id, q_index)
  chunk_id   str   gold doc
  q          str   2-6 word topic phrase
  self_check bool  Sonnet's confidence flag
  why        str   one-sentence rationale (debug)
  source     str   chunk source (denormalised for eval-set stratification)
  project    str   chunk project, nullable
"""

from .generate import generate_queries
from .schema import SynthQuery, SynthQuerySchema

__all__ = ["SynthQuery", "SynthQuerySchema", "generate_queries"]
