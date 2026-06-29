"""Run Sonnet 4.6 over chunks via Bedrock with cached system prompt and
parallel in-flight requests.

Design:
  - System prompt loaded from prompt.md once; passed with cache_control.
  - User content per call: the chunk's text + parent_text, role-tagged.
  - Parallelism via asyncio.gather over a Semaphore.
  - Retries with exponential backoff for transient errors.
  - Output streamed to parquet in batches; never holds all queries in RAM.
  - Resumes: skips chunk_ids already present in the output parquet.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import orjson
import pyarrow as pa
import pyarrow.parquet as pq
from anthropic import AsyncAnthropicBedrock
from anthropic.types import MessageParam, TextBlock, TextBlockParam
from pydantic import ValidationError
from tqdm.asyncio import tqdm as atqdm

from sxc.chunk.schema import Chunk

from .schema import ModelOutput, SynthQuery, SynthQuerySchema

PROMPT_PATH = Path(__file__).parent / "prompt.md"
DEFAULT_MODEL_ID = "eu.anthropic.claude-sonnet-4-5-20250929-v1:0"
DEFAULT_AWS_REGION = "eu-west-1"
# ^ Bedrock cross-region inference profile. Sonnet 4.6 has no profile in
#   the user's account; Sonnet 4.5 is the closest active family member
#   reachable via on-demand chat. Override with --model / AWS_REGION.

DEFAULT_MAX_TOKENS = 512
DEFAULT_TEMPERATURE = 0.2
DEFAULT_CONCURRENCY = 64
RETRY_ATTEMPTS = 4
RETRY_BASE_DELAY = 1.5

# Bedrock error classes that will never succeed on retry with the same input.
# `BadRequestError` (HTTP 400, e.g. "Input is too long for requested model")
# is deterministic per chunk: re-calling burns quota for the same failure every
# run forever. We sentinel these so resume's skip-set treats them as done.
# Transient classes (RateLimitError 429, json_decode, schema_invalid) are NOT
# listed -- those legitimately retry and usually pass on a later, calmer run.
PERMANENT_ERROR_PREFIXES = ("BadRequestError:",)


def _is_permanent_error(err: str | None) -> bool:
    return err is not None and err.startswith(PERMANENT_ERROR_PREFIXES)


# ----- helpers ----------------------------------------------------------------


def _load_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def _qid(chunk_id: str, idx: int) -> str:
    h = hashlib.blake2b(digest_size=10)
    h.update(chunk_id.encode("ascii"))
    h.update(b":")
    h.update(str(idx).encode("ascii"))
    return h.hexdigest()


def _user_payload(chunk_text: str, parent_text: str) -> str:
    """Single user message body. Parent first as background, chunk last
    so the model's attention biases toward the labeled unit."""
    return (
        "PARENT_WINDOW (context only, do not label):\n"
        "----------------------------------------\n"
        f"{parent_text}\n"
        "----------------------------------------\n\n"
        "CHUNK (label this):\n"
        "----------------------------------------\n"
        f"{chunk_text}\n"
        "----------------------------------------"
    )


# ----- input streaming --------------------------------------------------------


def _iter_chunks(parquet_path: Path, *, skip_ids: set[str]) -> Iterator[Chunk]:
    """Stream chunks from parquet, skipping any whose id is already done."""
    pf = pq.ParquetFile(parquet_path)
    cols = ["chunk_id", "source", "session_id", "project", "ts", "role", "text", "parent_text"]
    for batch in pf.iter_batches(batch_size=2048, columns=cols):
        d = batch.to_pydict()
        n = len(d["chunk_id"])
        for i in range(n):
            cid = d["chunk_id"][i]
            if cid in skip_ids:
                continue
            yield Chunk(
                chunk_id=cid,
                source=d["source"][i],
                session_id=d["session_id"][i],
                project=d["project"][i],
                ts=d["ts"][i],
                role=d["role"][i],
                text=d["text"][i],
                parent_text=d["parent_text"][i],
                prev_chunk_id=None,
                next_chunk_id=None,
            )


def _existing_ids(out: Path) -> set[str]:
    if not out.exists():
        return set()
    try:
        t = pq.read_table(out, columns=["chunk_id"])
    except (OSError, pa.lib.ArrowInvalid):
        return set()
    return {x for x in t.column("chunk_id").to_pylist() if isinstance(x, str)}


# ----- Bedrock call -----------------------------------------------------------


def _system_blocks(prompt: str) -> list[TextBlockParam]:
    """System prompt as a single cached text block."""
    return [
        {
            "type": "text",
            "text": prompt,
            "cache_control": {"type": "ephemeral"},
        }
    ]


def _parse_model_output(raw: str) -> tuple[ModelOutput | None, str | None]:
    """Shared post-processing for BOTH backends (Bedrock and local OpenAI).

    The model is asked to emit bare JSON; this strips accidental ```json
    fences, parses, and validates against the pydantic schema. Returns
    (parsed, None) on success or (None, error_string) on failure.
    """
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    try:
        obj: Any = json.loads(raw)
    except json.JSONDecodeError as e:
        return None, f"json_decode: {e.msg}"
    try:
        return ModelOutput.model_validate(obj), None
    except ValidationError as e:
        return None, f"schema_invalid: {e.errors()[0]['msg']}"


async def _call_one(
    client: AsyncAnthropicBedrock,
    *,
    model_id: str,
    system_blocks: list[TextBlockParam],
    chunk: Chunk,
    max_tokens: int,
    temperature: float,
) -> tuple[Chunk, ModelOutput | None, str | None]:
    """Bedrock path. Returns (chunk, parsed_output|None, error|None). Never raises."""
    body = _user_payload(chunk.text, chunk.parent_text)
    messages: list[MessageParam] = [{"role": "user", "content": body}]
    last_err: str | None = None
    for attempt in range(RETRY_ATTEMPTS):
        try:
            resp = await client.messages.create(
                model=model_id,
                system=system_blocks,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            text_parts = [blk.text for blk in resp.content if isinstance(blk, TextBlock)]
            parsed, err = _parse_model_output("".join(text_parts))
            return chunk, parsed, err
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            if attempt + 1 < RETRY_ATTEMPTS:
                await asyncio.sleep(
                    RETRY_BASE_DELAY * (2**attempt) + random.uniform(0, 0.5)
                )
            else:
                return chunk, None, last_err
    return chunk, None, last_err


async def _call_one_openai(
    http: "httpx.AsyncClient",
    *,
    base_url: str,
    api_key: str,
    model_id: str,
    system_text: str,
    chunk: Chunk,
    max_tokens: int,
    temperature: float,
) -> tuple[Chunk, ModelOutput | None, str | None]:
    """Local/OpenAI-compatible path (e.g. llama-server `/v1`).

    Same contract as `_call_one`: (chunk, parsed|None, error|None), never
    raises. Speaks plain `POST {base}/chat/completions` via httpx — no
    `openai` package needed, and llama.cpp ignores Anthropic-only concepts
    like cache_control, so the system prompt goes in as one plain message.
    """
    body = _user_payload(chunk.text, chunk.parent_text)
    payload = {
        "model": model_id,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system_text},
            {"role": "user", "content": body},
        ],
    }
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    last_err: str | None = None
    for attempt in range(RETRY_ATTEMPTS):
        try:
            resp = await http.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"] or ""
            parsed, err = _parse_model_output(raw)
            return chunk, parsed, err
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            if attempt + 1 < RETRY_ATTEMPTS:
                await asyncio.sleep(
                    RETRY_BASE_DELAY * (2**attempt) + random.uniform(0, 0.5)
                )
            else:
                return chunk, None, last_err
    return chunk, None, last_err


# ----- output ------------------------------------------------------------------


def _to_arrow_batch(rows: list[SynthQuery]) -> pa.RecordBatch:
    cols: dict[str, list[Any]] = {f.name: [] for f in SynthQuerySchema}
    for r in rows:
        cols["query_id"].append(r.query_id)
        cols["chunk_id"].append(r.chunk_id)
        cols["q"].append(r.q)
        cols["self_check"].append(r.self_check)
        cols["why"].append(r.why)
        cols["source"].append(r.source)
        cols["project"].append(r.project)
    return pa.RecordBatch.from_pydict(cols, schema=SynthQuerySchema)


# ----- top-level orchestration -------------------------------------------------


async def _run_async(
    *,
    chunks_parquet: Path,
    out_parquet: Path,
    model_id: str,
    concurrency: int,
    max_tokens: int,
    temperature: float,
    log_path: Path | None,
    limit: int | None,
) -> tuple[int, int]:
    """Returns (n_chunks_processed, n_queries_emitted).

    Crash-safe storage strategy: write each query as one line to an
    append-only NDJSON sidecar (`<out>.ndjson`), `fsync` per flush. Once
    the run finishes cleanly, convert the NDJSON to a single parquet at
    `out_parquet`. If the process is killed mid-run, the NDJSON survives
    and a re-run picks up where it stopped (by chunk_id).
    """
    prompt = _load_prompt()
    system_blocks = _system_blocks(prompt)

    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    ndjson_path = out_parquet.with_suffix(".ndjson")

    skip: set[str] = set()
    if ndjson_path.exists():
        # Resume: collect every chunk_id we've already emitted at least one
        # query for. (A chunk that produced zero queries gets re-run; that's
        # acceptable noise compared to the cost of tracking empty results.)
        with ndjson_path.open("rb") as f:
            for raw in f:
                try:
                    obj = orjson.loads(raw)
                except orjson.JSONDecodeError:
                    continue
                cid = obj.get("chunk_id")
                if isinstance(cid, str):
                    skip.add(cid)
    if out_parquet.exists():
        skip.update(_existing_ids(out_parquet))

    nd_fh = ndjson_path.open("ab")
    log_fh = log_path.open("a", encoding="utf-8") if log_path else None

    sem = asyncio.Semaphore(concurrency)

    # Backend dispatch, env-gated:
    #   SXC_SYNTH_BASE_URL set   -> local OpenAI-compatible server (llama-server
    #                               et al). Model id + key via SXC_SYNTH_MODEL /
    #                               SXC_SYNTH_API_KEY. Free + GPU-idle-time use.
    #   SXC_SYNTH_BASE_URL unset -> Bedrock (existing behavior, untouched).
    base_url = os.environ.get("SXC_SYNTH_BASE_URL")
    if base_url:
        import httpx

        http = httpx.AsyncClient(timeout=120.0)
        api_key = os.environ.get("SXC_SYNTH_API_KEY", "none")
        # system_blocks is Anthropic-shaped ([{type, text, cache_control}]);
        # the OpenAI path just needs the raw prompt text.
        system_text = "\n".join(b["text"] for b in system_blocks)
        client = None
    else:
        aws_region = os.environ.get("AWS_REGION") or DEFAULT_AWS_REGION
        client = AsyncAnthropicBedrock(aws_region=aws_region)

    n_in = 0
    n_out = 0
    flush_every = 256
    since_flush = 0

    async def _bound(c: Chunk) -> tuple[Chunk, ModelOutput | None, str | None]:
        async with sem:
            if base_url:
                return await _call_one_openai(
                    http,
                    base_url=base_url,
                    api_key=api_key,
                    model_id=model_id,
                    system_text=system_text,
                    chunk=c,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            return await _call_one(
                client,
                model_id=model_id,
                system_blocks=system_blocks,
                chunk=c,
                max_tokens=max_tokens,
                temperature=temperature,
            )

    iterator = _iter_chunks(chunks_parquet, skip_ids=skip)
    if limit is not None:
        iterator = (c for i, c in enumerate(iterator) if i < limit)

    pending: set[asyncio.Task[tuple[Chunk, ModelOutput | None, str | None]]] = set()
    progress = atqdm(desc="synth", unit="chunk")

    def _emit(rows: list[SynthQuery]) -> None:
        nonlocal since_flush
        for r in rows:
            nd_fh.write(orjson.dumps(r.model_dump()) + b"\n")
        since_flush += len(rows)
        if since_flush >= flush_every:
            nd_fh.flush()
            os.fsync(nd_fh.fileno())
            since_flush = 0

    def _emit_sentinel(chunk_id: str) -> None:
        # "done, no queries" marker for resume. Shape is deliberately NOT a
        # SynthQuery (chunk_id only) so the parquet compaction step drops it.
        nonlocal since_flush
        nd_fh.write(orjson.dumps({"chunk_id": chunk_id}) + b"\n")
        since_flush += 1
        if since_flush >= flush_every:
            nd_fh.flush()
            os.fsync(nd_fh.fileno())
            since_flush = 0

    def _record(chunk: Chunk, parsed: ModelOutput | None, err: str | None) -> None:
        nonlocal n_out
        progress.update(1)
        if err is not None or parsed is None:
            if log_fh:
                log_fh.write(
                    orjson.dumps({"chunk_id": chunk.chunk_id, "err": err}).decode() + "\n"
                )
            # Permanent failures (e.g. chunk too long for the model) can never
            # succeed on retry -- sentinel them so resume stops re-calling and
            # re-burning quota every run. Transient errors get NO sentinel:
            # they fall through to be retried on the next run.
            if _is_permanent_error(err):
                _emit_sentinel(chunk.chunk_id)
            return
        rows: list[SynthQuery] = []
        for idx, item in enumerate(parsed.queries):
            q_clean = item.q.strip().lower()
            try:
                rows.append(
                    SynthQuery(
                        query_id=_qid(chunk.chunk_id, idx),
                        chunk_id=chunk.chunk_id,
                        q=q_clean,
                        self_check=item.self_check,
                        why=item.why,
                        source=chunk.source,
                        project=chunk.project,
                    )
                )
            except ValidationError as e:
                if log_fh:
                    log_fh.write(
                        orjson.dumps(
                            {
                                "chunk_id": chunk.chunk_id,
                                "err": f"row_invalid: {e.errors()[0]['msg']}",
                            }
                        ).decode()
                        + "\n"
                    )
        if rows:
            _emit(rows)
            n_out += len(rows)
        else:
            # Chunk was attempted and parsed OK but produced no usable query
            # (model returned an empty list, or every row failed validation).
            # Without a record here, resume's skip-set never learns this chunk
            # is done, so every later run re-calls the model on it for nothing.
            # Write a chunk_id-only sentinel: resume skips it (skip-set keys on
            # chunk_id), and _ndjson_to_parquet drops it (fails SynthQuery
            # validation -> continue), so it never reaches the parquet.
            _emit_sentinel(chunk.chunk_id)

    try:
        for c in iterator:
            n_in += 1
            t = asyncio.create_task(_bound(c))
            pending.add(t)
            if len(pending) >= concurrency * 2:
                done, pending = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )
                for fut in done:
                    chunk, parsed, err = fut.result()
                    _record(chunk, parsed, err)
        if pending:
            done, _ = await asyncio.wait(pending)
            for fut in done:
                chunk, parsed, err = fut.result()
                _record(chunk, parsed, err)
    finally:
        progress.close()
        nd_fh.flush()
        os.fsync(nd_fh.fileno())
        nd_fh.close()
        if log_fh is not None:
            log_fh.close()

    # Compact NDJSON -> parquet.
    _ndjson_to_parquet(ndjson_path, out_parquet)
    return n_in, n_out


def _ndjson_to_parquet(ndjson_path: Path, out_parquet: Path) -> None:
    """Stream the NDJSON sidecar into one parquet file."""
    if not ndjson_path.exists():
        return
    if out_parquet.exists():
        out_parquet.unlink()
    writer = pq.ParquetWriter(out_parquet, SynthQuerySchema, compression="zstd")
    buf: list[SynthQuery] = []
    BATCH = 4096
    try:
        with ndjson_path.open("rb") as f:
            for raw in f:
                if not raw.strip():
                    continue
                try:
                    obj = orjson.loads(raw)
                except orjson.JSONDecodeError:
                    continue
                try:
                    buf.append(SynthQuery.model_validate(obj))
                except ValidationError:
                    continue
                if len(buf) >= BATCH:
                    writer.write_batch(_to_arrow_batch(buf))
                    buf.clear()
        if buf:
            writer.write_batch(_to_arrow_batch(buf))
    finally:
        writer.close()


def generate_queries(
    chunks_parquet: str | os.PathLike[str],
    out_parquet: str | os.PathLike[str],
    *,
    model_id: str = DEFAULT_MODEL_ID,
    concurrency: int = DEFAULT_CONCURRENCY,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    log_path: str | os.PathLike[str] | None = None,
    limit: int | None = None,
) -> tuple[int, int, float]:
    """Public sync entry. Returns (n_chunks, n_queries, elapsed_sec).

    Incremental by construction: resumes from the NDJSON sidecar, skipping any
    chunk_id already synthed, so a re-run only processes new chunks.
    """
    t0 = time.time()
    n_in, n_out = asyncio.run(
        _run_async(
            chunks_parquet=Path(chunks_parquet),
            out_parquet=Path(out_parquet),
            model_id=model_id,
            concurrency=concurrency,
            max_tokens=max_tokens,
            temperature=temperature,
            log_path=Path(log_path) if log_path else None,
            limit=limit,
        )
    )
    return n_in, n_out, time.time() - t0
