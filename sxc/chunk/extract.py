"""Walk raw transcript dirs and emit chunks.parquet.

Schema variants handled:

claude  ~/.claude/projects/<encoded-cwd>/<session-uuid>.jsonl
        line types we keep:
          {type:"user",      message:{role,content:str|list[{type,text}]}, ...}
          {type:"assistant", message:{role,content:str|list[{type,text}]}, ...}
        ignore: queue-operation, system-meta, tool-result blocks (kept as
        flattened "[tool_use] name=…" / "[tool_result] …" so retrievers can
        still surface command/output context).

codex   ~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl
        {type:"response_item", payload:{type:"message",
                                        role:"user"|"assistant"|"system",
                                        content:[{type:"input_text"|"output_text", text}]}}
        ignore: event_msg, turn_context, session_meta.

hermes  ~/.hermes/sessions/<ts>_<id>.jsonl
        {role:"user"|"assistant"|"tool", content:str, timestamp:str}
        first line is often {role:"session_meta", tools:[...]} — skip.

eigen   ~/.eigen/{sessions,daemon/sessions,daemon-dev/sessions}/<id>.jsonl
        one llm.Message per line (Go field names, capitalized):
          {Role:"user"|"assistant"|"tool", Text:str, ToolCalls:[{ID,Name,Arguments}],
           ToolName, ToolError, Reasoning, ...}
        no per-line timestamp; project comes from the <id>.meta.json sidecar's
        "dir". Tool calls flatten to "[tool_use] Name {args}", tool results to
        "[tool_result] …" so command/output context survives.

readme  data/readmes/<slug>.md
        single doc, paragraph-split.

memory  data/memories/<slug>.md
        eigen + hermes curated/dreamed memory. Markdown, paragraph-split like
        readme. Slug is "<agent>__<project>" (eigen) or "hermes__<file>".

skill   data/skills/<slug>.md
        eigen + hermes SKILL.md catalog (incl. eigen skills-proposed). Markdown,
        paragraph-split. Slug is "<agent>__<skill-name>".

paper   data/papers/<slug>.md
        long-form research / knowledge syntheses. Markdown, paragraph-split.

surreal data/surreal/<slug>.md
        revuto reviewer-namespace tables dumped to one markdown doc per database
        (data/surreal/<db>.md), embedding vectors stripped. Paragraph-split.

Chunking policy (uniform across sources):
  - drop messages <40 chars unless they contain a reaction marker
  - flatten tool blocks to one line each
  - merge consecutive short turns from same role into one logical turn
  - parent_text = sliding window of ±N tokens around the chunk
  - chunk text  = the messages whose budget hits ~256 tokens (no overlap;
                  parent_text supplies overlap implicitly)
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import orjson
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from .schema import Chunk, ChunkSchema, Role, Source

# ----- tunables ---------------------------------------------------------------

CHUNK_TOK_TARGET = 256
PARENT_TOK_TARGET = 1024
MIN_MSG_CHARS = 40
TOOL_BLOCK_CAP_CHARS = 400
REACTION_RE = re.compile(
    r"\b(love|cool|nice|brilliant|elegant|drawn to|excited|i want to|let me try|building)\b",
    re.IGNORECASE,
)
CHARS_PER_TOK = 4
CHUNK_CHAR_TARGET = CHUNK_TOK_TARGET * CHARS_PER_TOK
PARENT_CHAR_TARGET = PARENT_TOK_TARGET * CHARS_PER_TOK
# ColBERT sees only a capped passage prefix. Large injected AGENTS/CLAUDE
# blocks often differ only in their tail, so de-dupe on the retrieval-visible
# prefix instead of letting them fill a result page with near-identical hits.
DEDUPE_PREFIX_CHARS = CHUNK_CHAR_TARGET * 2


# ----- normalised intermediate (hot path: dataclass slots, no pydantic cost) --


@dataclass(slots=True)
class Turn:
    role: Role
    text: str
    ts: float | None


# ----- helpers ----------------------------------------------------------------


def _hash_id(*parts: str) -> str:
    h = hashlib.blake2b(digest_size=12)
    for p in parts:
        h.update(p.encode("utf-8", "replace"))
        h.update(b"\x00")
    return h.hexdigest()


def _dedupe_key(text: str) -> str:
    """Stable content key for repeated chunks across sessions/tools."""
    norm = re.sub(r"\s+", " ", text).strip().lower()
    if len(norm) > DEDUPE_PREFIX_CHARS:
        norm = norm[:DEDUPE_PREFIX_CHARS]
    return _hash_id("dedupe", norm)


def _duplicate_rank(count: int) -> float:
    """Bounded rank penalty for repeated chunks."""
    if count <= 1:
        return 1.0
    return float(1.0 + min(4.0, math.log2(count)))


def _coerce_ts(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            iso = v[:-1] + "+00:00" if v.endswith("Z") else v
            return datetime.fromisoformat(iso).timestamp()
        except ValueError:
            try:
                return float(v)
            except ValueError:
                return None
    return None


def _flatten_content(content: Any) -> str:
    """Anthropic message.content can be str or list[block]. Flatten to text."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    out: list[str] = []
    for blk in content:
        if not isinstance(blk, dict):
            continue
        t = blk.get("type")
        if t in ("text", "input_text", "output_text"):
            out.append(str(blk.get("text", "")))
        elif t == "tool_use":
            name = str(blk.get("name", "?"))
            inp = blk.get("input", {})
            inp_s = orjson.dumps(inp).decode("utf-8", "replace")[:TOOL_BLOCK_CAP_CHARS]
            out.append(f"[tool_use] {name} {inp_s}")
        elif t == "tool_result":
            inner = blk.get("content", "")
            if isinstance(inner, list):
                inner = " ".join(
                    str(b.get("text", "")) for b in inner if isinstance(b, dict)
                )
            out.append(f"[tool_result] {str(inner)[:TOOL_BLOCK_CAP_CHARS]}")
    return "\n".join(s for s in out if s)


def _coerce_role(r: Any) -> Role | None:
    if r in ("user", "assistant", "tool", "doc"):
        return cast(Role, r)
    return None


# ----- per-source iterators (yield Turn) --------------------------------------


def _iter_claude(path: Path) -> Iterator[Turn]:
    try:
        with path.open("rb") as f:
            for raw in f:
                if not raw.strip():
                    continue
                try:
                    obj = orjson.loads(raw)
                except orjson.JSONDecodeError:
                    continue
                if not isinstance(obj, dict):
                    continue
                t = obj.get("type")
                if t not in ("user", "assistant"):
                    continue
                msg = obj.get("message") or {}
                if not isinstance(msg, dict):
                    continue
                role = _coerce_role(msg.get("role") or t)
                if role is None:
                    continue
                text = _flatten_content(msg.get("content", ""))
                if not text:
                    continue
                yield Turn(role=role, text=text, ts=_coerce_ts(obj.get("timestamp")))
    except OSError:
        return


def _iter_codex(path: Path) -> Iterator[Turn]:
    try:
        with path.open("rb") as f:
            for raw in f:
                if not raw.strip():
                    continue
                try:
                    obj = orjson.loads(raw)
                except orjson.JSONDecodeError:
                    continue
                if not isinstance(obj, dict):
                    continue
                if obj.get("type") != "response_item":
                    continue
                payload = obj.get("payload") or {}
                if not isinstance(payload, dict):
                    continue
                if payload.get("type") != "message":
                    continue
                role = _coerce_role(payload.get("role"))
                if role is None or role not in ("user", "assistant"):
                    continue
                text = _flatten_content(payload.get("content", ""))
                if not text:
                    continue
                yield Turn(role=role, text=text, ts=_coerce_ts(obj.get("timestamp")))
    except OSError:
        return


def _iter_hermes(path: Path) -> Iterator[Turn]:
    try:
        with path.open("rb") as f:
            for raw in f:
                if not raw.strip():
                    continue
                try:
                    obj = orjson.loads(raw)
                except orjson.JSONDecodeError:
                    continue
                if not isinstance(obj, dict):
                    continue
                role = _coerce_role(obj.get("role"))
                if role is None or role not in ("user", "assistant"):
                    continue  # drop tool / session_meta
                content = obj.get("content", "")
                text = content if isinstance(content, str) else _flatten_content(content)
                if not text:
                    continue
                yield Turn(role=role, text=text, ts=_coerce_ts(obj.get("timestamp")))
    except OSError:
        return


def _iter_eigen(path: Path) -> Iterator[Turn]:
    """eigen daemon/local transcripts: one llm.Message per line, Go-capitalized
    field names. assistant turns may carry only ToolCalls (no Text); tool turns
    carry the result in Text. Flatten tool calls/results to text so command and
    output context is retained, mirroring the claude/codex handling. No per-line
    timestamp (ordering is line order; the sidecar carries session-level meta)."""
    try:
        with path.open("rb") as f:
            for raw in f:
                if not raw.strip():
                    continue
                try:
                    obj = orjson.loads(raw)
                except orjson.JSONDecodeError:
                    continue
                if not isinstance(obj, dict):
                    continue
                role = _coerce_role(obj.get("Role"))
                if role is None:
                    continue
                parts: list[str] = []
                text = obj.get("Text")
                if isinstance(text, str) and text:
                    if role == "tool":
                        name = str(obj.get("ToolName") or "")
                        tag = "[tool_result]" + (f" {name}" if name else "")
                        parts.append(f"{tag} {text[:TOOL_BLOCK_CAP_CHARS]}")
                    else:
                        parts.append(text)
                for call in obj.get("ToolCalls") or []:
                    if not isinstance(call, dict):
                        continue
                    name = str(call.get("Name", "?"))
                    inp = call.get("Arguments", {})
                    inp_s = orjson.dumps(inp).decode("utf-8", "replace")[
                        :TOOL_BLOCK_CAP_CHARS
                    ]
                    parts.append(f"[tool_use] {name} {inp_s}")
                merged = "\n".join(p for p in parts if p)
                if not merged:
                    continue
                yield Turn(role=role, text=merged, ts=None)
    except OSError:
        return


def _iter_readme(path: Path) -> Iterator[Turn]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        yield Turn(role="doc", text=para, ts=None)


# ----- session → chunks --------------------------------------------------------


def _normalise(turns: list[Turn]) -> list[Turn]:
    """Drop tiny turns w/o reaction markers; merge consecutive same-role
    until merged size approaches the chunk budget — beyond that, split so the
    packer can choose chunk boundaries."""
    keep: list[Turn] = []
    for t in turns:
        text = t.text.strip()
        if not text:
            continue
        if len(text) < MIN_MSG_CHARS and not REACTION_RE.search(text):
            continue
        if (
            keep
            and keep[-1].role == t.role
            and len(keep[-1].text) + len(text) + 1 <= CHUNK_CHAR_TARGET
        ):
            keep[-1].text = keep[-1].text + "\n" + text
            keep[-1].ts = keep[-1].ts or t.ts
        else:
            keep.append(Turn(role=t.role, text=text, ts=t.ts))
    return keep


def _pack(turns: list[Turn], target: int) -> list[tuple[int, int]]:
    """Greedy pack consecutive turns up to `target` chars. Half-open ranges."""
    spans: list[tuple[int, int]] = []
    i = 0
    n = len(turns)
    while i < n:
        size = 0
        j = i
        while j < n and (size == 0 or size + len(turns[j].text) <= target):
            size += len(turns[j].text) + 1
            j += 1
        spans.append((i, max(j, i + 1)))
        i = max(j, i + 1)
    return spans


def _expand_parent(turns: list[Turn], i: int, j: int, target: int) -> str:
    lo, hi = i, j
    size = sum(len(turns[k].text) + 1 for k in range(lo, hi))
    while size < target and (lo > 0 or hi < len(turns)):
        if lo > 0:
            lo -= 1
            size += len(turns[lo].text) + 1
        if size >= target:
            break
        if hi < len(turns):
            size += len(turns[hi].text) + 1
            hi += 1
    return "\n\n".join(f"[{turns[k].role}] {turns[k].text}" for k in range(lo, hi))


def _session_chunks(
    turns: list[Turn], *, source: Source, session_id: str, project: str | None
) -> list[Chunk]:
    turns = _normalise(turns)
    if not turns:
        return []
    spans = _pack(turns, CHUNK_CHAR_TARGET)
    chunks: list[Chunk] = []
    for span_idx, (i, j) in enumerate(spans):
        text = "\n\n".join(f"[{turns[k].role}] {turns[k].text}" for k in range(i, j))
        ts = next((turns[k].ts for k in range(i, j) if turns[k].ts is not None), None)
        role: Role = (
            "user"
            if any(turns[k].role == "user" for k in range(i, j))
            else turns[i].role
        )
        cid = _hash_id(source, session_id, str(span_idx), text[:128])
        dedupe_key = _dedupe_key(text)
        parent = _expand_parent(turns, i, j, PARENT_CHAR_TARGET)
        chunks.append(
            Chunk(
                chunk_id=cid,
                source=source,
                session_id=session_id,
                project=project,
                ts=ts,
                role=role,
                text=text,
                parent_text=parent,
                prev_chunk_id=None,
                next_chunk_id=None,
                dedupe_key=dedupe_key,
                duplicate_count=1,
                duplicate_rank=1.0,
                duplicate_sources=source,
                duplicate_projects=project,
                duplicate_session_ids=session_id,
            )
        )
    for idx, c in enumerate(chunks):
        c.prev_chunk_id = chunks[idx - 1].chunk_id if idx > 0 else None
        c.next_chunk_id = chunks[idx + 1].chunk_id if idx + 1 < len(chunks) else None
    return chunks


def _prefer_canonical(candidate: Chunk, current: Chunk) -> bool:
    """Pick the representative with the freshest/richest context."""
    c_ts = candidate.ts if candidate.ts is not None else -1.0
    cur_ts = current.ts if current.ts is not None else -1.0
    if c_ts != cur_ts:
        return c_ts > cur_ts
    return len(candidate.parent_text) > len(current.parent_text)


def _limited_join(values: set[str], *, limit: int = 16) -> str | None:
    if not values:
        return None
    ordered = sorted(values)
    shown = ordered[:limit]
    if len(ordered) > limit:
        shown.append(f"+{len(ordered) - limit} more")
    return ",".join(shown)


def _dedupe_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """Collapse repeated chunk text, preserving repetition as rank metadata."""
    canonical: dict[str, Chunk] = {}
    counts: dict[str, int] = {}
    sources: dict[str, set[str]] = {}
    projects: dict[str, set[str]] = {}
    sessions: dict[str, set[str]] = {}

    for c in chunks:
        key = c.dedupe_key
        counts[key] = counts.get(key, 0) + 1
        sources.setdefault(key, set()).add(c.source)
        if c.project:
            projects.setdefault(key, set()).add(c.project)
        sessions.setdefault(key, set()).add(c.session_id)
        if key not in canonical or _prefer_canonical(c, canonical[key]):
            canonical[key] = c

    out = list(canonical.values())
    for c in out:
        count = counts[c.dedupe_key]
        c.chunk_id = c.dedupe_key
        c.prev_chunk_id = None
        c.next_chunk_id = None
        c.duplicate_count = count
        c.duplicate_rank = _duplicate_rank(count)
        c.duplicate_sources = _limited_join(sources.get(c.dedupe_key, set()))
        c.duplicate_projects = _limited_join(projects.get(c.dedupe_key, set()))
        c.duplicate_session_ids = _limited_join(sessions.get(c.dedupe_key, set()))
    return out


# ----- top-level walk ----------------------------------------------------------


def _claude_project(slug_dir: str) -> str | None:
    """Decode a Claude project slug dir, e.g. '-home-user-projects-myapp' → 'myapp'."""
    name = slug_dir.lstrip("-")
    if "projects-" in name:
        tail = name.split("projects-", 1)[1]
        return tail.split("-")[0] or None
    parts = name.split("-")
    return parts[-1] if parts else None


def _readme_project(slug: str) -> str:
    return slug.split("__", 1)[0]


def _eigen_project(sess: Path) -> str | None:
    """Project for an eigen session = basename of the `dir` in its meta sidecar
    (<id>.meta.json). '/home/user/projects/myapp' → 'myapp'; the bare
    '~/projects' root → 'projects'. Missing/garbled sidecar → None."""
    meta = sess.with_suffix(".meta.json")
    try:
        obj = orjson.loads(meta.read_bytes())
    except (OSError, orjson.JSONDecodeError):
        return None
    if not isinstance(obj, dict):
        return None
    d = obj.get("dir")
    if not isinstance(d, str) or not d:
        return None
    return Path(d).name or None


def _walk(data_dir: Path) -> Iterator[Chunk]:
    claude_root = data_dir / "sessions" / "claude"
    codex_root = data_dir / "sessions" / "codex"
    hermes_root = data_dir / "sessions" / "hermes"
    eigen_root = data_dir / "sessions" / "eigen"
    readme_root = data_dir / "readmes"
    # extra doc corpora: distilled knowledge, not raw transcripts. Each is markdown
    # staged as "<slug>.md" by refresh.sh, paragraph-split exactly like readmes.
    doc_roots: list[tuple[str, Path]] = [
        ("memory", data_dir / "memories"),
        ("skill", data_dir / "skills"),
        ("paper", data_dir / "papers"),
        ("surreal", data_dir / "surreal"),
    ]

    if claude_root.exists():
        for proj_dir in sorted(p for p in claude_root.iterdir() if p.is_dir()):
            project = _claude_project(proj_dir.name)
            for sess in tqdm(
                sorted(proj_dir.glob("*.jsonl")),
                desc=f"claude/{proj_dir.name}",
                leave=False,
            ):
                yield from _session_chunks(
                    list(_iter_claude(sess)),
                    source="claude",
                    session_id=sess.stem,
                    project=project,
                )

    if codex_root.exists():
        files = sorted(codex_root.rglob("*.jsonl"))
        for sess in tqdm(files, desc="codex"):
            yield from _session_chunks(
                list(_iter_codex(sess)),
                source="codex",
                session_id=sess.stem,
                project=None,
            )

    if hermes_root.exists():
        for sess in tqdm(sorted(hermes_root.glob("*.jsonl")), desc="hermes"):
            yield from _session_chunks(
                list(_iter_hermes(sess)),
                source="hermes",
                session_id=sess.stem,
                project=None,
            )

    if eigen_root.exists():
        for sess in tqdm(sorted(eigen_root.glob("*.jsonl")), desc="eigen"):
            yield from _session_chunks(
                list(_iter_eigen(sess)),
                source="eigen",
                session_id=sess.stem,
                project=_eigen_project(sess),
            )

    if readme_root.exists():
        for md in tqdm(sorted(readme_root.glob("*.md")), desc="readmes"):
            yield from _session_chunks(
                list(_iter_readme(md)),
                source="readme",
                session_id=md.stem,
                project=_readme_project(md.stem),
            )

    for src, root in doc_roots:
        if not root.exists():
            continue
        for md in tqdm(sorted(root.glob("*.md")), desc=src):
            yield from _session_chunks(
                list(_iter_readme(md)),
                source=cast(Source, src),
                session_id=md.stem,
                project=_readme_project(md.stem),
            )


# ----- writer ------------------------------------------------------------------


def _to_arrow_batch(chunks: list[Chunk]) -> pa.RecordBatch:
    cols: dict[str, list[Any]] = {f.name: [] for f in ChunkSchema}
    for c in chunks:
        cols["chunk_id"].append(c.chunk_id)
        cols["source"].append(c.source)
        cols["session_id"].append(c.session_id)
        cols["project"].append(c.project)
        cols["ts"].append(c.ts)
        cols["role"].append(c.role)
        cols["text"].append(c.text)
        cols["parent_text"].append(c.parent_text)
        cols["prev_chunk_id"].append(c.prev_chunk_id)
        cols["next_chunk_id"].append(c.next_chunk_id)
        cols["dedupe_key"].append(c.dedupe_key)
        cols["duplicate_count"].append(c.duplicate_count)
        cols["duplicate_rank"].append(c.duplicate_rank)
        cols["duplicate_sources"].append(c.duplicate_sources)
        cols["duplicate_projects"].append(c.duplicate_projects)
        cols["duplicate_session_ids"].append(c.duplicate_session_ids)
    return pa.RecordBatch.from_pydict(cols, schema=ChunkSchema)


def build_chunks(
    data_dir: str | os.PathLike[str],
    out: str | os.PathLike[str],
    *,
    batch_size: int = 5000,
) -> int:
    """Walk data_dir, write chunks.parquet, return total count."""
    data_path = Path(data_dir)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    raw = list(_walk(data_path))
    unique = _dedupe_chunks(raw)

    writer: pq.ParquetWriter | None = None
    total = 0
    try:
        for start in range(0, len(unique), batch_size):
            batch_chunks = unique[start : start + batch_size]
            if batch_chunks:
                batch = _to_arrow_batch(batch_chunks)
                if writer is None:
                    writer = pq.ParquetWriter(out_path, ChunkSchema, compression="zstd")
                writer.write_batch(batch)
                total += len(batch_chunks)
    finally:
        if writer is not None:
            writer.close()
    return total
