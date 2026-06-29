"""Live relevance judge — filters retrieval candidates through the trained
Qwen3.5-9B judge served by llama-server (see ~/.local/bin/serve-judge-9b).

Every decision is appended to data/eval/judge_live.jsonl with the P(1)
margin: low-margin pairs are tomorrow's human-review queue, so live traffic
grows the anchor set for free (the flywheel's data collection loop).

Fail-open by design: if the judge server is down, slow, or returns garbage,
the chunk PASSES and the failure is logged. Retrieval must never break
because a side-car model is offline.

Env:
  SXC_JUDGE=0          disable filtering entirely (default: on)
  SXC_JUDGE_URL        default http://127.0.0.1:8003/v1
  SXC_JUDGE_TIMEOUT    seconds per call (default 8)

The prompt MUST stay byte-identical to PROMPT in eval/train_judge.py —
the model was trained on it; paraphrases shift the threshold.
"""

from __future__ import annotations

import json
import math
import os
import time
import urllib.request
from pathlib import Path

_LOG = Path(__file__).resolve().parents[2] / "data" / "eval" / "judge_live.jsonl"

_PROMPT = """A user is researching the topic given by a short search query \
(queries may contain typos — read them charitably).
Would the chunk below help someone researching that topic? Even a chunk that \
only touches the topic while doing something else counts as helpful.

Query: {query}

Chunk:
{chunk}

Reply with ONLY one digit: 1 = helpful, 0 = not."""


def enabled() -> bool:
    return os.environ.get("SXC_JUDGE", "1") != "0"


def _p1_from_logprobs(choice: dict) -> float | None:
    """P(1) over {0,1} from the first generated token's top_logprobs."""
    try:
        top = choice["logprobs"]["content"][0]["top_logprobs"]
        lp = {e["token"].strip(): e["logprob"] for e in top}
        if "0" not in lp and "1" not in lp:
            return None
        e0 = math.exp(lp.get("0", -100.0))
        e1 = math.exp(lp.get("1", -100.0))
        return e1 / (e0 + e1)
    except (KeyError, IndexError, TypeError):
        return None


def judge(query: str, chunk_text: str) -> tuple[int | None, float | None]:
    """Returns (digit, p1). digit None = judge unavailable (caller fails open)."""
    base = os.environ.get("SXC_JUDGE_URL", "http://127.0.0.1:8003/v1")
    timeout = float(os.environ.get("SXC_JUDGE_TIMEOUT", "8"))
    payload = {
        "model": "judge-9b",
        "temperature": 0.0,
        "max_tokens": 4,
        "logprobs": True,
        "top_logprobs": 6,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": _PROMPT.format(
            query=query, chunk=chunk_text[:1500])}],
    }
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {os.environ.get('SXC_JUDGE_KEY', 'local')}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            choice = json.loads(resp.read())["choices"][0]
    except Exception:
        return None, None
    content = choice["message"]["content"]
    digit = 1 if "1" in content else 0 if "0" in content else None
    return digit, _p1_from_logprobs(choice)


def log_decision(query: str, chunk_id: str, retriever: str,
                 digit: int | None, p1: float | None, kept: bool,
                 latency_ms: float) -> None:
    """Append-only decision log; never raises into the retrieval path."""
    try:
        _LOG.parent.mkdir(parents=True, exist_ok=True)
        with _LOG.open("a") as f:
            f.write(json.dumps({
                "ts": time.time(), "query": query, "chunk_id": chunk_id,
                "retriever": retriever, "digit": digit,
                "p1": round(p1, 4) if p1 is not None else None,
                "kept": kept, "latency_ms": round(latency_ms, 1),
            }) + "\n")
    except OSError:
        pass
