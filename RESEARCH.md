# Research notes: retriever benchmark, a trained relevance judge, and a clean synthetic-data negative

These are the experiment write-ups behind sxc's design choices. Everything was run on a
real corpus of ~97k text chunks (agent transcripts, notes, docs) with the pipeline in
`eval/`-style scripts; numbers below are the actual measured results, not illustrations.
Voice throughout is past tense: this is what was done and what was found.

## 1. Retriever benchmark (why ColBERT is the default)

All four retrievers were benchmarked on 2,000 stratified synthetic gold pairs
(query generated from a known chunk; retriever scored on whether it recovered that
chunk). Aggregates over the full 8,000-row results parquet:

| retriever  | nDCG@10 | recall@50 |   MRR | mean latency |
|------------|--------:|----------:|------:|-------------:|
| colbert-v2 |   0.361 |     0.672 | 0.321 |      38.7 ms |
| hybrid     |   0.361 |     0.675 | 0.321 |     855.9 ms |
| splade-v3  |   0.327 |     0.633 | 0.288 |     794.5 ms |
| bm25       |   0.252 |     0.391 | 0.226 |       6.6 ms |

Hybrid (SPLADE first-stage → ColBERT rerank) matched ColBERT's quality at 22x the
latency, because the first stage was the bottleneck and the rerank recovered ColBERT's
ranking anyway. ColBERT-v2 alone won on the quality/latency frontier and became the
default retriever. BM25 stayed as the zero-dependency fallback.

## 2. The relevance judge (shipped)

Goal: a binary judge — "would this chunk help someone researching this seed topic?" —
good enough to filter synthetic training pairs and to sit in the live serving path.

Setup: Qwen3.5-9B base (int8) + rank-16 LoRA, binary target. Trained on a 1,101-pair
anchor set labeled by several voices with a strict precedence (human gold overrode the
frontier-model teacher, which overrode local-model labels), split at the seed level,
with a frozen 127-pair human-backed test set used once per round.

| judge                    | test acc | balanced acc | class-0 | class-1 |
|--------------------------|---------:|-------------:|--------:|--------:|
| prompted 27B few-shot    |    0.740 |        0.823 |   0.958 |   0.689 |
| **9B LoRA r1 (shipped)** |    0.898 |    **0.937** | **1.000** |   0.874 |
| r2 (+human pass 2)       |    0.937 |        0.913 |   0.875 |   0.951 |
| r3 (+lexical traps)      |    0.850 |        0.891 |   0.958 |   0.825 |

r1 was chosen on a noise-is-worse criterion: the later rounds added labels but traded
away class-0 precision, and false positives were the expensive error for a filter.

Findings that generalized:

- **A 9B LoRA beat the prompted 27B it was distilled from** (0.937 vs 0.823 balanced
  accuracy). Fine-tuning on ~1.1k carefully-adjudicated pairs was worth more than 3x
  the parameters plus few-shot examples.
- **Thinking mode made the prompted judge worse.** Given room to reason, it
  rationalized its own strictness instead of correcting it.
- **A graded 0/1/2 scale carried no signal over binary** — the few-shot examples did
  all the work.
- **List-mention is not relevance.** Roughly 40% of disputed teacher labels were chunks
  that merely mentioned the seed term inside a list; encoding that one rule resolved
  most teacher/human disagreement.
- **A 0.6B student memorized instead of learning** (0.642 balanced accuracy); 9B was
  the smallest size that actually learned the task (0.937).

The shipped judge serves as a Q8_0 GGUF behind an OpenAI-compatible endpoint (the
`SXC_JUDGE_URL` default in this repo) and runs as a fail-open live filter in
`sxc/serve/judge.py`, hooked into `get_context`. Decisions are logged with P(1)
margins, forming a review queue for future label rounds.

## 3. Synth-v2: a clean negative on seed-style training data

Hypothesis tested: the production retriever was fine-tuned on ~120k rich synthetic
*questions* (Sonnet-generated), but real traffic is short 2-3 word *seed* queries — so
training directly on seed-style queries should help.

Pipeline: a 27B model (Qwen3.6-FP8 on vLLM, rented EC2) generated 3 seed-style queries
per chunk for all 97,199 chunks; the 9B judge then filtered every (seed, chunk) pair.
**291,399 judged pairs, 0.906 accept rate, zero pipeline errors, ~$30 of EC2.** Three
retrievers were trained on the result and compared against production on 60 real seed
queries, judged p@5:

| retriever          | training data                    | judged p@5 | head-to-head vs v1 |
|--------------------|----------------------------------|-----------:|--------------------|
| **ft-v1 (production)** | 120k Sonnet questions        |  **0.727** | —                  |
| ft-v2slice         | 67k seeds, naive contrastive     |      0.697 | 15–7 (38 ties)     |
| ft-v2slice-nd      | 50k seeds, cap-8 + no-duplicates |      0.673 | 17–5 (38 ties)     |
| ft-v2stream        | 264k seeds, 3-stage continual    |      0.690 | 17–8 (35 ties)     |

**Seed-style synthetic training lost to question-style at every scale and configuration
tested.** 4x more data did not close the gap. The reading: rich multi-token questions
teach token-level matching that transfers *down* to short seed queries; narrow
2.5-word seeds don't generalize *up*. Training on the test distribution lost to
training on a richer one.

Confound checked: 56.7% of pairs shared seed text (one seed, "zoxide configuration",
matched 970 chunks), which poisons in-batch negatives. Fixing that collision (the `nd`
run) did **not** recover the gap, so collisions weren't the cause.

## 4. The open leg

The judge-filtering pass produced **27,456 judge-rejected pairs**: high lexical overlap,
judged irrelevant — exactly the profile of hard negatives. All v2 training runs used
positives only. Adding these explicit negatives to the contrastive loss was the most
promising lever left untested; the untested hard-negatives leg is the next step. Two
smaller untested variants: a mixed diet (Sonnet questions + judge-cleaned seeds), and
keeping seeds as eval-only.

## 5. Operational notes

Lessons from running the pipeline on rented GPUs:

- **Rent wall-clock, not capability**: N replicas of the whole work unit on one
  multi-GPU box beat heterogeneous fleets; new capacity went to the measured
  bottleneck (one judge instance drained ~2-3 generator instances).
- **Stream between stages**: outputs flowed to the next stage as produced.
  A collect-then-train loop with a mark-seen-at-read bug silently leaked 120k pairs
  for 50 minutes before it was caught — buffer pending work explicitly.
- `pkill` self-match: pattern-kill commands ran alone and bracket-patterned
  (`[p]attern`), never inside compound commands.

## Artifacts

The datasets and adapters live in the research workspace, not in this repo. For
reproduction, the canonical dataset was a parquet of
`(query, chunk_id, judge_digit, judge_p1)` rows plus the raw judgment log and a
generation manifest; the judge anchor set was a JSONL of labeled pairs across all
voices with the human-backed test split frozen. The `bench`, `synth`, and judge-serving
entry points in this repo are the same code paths those experiments ran through.
