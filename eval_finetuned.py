"""Fine-tuned vs zero-shot ColBERTv2 on the held-out protocol, full corpus.

Ranks each held-out query's gold chunk among ALL chunks (same difficulty as
the nx-playground full-corpus evals). Pool encoded once per model, scores
computed in sharded torch MaxSim on GPU.

Run: ~/.venvs/pylate/bin/python eval_finetuned.py
"""

import numpy as np
import pyarrow.parquet as pq
import torch
from pylate import models

DATA = "data"
N_EVAL = 500
SEED = 7
DOC_LEN = 180  # pylate doc tokens default-ish; same for both models

q = pq.read_table(f"{DATA}/synth_queries.parquet", columns=["chunk_id", "q"]).to_pandas()
c = pq.read_table(f"{DATA}/chunks.parquet", columns=["chunk_id", "text"]).to_pandas()
joined = q.merge(c, on="chunk_id", how="inner")
joined = joined.sample(n=min(120_000, len(joined)), random_state=SEED).reset_index(drop=True)
heldout = joined.iloc[:1500].head(N_EVAL)  # the rows the fine-tune NEVER saw

all_ids = c["chunk_id"].tolist()
id_to_pos = {cid: i for i, cid in enumerate(all_ids)}
docs = c["text"].tolist()
queries = heldout["q"].tolist()
gold = [id_to_pos[cid] for cid in heldout["chunk_id"]]
print(f"pool: {len(docs)} docs; eval queries: {len(queries)}")


def evaluate(model_path: str, label: str) -> None:
    model = models.ColBERT(model_name_or_path=model_path, device="cuda")

    d_emb = model.encode(
        docs, batch_size=256, is_query=False, show_progress_bar=True,
        convert_to_numpy=True,
    )
    q_emb = model.encode(queries, batch_size=64, is_query=True, convert_to_numpy=True)

    # MaxSim, sharded over docs. Variable-length doc embeddings -> pad per shard.
    q_t = [torch.tensor(e, device="cuda", dtype=torch.float16) for e in q_emb]
    ranks = np.zeros(len(queries), dtype=np.int64)
    gold_scores = np.full(len(queries), -1e9, dtype=np.float32)
    all_scores = np.zeros((len(queries), len(docs)), dtype=np.float32)

    shard = 2048
    for start in range(0, len(docs), shard):
        chunk = d_emb[start:start + shard]
        max_len = max(e.shape[0] for e in chunk)
        padded = np.zeros((len(chunk), max_len, chunk[0].shape[1]), dtype=np.float16)
        mask = np.zeros((len(chunk), max_len), dtype=bool)
        for i, e in enumerate(chunk):
            padded[i, : e.shape[0]] = e
            mask[i, : e.shape[0]] = True

        d_t = torch.tensor(padded, device="cuda")
        m_t = torch.tensor(mask, device="cuda")

        for qi, qe in enumerate(q_t):
            # sim: {shard, max_len, q_len}; mask pads to -inf before max
            sim = torch.einsum("dlh,qh->dlq", d_t, qe.to(d_t.dtype))
            sim = sim.masked_fill(~m_t.unsqueeze(-1), float("-inf"))
            scores = sim.max(dim=1).values.sum(dim=1)  # {shard}
            all_scores[qi, start:start + len(chunk)] = scores.float().cpu().numpy()

    for qi in range(len(queries)):
        gs = all_scores[qi, gold[qi]]
        ranks[qi] = int((all_scores[qi] > gs).sum()) + 1

    mrr = float(np.mean(1.0 / ranks))
    rec = {k: float(np.mean(ranks <= k)) for k in (1, 10, 50)}
    print(f"{label}: MRR={mrr:.4f} r@1={rec[1]:.3f} r@10={rec[10]:.3f} r@50={rec[50]:.3f}")


evaluate("colbert-ir/colbertv2.0", "zero-shot ")
evaluate("models/colbertv2-sxc-ft/final", "fine-tuned")
