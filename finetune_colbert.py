"""Fine-tune real ColBERTv2 on the sxc synth pairs (in-batch contrastive).

Same data protocol as the nx-playground from-scratch experiments:
  - seed-7 sample of joined (query, chunk) pairs
  - first 1500 rows held out (never trained) -> comparable eval later

Low lr (1e-5): pretrained weights are an asset being adjusted, not demolished
(the catastrophic-forgetting lesson, measured in the Elixir experiments).

Run:  ~/.venvs/pylate/bin/python finetune_colbert.py
"""

import pyarrow.parquet as pq
from datasets import Dataset
from pylate import losses, models
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)

DATA = "data"
OUT = "models/colbertv2-sxc-ft"
N_PAIRS = 120_000
SEED = 7
BATCH = 32
EPOCHS = 2
LR = 1e-5

# ---- load + join, replicating the Elixir protocol ----
q = pq.read_table(f"{DATA}/synth_queries.parquet", columns=["chunk_id", "q"]).to_pandas()
c = pq.read_table(f"{DATA}/chunks.parquet", columns=["chunk_id", "text"]).to_pandas()
joined = q.merge(c, on="chunk_id", how="inner")

# pandas sample != polars sample, but the principle holds: fixed-seed sample,
# carve a held-out head that training never sees.
joined = joined.sample(n=min(N_PAIRS, len(joined)), random_state=SEED).reset_index(drop=True)
train_df = joined.iloc[1500:]
print(f"pairs: {len(joined)}  train: {len(train_df)}  held-out: 1500")

train_dataset = Dataset.from_dict(
    {
        "query": train_df["q"].tolist(),
        "positive": train_df["text"].tolist(),
    }
)

# ---- model + loss ----
model = models.ColBERT(model_name_or_path="colbert-ir/colbertv2.0")
loss = losses.Contrastive(model=model)

args = SentenceTransformerTrainingArguments(
    output_dir=OUT,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH,
    learning_rate=LR,
    bf16=True,
    logging_steps=200,
    save_strategy="epoch",
    save_total_limit=2,
    report_to=[],
)

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    loss=loss,
)

trainer.train()
model.save_pretrained(f"{OUT}/final")
print(f"saved -> {OUT}/final")
