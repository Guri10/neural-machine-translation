from __future__ import annotations
import gc, logging
from pathlib import Path
from typing import Dict

import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW                       
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
)

MODEL_NAME   = "facebook/bart-base"  
BATCH_SIZE   = 8
GRAD_ACCUM   = 4                     
EPOCHS       = 3
LR           = 5e-5
MAX_LEN      = 128

DATA_DIR     = Path("/content/neural-machine-translation/data/processed")
CKPT_DIR     = Path("/content/neural-machine-translation/checkpoints")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname).4s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
log.info("Running on %s", DEVICE)


class TranslationDS(Dataset):
    """CSV row → model input dict (tokenised)."""

    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int = MAX_LEN):
        self.df, self.tok, self.max_len = df.reset_index(drop=True), tokenizer, max_len

    def __len__(self): return len(self.df)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        model_inputs = self.tok(
            row["en"],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        with self.tok.as_target_tokenizer():
            labels = self.tok(
                row["fr"],
                truncation=True,
                padding="max_length",
                max_length=self.max_len,
                return_tensors="pt",
            )
        model_inputs["labels"] = labels["input_ids"]
        return {k: v.squeeze(0) for k, v in model_inputs.items()}


def bleu_score(model, loader, tokenizer, metric):
    model.eval()
    preds, refs = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            gen = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=MAX_LEN,
                num_beams=4,
            )
            preds += tokenizer.batch_decode(gen, skip_special_tokens=True)
            refs  += tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)
    model.train()
    return metric.compute(predictions=preds,
                          references=[[r] for r in refs])["score"]


def train_epoch(model, loader, optim, scaler):
    total = 0.0
    for step, batch in enumerate(loader):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        with autocast():
            loss = model(**batch).loss / GRAD_ACCUM
        scaler.scale(loss).backward()

        if (step + 1) % GRAD_ACCUM == 0:
            scaler.step(optim); scaler.update()
            optim.zero_grad(set_to_none=True)
        total += loss.item() * GRAD_ACCUM

        if step % 100 == 0:
            log.info("step %d | loss %.3f", step, loss.item() * GRAD_ACCUM)
        gc.collect()
    return total / len(loader)


def main():
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_csv = DATA_DIR / "val.csv"
    if test_csv.exists():
        val_df = pd.read_csv(test_csv)
        log.info("Validation: using val.csv (%d rows)", len(val_df))
    else:
        val_df = train_df.sample(frac=0.02, random_state=42)
        train_df = train_df.drop(val_df.index).reset_index(drop=True)
        log.info("Validation: sampled %d rows (2%% of train)", len(val_df))

    tok   = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)

    collate = DataCollatorForSeq2Seq(tok, model=model)
    train_dl = DataLoader(
        TranslationDS(train_df, tok), batch_size=BATCH_SIZE,
        shuffle=True,  collate_fn=collate, num_workers=2, pin_memory=True,
    )
    val_dl = DataLoader(
        TranslationDS(val_df, tok),   batch_size=BATCH_SIZE,
        shuffle=False, collate_fn=collate, num_workers=2, pin_memory=True,
    )

    optim  = AdamW(model.parameters(), lr=LR)
    scaler = GradScaler()
    sacre  = evaluate.load("sacrebleu")

    for epoch in range(1, EPOCHS + 1):
        log.info("Epoch %d/%d", epoch, EPOCHS)
        tr_loss = train_epoch(model, train_dl, optim, scaler)
        bleu    = bleu_score(model, val_dl, tok, sacre)
        log.info("epoch %d | train-loss %.4f | BLEU %.2f", epoch, tr_loss, bleu)

        ckpt = CKPT_DIR / f"epoch{epoch}"
        ckpt.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(ckpt); tok.save_pretrained(ckpt)
        log.info("Saved checkpoint → %s", ckpt)

    log.info("Done ✨")


if __name__ == "__main__":
    main()
