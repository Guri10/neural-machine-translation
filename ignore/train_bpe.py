import os
import sys
# Add root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from datetime import datetime

from bpe_tokenizer import BPETokenizer
from dataset_bpe import BPETranslationDataset, bpe_collate_fn
from models.encoder import Encoder
from models.attention import Attention
from models.decoder import Decoder
from models.seq2seq import Seq2Seq

# Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 15
BATCH_SIZE = 32
EMB_DIM = 256
HIDDEN_DIM = 512
DROPOUT = 0.5
CLIP = 1
BLEU_EVAL_SAMPLES = 100

# Tokenizers
bpe_src = BPETokenizer("data/bpe/bpe.model")
bpe_tgt = BPETokenizer("data/bpe/bpe.model")

# Datasets
train_data = BPETranslationDataset("data/processed/train.csv", bpe_src, bpe_tgt)
val_data = BPETranslationDataset("data/processed/val.csv", bpe_src, bpe_tgt)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=bpe_collate_fn)
val_pairs = list(zip(val_data.src_sentences, val_data.tgt_sentences))

# Model
attn = Attention(HIDDEN_DIM, HIDDEN_DIM)
encoder = Encoder(len(bpe_src), EMB_DIM, HIDDEN_DIM, 1, DROPOUT)
decoder = Decoder(len(bpe_tgt), EMB_DIM, HIDDEN_DIM, 1, DROPOUT, attn)
model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=0)
smoothie = SmoothingFunction().method4
best_bleu = 0

def translate(model, sentence, tokenizer_src, tokenizer_tgt, device, max_len=50):
    model.eval()
    ids = tokenizer_src.encode(sentence)[:max_len]
    ids = [tokenizer_src.sos_id()] + ids + [tokenizer_src.eos_id()]
    src_tensor = torch.tensor(ids).unsqueeze(0).to(device)

    with torch.no_grad():
        enc_out, hidden, cell = model.encoder(src_tensor)

    outputs = []
    input_token = torch.tensor([tokenizer_tgt.sos_id()]).to(device)

    for _ in range(max_len):
        with torch.no_grad():
            output, hidden, cell, _ = model.decoder(input_token, hidden, cell, enc_out)
            top1 = output.argmax(1).item()
        if top1 == tokenizer_tgt.eos_id():
            break
        outputs.append(top1)
        input_token = torch.tensor([top1]).to(device)

    return tokenizer_tgt.decode(outputs)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    start = time.time()

    for batch_idx, (src, tgt) in enumerate(train_loader):
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
        optimizer.zero_grad()
        output = model(src, tgt)

        output = output[:, 1:].reshape(-1, output.shape[-1])
        tgt = tgt[:, 1:].reshape(-1)

        loss = criterion(output, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optimizer.step()
        epoch_loss += loss.item()

        if (batch_idx + 1) % 50 == 0:
            print(f"✅ Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

    avg_train_loss = epoch_loss / len(train_loader)

    # BLEU on val
    bleu_scores = []
    for src_text, tgt_text in val_pairs[:BLEU_EVAL_SAMPLES]:
        pred = translate(model, src_text, bpe_src, bpe_tgt, DEVICE)
        ref = word_tokenize(tgt_text, language="french")
        hyp = word_tokenize(pred, language="french")
        bleu_scores.append(sentence_bleu([ref], hyp, smoothing_function=smoothie))

    avg_bleu = sum(bleu_scores) / len(bleu_scores)

    # Save best model
    if avg_bleu > best_bleu:
        best_bleu = avg_bleu
        os.makedirs("checkpoints", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = f"checkpoints/seq2seq_bpe_best_bleu_{timestamp}.pt"
        torch.save(model.state_dict(), path)
        print(f"📦 Saved best model with BLEU {avg_bleu:.4f} to {path}")

    print(f"\n🔁 Epoch {epoch+1}/{EPOCHS}")
    print(f"🧮 Train Loss: {avg_train_loss:.4f}")
    print(f"📊 Val BLEU: {avg_bleu:.4f}")
    print(f"⏱️  Time: {time.time() - start:.2f}s")
    print("-" * 60)
