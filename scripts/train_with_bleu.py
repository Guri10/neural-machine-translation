import os
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import nltk
nltk.download("punkt")
from datetime import datetime

# Add root directory to sys.path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.dataset import TranslationDataset, collate_fn
from models.encoder import Encoder
from models.attention import Attention
from models.decoder import Decoder
from models.seq2seq import Seq2Seq

# Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EMB_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 1
DROPOUT = 0.5
EPOCHS = 15
CLIP = 1

# Load datasets
train_data = TranslationDataset("data/processed/train.csv")
val_data = TranslationDataset("data/processed/val.csv", train_data.en_vocab, train_data.fr_vocab)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_pairs = list(zip(val_data.en_sentences, val_data.fr_sentences))

# Build model
attn = Attention(HIDDEN_DIM, HIDDEN_DIM)
encoder = Encoder(len(train_data.en_vocab), EMB_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
decoder = Decoder(len(train_data.fr_vocab), EMB_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT, attn)
model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=train_data.fr_vocab.stoi["<pad>"])
smoothie = SmoothingFunction().method4
best_bleu = 0

def translate_with_attention(model, sentence, en_vocab, fr_vocab, device, max_len=50):
    model.eval()
    tokens = en_vocab.tokenizer(sentence)
    indices = [en_vocab.stoi.get(t, en_vocab.stoi["<unk>"]) for t in tokens]
    src_tensor = torch.tensor([en_vocab.stoi["<sos>"]] + indices + [en_vocab.stoi["<eos>"]]).unsqueeze(0).to(device)

    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(src_tensor)

    outputs = []
    attentions = []
    input_token = torch.tensor([fr_vocab.stoi["<sos>"]]).to(device)

    for _ in range(max_len):
        with torch.no_grad():
            output, hidden, cell, attn_weights = model.decoder(input_token, hidden, cell, encoder_outputs)
            top1 = output.argmax(1).item()
            attentions.append(attn_weights.squeeze().cpu())
        if top1 == fr_vocab.stoi["<eos>"]:
            break
        outputs.append(top1)
        input_token = torch.tensor([top1]).to(device)

    return outputs, attentions, tokens

def plot_attention(attns, src_tokens, tgt_tokens, save_path):
    import numpy as np
    attn_matrix = torch.stack(attns).detach().numpy()  # [tgt_len, src_len]
    fig, ax = plt.subplots(figsize=(max(8, len(src_tokens)*0.6), max(4, len(tgt_tokens)*0.5)))
    sns.heatmap(attn_matrix, xticklabels=src_tokens, yticklabels=tgt_tokens, cmap="YlGnBu", ax=ax)
    ax.set_xlabel("Source Tokens")
    ax.set_ylabel("Target Tokens")
    plt.xticks(rotation=45)
    plt.tight_layout()
    os.makedirs("attention_maps", exist_ok=True)
    plt.savefig(save_path)
    plt.close()

# Training loop
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    start = time.time()

    for batch_idx, (src, trg) in enumerate(train_loader):
        src, trg = src.to(DEVICE), trg.to(DEVICE)
        optimizer.zero_grad()
        output = model(src, trg)

        output = output[:, 1:].reshape(-1, len(train_data.fr_vocab))
        trg = trg[:, 1:].reshape(-1)

        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optimizer.step()

        epoch_loss += loss.item()
        if (batch_idx + 1) % 50 == 0:
            print(f"✅ Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

    avg_train_loss = epoch_loss / len(train_loader)

    # BLEU on validation set
    bleu_scores = []
    for src, tgt in val_pairs[:100]:
        pred = " ".join([
            train_data.fr_vocab.itos[i] for i in
            translate_with_attention(model, src, train_data.en_vocab, train_data.fr_vocab, DEVICE)[0]
        ])
        pred_tok = word_tokenize(pred, language="french")
        tgt_tok = [word_tokenize(tgt, language="french")]
        bleu_scores.append(sentence_bleu(tgt_tok, pred_tok, smoothing_function=smoothie))
    avg_bleu = sum(bleu_scores) / len(bleu_scores)

    # Save best model
    if avg_bleu > best_bleu:
        best_bleu = avg_bleu
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        os.makedirs("checkpoints", exist_ok=True)
        save_path = f"checkpoints/seq2seq_best_bleu_{timestamp}.pt"
        torch.save(model.state_dict(), save_path)
        print(f"📦 Saved new best model (BLEU: {avg_bleu:.4f}) at {save_path}")

    # 🔍 Save attention heatmap for first val example
    src_sample = val_pairs[0][0]
    tgt_sample = val_pairs[0][1]
    pred_ids, attns, src_tokens = translate_with_attention(model, src_sample, train_data.en_vocab, train_data.fr_vocab, DEVICE)
    tgt_tokens = [train_data.fr_vocab.itos[i] for i in pred_ids]
    src_display = src_tokens[:attns[0].shape[0]]
    plot_attention(attns, src_display, tgt_tokens, f"attention_maps/epoch{epoch+1}.png")
    print(f"🖼️ Saved attention map: attention_maps/epoch{epoch+1}.png")

    print(f"\n🔁 Epoch {epoch+1}/{EPOCHS}")
    print(f"🧮 Train Loss: {avg_train_loss:.4f}")
    print(f"📊 Val BLEU: {avg_bleu:.4f}")
    print(f"⏱️  Time: {time.time() - start:.2f}s")
    print("-" * 60)
