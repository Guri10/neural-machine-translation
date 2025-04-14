import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import nltk
nltk.download("punkt")
nltk.download('punkt_tab')
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
CHECKPOINT_PATH = "checkpoints/seq2seq_best_bleu.pt"

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

# Optimizer & loss
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=train_data.fr_vocab.stoi["<pad>"])
smoothie = SmoothingFunction().method4
best_bleu = 0

def translate(model, sentence, en_vocab, fr_vocab, device, max_len=50):
    model.eval()
    tokens = en_vocab.tokenizer(sentence)
    indices = [en_vocab.stoi.get(t, en_vocab.stoi["<unk>"]) for t in tokens]
    src_tensor = torch.tensor([en_vocab.stoi["<sos>"]] + indices + [en_vocab.stoi["<eos>"]]).unsqueeze(0).to(device)

    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(src_tensor)

    outputs = []
    input_token = torch.tensor([fr_vocab.stoi["<sos>"]]).to(device)

    for _ in range(max_len):
        with torch.no_grad():
            output, hidden, cell, _ = model.decoder(input_token, hidden, cell, encoder_outputs)
            top1 = output.argmax(1).item()
        if top1 == fr_vocab.stoi["<eos>"]:
            break
        outputs.append(top1)
        input_token = torch.tensor([top1]).to(device)

    return " ".join([fr_vocab.itos[i] for i in outputs])

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

    # ✅ Log every N batches
    if (batch_idx + 1) % 50 == 0 or (batch_idx == 0):
        print(f"✅ Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")


    avg_train_loss = epoch_loss / len(train_loader)

    # BLEU on validation set
    bleu_scores = []
    for src, tgt in val_pairs[:100]:  # limit for speed
        pred = translate(model, src, train_data.en_vocab, train_data.fr_vocab, DEVICE)
        pred_tok = word_tokenize(pred, language="french")
        tgt_tok = [word_tokenize(tgt, language="french")]
        bleu = sentence_bleu(tgt_tok, pred_tok, smoothing_function=smoothie)
        bleu_scores.append(bleu)

    avg_bleu = sum(bleu_scores) / len(bleu_scores)



    if avg_bleu > best_bleu:
        best_bleu = avg_bleu
        os.makedirs("checkpoints", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        save_path = f"checkpoints/seq2seq_best_bleu_{timestamp}.pt"
        torch.save(model.state_dict(), save_path)
        print(f"📦 Saved new best model (BLEU: {avg_bleu:.4f}) at {save_path}")


    print(f"\n🔁 Epoch {epoch+1}/{EPOCHS}")
    print(f"🧮 Train Loss: {avg_train_loss:.4f}")
    print(f"📊 Val BLEU: {avg_bleu:.4f}")
    print(f"⏱️  Time: {time.time() - start:.2f}s")
    print("-" * 60)
