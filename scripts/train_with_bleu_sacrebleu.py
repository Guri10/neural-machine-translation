import os
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import sacrebleu
from torch.utils.data import DataLoader
from nltk.tokenize import word_tokenize
import nltk
nltk.download("punkt")
nltk.download("punkt_tab")
from datetime import datetime
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
PATIENCE = 4

# Load datasets
train_data = TranslationDataset("data/processed/train.csv")
val_data = TranslationDataset("data/processed/val.csv", train_data.en_vocab, train_data.fr_vocab)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
val_pairs = list(zip(val_data.en_sentences, val_data.fr_sentences))

# Build model
attn = Attention(HIDDEN_DIM, HIDDEN_DIM)
encoder = Encoder(len(train_data.en_vocab), EMB_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
decoder = Decoder(len(train_data.fr_vocab), EMB_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT, attn)
model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=train_data.fr_vocab.stoi["<pad>"])

# Tracking bests
best_bleu = 0
best_val_loss = float("inf")
epochs_since_improvement = 0
bleu_history = []
val_loss_history = []

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

    return outputs

# Training loop
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    start_time = time.time()

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

    # BLEU scoring
    predictions = []
    references = []
    for src, tgt in val_pairs:
        pred_ids = translate(model, src, train_data.en_vocab, train_data.fr_vocab, DEVICE)
        pred = " ".join([train_data.fr_vocab.itos[i] for i in pred_ids])
        predictions.append(pred)
        references.append(tgt)

    bleu_result = sacrebleu.corpus_bleu(predictions, [references])
    avg_bleu = bleu_result.score

    # Validation loss
    val_loss_total = 0
    model.eval()
    with torch.no_grad():
        for src, trg in val_loader:
            src, trg = src.to(DEVICE), trg.to(DEVICE)
            output = model(src, trg)
            output = output[:, 1:].reshape(-1, len(train_data.fr_vocab))
            trg = trg[:, 1:].reshape(-1)
            loss = criterion(output, trg)
            val_loss_total += loss.item()
    avg_val_loss = val_loss_total / len(val_loader)

    # Save best model if BLEU and val_loss improved
    if avg_bleu > best_bleu and avg_val_loss < best_val_loss:
        best_bleu = avg_bleu
        best_val_loss = avg_val_loss
        epochs_since_improvement = 0
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        os.makedirs("checkpoints", exist_ok=True)
        save_path = f"checkpoints/seq2seq_best_bleu_{timestamp}.pt"
        torch.save(model.state_dict(), save_path)
        print(f"📦 Saved best model: BLEU={avg_bleu:.2f}, Val Loss={avg_val_loss:.4f} at {save_path}")
    else:
        epochs_since_improvement += 1
        print(f"⚠️ No improvement. Patience left: {PATIENCE - epochs_since_improvement}")

    # Log scores
    bleu_history.append(avg_bleu)
    val_loss_history.append(avg_val_loss)

    print(f"\n🔁 Epoch {epoch+1}/{EPOCHS}")
    print(f"🧮 Train Loss: {avg_train_loss:.4f}")
    print(f"📉 Val Loss: {avg_val_loss:.4f}")
    print(f"📊 SacreBLEU: {avg_bleu:.2f}")
    print(f"⏱️ Time: {time.time() - start_time:.2f}s")
    print("-" * 60)

    # Early stopping
    if epochs_since_improvement >= PATIENCE:
        print("🛑 Early stopping triggered.")
        break

# 🔥 Plotting after training
plt.figure(figsize=(8, 6))
plt.plot(bleu_history, label='SacreBLEU Score')
plt.plot(val_loss_history, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.title('Training Progress')
plt.legend()
plt.grid()
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/bleu_val_loss_curve.png")
plt.show()
