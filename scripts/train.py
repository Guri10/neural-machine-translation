import os
import sys
# Add root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scripts.dataset import TranslationDataset, collate_fn
from models.encoder import Encoder
from models.attention import Attention
from models.decoder import Decoder
from models.seq2seq import Seq2Seq
import time
from datetime import datetime

# Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# BATCH_SIZE = 32
# EMB_DIM = 256
# HIDDEN_DIM = 512
# NUM_LAYERS = 1
# DROPOUT = 0.5
# EPOCHS = 5
# CLIP = 1

BATCH_SIZE = 16         # smaller batch = faster on small data
EMB_DIM = 128           # lower embedding size = faster lookup + smaller model
HIDDEN_DIM = 256        # lower hidden size = smaller LSTM + faster forward/backward pass
NUM_LAYERS = 1          # keep it at 1 for now
EPOCHS = 1              # just 1 epoch to test if it's all working
DROPOUT = 0.2           # lower dropout = faster training
CLIP = 1               # gradient clipping


# Load dataset and build vocab
train_data = TranslationDataset("data/processed/train.csv")
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

en_vocab_size = len(train_data.en_vocab)
fr_vocab_size = len(train_data.fr_vocab)

# Initialize model components
attn = Attention(HIDDEN_DIM, HIDDEN_DIM)
encoder = Encoder(en_vocab_size, EMB_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
decoder = Decoder(fr_vocab_size, EMB_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT, attn)
model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)

# Optimizer and Loss
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=train_data.fr_vocab.stoi["<pad>"])

# Training loop
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    start_time = time.time()

    print(f"\n🔁 Starting epoch {epoch+1}/{EPOCHS}...")

    for batch_idx, (src, trg) in enumerate(train_loader):
        src, trg = src.to(DEVICE), trg.to(DEVICE)

        optimizer.zero_grad()
        output = model(src, trg)  # output: [batch_size, trg_len, vocab_size]

        # Remove <sos> token for loss computation
        output = output[:, 1:].reshape(-1, fr_vocab_size)
        trg = trg[:, 1:].reshape(-1)

        loss = criterion(output, trg)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optimizer.step()

        epoch_loss += loss.item()

        # Log every N batches
        if (batch_idx + 1) % 100 == 0 or batch_idx == 0:
            print(f"  ✅ Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f}")

    avg_loss = epoch_loss / len(train_loader)
    epoch_time = time.time() - start_time

    print(f"\n📊 Epoch {epoch+1} completed in {epoch_time:.2f}s | Average Loss: {avg_loss:.4f}")
    print("-" * 60)


timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
torch.save(model.state_dict(), f"checkpoints/seq2seq_model_{timestamp}.pt")
print(f"✅ Model saved to seq2seq_model_{timestamp}.pt")
print("Training completed.")