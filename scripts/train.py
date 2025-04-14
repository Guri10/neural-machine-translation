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

# Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EMB_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 1
DROPOUT = 0.5
EPOCHS = 5
CLIP = 1

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

    for batch_idx, (src, trg) in enumerate(train_loader):
        src, trg = src.to(DEVICE), trg.to(DEVICE)

        optimizer.zero_grad()
        output = model(src, trg)

        # output: [batch_size, trg_len, vocab_size]
        # target: [batch_size, trg_len]
        output = output[:, 1:].reshape(-1, fr_vocab_size)
        trg = trg[:, 1:].reshape(-1)

        loss = criterion(output, trg)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss / len(train_loader):.4f}")
