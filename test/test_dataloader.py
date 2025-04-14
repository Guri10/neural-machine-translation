
import os
import sys
# Add root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import DataLoader
from scripts.dataset import TranslationDataset, collate_fn

# Load dataset from processed CSV
dataset = TranslationDataset("data/processed/train.csv")

# Create DataLoader
loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

# Preview one batch
for en_batch, fr_batch in loader:
    print("English batch shape:", en_batch.shape)
    print("French batch shape:", fr_batch.shape)
    print("Sample English (indices):", en_batch[0])
    print("Sample French (indices):", fr_batch[0])
    break
