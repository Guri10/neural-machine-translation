import os
import sys
# Add root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from scripts.vocab import Vocabulary

class TranslationDataset(Dataset):
    def __init__(self, data_path, en_vocab=None, fr_vocab=None, freq_threshold=2, max_len=50):
        self.df = pd.read_csv(data_path)
        self.max_len = max_len

        self.en_sentences = self.df['en'].tolist()
        self.fr_sentences = self.df['fr'].tolist()

        self.en_vocab = en_vocab if en_vocab else Vocabulary(freq_threshold)
        self.fr_vocab = fr_vocab if fr_vocab else Vocabulary(freq_threshold)

        if not en_vocab:
            self.en_vocab.build_vocab(self.en_sentences)
        if not fr_vocab:
            self.fr_vocab.build_vocab(self.fr_sentences)

    def __len__(self):
        return len(self.en_sentences)

    def __getitem__(self, idx):
        en = self.en_sentences[idx]
        fr = self.fr_sentences[idx]

        # Numericalize + add <sos> and <eos>
        en_tensor = [self.en_vocab.stoi["<sos>"]] + \
                    self.en_vocab.numericalize(en) + \
                    [self.en_vocab.stoi["<eos>"]]

        fr_tensor = [self.fr_vocab.stoi["<sos>"]] + \
                    self.fr_vocab.numericalize(fr) + \
                    [self.fr_vocab.stoi["<eos>"]]

        return torch.tensor(en_tensor), torch.tensor(fr_tensor)

def collate_fn(batch):
    en_batch = [item[0] for item in batch]
    fr_batch = [item[1] for item in batch]

    en_batch = pad_sequence(en_batch, padding_value=0, batch_first=True)
    fr_batch = pad_sequence(fr_batch, padding_value=0, batch_first=True)

    return en_batch, fr_batch
