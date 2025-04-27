import os
import sys
# Add root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import Dataset
import pandas as pd
from torch.nn.utils.rnn import pad_sequence

class BPETranslationDataset(Dataset):
    def __init__(self, data_path, tokenizer_src, tokenizer_tgt, max_len=50):
        self.df = pd.read_csv(data_path)
        self.src_sentences = self.df['en'].tolist()
        self.tgt_sentences = self.df['fr'].tolist()
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.max_len = max_len

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, idx):
        src = self.tokenizer_src.encode(self.src_sentences[idx])[:self.max_len]
        tgt = self.tokenizer_tgt.encode(self.tgt_sentences[idx])[:self.max_len]

        src = [self.tokenizer_src.sos_id()] + src + [self.tokenizer_src.eos_id()]
        tgt = [self.tokenizer_tgt.sos_id()] + tgt + [self.tokenizer_tgt.eos_id()]

        return torch.tensor(src), torch.tensor(tgt)

def bpe_collate_fn(batch):
    src_batch = [item[0] for item in batch]
    tgt_batch = [item[1] for item in batch]

    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=0)

    return src_batch, tgt_batch