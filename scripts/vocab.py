import os
import sys
# Add root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from collections import Counter
import re

SPECIAL_TOKENS = {
    "<pad>": 0,
    "<sos>": 1,
    "<eos>": 2,
    "<unk>": 3
}

class Vocabulary:
    def __init__(self, freq_threshold=2):
        self.freq_threshold = freq_threshold
        self.itos = {i: tok for tok, i in SPECIAL_TOKENS.items()}
        self.stoi = {tok: i for tok, i in SPECIAL_TOKENS.items()}

    def __len__(self):
        return len(self.itos)

    def tokenizer(self, text):
        # basic lowercase + punctuation removal
        text = re.sub(r"[^\w\s']", '', text.lower())
        return text.strip().split()

    def build_vocab(self, sentence_list):
        frequencies = Counter()
        idx = len(SPECIAL_TOKENS)

        for sentence in sentence_list:
            tokens = self.tokenizer(sentence)
            frequencies.update(tokens)

        for word, freq in frequencies.items():
            if freq >= self.freq_threshold and word not in self.stoi:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

    def numericalize(self, text):
        tokens = self.tokenizer(text)
        return [self.stoi.get(token, self.stoi["<unk>"]) for token in tokens]
