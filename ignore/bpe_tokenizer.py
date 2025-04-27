import os
import sys
# Add root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sentencepiece import SentencePieceProcessor

class BPETokenizer:
    def __init__(self, model_path):
        self.sp = SentencePieceProcessor()
        self.sp.load(model_path)

    def encode(self, text):
        return self.sp.encode(text, out_type=int)

    def decode(self, ids):
        return self.sp.decode(ids)

    def __len__(self):
        return self.sp.get_piece_size()

    def pad_id(self): return self.sp.pad_id()
    def sos_id(self): return self.sp.bos_id()
    def eos_id(self): return self.sp.eos_id()
