import os
import sys
# Add root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from models.encoder import Encoder
from models.attention import Attention
from models.decoder import Decoder
from models.seq2seq import Seq2Seq
from scripts.dataset import TranslationDataset
from scripts.inference_beam_eval import translate_sentence_beam

# Setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load vocab
dataset = TranslationDataset("data/processed/train.csv")
en_vocab = dataset.en_vocab
fr_vocab = dataset.fr_vocab

# Load model (config must match training)
attn = Attention(512, 512)
encoder = Encoder(len(en_vocab), 256, 512, 1, 0.5)
decoder = Decoder(len(fr_vocab), 256, 512, 1, 0.5, attn)
model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)

# Load checkpoint
model_path = "checkpoints/seq2seq_best_bleu_20250414-230424.pt"  # Update with your best checkpoint
model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model.eval()

# User input loop
print("📝 Enter an English sentence (type 'exit' to quit):")
while True:
    sentence = input("\n👉 Your input: ").strip()
    if sentence.lower() in ["exit", "quit"]:
        print("👋 Exiting.")
        break

    translation = translate_sentence_beam(model, sentence, en_vocab, fr_vocab, DEVICE, beam_width=5)
    print(f"🇫🇷 Translation: {translation}")
