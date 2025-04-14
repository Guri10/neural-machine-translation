import os
import sys
# Add root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import pandas as pd
import argparse
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import nltk

from models.encoder import Encoder
from models.attention import Attention
from models.decoder import Decoder
from models.seq2seq import Seq2Seq
from scripts.dataset import TranslationDataset

# nltk.download("punkt")
# nltk.download('punkt_tab')
smoothie = SmoothingFunction().method4

def translate_sentence(model, sentence, en_vocab, fr_vocab, device, max_len=50):
    model.eval()
    tokens = en_vocab.tokenizer(sentence)
    numericalized = [en_vocab.stoi.get(word, en_vocab.stoi["<unk>"]) for word in tokens]
    tensor = torch.tensor([en_vocab.stoi["<sos>"]] + numericalized + [en_vocab.stoi["<eos>"]]).unsqueeze(0).to(device)

    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(tensor)

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

    translated_tokens = [fr_vocab.itos[idx] for idx in outputs]
    return " ".join(translated_tokens)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="checkpoints/seq2seq_model_20250414-213821.pt")
    parser.add_argument('--num_samples', type=int, default=100)
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load vocab
    dataset = TranslationDataset("data/processed/train.csv")
    en_vocab = dataset.en_vocab
    fr_vocab = dataset.fr_vocab

    # Load model (trained with full config)
    attn = Attention(512, 512)
    encoder = Encoder(len(en_vocab), 256, 512, 1, 0.5)
    decoder = Decoder(len(fr_vocab), 256, 512, 1, 0.5, attn)
    model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)
    model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
    model.eval()

    # Load test data
    test_df = pd.read_csv("data/processed/test.csv")

    print(f"\n🔍 Testing on {args.num_samples} examples from test.csv...\n")
    bleu_scores = []

    for i in range(args.num_samples):
        src = test_df.iloc[i]['en']
        tgt = test_df.iloc[i]['fr']

        pred = translate_sentence(model, src, en_vocab, fr_vocab, DEVICE)

        print(f"🗣️  Input     : {src}")
        print(f"🔁 Predicted : {pred}")
        print(f"🎯 Target    : {tgt}")
        print("-" * 50)

        pred_tokens = word_tokenize(pred, language="french")
        tgt_tokens = [word_tokenize(tgt, language="french")]

        score = sentence_bleu(tgt_tokens, pred_tokens, smoothing_function=smoothie)
        bleu_scores.append(score)

    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    print(f"\n📊 Average BLEU score over {args.num_samples} samples: {avg_bleu:.4f}")
