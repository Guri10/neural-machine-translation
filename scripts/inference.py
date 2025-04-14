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
import argparse
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize


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

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--model_path', type=str, default="checkpoints/seq2seq_model_20250414-204901.pt")
#     parser.add_argument('--sentence', type=str, required=True)
#     args = parser.parse_args()

#     DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Load vocab from training data
#     dataset = TranslationDataset("data/processed/train.csv")
#     en_vocab = dataset.en_vocab
#     fr_vocab = dataset.fr_vocab

#     # Model sizes should match training
#     # attn = Attention(512, 512)
#     # encoder = Encoder(len(en_vocab), 256, 512, 1, 0.5)
#     # decoder = Decoder(len(fr_vocab), 256, 512, 1, 0.5, attn)

#     # Match training config from Colab quick run
#     attn = Attention(256, 256)
#     encoder = Encoder(len(en_vocab), 128, 256, 1, 0.0)  # dropout=0.0 since only 1 layer
#     decoder = Decoder(len(fr_vocab), 128, 256, 1, 0.0, attn)

#     model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)
#     model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))

#     # Translate input
#     translation = translate_sentence(model, args.sentence, en_vocab, fr_vocab, DEVICE)
#     print(f"\n🗣️  English: {args.sentence}")
#     print(f"🇫🇷 French: {translation}")


import pandas as pd

if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset + vocabs
    dataset = TranslationDataset("data/processed/train.csv")
    en_vocab = dataset.en_vocab
    fr_vocab = dataset.fr_vocab

    # Load model with matching config
    attn = Attention(256, 256)
    encoder = Encoder(len(en_vocab), 128, 256, 1, 0.0)
    decoder = Decoder(len(fr_vocab), 128, 256, 1, 0.0, attn)
    model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)
    model.load_state_dict(torch.load("checkpoints/seq2seq_model_20250414-204901.pt", map_location=DEVICE))
    model.eval()

    # Load test set
    test_df = pd.read_csv("data/processed/test.csv")

    # Translate first 10 sentences
    for i in range(10):
        src = test_df.iloc[i]["en"]
        tgt = test_df.iloc[i]["fr"]
        pred = translate_sentence(model, src, en_vocab, fr_vocab, DEVICE)
        
        print(f"\n🗣️  Input     : {src}")
        print(f"🔁 Predicted : {pred}")
        print(f"🎯 Target    : {tgt}")
        print("-" * 50)

    smoothie = SmoothingFunction().method4

    bleu_scores = []

    for i in range(50):  # Evaluate on first 50 samples
        src = test_df.iloc[i]["en"]
        ref = test_df.iloc[i]["fr"]

        pred = translate_sentence(model, src, en_vocab, fr_vocab, DEVICE)

        pred_tokens = word_tokenize(pred, language="french")
        ref_tokens = [word_tokenize(ref, language="french")]

        score = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoothie)
        bleu_scores.append(score)

    average_bleu = sum(bleu_scores) / len(bleu_scores)
    print(f"\n📊 Average BLEU score on test set: {average_bleu:.4f}")


