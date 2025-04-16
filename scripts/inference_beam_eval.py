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
import heapq
from models.encoder import Encoder
from models.attention import Attention
from models.decoder import Decoder
from models.seq2seq import Seq2Seq
from scripts.dataset import TranslationDataset

# nltk.download("punkt")
# nltk.download('punkt_tab')
smoothie = SmoothingFunction().method4

def translate_sentence_beam(model, sentence, en_vocab, fr_vocab, device, max_len=50, beam_width=3):
    model.eval()
    tokens = en_vocab.tokenizer(sentence)
    numericalized = [en_vocab.stoi.get(word, en_vocab.stoi["<unk>"]) for word in tokens]
    src_tensor = torch.tensor([en_vocab.stoi["<sos>"]] + numericalized + [en_vocab.stoi["<eos>"]]).unsqueeze(0).to(device)

    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(src_tensor)

    # Beam stores: (log_prob, [token_indices], hidden, cell)
    beams = [(0.0, [fr_vocab.stoi["<sos>"]], hidden, cell)]

    completed_sequences = []

    for _ in range(max_len):
        new_beams = []

        for log_prob, seq, hidden, cell in beams:
            input_token = torch.tensor([seq[-1]]).to(device)

            with torch.no_grad():
                output, hidden, cell, _ = model.decoder(input_token, hidden, cell, encoder_outputs)

            probs = torch.log_softmax(output, dim=1).squeeze(0)  # [vocab_size]

            top_probs, top_indices = probs.topk(beam_width)

            for i in range(beam_width):
                next_token = top_indices[i].item()
                next_log_prob = log_prob + top_probs[i].item()

                new_seq = seq + [next_token]
                new_beams.append((next_log_prob, new_seq, hidden, cell))

        # Keep top `beam_width` beams
        beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_width]

        # Check if any sequences are completed
        for beam in beams:
            if beam[1][-1] == fr_vocab.stoi["<eos>"]:
                completed_sequences.append(beam)

        # Stop early if enough completed
        if len(completed_sequences) >= beam_width:
            break

    # Choose best completed or fallback to best partial
    final_seq = max(completed_sequences, key=lambda x: x[0])[1] if completed_sequences else beams[0][1]
    translated_tokens = [fr_vocab.itos[idx] for idx in final_seq[1:-1]]  # skip <sos> and <eos>

    return " ".join(translated_tokens)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="checkpoints/seq2seq_best_bleu_20250416-214530.pt")
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--beam_width', type=int, default=1)
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

        pred = translate_sentence_beam(model, src, en_vocab, fr_vocab, DEVICE, beam_width=args.beam_width)


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
