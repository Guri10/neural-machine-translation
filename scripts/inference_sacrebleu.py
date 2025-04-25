import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import sacrebleu
import argparse
from torch.utils.data import DataLoader
from scripts.dataset import TranslationDataset, collate_fn
from models.encoder import Encoder
from models.attention import Attention
from models.decoder import Decoder
from models.seq2seq import Seq2Seq

# Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMB_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 1
DROPOUT = 0.5

def load_model(model_path, en_vocab_size, fr_vocab_size):
    attn = Attention(HIDDEN_DIM, HIDDEN_DIM)
    encoder = Encoder(en_vocab_size, EMB_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
    decoder = Decoder(fr_vocab_size, EMB_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT, attn)
    model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model

def translate_beam(model, src_sentence, en_vocab, fr_vocab, beam_width=5, max_len=50):
    model.eval()
    tokens = en_vocab.tokenizer(src_sentence)
    indices = [en_vocab.stoi.get(t, en_vocab.stoi["<unk>"]) for t in tokens]
    src_tensor = torch.tensor([en_vocab.stoi["<sos>"]] + indices + [en_vocab.stoi["<eos>"]]).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(src_tensor)

    beams = [(0.0, [fr_vocab.stoi["<sos>"]], hidden, cell)]

    for _ in range(max_len):
        new_beams = []
        for log_prob, seq, hidden, cell in beams:
            input_token = torch.tensor([seq[-1]]).to(DEVICE)
            with torch.no_grad():
                output, hidden, cell, _ = model.decoder(input_token, hidden, cell, encoder_outputs)
            probs = torch.log_softmax(output, dim=1).squeeze(0)
            top_probs, top_indices = probs.topk(beam_width)

            for i in range(beam_width):
                next_token = top_indices[i].item()
                new_log_prob = log_prob + top_probs[i].item()
                new_seq = seq + [next_token]
                new_beams.append((new_log_prob, new_seq, hidden, cell))

        beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_width]

        # Early stopping if <eos> reached in all beams
        if all(b[1][-1] == fr_vocab.stoi["<eos>"] for b in beams):
            break

    final_seq = max(beams, key=lambda x: x[0])[1]
    translated_tokens = [fr_vocab.itos[idx] for idx in final_seq[1:-1]]
    return " ".join(translated_tokens)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model checkpoint")
    parser.add_argument('--num_samples', type=int, default=500, help="Number of samples to evaluate")
    parser.add_argument('--beam_width', type=int, default=5, help="Beam width for beam search decoding")
    args = parser.parse_args()

    # Load dataset and vocab
    train_dataset = TranslationDataset("data/processed/train.csv")
    test_dataset = TranslationDataset("data/processed/test.csv", train_dataset.en_vocab, train_dataset.fr_vocab)
    en_vocab = train_dataset.en_vocab
    fr_vocab = train_dataset.fr_vocab
    dataset = test_dataset


    model = load_model(args.model_path, len(en_vocab), len(fr_vocab))

    predictions = []
    references = []

    for idx in range(min(args.num_samples, len(dataset))):
        src_sentence = dataset.en_sentences[idx]
        tgt_sentence = dataset.fr_sentences[idx]

        pred_sentence = translate_beam(model, src_sentence, en_vocab, fr_vocab, beam_width=args.beam_width)
        
        predictions.append(pred_sentence)
        references.append(tgt_sentence)

        if idx < 10:  # Print first 10 examples
            print(f"\n👉 Input: {src_sentence}")
            print(f"🔁 Predicted: {pred_sentence}")
            print(f"🎯 Reference: {tgt_sentence}")

    bleu_result = sacrebleu.corpus_bleu(predictions, [references])
    print(f"\n📊 Corpus SacreBLEU Score: {bleu_result.score:.2f}")
