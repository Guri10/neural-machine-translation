# 🔤 Neural Machine Translation (English ↔ French)

This project implements a custom neural machine translation (NMT) system using a sequence-to-sequence architecture with attention.

---

## 🚀 Features

- Encoder-Decoder model with Luong-style attention
- Beam search decoding
- BLEU score evaluation on validation/test sets
- Clean modular PyTorch code
- Fully configurable training + inference pipeline

---

## 🧠 Model Architecture

- **Encoder**: LSTM with embeddings
- **Attention**: Bahdanau’s additive attention
- **Decoder**: LSTM with context vector + output projection
- **Training**: Teacher forcing + CrossEntropyLoss

---

## 📊 Dataset

- Source: [Tatoeba English-French sentence pairs](https://tatoeba.org/en/downloads)
- Preprocessed to `train.csv`, `val.csv`, `test.csv` in `data/processed/`

---

## 🏋️ Training

```bash
python scripts/train.py
```
