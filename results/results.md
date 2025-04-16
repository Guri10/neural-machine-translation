# 🧪 Machine Translation: BLEU Score Tracking

This file logs model performance metrics during training and inference.

---

## ✅ Training Configuration

| Param         | Value                         |
| ------------- | ----------------------------- |
| Epochs        | 15                            |
| Batch Size    | 32                            |
| Embedding Dim | 256                           |
| Hidden Dim    | 512                           |
| Dropout       | 0.5                           |
| Beam Widths   | 1 (greedy), 3, 5              |
| Optimizer     | Adam                          |
| Loss          | CrossEntropy (ignore `<pad>`) |

---

## 📈 BLEU Score Summary

| Checkpoint                        | Epoch | Beam Width | Samples | BLEU Score |
| --------------------------------- | ----- | ---------- | ------- | ---------- |
| seq2seq_model_20250414-213821.pt  | 05    | 1          | 500     | 0.1880     |
| seq2seq_model_20250414-213821.pt  | 05    | 1          | 1000    | 0.1880     |
| seq2seq_model_20250414-213821.pt  | 05    | 3          | 100     | 0.1922     |
| seq2seq_model_20250414-213821.pt  | 05    | 5          | 100     | 0.1934     |
| seq2seq_model_20250414-213821.pt  | 05    | 5          | 500     | 0.1985     |
| seq2seq_best_bleu_20250414-230424 | 15    | 1          | 10      | 0.2274     |
| seq2seq_best_bleu_20250414-230424 | 15    | 3          | 10      | 0.2254     |
| seq2seq_best_bleu_20250414-230424 | 15    | 5          | 10      | 0.2267     |
| seq2seq_best_bleu_20250414-230424 | 15    | 1          | 100     | 0.2022     |
| seq2seq_best_bleu_20250414-230424 | 15    | 3          | 100     | 0.2056     |
| seq2seq_best_bleu_20250414-230424 | 15    | 5          | 100     | 0.2095     |
| seq2seq_best_bleu_20250414-230424 | 15    | 1          | 500     | 0.2093     |
| seq2seq_best_bleu_20250414-230424 | 15    | 3          | 500     | 0.2160     |
| seq2seq_best_bleu_20250414-230424 | 15    | 5          | 500     | 0.2167     |
| seq2seq_best_bleu_20250414-230424 | 15    | 1          | 1000    | 0.1979     |
| seq2seq_best_bleu_20250414-230424 | 15    | 3          | 1000    | 0.2066     |
| seq2seq_best_bleu_20250414-230424 | 15    | 5          | 1000    | 0.2078     |
| seq2seq_best_bleu_20250414-230424 | 15    | 5          | 2000    | 0.2111     |
| seq2seq_best_bleu_20250414-230424 | 15    | 5          | 3000    | 0.2123     |

📝 Add more rows as new models are evaluated.

---

## 🧠 Observations

- Beam search gives slightly higher BLEU (~0.01 gain).
- BLEU plateaus around 0.19–0.20 with current training setup.
- Need longer training or subword tokenization for significant gains.

---

## 🔜 Next Steps

- Try 20+ epochs
- Experiment with BPE tokenization
- Visualize attention weights
