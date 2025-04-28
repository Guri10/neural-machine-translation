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

| Checkpoint                                 | Epoch | Beam Width | Samples | BLEU Score | Train Dataset size |
| ------------------------------------------ | ----- | ---------- | ------- | ---------- | ------------------ |
| seq2seq_model_20250414-213821.pt           | 05    | 1          | 500     | 0.1880     | 25%                |
| seq2seq_model_20250414-213821.pt           | 05    | 1          | 1000    | 0.1880     | 25%                |
| seq2seq_model_20250414-213821.pt           | 05    | 3          | 100     | 0.1922     | 25%                |
| seq2seq_model_20250414-213821.pt           | 05    | 5          | 100     | 0.1934     | 25%                |
| seq2seq_model_20250414-213821.pt           | 05    | 5          | 500     | 0.1985     | 25%                |
| seq2seq_best_bleu_20250414-230424          | 15    | 1          | 10      | 0.2274     | 50%                |
| seq2seq_best_bleu_20250414-230424          | 15    | 3          | 10      | 0.2254     | 50%                |
| seq2seq_best_bleu_20250414-230424          | 15    | 5          | 10      | 0.2267     | 50%                |
| seq2seq_best_bleu_20250414-230424          | 15    | 1          | 100     | 0.2022     | 50%                |
| seq2seq_best_bleu_20250414-230424          | 15    | 3          | 100     | 0.2056     | 50%                |
| seq2seq_best_bleu_20250414-230424          | 15    | 5          | 100     | 0.2095     | 50%                |
| seq2seq_best_bleu_20250414-230424          | 15    | 1          | 500     | 0.2093     | 50%                |
| seq2seq_best_bleu_20250414-230424          | 15    | 3          | 500     | 0.2160     | 50%                |
| seq2seq_best_bleu_20250414-230424          | 15    | 5          | 500     | 0.2167     | 50%                |
| seq2seq_best_bleu_20250414-230424          | 15    | 1          | 1000    | 0.1979     | 50%                |
| seq2seq_best_bleu_20250414-230424          | 15    | 3          | 1000    | 0.2066     | 50%                |
| seq2seq_best_bleu_20250414-230424          | 15    | 5          | 1000    | 0.2078     | 50%                |
| seq2seq_best_bleu_20250414-230424          | 15    | 5          | 2000    | 0.2111     | 50%                |
| seq2seq_best_bleu_20250414-230424          | 15    | 5          | 3000    | 0.2123     | 50%                |
| seq2seq_best_bleu_20250416-214530          | 15    | 5          | 1000    | 0.2165     | 50%                |
| seq2seq_best_bleu_20250416-214530          | 15    | 5          | 2000    | 0.2271     | 50%                |
| seq2seq_fulldata_best_bleu_20250427-000059 | 10    | 5          | 10      | 0.2544     | 100%               |
| seq2seq_fulldata_best_bleu_20250427-000059 | 10    | 5          | 100     | 0.2712     | 100%               |
| seq2seq_fulldata_best_bleu_20250427-000059 | 10    | 5          | 1000    | 0.2638     | 100%               |
| seq2seq_fulldata_best_bleu_20250427-000059 | 10    | 5          | 2000    | 0.2667     | 100%               |
| seq2seq_fulldata_best_bleu_20250427-000059 | 10    | 5          | 3000    | 0.2677     | 100%               |
| seq2seq_fulldata_best_bleu_20250427-000059 | 10    | 5          | 41000   | 0.2684     | 100%               |

📝 Add more rows as new models are evaluated.

---

## 🧠 Observations

- ✅ Training for 15 epochs significantly improved BLEU, especially with beam search.

- 🚀 Beam search (width = 5) consistently outperforms greedy decoding by ~0.01–0.02 BLEU across all sample sizes.

- 📈 BLEU continues to climb with more evaluation samples:

- - From 0.1985 (500) → 0.2111 (2000) → 0.2123 (3000)

- 🔁 The model generalizes well across longer test sets — BLEU doesn’t collapse as sample size increases.

- 📉 BLEU on greedy decoding plateaus around 0.188–0.209, while beam width 5 pushes it past 0.21

- 🔎 Training was done on only ~24% of the dataset (~100k samples) — so there’s headroom for improvement with more data.

---

## 🔜 Next Steps

- 📈 Train on more data — Try increasing MAX_SAMPLES to 200k–300k or the full 417k for even better performance.

- ⏳ Train longer — Go beyond 15 epochs with early stopping based on validation BLEU.

- ✂️ Switch to subword tokenization (BPE) — Helps with OOV words and morphology, expected BLEU gain: +0.03 to +0.07

- 🧪 Tune decoding parameters — Experiment with beam width 6–10, or apply length normalization during beam search.

- 📊 Track per-epoch BLEU in a CSV — For plotting learning curves and analyzing training dynamics.

- 🧠 Add attention visualization — Helps interpret and debug alignment between source and target.

- 📦 Create a leaderboard-style summary — Show checkpoint, training config, and BLEU in one place (great for reports/presentations).
