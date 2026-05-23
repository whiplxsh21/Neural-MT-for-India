# Building a Neural Machine Translation System for India

**Dhruv Gupta · 240354**

Neural Machine Translation from English to Hindi and Bengali, built entirely from scratch in PyTorch. Three architectures explored across 25 training-phase submissions, ending with an optimized Transformer that ranked **14th on the leaderboard**.

---

## Competition Results

| Metric | Score |
|--------|-------|
| **Leaderboard Rank (Train Phase)** | **14th** |
| chrF++ | 0.412 |
| ROUGE | 0.431 |
| BLEU | 0.157 |
| Total Submissions (Training Phase) | 25 |

---

## Problem

Machine Translation for Indian languages is fundamentally different from translating between European languages. Two main challenges: tokenization is hard due to semantic and structural differences, and most Indian languages have very limited digital presence, meaning dataset availability is the primary bottleneck. Projects like IndicNLP and IndicBERT have improved the situation, but resources remain scarce.

**Setup:** A media company that publishes in English wants to expand to Hindi and Bengali. The constraint was to build an encoder-decoder NMT system using only PyTorch, from scratch, with no pre-trained resources from HuggingFace or similar. Evaluation via chrF++, BLEU, and ROUGE.

**Rules:**
- All models trained from scratch in PyTorch (no HuggingFace, no pre-built seq2seq modules)
- Only static embeddings like Word2Vec or GloVe permitted as pre-trained resources (unused)
- Only the provided datasets allowed

---

## Data

| | English to Hindi | English to Bengali |
|-|-----------------|-------------------|
| Training pairs | 80,797 | 68,849 |
| Test sentences | 23,085 | 19,672 |
| **Total train** | | **149,646** |
| **Total test** | | **42,757** |

2,791 duplicate sentence pairs were found across the combined dataset (~1.8% of data). Not removed, since they posed minimal bias risk.

---

## Data Analysis (EDA)

EDA ran before any model work and directly shaped architecture decisions, not the other way around.

### Sentence Length Distribution

**Finding:** Bengali was the most compact, with 95.39% of sentences under 25 tokens. English and Hindi both had ~85% of sentences under 25 tokens. This set `MAX_LENGTH = 25` globally, capturing the vast majority of the corpus without padding waste.

<img width="740" height="369" alt="image" src="https://github.com/user-attachments/assets/2657476d-78aa-4c3e-8b84-68f3e4a212ae" />


### Source-Target Length Correlation

English-Hindi lengths had a Pearson correlation of 0.92, and English-Bengali 0.88. Both pairs cluster tightly around the y=x line. This confirmed that the sequence lengths are mostly equivalent across languages, so no special positional encoding adjustments were needed for the transformer.

<img width="668" height="450" alt="image" src="https://github.com/user-attachments/assets/624d4524-da3e-464f-a756-ea06810e2404" />


### Vocabulary Distribution

Top tokens across all three languages are dominated by grammatical function words (stop words, auxiliary verbs, conjunctions, prepositions). Pruning these would have broken sentence structure, so the full vocabulary was kept.

Final vocabulary sizes:
- English: ~57k types
- Hindi: ~75k types
- Bengali: ~105k types

The Bengali vocabulary is nearly twice the size of Hindi's, with a type-token ratio of 0.108 vs 0.053. Bengali is morphologically richer, meaning the decoder is predicting over a much larger output space at each step.

<img width="648" height="272" alt="image" src="https://github.com/user-attachments/assets/16b0a696-6168-4ea6-ae0f-bb79e228a14f" />


---

## Preprocessing Pipeline

```
raw text
  → HTML entity decode
  → normalise backslashes
  → lowercase
  → remove punctuation and digits
  → (target only: strip stray English characters)
  → NLTK word tokenize
  → insert <SOS> / <EOS>
  → pad or truncate to length 25
```

`nltk.word_tokenize` was sufficient for this task. BPE was experimented with but dropped due to submission issues. All three languages use a `Vocab` object built per-language, including SOS, EOS, and PAD tokens.

---

## Model Architecture

The project replicated the historical order of development in NMT, systematically increasing complexity across 25 submissions.

### Architecture Progression and chrF++ Scores

| Submission Range | Architecture | Best chrF++ |
|-----------------|-------------|-------------|
| Subs 1-5 | Stacked Bi-LSTM (2-layer) | 0.21 |
| Subs 6-12 | Bi-GRU + Bahdanau Attention | 0.27 |
| Subs 13-16 | Baseline Transformer (3-layer) | 0.34 |
| **Subs 17-25** | **Optimized Transformer** | **0.41** |

---

### Phase 1: Baseline Seq2Seq (Submissions 1-5)

Vanilla RNN, LSTM, GRU, and their bidirectional variants. Single-layer models couldn't capture long-term dependencies. The best score in this phase came from Bi-LSTM at 0.21. The core problem with all of these is the fixed-size context vector: the entire source sentence gets compressed into one vector, which is a hard bottleneck.

---

### Phase 2: Attention Models (Submissions 6-12)

Added Bahdanau and Luong attention mechanisms on top of Bi-GRU and Bi-LSTM. The decoder now attends over all encoder hidden states rather than just the final one. Long-range dependencies improved substantially. Score jumped to ~0.27. Stacked LSTMs and Stacked Bi-LSTMs were also tried but didn't beat the attention models. Below figure depicts the encoder-decoder architecture for these models:

<img width="714" height="442" alt="image" src="https://github.com/user-attachments/assets/6beac65d-c58f-4561-bdd9-9ca8b59409f5" />


---

### Phase 3: Baseline Transformer (Submissions 13-16)

Standard 3-layer transformer trained for ~30 epochs. Even the vanilla transformer severely outperformed the best Seq2Seq models, scoring ~0.35. Parallelization and self-attention were the differentiators.

---

### Phase 4: Optimized Transformer (Submissions 17-25, Final)

Two changes on top of the vanilla transformer drove the final score to 0.41:

**1. Noam Learning Rate Schedule** (from Vaswani et al., "Attention is All You Need"):
```
lr = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))
```
Training with standard Adam was unstable. This schedule linearly warms up for 4000 steps, then decays as the inverse square root of the step count. No manual LR tuning required.

**2. Weight Tying** (from Press & Wolf, 2017):
The decoder's token embedding matrix and the final linear projection layer share weights:
```python
self.decoder.fc_out.weight = self.decoder.tok_embedding.weight
```
This reduces total parameters by millions and forces the embedding space and output logit space to stay semantically aligned. It produced the single largest performance jump of any individual change. The graph below depicts the best performing model’s architecture:

<img width="745" height="671" alt="image" src="https://github.com/user-attachments/assets/280754c8-3b5b-47d4-baa3-1dc437dd47ee" />


---

### Final Model Hyperparameters

| Hyperparameter | Value | Rationale |
|---------------|-------|-----------|
| D_MODEL | 256 | Tradeoff between capacity and compute cost |
| N_HEADS | 8 | D_MODEL must be divisible by N_HEADS |
| N_LAYERS | 3 | Original paper used 6; halved for faster training |
| D_FF | 512 | 2x D_MODEL by convention |
| DROPOUT | 0.1 | Standard rate |
| BATCH_SIZE | 64 | Largest that fits in GPU memory |
| NUM_EPOCHS | 21 | Could have increased further |
| WARMUP_STEPS | 4000 | Directly from the Attention is All You Need paper |
| MAX_LENGTH | 25 | Derived from EDA (see above) |
| CLIP | 1.0 | Gradient clipping to prevent explosion |
| Weight init | Xavier uniform | |

---

## Training Results

**Inference:** Greedy decoding (argmax at each step, fed back as next input, stops on `<EOS>` or max length). Beam search was not implemented due to compute constraints, but would likely improve scores.

### English to Bengali

21 epochs · 1,076 batches/epoch · ~82 min training · ~32 min inference

| Epoch | Loss |
|-------|------|
| 1 | 8.999 |
| 5 | 6.075 |
| 10 | 4.875 |
| 15 | 4.474 |
| **21** | **4.125** |


### English to Hindi

21 epochs · 1,263 batches/epoch · ~28 min training · ~36 min inference

| Epoch | Loss |
|-------|------|
| 1 | 7.678 |
| 5 | 4.162 |
| 10 | 3.200 |
| 15 | 2.850 |
| **21** | **2.602** |


Hindi converged faster and reached a lower final loss. This traces directly back to vocabulary size: the Bengali decoder predicts over 105k tokens vs 75k for Hindi. At each step, Bengali is choosing from ~40% more candidates.

### Final Vocabulary Sizes

| | English (source) | Target |
|-|-----------------|--------|
| Bengali model | 53,923 | 105,528 |
| Hindi model | 57,278 | 75,562 |

**Total translations produced: 42,757** (merged into `answer.csv` and zipped as `submission.zip`)

---

## Error Analysis

**What worked:** The optimized transformer was superior across all settings. Attention resolved the long-range dependency problem that broke the baseline Seq2Seq models. Weight tying and Noam scheduling were the two highest-impact single changes.

**What didn't work:** Stacked LSTMs and Stacked Bi-LSTMs showed minimal improvement over their single-layer counterparts.

**Key insight:** The model performed better on Hindi than Bengali despite Bengali having a smaller training set. Bengali's morphological richness and larger vocabulary made it genuinely harder to translate, not just a dataset size issue.

**Three ways to improve the final 0.41 score:**
1. Switch from greedy decoding to beam search -- this would explore a wider range of candidate translations at inference time
2. Replace `nltk.word_tokenize` with BPE or SentencePiece to handle OOV words and capture morphological patterns that are critical in Indic languages
3. Scale up: the papers I referenced used 6 transformer layers with much higher embedding dimensions. Compute was the only constraint here

**Attention heatmaps** were not implemented (the `MultiHeadAttention` module returns `attention_weights` but extracting and visualizing them was cut for time). This would be a useful next step for interpretability.

---

## Conclusion

A complete NMT pipeline from scratch, producing 42,757 translations across Hindi and Bengali with a final chrF++ of 0.412 and a rank of 14th. The clearest result from the whole project: transformers significantly outperform RNN-based models for this task, and the two optimizations (Noam scheduling + weight tying) pushed well beyond the vanilla transformer baseline.

**Future directions:**
1. Beam search decoder in place of greedy decoding
2. Sub-word tokenizer (BPE or SentencePiece) to handle morphologically rich languages
3. Larger model: 6 layers, higher d_model to increase capacity

---

## Files

```
.
├── FINAL CODE.ipynb                          <- final Transformer (Bengali + Hindi)
├── EDA with traintest data.ipynb             <- EDA notebook
├── FINAL REPORT.pdf                          <- full project report
├── previous attempts/
│   ├── simple_lstm.ipynb                     <- LSTM Seq2Seq prototype
│   ├── Bi-GRU Attention.ipynb                <- Bi-GRU + attention prototype
│   └── optimized_transformer_wth_bpe.ipynb   <- Transformer variant with BPE
└── data/
    ├── train_data1.json                      <- 149,646 training pairs
    ├── val_data1.json
    └── test_data1.json                       <- 42,757 test sentences
```

---

## Tech Stack

PyTorch · NLTK · indic-nlp-library · scikit-learn · pandas · NumPy · Matplotlib · Seaborn · Plotly

Trained on Kaggle (GPU) and Google Colab (GPU).

---

## References

1. P. Koehn and R. Knowles, "Six challenges for neural machine translation," WNMT 2017.
2. P. Koehn, "Europarl: A parallel corpus for statistical machine translation," MT Summit X, 2005.
3. R. Kumar, S. Sanyal, and P. Talukdar, "A survey of NMT for Indian languages," ICON 2018.
4. D. Bahdanau, K. Cho, and Y. Bengio, "Neural machine translation by jointly learning to align and translate," arXiv:1409.0473, 2014.
5. M.-T. Luong, H. Pham, and C. D. Manning, "Effective approaches to attention-based NMT," arXiv:1508.04025, 2015.
6. A. Vaswani et al., "Attention is all you need," NeurIPS 2017.
7. O. Press and L. Wolf, "Using the output embedding to improve language models," EACL 2017.
