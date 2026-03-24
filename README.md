# lecture6 — GPT-Style Transformer on Medical Dialog

A minimal, educational GPT-style transformer trained on Reddit doctor-patient Q&A (`medical_dialog.txt`). Covers the full ML pipeline: data loading, tokenization, training, evaluation, and text generation. Supports both character-level and BPE subword tokenization.

---

## Quick Start

```bash
pip install -r requirements.txt

# Train (BPE tokenizer, full model, ~1hr on RTX 4090)
python train.py \
  --input medical_dialog.txt \
  --tokenizer bpe --bpe-vocab-size 2000 --bpe-model medical_bpe.model \
  --n-embd 384 --n-head 6 --n-layer 6 \
  --context-size 256 --batch-size 64 --dropout 0.35 \
  train --steps 15000 --lr 3e-4 --warmup-steps 500 \
  --checkpoint-freq 3000 --save medical_bpe.pth

# Generate with a prompt
python train.py \
  --input medical_dialog.txt \
  --tokenizer bpe --bpe-model medical_bpe.model \
  --n-embd 384 --n-head 6 --n-layer 6 --context-size 256 \
  eval --load medical_bpe.pth \
  --prompt "Patient: I have chest pain and shortness of breath. Doctor:" \
  --token-count 200
```

---

## Project Structure

```
lecture6/
├── gpt.py                   # Model architecture (Head, MultiHeadAttention, Block, GPTLanguageModel)
├── train.py                 # CLI entry point — training and inference
├── util.py                  # CharacterTokenizer, BPETokenizer, Dataset
├── loss.py                  # estimate_loss() — train/val loss evaluation
├── metrics.py               # Metrics — perplexity, ROUGE, BERTScore, masked accuracy
├── prepare_medical_dialog.py # Regenerate medical_dialog.txt from HuggingFace
└── requirements.txt
```

---

## Architecture

```
Input text (raw .txt)
        │
        ▼
┌─────────────────────┐
│  Tokenizer          │  util.py
│  CharacterTokenizer │  char → index  (vocab ~100)
│  or BPETokenizer    │  subword → index (vocab 2000)
└────────┬────────────┘
         │  encode() → list[int]
         ▼
┌─────────────────────┐
│      Dataset        │  util.py
│  90% train /        │  get_batch('train'|'val', device)
│  10%  val split     │  returns X:(B,T), Y:(B,T) shifted by 1
└────────┬────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────┐
│                   GPTLanguageModel                        │  gpt.py
│                                                           │
│  token_embedding_table  (vocab_size × n_embd)            │
│         +                                                 │
│  position_embedding_table (context_size × n_embd)        │
│         │                                                 │
│         ▼                                                 │
│  ┌─────────────────────────────────────┐  × n_layer      │
│  │             Block                   │                  │
│  │  x → LayerNorm → MultiHeadAttention │                  │
│  │              ↓ (+ residual)         │                  │
│  │  x → LayerNorm → FeedForward        │                  │
│  │              ↓ (+ residual)         │                  │
│  └─────────────────────────────────────┘                  │
│         │                                                 │
│         ▼                                                 │
│  LayerNorm → Linear (n_embd → vocab_size)                 │
│         │                                                 │
│         ▼                                                 │
│     logits (B, T, vocab_size)                             │
└──────────────────────────────────────────────────────────┘
         │
         ▼ (training)                   ▼ (inference)
  cross_entropy(logits, targets)   softmax → multinomial sample
  loss.backward()                  → next token → append → repeat
  optimizer.step()
  scheduler.step()  ← warmup+cosine
```

### KV-Cache (inference)

```
Step 1: process tokens [0..T-1]  → cache K,V for all positions
Step 2: process only token [T]   → concatenate with cached K,V → attend over full history
Step N: only 1 token processed   → O(1) compute per step instead of O(T²)

Cache is reset when its length would exceed context_size.
```

---

## Installation

```bash
pip install -r requirements.txt
```

```bash
# requirements.txt includes:
torch numpy
evaluate rouge_score bert_score
sentencepiece   # for BPE tokenizer
datasets transformers
```

---

## CLI Reference

### Argument structure

Global flags must come **before** the subcommand; subcommand flags come **after**:

```
python train.py [GLOBAL FLAGS] train [TRAIN FLAGS]
python train.py [GLOBAL FLAGS] eval  [EVAL FLAGS]
```

### Global flags

| Flag               | Default     | Description                              |
|--------------------|-------------|------------------------------------------|
| `--input`          | `input.txt` | Path to training text                    |
| `--tokenizer`      | `char`      | `char` or `bpe`                          |
| `--bpe-vocab-size` | `2000`      | BPE vocabulary size (ignored for char)   |
| `--bpe-model`      | `bpe.model` | Path to save/load sentencepiece model    |
| `--context-size`   | `256`       | Sequence length (tokens)                 |
| `--batch-size`     | `32`        | Batch size                               |
| `--n-embd`         | `384`       | Embedding dimension                      |
| `--n-head`         | `6`         | Attention heads per block                |
| `--n-layer`        | `6`         | Number of transformer blocks             |
| `--dropout`        | `0.2`       | Dropout rate                             |
| `--seed`           | (none)      | Random seed                              |

### Train flags (after `train`)

| Flag                | Default     | Description                                        |
|---------------------|-------------|----------------------------------------------------|
| `--steps`           | `5000`      | Number of training steps                           |
| `--save`            | `model.pth` | Output checkpoint path                             |
| `--lr`              | `1e-3`      | Peak learning rate                                 |
| `--warmup-steps`    | `0`         | Linear warmup steps before cosine decay (0 = off) |
| `--report`          | `500`       | Loss reporting interval                            |
| `--checkpoint-freq` | `0`         | Save checkpoint every N steps (0 = disabled)       |

### Eval flags (after `eval`)

| Flag            | Default     | Description        |
|-----------------|-------------|--------------------|
| `--load`        | `model.pth` | Checkpoint to load |
| `--prompt`      | (none)      | Optional seed text |
| `--token-count` | `300`       | Tokens to generate |

> **Important:** Always pass the same `--tokenizer`, `--bpe-model`, and architecture flags (`--n-embd`, `--n-head`, `--n-layer`, `--context-size`) that were used during training, or the checkpoint will fail to load.

---

## Evaluation Metrics

Computed every `--report` steps during training:

| Metric            | Description                                                        |
|-------------------|--------------------------------------------------------------------|
| `perplexity`      | `exp(cross_entropy)` — lower is better; not comparable across tokenizers |
| `rouge1`          | Unigram overlap between generated and reference text               |
| `rougeL`          | Longest common subsequence similarity                              |
| `bertscore`       | Semantic similarity via RoBERTa embeddings (F1)                   |
| `masked_accuracy` | Argmax accuracy on randomly masked 15% of token positions          |

> **Note on comparing char vs BPE loss/perplexity:** Raw loss and perplexity are **not** directly comparable between tokenizers because the prediction task difficulty scales with vocab size (char ~100, BPE ~2000). Use ROUGE and BERTScore for honest cross-tokenizer comparison.

---

## Tokenizers

### CharacterTokenizer (`util.py`)
- Vocab = all unique characters in the input file (~100 for medical dialog)
- One token = one character
- 256 context tokens ≈ 40 words of history
- No external files needed — vocab is derived from the input at runtime

### BPETokenizer (`util.py`)
- Backed by `sentencepiece`, trained on the input file at first run
- One token = a subword chunk ("diag", "nosis", "_pain")
- 256 context tokens ≈ 150 words of history — 3-4× more context per token
- Saves `<bpe-model>.model` and `<bpe-model>.vocab` on first run
- **Must pass `--bpe-model` path at eval time** — vocab cannot be reconstructed from the raw text

---

## Training Experiments

All runs use `medical_dialog.txt` (~3MB, ~18,900 lines of Reddit doctor-patient Q&A).

### Results Summary

| Run | Tokenizer | Model | Params | Steps | Best Val Loss | ROUGE-1 | BERTScore | Notes |
|-----|-----------|-------|--------|-------|--------------|---------|-----------|-------|
| Char baseline | Char | 256/4/4 | 3.2M | 5k | 0.81 | 0.174 | 0.843 | Plateaued at capacity |
| Char extended | Char | 256/4/4 | 3.2M | 8k | 0.84 | 0.161 | 0.838 | No improvement over 5k |
| BPE small | BPE-2000 | 256/4/4 | 4.2M | 15k | 2.23* | 0.353 | 0.852 | **2× ROUGE gain** from BPE alone |
| BPE big | BPE-2000 | 384/6/6 | 12.3M | 20k | 3.46* | 0.356 | 0.856 | Overfit — dropout too low (0.2) |
| BPE big v3 | BPE-2000 | 384/6/6 | 12.3M | 15k | TBD | TBD | TBD | dropout=0.35 + cosine LR |

*BPE loss not comparable to char loss — different vocab size. See note above.

### Key findings

- **BPE vs Char:** ROUGE-1 jumped from 0.17 → 0.35 using the same model size. Subword tokenization gives the model 3–4× more context per token, making the biggest single improvement.
- **Bigger model without regularization:** The 12.3M param model overfit severely (train loss 0.19, val loss 3.46). More dropout (0.35) and cosine LR warmup added in v3.
- **Val < train loss:** Normal when dropout is active during training — not a bug.

---

## Sample Outputs

Outputs shown are free generation (no prompt) unless noted.

### Char model — 5k steps (3.2M params)
```
Sore through was before, symptoms are side addically toleracturies fingers small
coming an electrauma anPle. She have an Im not exposed testion. Take a chronic
passing do ptorgood verital or the decapes and finger as leg as it hard nervous
her biopsy expected to 1/0-1.50 with this wonder this after
```
*Character-level gibberish, but medical vocabulary present: "biopsy", "chronic".*

### Char model — 8k steps, prompted ("Doctor: ")
```
Doctor: i have consequently had a father too, and he particulty seizure contact
for symptoms. Problems like you need to do an infectious...
Doctor, I'm experiencing delusion due to cat of the cycles, because of the eyes,
and I'm worried about 160 post relosing of Important to melising, not it insulins
or Heavil home, which drug should I take?
```
*Recognizable Q&A structure emerging. Words mostly real but grammar broken.*

### BPE small — 15k steps (4.2M params)
```
orsaks, but probably because it goes down to the doc did gets fevers and infected.
Nerealation to see if there's a possibility I will fi (no ramily slowly for
depression and treated steroids etc). But can be done than an inp but not allow
would be a one thing for sure to emetry had a specialist who won't see if there
was something or was it to worry.
```
*Full words and medical terms used contextually. Sentence structure much more coherent.*

### BPE big — 20k steps (12.3M params, overfit)
```
actone, and surgery, since nothing surgery, they are some different things. The
first step is to relaxated in the brain yesterday but the least, and how often
to be present. The patiently this is the cause of death has a set and is unusual.
Im worried I snack well, and ask for any further questions. Thank you so much
for your help. I think you are step, how psychologically very...
```
*Most fluent output — "Thank you so much for your help" is a real phrase from training data, showing memorization. Val loss was rising at this point.*

---

## Running on Cloud GPU

Recommended: RunPod with RTX 4090 on-demand instance (~$0.74/hr).

```bash
# Clone repo
git clone https://github.com/jjchong5/GPT2-test-model /workspace/lecture6
cd /workspace/lecture6
pip install -r requirements.txt

# Train overnight (background, logs to file)
nohup python train.py \
  --input medical_dialog.txt \
  --tokenizer bpe --bpe-vocab-size 2000 --bpe-model medical_bpe.model \
  --n-embd 384 --n-head 6 --n-layer 6 \
  --context-size 256 --batch-size 64 --dropout 0.35 \
  train --steps 15000 --lr 3e-4 --warmup-steps 500 \
  --checkpoint-freq 3000 --report 1000 \
  --save medical_bpe_v3.pth \
> training_log.txt 2>&1 &

# Monitor
tail -f training_log.txt
```

Also works on Google Colab (T4 GPU, free tier).

---

## Data

`medical_dialog.txt` is Reddit doctor-patient Q&A (~3MB, ~18,900 lines). To regenerate:

```bash
python prepare_medical_dialog.py
# Source: knowrohit07/know_medical_dialogue_v2 on HuggingFace
# Columns: instruction (patient question) + output (doctor response)
```

---

## Hardware

The script auto-selects the best available device:

```
CUDA (NVIDIA GPU) → MPS (Apple Silicon) → CPU
```

CPU training is feasible but slow. Recommended: any NVIDIA GPU with 8GB+ VRAM.

---

## What This Model Does (and Doesn't Do)

This model learns to **continue text in the style of doctor-patient conversations**. Given a prompt, it generates plausible-sounding medical dialogue. It does not diagnose accurately.

The gap between this model and a system like GPT-4 is not primarily training time — it comes down to three things:

### 1. Instruction tuning + RLHF
This model does pure **next-token prediction** — it continues text that looks like the training data. GPT-4 was explicitly taught to *answer questions well* via:
- **Instruction fine-tuning:** thousands of curated (question → ideal answer) pairs
- **RLHF (Reinforcement Learning from Human Feedback):** human raters ranked model outputs; a reward model was trained on those rankings; the LLM was then fine-tuned to maximize that reward

This is why GPT-4 gives structured advice rather than hallucinating. Our model has no concept of "correct" — only "probable next token."

### 2. Training data
~3MB of Reddit Q&A vs. hundreds of billions of tokens including medical textbooks, clinical guidelines, PubMed papers, and curated instruction datasets. More data = more facts memorized = more accurate responses.

### 3. Scale — but less than you'd think
GPT-4 is estimated at ~1.8 **trillion** parameters (using MoE). Ours: 12M. That's 150,000× bigger. But a **fine-tuned 7B Llama model** (600× our size) already produces clinically useful responses. Scale matters; it's not the whole story.

---

## Future Work

Improvements roughly ordered by impact-to-effort ratio:

### High impact, moderate effort
- **Top-p (nucleus) sampling** — filter low-probability tokens at generation time; dramatically improves output coherence with zero retraining. Add `--top-p` and `--temperature` flags to `eval`.
- **Instruction fine-tuning** — reformat `medical_dialog.txt` as explicit `[INST] patient question [/INST] doctor answer` pairs and fine-tune. Teaches the model to *answer* rather than *continue*. This is the single biggest quality jump possible without changing scale.
- **Weight tying** — share input embedding and output projection weights (`lm_head.weight = token_embedding_table.weight`). Reduces parameters, improves generalization on small models. One line of code.

### Moderate impact, moderate effort
- **Context size 512** — 512 BPE tokens ≈ a full doctor-patient exchange in context. Currently 256 ≈ 150 words. Halve batch size to fit in GPU memory.
- **Larger BPE vocabulary (4000–8000)** — fewer tokens per sequence, more semantic precision per token.
- **Direct Preference Optimization (DPO)** — a simpler alternative to RLHF that doesn't require a separate reward model. Given pairs of (good response, bad response), trains the model to prefer the good one.

### Lower impact or higher effort
- **Mixture of Experts (MoE)** — routes each token to one of N expert feedforward sub-networks. Gives more parameters for the same compute. Pays off at 1B+ parameters; overhead not worth it at our scale.
- **Rotary Position Embeddings (RoPE)** — replaces learned position embeddings with rotation-based encoding. Better generalization to sequence lengths not seen during training.
- **Retrieval-Augmented Generation (RAG)** — at inference time, retrieve relevant medical documents and prepend them to the context. Gets factual accuracy without needing the model to memorize facts. Orthogonal to model training.
- **More/better data** — medical textbooks, clinical notes, PubMed abstracts. More domain-specific data > more general data for this use case.
