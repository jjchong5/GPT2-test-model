# lecture6 — Character-Level GPT

A minimal, educational implementation of a GPT-style transformer trained on the Tiny Shakespeare dataset. Covers the full ML pipeline: data loading, tokenization, training, evaluation, and text generation.

---

## Project Structure

```
lecture6/
├── gpt.py       # Model architecture (Head, MultiHeadAttention, Block, GPTLanguageModel)
├── train.py     # CLI entry point — training and inference
├── util.py      # CharacterTokenizer and Dataset
├── loss.py      # estimate_loss() — train/val loss evaluation
├── metrics.py   # Metrics class — perplexity, ROUGE, BERTScore, masked accuracy
└── tinyshakespeare_model_5000steps.pth   # Pre-trained checkpoint (5000 steps)
```

---

## Architecture

```
Input text (raw .txt)
        │
        ▼
┌─────────────────────┐
│  CharacterTokenizer │  util.py
│  vocab = sorted(    │  char → index (e.g. 'A'→0, 'B'→1 ...)
│    unique chars)    │
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
│  │                                     │                  │
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
```

### Block internals

```
x ──┬──► LayerNorm ──► MultiHeadAttention ──► + ──► x'
    │                       │                 │
    │                  (KV-cache here)        │
    │                                         │
x'──┴──► LayerNorm ──► FeedForward       ──► + ──► output
```

### MultiHeadAttention

```
x ──► [Head_1, Head_2, ..., Head_h]   (run in parallel, each with its own KV-cache)
         │        │               │
         └────────┴───────────────┘
                  concat(dim=-1)
                      │
                  Linear proj
                      │
                   Dropout
```

### Single Attention Head

```
x ──► Q projection (n_embd → head_size)
x ──► K projection (n_embd → head_size)  ─── concat with cached K ──┐
x ──► V projection (n_embd → head_size)  ─── concat with cached V ──┤
                                                                      │
Q @ K^T * scale → causal mask → softmax → dropout → @ V ────────────┘
                                                          │
                                                        output + new_cache
```

### KV-Cache (inference)

During generation the model avoids recomputing the full attention matrix every step:

```
Step 1: process tokens [0..T-1]  → cache K,V for all positions
Step 2: process only token [T]   → concatenate with cached K,V → attend over full history
Step N: only 1 token processed   → O(1) compute per step instead of O(T²)

Cache is reset when its length would exceed context_size.
```

---

## Data Processing

### 1. Prepare input text

Any plain `.txt` file works. `input.txt` (Tiny Shakespeare) is included and was used to train the pre-trained checkpoint.

`medical_dialog.txt` (~3MB, ~18,900 lines of Reddit doctor-patient Q&A) is included and is the primary training target.

To download Tiny Shakespeare from source:

```bash
curl -o input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

### 2. Tokenization (`util.py`)

```python
tokenizer = CharacterTokenizer(content)
# tokenizer.vocab  → sorted list of unique characters
# tokenizer.encode("Hello") → [idx, idx, ...]
# tokenizer.decode([idx, ...]) → "Hello"
```

- Vocabulary is derived entirely from the input file — no external vocab needed.
- Vocabulary size equals the number of unique characters in the file (~65 for Shakespeare).

### 3. Dataset batching (`util.py`)

```python
dataset = Dataset(data_tensor, context_size=256, batch_size=32)
X, Y = dataset.get_batch('train', device)
# X shape: (32, 256) — input sequences
# Y shape: (32, 256) — same sequences shifted right by 1 (next-token targets)
```

- Train/val split is 90/10 by default.
- Batch positions are sampled randomly at each step (no epochs).

---

## Installation

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install torch numpy
pip install evaluate rouge_score bert_score datasets transformers
```

`evaluate` is the HuggingFace evaluation library used in `metrics.py`. `rouge_score` and `bert_score` are its backends — they are **not** installed automatically by `evaluate` and must be listed explicitly.

---

## Usage

### Argument structure

Global flags must come **before** the subcommand; subcommand flags come **after**:

```
python train.py [GLOBAL FLAGS] train [TRAIN FLAGS]
python train.py [GLOBAL FLAGS] eval  [EVAL FLAGS]
```

### Train

```bash
python train.py --input input.txt train --steps 5000 --save model.pth
```

**Global flags** (before `train`/`eval`):

| Flag             | Default     | Description                  |
|------------------|-------------|------------------------------|
| `--input`        | `input.txt` | Path to training text        |
| `--context-size` | `256`       | Sequence length (tokens)     |
| `--batch-size`   | `32`        | Batch size                   |
| `--n-embd`       | `384`       | Embedding dimension          |
| `--n-head`       | `6`         | Attention heads per block    |
| `--n-layer`      | `6`         | Number of transformer blocks |
| `--dropout`      | `0.2`       | Dropout rate                 |
| `--seed`         | (none)      | Random seed                  |

**Train-specific flags** (after `train`):

| Flag       | Default     | Description              |
|------------|-------------|--------------------------|
| `--steps`  | `5000`      | Number of training steps |
| `--save`   | `model.pth` | Output checkpoint path   |
| `--lr`     | `1e-3`      | Learning rate            |
| `--report` | `500`       | Loss reporting interval  |

### Generate (inference)

```bash
# Generate from blank context
python train.py --input input.txt eval --load tinyshakespeare_model_5000steps.pth

# Generate with a prompt
python train.py --input input.txt eval --load tinyshakespeare_model_5000steps.pth --prompt "ROMEO: " --token-count 500
```

**Eval-specific flags** (after `eval`):

| Flag            | Default     | Description        |
|-----------------|-------------|--------------------|
| `--load`        | `model.pth` | Checkpoint to load |
| `--prompt`      | (none)      | Optional seed text |
| `--token-count` | `300`       | Tokens to generate |

---

## Evaluation Metrics

Computed every `--report` steps during training:

| Metric              | Description                                                 |
| ------------------- | ----------------------------------------------------------- |
| `perplexity`      | `exp(cross_entropy_loss)` — lower is better              |
| `rouge1`          | Unigram overlap between generated and reference text        |
| `rougeL`          | Longest common subsequence similarity                       |
| `bertscore`       | Semantic similarity via BERT embeddings (F1)                |
| `masked_accuracy` | Argmax prediction accuracy on randomly masked 15% of tokens |

---

## Training on Medical Dialog

The primary dataset is `medical_dialog.txt` (Reddit doctor-patient Q&A). To regenerate it from HuggingFace:

```bash
python prepare_medical_dialog.py
```

Train with recommended settings for a cloud GPU (RTX 3090/4090/5090):

```bash
# Quick run (~20 min on GPU)
python train.py \
  --input medical_dialog.txt \
  --n-embd 256 --n-head 4 --n-layer 4 \
  --context-size 128 --batch-size 64 \
  train --steps 5000 --lr 3e-4 \
  --checkpoint-freq 2000 --save medical_model.pth

# Generate a doctor response
python train.py \
  --input medical_dialog.txt \
  --n-embd 256 --n-head 4 --n-layer 4 \
  --context-size 128 \
  eval --load medical_model.pth \
  --prompt "Doctor: " --token-count 300
```

---

## Hardware

The script auto-selects the best available device:

```
CUDA (NVIDIA GPU) → MPS (Apple Silicon) → CPU
```

A CPU warning is printed if neither GPU backend is available. Training 5000 steps on CPU is feasible but slow.
