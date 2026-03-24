# CLAUDE.md — lecture6 GPT Codebase

## Project Purpose

Educational character-level GPT implementation. Originally scaffolded on Tiny Shakespeare; primary training target is `medical_dialog.txt` (Reddit doctor-patient Q&A, ~3MB, 18,920 lines). Focus is on clarity over performance — every component is minimal and directly traceable to the theory.

---

## Infrastructure

Training runs on **RunPod** (cloud GPU), not locally or on Colab. Colab was tried but switched to RunPod for parallel hyperparameter runs and no session timeout limits.

- GPU: RTX 5090 (or 3090/4090 as available)
- Use **spot/interruptible instances** for cost savings (~50-70% cheaper)
- Use a **Network Volume** to persist checkpoints across pod interruptions
- `train.py` supports `--checkpoint-freq N` to save `model_stepN.pth` every N steps — critical for spot instance resilience
- Install deps on pod: `pip install evaluate rouge_score bert_score`

---

## File Responsibilities

| File | Role |
|---|---|
| `gpt.py` | Model architecture only — no I/O, no training loop |
| `train.py` | CLI entry point; wires all components together |
| `util.py` | Tokenization + dataset batching |
| `loss.py` | Validation loss estimation (no_grad context) |
| `metrics.py` | Extended evaluation metrics (ROUGE, BERTScore, masked accuracy) |

---

## Architecture Overview

```
                        train.py (CLI)
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
      util.py            gpt.py            metrics.py
  CharacterTokenizer  GPTLanguageModel      Metrics
      Dataset              │               loss.py
                           │            estimate_loss()
                     ┌─────┴──────┐
                     │            │
                   Block × n_layer
                     │
             ┌───────┴────────┐
             │                │
    MultiHeadAttention    FeedForward
             │
        Head × n_head
    (Q, K, V projections
      + KV-cache)
```

### Forward pass (training)

```
input idx (B, T)
    │
    ├─► token_embedding_table[idx]       → (B, T, n_embd)
    ├─► position_embedding_table[0..T-1] → (T, n_embd)
    │         add
    ▼
    x (B, T, n_embd)
    │
    ▼  [for each Block]
    ├─► LayerNorm → MultiHeadAttention → + residual
    └─► LayerNorm → FeedForward        → + residual
    │
    ▼
    LayerNorm (final)
    │
    ▼
    lm_head Linear → logits (B, T, vocab_size)
    │
    ▼  (if targets provided)
    cross_entropy(logits.view(B*T, C), targets.view(B*T)) → scalar loss
```

### Attention head (single head)

```
x (B, T, n_embd)
    │
    ├─► key   Linear → k (B, T, head_size)   ─┐
    ├─► query Linear → q (B, T, head_size)    │  concatenate with kv_cache if present
    └─► value Linear → v (B, T, head_size)   ─┘
                                               │
    wei = q @ k^T * head_size^(-0.5)           │
    wei = causal_mask(wei)  ← tril buffer      │
    wei = softmax(wei)                         │
    wei = dropout(wei)                         │
    out = wei @ v  → (B, T, head_size)         │
    return out, new_cache ─────────────────────┘
```

### KV-Cache during generation

```
generate() loop:
  if cache exists and fits within context_size:
    idx_input = idx[:, -1:]          # only last token
    forward(idx_input, kv_cache=...) # O(1) attention per step
  else:
    idx_input = idx[:, -context_size:]
    forward(idx_input)               # full recompute
    start fresh cache

  logits[:, -1, :] → softmax → multinomial → append to idx
```

---

## Data Processing Pipeline

### Step 1 — Raw text input

Place any plain `.txt` file in the project root. The tokenizer is built entirely from its contents — no external vocab files needed.

```bash
# Tiny Shakespeare (used for the pre-trained checkpoint)
curl -o input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

### Step 2 — CharacterTokenizer (`util.py:4-14`)

```python
tokenizer = CharacterTokenizer(content)
```

- `vocab`: `sorted(set(content))` — all unique characters, alphabetically ordered
- `char_to_idx`: `{'\\n': 0, ' ': 1, '!': 2, ...}`
- `idx_to_char`: reverse mapping
- `encode(text)` → `list[int]`
- `decode(indices)` → `str`

The vocab is derived fresh from the input file each run. When loading a saved model, use the same input file to reproduce the same vocab — **vocab must match the checkpoint**.

### Step 3 — Tensor conversion (`train.py:78`)

```python
data = torch.tensor(tokenizer.encode(content), dtype=torch.long)
```

Entire corpus becomes a 1-D `LongTensor` of token indices.

### Step 4 — Dataset split and batching (`util.py:17-32`)

```python
dataset = Dataset(data, context_size=256, batch_size=32)
# Internally: train_data = data[:90%], val_data = data[90%:]
```

`get_batch(split, device)`:
```
random offsets ix: shape (batch_size,)
X = data[ix : ix+context_size]        → (batch_size, context_size)
Y = data[ix+1 : ix+context_size+1]    → (batch_size, context_size)  ← shifted by 1
```

Y is X shifted right by one position: the model predicts the next character at every position simultaneously.

---

## Training Loop (`train.py:13-31`)

```python
for step in range(steps):
    xb, yb = data.get_batch('train', device)
    _, loss = model(xb, yb)            # forward
    optimizer.zero_grad(set_to_none=True)
    loss.backward()                    # backward
    optimizer.step()                   # update

    if step % report_frequency == 0:
        losses = estimate_loss(data, model)   # loss.py — 100 batches, no_grad
        metrics_dict = metrics(data, model, tokenizer)  # metrics.py — 5 steps
        print(...)
```

Optimizer: AdamW, default lr=1e-3, no LR schedule.

---

## Evaluation (`loss.py`, `metrics.py`)

### estimate_loss (`loss.py`)

Averages cross-entropy over 100 random batches from each split. Called under `torch.no_grad()`. Returns `{'train': float, 'val': float}`.

### Metrics (`metrics.py`)

Each call averages 5 steps. Each step computes:

1. **Perplexity** — `exp(cross_entropy)` on a val batch
2. **ROUGE-1 / ROUGE-L** — generate `context_size // 2` tokens from first half of batch as prompt; compare to held-out second half
3. **BERTScore** — same generated vs reference texts, semantic similarity F1
4. **Masked accuracy** — forward pass, mask 15% of positions randomly, check `argmax == target`

Requires the `evaluate` package (`pip install evaluate`).

---

## Running the Code

### Argument order

Global flags (`--input`, `--context-size`, `--n-embd`, etc.) must come **before** the subcommand. Subcommand flags (`--steps`, `--save`, `--load`, etc.) come **after**.

```
python train.py [GLOBAL FLAGS] train [TRAIN FLAGS]
python train.py [GLOBAL FLAGS] eval  [EVAL FLAGS]
```

### Train from scratch

```bash
python train.py --input input.txt train --steps 5000 --save model.pth
```

### Generate from pre-trained checkpoint

```bash
python train.py --input input.txt eval --load tinyshakespeare_model_5000steps.pth
```

`--input` is required even for eval because the tokenizer vocab is rebuilt from it.

### Generate with a prompt

```bash
python train.py --input input.txt eval --load tinyshakespeare_model_5000steps.pth \
  --prompt "ROMEO: " --token-count 500
```

### Custom hyperparameters

```bash
python train.py --input input.txt \
  --n-embd 128 --n-head 4 --n-layer 4 --context-size 128 \
  --batch-size 64 \
  train --steps 3000 --save small_model.pth
```

---

## Key Constraints

- **Vocab is not saved with the checkpoint.** Always pass the same `--input` file used during training when loading a model. A different file = different vocab = garbage output.
- `context_size` must match between training and the loaded checkpoint. It is embedded in the position embedding table shape.
- `n_embd`, `n_head`, `n_layer` must also match the checkpoint — they define model weight shapes.

---

## Dependencies

```bash
pip install torch numpy
pip install evaluate rouge_score bert_score
```

| Package | Used in | Notes |
|---|---|---|
| `torch` | all files | core deep learning framework |
| `numpy` | `metrics.py`, `train.py` | array ops and metric aggregation |
| `evaluate` | `metrics.py` | HuggingFace evaluation library (`evaluate.load(...)`) |
| `rouge_score` | `metrics.py` (via `evaluate`) | ROUGE backend — **not** auto-installed by `evaluate` |
| `bert_score` | `metrics.py` (via `evaluate`) | BERTScore backend — **not** auto-installed by `evaluate` |
| `transformers` | pulled in by `bert_score` | provides BERT model for semantic scoring |

### Available training datasets

| File | Status | Description |
|---|---|---|
| `input.txt` | Ready | Tiny Shakespeare — used for the pre-trained checkpoint |
| `medical_dialog.txt` | **Ready** | Doctor-patient Q&A — primary dataset. Needs cleaning: remove embedded URLs before training. |

To regenerate `medical_dialog.txt`:

```bash
python prepare_medical_dialog.py
```

Source dataset: `knowrohit07/know_medical_dialogue_v2` (HuggingFace, ~6K rows, `instruction` + `output` columns).

> **Note:** The original source `UCSD26/medical_dialog` no longer loads because it uses an old dataset script format (`medical_dialog.py`) that was dropped in `datasets >= 3.0`.
