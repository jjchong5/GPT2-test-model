#!/usr/bin/env python3
import argparse

import torch
import numpy as np

from util import CharacterTokenizer, BPETokenizer, Dataset
from gpt import GPTLanguageModel
from loss import estimate_loss
from metrics import Metrics


def train(data, model, tokenizer, steps, report_frequency, lr, save_path, checkpoint_freq, warmup_steps=0):
  device = next(model.parameters()).device
  optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
  metrics = Metrics()

  if warmup_steps > 0:
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    warmup    = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    decay     = CosineAnnealingLR(optimizer, T_max=max(1, steps - warmup_steps), eta_min=lr * 0.01)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, decay], milestones=[warmup_steps])
  else:
    scheduler = None

  for step in range(steps):
    xb, yb = data.get_batch('train', device)

    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if scheduler:
      scheduler.step()
    if step % report_frequency == 0 or step == steps - 1:
      losses = estimate_loss(data, model)
      print(f"Step {step}, train loss: {losses['train']:.4f} val loss: {losses['val']:.4f}")

      metrics_dict = metrics(data, model, tokenizer)
      print("Metrics:", metrics_dict)

      print()

    if checkpoint_freq and step > 0 and step % checkpoint_freq == 0:
      ckpt_path = save_path.replace('.pth', f'_step{step}.pth')
      torch.save(model.state_dict(), ckpt_path)
      print(f"Checkpoint saved: {ckpt_path}")

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default="input.txt")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--tokenizer", type=str, default="char", choices=["char", "bpe"],
                    help="Tokenizer type: char (default) or bpe (subword sentencepiece)")
parser.add_argument("--bpe-vocab-size", type=int, default=2000,
                    help="Vocab size when training a BPE tokenizer (ignored for char)")
parser.add_argument("--bpe-model", type=str, default="bpe.model",
                    help="Path to save/load the sentencepiece .model file")

parser.add_argument("--context-size", type=int, default=256)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--n-embd", type=int, default=384)
parser.add_argument("--n-head", type=int, default=6)
parser.add_argument("--n-layer", type=int, default=6)
parser.add_argument("--dropout", type=float, default=0.2)

subparsers = parser.add_subparsers(dest="command", required=True)

train_parser = subparsers.add_parser("train")
train_parser.add_argument("--save", type=str, default="model.pth")
train_parser.add_argument("--steps", type=int, default=5000)
train_parser.add_argument("--report", type=int, default=500)
train_parser.add_argument("--lr", type=float, default=1e-3)
train_parser.add_argument("--checkpoint-freq", type=int, default=0, help="Save a checkpoint every N steps (0 = disabled)")
train_parser.add_argument("--warmup-steps", type=int, default=0, help="LR warmup steps before cosine decay (0 = constant LR)")

eval_parser = subparsers.add_parser("eval")
eval_parser.add_argument("--load", type=str, default="model.pth")
eval_parser.add_argument("--prompt", type=str)
eval_parser.add_argument("--token-count", type=int, default=300)

args = parser.parse_args()

if args.seed:
  torch.manual_seed(args.seed)
batch_size = args.batch_size
context_size = args.context_size
n_embd = args.n_embd
n_head = args.n_head
n_layer = args.n_layer
dropout = args.dropout

# replace this with ur hw backend if needed
device = 'cuda' if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
if device == "cpu":
  print("WARNING: Running on cpu!")

with open(args.input, "r") as f:
  content = f.read()

if args.tokenizer == "bpe":
  tokenizer = BPETokenizer(args.input, model_path=args.bpe_model, vocab_size=args.bpe_vocab_size)
else:
  tokenizer = CharacterTokenizer(content)

data = torch.tensor(tokenizer.encode(content), dtype=torch.long)
dataset = Dataset(data, context_size, batch_size)

model = GPTLanguageModel(tokenizer.vocab_size, n_embd, context_size, n_head, n_layer)
model = model.to(device)

print(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6}")
print(f"Using device: {device}")
print()

if args.command == "eval":
  print("=" * 20, "INFERENCE", "=" * 20)
  model.load_state_dict(torch.load(args.load))
  model.eval()
elif args.command == "train":
  print("=" * 20, "TRAINING", "=" * 20)
  train(dataset, model, tokenizer, args.steps, args.report, args.lr, args.save, args.checkpoint_freq, args.warmup_steps)
  torch.save(model.state_dict(), args.save)
  print("=" * 50)

context = torch.zeros((1, 1), dtype=torch.long, device=device)
max_tokens = 300
if args.command == "eval":
  if args.prompt is not None:
    context = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
  max_tokens = args.token_count
print(
  tokenizer.decode(
    model.generate(start_idx=context, number_of_tokens=max_tokens, use_cache=True)[0].tolist()
  )
)
