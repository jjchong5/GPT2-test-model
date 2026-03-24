import torch
from torch import nn
import torch.nn.functional as F


class Head(nn.Module):
  def __init__(self, head_size, n_embd, context_size):
    super().__init__()
    self.key = nn.Linear(n_embd, head_size, bias=False)
    self.query = nn.Linear(n_embd, head_size, bias=False)
    self.value = nn.Linear(n_embd, head_size, bias=False)
    self.dropout = nn.Dropout(0.2)
    self.register_buffer('tril', torch.tril(torch.ones(context_size, context_size)))

  def forward(self, x, kv_cache=None):
    B,T,C = x.shape
    k = self.key(x)   # (B,T,C)
    q = self.query(x) # (B,T,C)
    v = self.value(x) # (B,T,C)

    if kv_cache is not None:
      k_prev, v_prev = kv_cache
      k = torch.cat([k_prev, k], dim=1)
      v = torch.cat([v_prev, v], dim=1)

    new_cache = (k, v)
    T_full = k.shape[1]

    # compute attention scores
    wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, T_full)
    wei = wei.masked_fill(self.tril[T_full-T:T_full, :T_full] == 0, float('-inf')) # (B, T, T_full)
    wei = F.softmax(wei, dim=-1) # (B, T, T_full)
    wei = self.dropout(wei)
    # perform the weighted aggregation of the values
    out = wei @ v # (B, T, T_full) @ (B, T_full, C) -> (B, T, C)
    return out, new_cache



class MultiHeadAttention(nn.Module):
  def __init__(self, num_heads, head_size, n_embd, context_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size, n_embd, context_size) for _ in range(num_heads)])
    self.proj = nn.Linear(head_size * num_heads, n_embd)
    self.dropout = nn.Dropout(0.2)

  def forward(self, x, kv_cache=None):
    if kv_cache is None:
      kv_cache = [None] * len(self.heads)
    results = [h(x, c) for h, c in zip(self.heads, kv_cache)]
    outs, new_caches = zip(*results)
    out = torch.cat(list(outs), dim=-1)
    out = self.proj(out)
    out = self.dropout(out)
    return out, list(new_caches)


class FeedFoward(nn.Module):
  def __init__(self, n_embd):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(n_embd, n_embd * 4),
      nn.ReLU(),
      nn.Linear(n_embd * 4, n_embd),
      nn.Dropout(0.2),
    )

  def forward(self, x):
    return self.net(x)


class Block(nn.Module):
  def __init__(self, n_embd, n_head, context_size):
    super().__init__()
    head_size = n_embd // n_head
    self.sa = MultiHeadAttention(n_head, head_size, n_embd, context_size)
    self.ffwd = FeedFoward(n_embd)

    self.ln1 = nn.LayerNorm(n_embd)
    self.ln2 = nn.LayerNorm(n_embd)

  def forward(self, x, kv_cache=None):
    sa_out, new_cache = self.sa(self.ln1(x), kv_cache)
    x = x + sa_out
    x = x + self.ffwd(self.ln2(x))
    return x, new_cache


class GPTLanguageModel(nn.Module):
  def __init__(self, vocab_size, n_embd=32, context_size=8, n_head=4, n_layer=4): #1
    super().__init__()
    self.context_size = context_size

    self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # lookup table, vocab_size x vocab_size
    self.position_embedding_table = nn.Embedding(context_size, n_embd)
    self.blocks = nn.ModuleList([Block(n_embd, n_head=n_head, context_size=context_size) for _ in range(n_layer)])

    self.ln_f = nn.LayerNorm(n_embd)
    self.lm_head = nn.Linear(n_embd, vocab_size)

  def generate(self, start_idx, number_of_tokens, use_cache=False):
    idx = start_idx
    if use_cache:
      kv_cache = None
      for _ in range(number_of_tokens):
        # drop cache if it would exceed position embedding table
        if kv_cache is not None and kv_cache[0][0][0].shape[1] >= self.context_size:
          kv_cache = None

        if kv_cache is not None:
          idx_input = idx[:, -1:]
          logits, _, kv_cache = self(idx_input, use_cache=True, kv_cache=kv_cache)
        else:
          idx_input = idx[:, -self.context_size:]
          if idx_input.shape[1] < self.context_size:
            # room to grow — start caching
            logits, _, kv_cache = self(idx_input, use_cache=True)
          else:
            # window is full — no benefit from caching
            logits, _ = self(idx_input)

        logits = logits[:, -1, :] # (batch_size, vocab_size)
        probs = F.softmax(logits, dim=1)
        idx_next = torch.multinomial(probs, num_samples=1) # (batch_size, 1)
        idx = torch.cat((idx, idx_next), dim=1) # (batch_size, t + 1)
      return idx
    else:
      for _ in range(number_of_tokens):
        # crop to last block_size of tokens
        idx_cond = idx[:, -self.context_size:]
        logits, loss = self(idx_cond)
        # apply softmax to get probabilities
        logits = logits[:, -1, :] # (batch_size, vocab_size)
        probs = F.softmax(logits, dim=1)
        idx_next = torch.multinomial(probs, num_samples=1) # (batch_size, 1)
        idx = torch.cat((idx, idx_next), dim=1) # (batch_size, t + 1)
      return idx

  def forward(self, idx, targets=None, use_cache=False, kv_cache=None):
    # idx (batch_size, block_size)
    B, T = idx.shape

    if kv_cache is not None and kv_cache[0] is not None:
      past_len = kv_cache[0][0][0].shape[1]  # layer 0 -> head 0 -> k tensor -> seq dim
    else:
      past_len = 0

    emb = self.token_embedding_table(idx) # (batch_size, block_size, n_embd)
    pos_emb = self.position_embedding_table(torch.arange(past_len, past_len + T, device=idx.device)) # (block_size, n_embd)
    x = emb + pos_emb # (batch_size, block_size, n_embd)

    if use_cache:
      if kv_cache is None:
        kv_cache = [None] * len(self.blocks)
      new_caches = []
      for block, cache in zip(self.blocks, kv_cache):
        x, new_cache = block(x, cache)
        new_caches.append(new_cache)
    else:
      new_caches = None
      for block in self.blocks:
        x, _ = block(x)

    x = self.ln_f(x)
    logits = self.lm_head(x) # (batch_size, block_size, vocab_size)

    if targets is not None:
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)
    else:
      loss = None

    if use_cache:
      return logits, loss, new_caches
    return logits, loss
