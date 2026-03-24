import torch


class CharacterTokenizer:
  def __init__(self, content):
    self.vocab = sorted(list(set(content)))
    self.vocab_size = len(self.vocab)
    self.char_to_idx = { ch:i for i,ch in enumerate(self.vocab) }
    self.idx_to_char = { i:ch for i,ch in enumerate(self.vocab) }

  def encode(self, xs):
    return [self.char_to_idx[x] for x in xs]

  def decode(self, xs):
    return ''.join([self.idx_to_char[x] for x in xs])


class BPETokenizer:
  """Subword BPE tokenizer backed by sentencepiece.

  If model_path already exists it is loaded directly.
  Otherwise sentencepiece is trained on input_path and saved to model_path.
  """
  def __init__(self, input_path: str, model_path: str = "bpe.model", vocab_size: int = 2000):
    import sentencepiece as spm
    import os
    if not os.path.exists(model_path):
      print(f"Training BPE tokenizer (vocab_size={vocab_size}) → {model_path}")
      spm.SentencePieceTrainer.train(
        input=input_path,
        model_prefix=model_path.replace(".model", ""),
        vocab_size=vocab_size,
        character_coverage=1.0,
        model_type="bpe",
        pad_id=3,
      )
      print("BPE tokenizer ready.")
    self.sp = spm.SentencePieceProcessor(model_file=model_path)
    self.vocab_size = self.sp.get_piece_size()

  def encode(self, text: str):
    return self.sp.encode(text)

  def decode(self, ids):
    return self.sp.decode(ids.tolist() if hasattr(ids, 'tolist') else list(ids))


class Dataset:
  def __init__(self, content, context_size, batch_size, split_factor=0.9):
    self.context_size = context_size
    self.batch_size = batch_size
    self.data = content
    assert split_factor > 0 and split_factor < 1
    n = int(len(self.data) * split_factor)
    self.train_data, self.val_data = self.data[:n], self.data[n:]

  def get_batch(self, split, device, y_shift=1):
    data = self.train_data if split == 'train' else self.val_data
    ix = torch.randint(len(data) - self.context_size - y_shift, (self.batch_size,))
    x = torch.stack([data[i:i+self.context_size] for i in ix])
    y = torch.stack([data[i+y_shift:i+self.context_size+y_shift] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y
