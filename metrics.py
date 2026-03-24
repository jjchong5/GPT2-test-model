import numpy as np
import torch


class Metrics:
  def __init__(self, number_of_steps=5, mask_ratio=0.15):
    import evaluate
    self.rouge = evaluate.load("rouge")
    self.bertscore = evaluate.load("bertscore")

    self.number_of_steps = number_of_steps
    self.mask_ratio = mask_ratio

  def step(self, data, model, tokenizer):
    device = next(model.parameters()).device

    # standard batch for perplexity
    x, y = data.get_batch('val', device)
    _, loss = model(x, y)
    perplexity = torch.exp(loss).item()

    # split context window in half: first half = prompt, second half = reference
    # keeps prompt + gen_len <= context_size so KV cache positions stay valid
    gen_len = data.context_size // 2
    prompt_len = data.context_size - gen_len

    # autoregressively generate gen_len tokens and then compare it to GT reference
    x, _ = data.get_batch('val', device)
    prompt = x[:, :prompt_len]
    y = x[:, prompt_len:]  # (batch_size, gen_len)

    gen_x = model.generate(prompt, gen_len, use_cache=True)
    gen_x = gen_x[:, -gen_len:]  # (batch_size, gen_len)

    generated_texts = [tokenizer.decode(i) for i in gen_x.detach().cpu().numpy()]
    reference_texts = [tokenizer.decode(i) for i in y.detach().cpu().numpy()]

    rouge_results = self.rouge.compute(predictions=generated_texts, references=reference_texts)
    bertscore_results = self.bertscore.compute(predictions=generated_texts, references=reference_texts, lang="en") # (batch_size)

    rouge_1 = rouge_results["rouge1"].item()
    rouge_L = rouge_results["rougeL"].item()
    bertscore = np.mean(bertscore_results["f1"]).item()

    # masked token accuracy: forward pass on a batch, randomly mask positions,
    # check if argmax prediction matches the target at those positions
    x_acc, y_acc = data.get_batch('val', device)
    logits, _ = model(x_acc)  # (B, T, vocab_size)
    B, T, C = logits.shape
    mask = torch.rand(B, T, device=device) < self.mask_ratio
    if not mask.any():
      mask[:, torch.randint(T, (B,))] = True
    preds = logits.argmax(dim=-1)  # (B, T)
    accuracy = (preds[mask] == y_acc[mask]).float().mean().item()

    return [perplexity, rouge_1, rouge_L, bertscore, accuracy]

  @torch.no_grad()
  def __call__(self, data, model, tokenizer):
    model.eval()

    all_metrics = []
    for _ in range(self.number_of_steps):
      metrics = self.step(data, model, tokenizer)
      all_metrics.append(metrics)

    model.train()

    agg_metrics = np.mean(np.array(all_metrics), axis=0).tolist()
    keys = ["perplexity", "rouge1", "rougeL", "bertscore", "masked_accuracy"]
    return dict(zip(keys, agg_metrics))