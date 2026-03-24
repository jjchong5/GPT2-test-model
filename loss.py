import torch


@torch.no_grad()
def estimate_loss(data, model, eval_iters=100):
    device = next(model.parameters()).device

    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = data.get_batch(split, device)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
