import torch
import numpy as np
from sklearn.metrics import f1_score
from rouge_score import rouge_scorer

def calculate_perplexity(model, val_loader, device="cpu"):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for val_batch in val_loader:
            val_batch = [t.to(device).long() for t in val_batch]
            val_x, val_y = val_batch
            val_logits, val_loss = model(val_x, val_y)
            total_loss += val_loss.item()
            num_batches += 1

    average_loss = total_loss / num_batches
    perplexity = np.exp(average_loss)
    model.train()

    return perplexity

