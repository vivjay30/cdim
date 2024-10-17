import torch
import torch.nn as nn
import torch.nn.functional as F

def discrete_kl_loss(pred, target, num_bins=20, epsilon=1e-8):
    # Determine range for binning
    with torch.no_grad():
        combined = torch.cat([pred, target])
        min_val = combined.min().item()
        max_val = combined.max().item()

    # Create bin edges
    bin_edges = torch.linspace(min_val, max_val, num_bins + 1, device=pred.device)
    bin_widths = bin_edges[1:] - bin_edges[:-1]

    # Compute soft histogram
    def soft_histogram(x):
        x_expanded = x.unsqueeze(-1)
        deltas = torch.abs(x_expanded - bin_edges[:-1].unsqueeze(0))
        weights = torch.clamp(1 - deltas / bin_widths, min=0, max=1)
        hist = weights.sum(dim=0) / len(x)
        return hist

    pred_hist = soft_histogram(pred)
    target_hist = soft_histogram(target)

    # Add epsilon and normalize
    pred_probs = (pred_hist + epsilon) / (pred_hist.sum() + num_bins * epsilon)
    target_probs = (target_hist + epsilon) / (target_hist.sum() + num_bins * epsilon)

    # Compute KL divergence
    kl_div = F.kl_div(pred_probs.log(), target_probs, reduction='sum')

    return kl_div
