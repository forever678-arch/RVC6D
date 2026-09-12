"""Detached diagnostics for a detector-bbox pose pipeline.

Ground-truth visible masks are deliberately confined to this module.  The
returned Python scalars may be logged and plotted, but cannot participate in
forward propagation, loss construction, post-processing, or backpropagation.
"""

import torch
import torch.nn.functional as F


@torch.no_grad()
def routing_mask_diagnostics(routing_probability, metric_mask_visib):
    """Compare latent pose routing with GT visibility for diagnostics only."""
    prediction = routing_probability.detach().float()
    target = F.interpolate(
        metric_mask_visib.detach().float().to(prediction.device),
        size=prediction.shape[-2:], mode='nearest') > 0.5
    predicted = prediction > 0.5

    intersection = (predicted & target).flatten(1).sum(dim=1).float()
    union = (predicted | target).flatten(1).sum(dim=1).float()
    predicted_count = predicted.flatten(1).sum(dim=1).float()
    target_count = target.flatten(1).sum(dim=1).float()
    pixels = float(target[0].numel()) if target.size(0) else 1.0

    visible_values = prediction.masked_select(target)
    background_values = prediction.masked_select(~target)
    return {
        'routing_iou_metric': float(
            (intersection / union.clamp_min(1.0)).mean().item()),
        'routing_precision_metric': float(
            (intersection / predicted_count.clamp_min(1.0)).mean().item()),
        'routing_recall_metric': float(
            (intersection / target_count.clamp_min(1.0)).mean().item()),
        'mask_visib_fraction_metric': float(
            (target_count / pixels).mean().item()),
        'routing_visible_mean_metric': float(
            visible_values.mean().item()) if visible_values.numel() else 0.0,
        'routing_background_mean_metric': float(
            background_values.mean().item()) if background_values.numel() else 0.0,
    }
