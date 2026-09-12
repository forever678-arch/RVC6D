"""Shared detector-bbox utilities used identically by train and test."""

import torch


def bbox_depth_valid_mask(bbox_mask, depth):
    """Return the inference-available valid region inside a detector crop."""
    if bbox_mask.ndim != 4 or bbox_mask.size(1) != 1:
        raise ValueError('bbox_mask must have shape (batch, 1, height, width)')
    if depth.shape != bbox_mask.shape:
        raise ValueError('depth and bbox_mask must have the same shape')
    return bbox_mask * (depth > 0.0).to(
        device=bbox_mask.device, dtype=bbox_mask.dtype)
