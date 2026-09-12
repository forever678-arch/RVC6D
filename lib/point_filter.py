"""Segmentation-free point reliability filters for detector bbox crops."""

import numpy as np


POINT_FILTER_CHOICES = ('valid', 'diameter_hard', 'mad_soft')


def compute_point_filter(depth, centered_xyz, diameter, mode='mad_soft',
                         minimum_support=32):
    """Return a soft reliability map and its nonzero support.

    All inputs are inference-available: bbox depth/XYZ and known object size.
    No GT visible/instance mask is accepted by this API.
    """
    if mode not in POINT_FILTER_CHOICES:
        raise ValueError(
            'point filter must be one of {}'.format(POINT_FILTER_CHOICES))
    depth = np.asarray(depth, dtype=np.float32)
    centered_xyz = np.asarray(centered_xyz, dtype=np.float32)
    if centered_xyz.shape != depth.shape + (3,):
        raise ValueError('centered_xyz must have shape depth.shape + (3,)')

    finite_xyz = np.isfinite(centered_xyz).all(axis=2)
    valid = np.isfinite(depth) & (depth > 0.0) & finite_xyz
    valid_weight = valid.astype(np.float32)
    if mode == 'valid' or not np.any(valid):
        return valid_weight, valid

    diameter = max(float(diameter), 1e-4)
    if mode == 'diameter_hard':
        radius = np.linalg.norm(centered_xyz, axis=2)
        support = valid & (radius <= 0.75 * diameter)
        if int(support.sum()) < int(minimum_support):
            support = valid
        return support.astype(np.float32), support

    valid_depth = depth[valid]
    median_depth = float(np.median(valid_depth))
    mad = float(np.median(np.abs(valid_depth - median_depth)))
    robust_sigma = max(1.4826 * mad, 0.02 * diameter, 1e-4)
    threshold = float(np.clip(
        3.0 * robust_sigma, 0.15 * diameter, 0.75 * diameter))
    temperature = max(0.10 * threshold, 0.01 * diameter, 1e-4)
    logit = np.clip(
        (threshold - np.abs(depth - median_depth)) / temperature,
        -30.0, 30.0)
    weight = (1.0 / (1.0 + np.exp(-logit))).astype(np.float32)
    weight *= valid_weight
    support = valid & (weight >= 0.05)
    if int(support.sum()) < int(minimum_support):
        return valid_weight, valid
    weight *= support.astype(np.float32)
    return weight, support
