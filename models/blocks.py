#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Shared convolutional building blocks used by the RVC6D functional groups."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib import sarr as sarr_lib


def _group_count(channels, maximum=8):
    groups = min(int(maximum), int(channels))
    while channels % groups:
        groups -= 1
    return groups


def _safe_unit_vector(values, fallback, minimum_norm=1e-3):
    norm = values.norm(dim=-1, keepdim=True)
    normalized = values / norm.clamp_min(minimum_norm)
    fallback = F.normalize(fallback, dim=-1, eps=minimum_norm)
    return torch.where(norm > minimum_norm, normalized, fallback)


def _symmetry_descriptor(cls_ids, dtype, device):
    kappa = sarr_lib.tless_sarr_kappa(cls_ids, dtype=dtype, device=device)
    continuous = kappa >= 1000.0
    finite = torch.where(
        continuous, torch.zeros_like(kappa), kappa).clamp_max(4.0) / 4.0
    return torch.cat((finite, continuous.to(dtype)), dim=1)


class ConvGNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 groups=1, activate=True):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride,
            padding=kernel_size // 2, groups=groups, bias=False)
        self.norm = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.activate = bool(activate)

    def forward(self, features):
        features = self.norm(self.conv(features))
        return F.silu(features, inplace=True) if self.activate else features


class ResidualDepthwiseBlock(nn.Module):
    def __init__(self, channels, dropout=0.0, kernel_size=5):
        super().__init__()
        hidden = channels * 2
        self.depthwise = ConvGNAct(
            channels, channels, kernel_size=kernel_size, groups=channels)
        self.expand = ConvGNAct(channels, hidden, kernel_size=1)
        self.project = ConvGNAct(
            hidden, channels, kernel_size=1, activate=False)
        self.dropout = (
            nn.Dropout2d(float(dropout)) if dropout > 0.0 else nn.Identity())
        nn.init.zeros_(self.project.norm.weight)
        nn.init.zeros_(self.project.norm.bias)

    def forward(self, features):
        residual = self.project(self.expand(self.depthwise(features)))
        return F.silu(features + self.dropout(residual), inplace=True)


def _finite_tensor(name, tensor):
    if not bool(torch.isfinite(tensor).all().item()):
        raise FloatingPointError('{} became non-finite'.format(name))


__all__ = [
    'ConvGNAct', 'ResidualDepthwiseBlock',
    '_finite_tensor', '_group_count', '_safe_unit_vector',
    '_symmetry_descriptor',
]
