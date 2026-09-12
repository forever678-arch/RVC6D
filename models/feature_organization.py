#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Functional group F: reliability-aware RGB-XYZ feature organization.

Dual scratch encoders, geometry descriptor, two-scale cross-modal fusion,
boundary detail preservation and axial context mixing, all conditioned on
the depth-validity / point-reliability signal.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.blocks import (
    ConvGNAct, ResidualDepthwiseBlock, _group_count)


class ActiveMultiKernelMixer(nn.Module):
    """RepViT/UIB-style token mixer with active multi-scale spatial paths."""

    def __init__(self, channels, expansion=2, dropout=0.0):
        super().__init__()
        hidden = channels * int(expansion)
        self.norm = nn.GroupNorm(8, channels)
        self.local = nn.Conv2d(
            channels, channels, 3, padding=1, groups=channels, bias=False)
        self.context = nn.Conv2d(
            channels, channels, 5, padding=4, dilation=2,
            groups=channels, bias=False)
        self.channel_mixer = nn.Sequential(
            nn.Conv2d(channels * 2, hidden, 1, bias=False),
            nn.GroupNorm(8, hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.GroupNorm(8, channels),
        )
        self.dropout = (
            nn.Dropout2d(float(dropout)) if dropout > 0.0 else nn.Identity())

    def forward(self, features):
        normalized = self.norm(features)
        mixed = torch.cat((
            self.local(normalized), self.context(normalized)), dim=1)
        return F.silu(
            features + self.dropout(self.channel_mixer(mixed)), inplace=True)


class ScratchStageEncoder(nn.Module):
    """Two-stage encoder whose residual transforms are active at step zero."""

    def __init__(self, in_channels, width=80, dropout=0.08):
        super().__init__()
        channels = width * 2
        self.stem = ConvGNAct(in_channels, width, kernel_size=5, stride=2)
        self.low = nn.Sequential(
            ActiveMultiKernelMixer(width, dropout=dropout),
            ActiveMultiKernelMixer(width, dropout=dropout))
        self.downsample = ConvGNAct(
            width, channels, kernel_size=3, stride=2)
        self.high = nn.Sequential(
            ActiveMultiKernelMixer(channels, dropout=dropout),
            ActiveMultiKernelMixer(channels, dropout=dropout),
            ActiveMultiKernelMixer(channels, dropout=dropout))

    def encode_low(self, inputs):
        return self.low(self.stem(inputs))

    def encode_high(self, low_features):
        return self.high(self.downsample(low_features))


class GeometryDescriptor(nn.Module):
    """Valid-aware absolute and local-relative organized XYZ descriptor."""

    output_channels = 9

    def forward(self, xyz, valid, reliability):
        valid = valid.to(xyz.dtype).clamp(0.0, 1.0)
        reliability = reliability.to(xyz.dtype).clamp(0.0, 1.0) * valid
        mass = F.avg_pool2d(valid, 5, stride=1, padding=2)
        local_mean = F.avg_pool2d(
            xyz * valid, 5, stride=1, padding=2) / mass.clamp_min(1e-4)
        relative = (xyz - local_mean) * valid
        shape = relative.square().sum(dim=1, keepdim=True).add(1e-8).sqrt()
        return torch.cat((
            xyz * valid, relative, shape, valid, reliability), dim=1)


class ActiveCrossFusion(nn.Module):
    """Aligned low-rank bidirectional fusion with geometry-quality routing."""

    def __init__(self, channels):
        super().__init__()
        rank = max(16, channels // 4)
        evidence_channels = rank * 4 + 3
        self.rgb_projection = nn.Conv2d(channels, rank, 1, bias=False)
        self.xyz_projection = nn.Conv2d(channels, rank, 1, bias=False)
        self.update = nn.Sequential(
            ConvGNAct(evidence_channels, channels * 2, kernel_size=1),
            nn.Conv2d(channels * 2, channels * 2, 1, bias=False))
        self.gate = nn.Sequential(
            ConvGNAct(evidence_channels, channels, kernel_size=1),
            nn.Conv2d(channels, channels * 2, 1),
            nn.Sigmoid())
        self.rgb_mixer = ActiveMultiKernelMixer(channels, expansion=1)
        self.xyz_mixer = ActiveMultiKernelMixer(channels, expansion=1)

    def forward(self, rgb, xyz, condition):
        condition = F.interpolate(
            condition, size=rgb.shape[-2:], mode='bilinear',
            align_corners=False)
        rgb_low = self.rgb_projection(rgb)
        xyz_low = self.xyz_projection(xyz)
        evidence = torch.cat((
            rgb_low, xyz_low, rgb_low * xyz_low,
            (rgb_low - xyz_low).abs(), condition), dim=1)
        rgb_update, xyz_update = self.update(evidence).chunk(2, dim=1)
        rgb_gate, xyz_gate = self.gate(evidence).chunk(2, dim=1)
        rgb = self.rgb_mixer(rgb + rgb_gate * xyz_update)
        xyz = self.xyz_mixer(xyz + xyz_gate * rgb_update)
        return rgb, xyz


class EvidenceFusion(nn.Module):
    """Fuse complementary, disagreement, and agreement evidence explicitly."""

    def __init__(self, channels, dropout=0.08):
        super().__init__()
        self.project = ConvGNAct(
            channels * 4 + 3, channels, kernel_size=1)
        self.refine = nn.Sequential(
            ActiveMultiKernelMixer(channels, dropout=dropout),
            ActiveMultiKernelMixer(channels, expansion=1, dropout=dropout))

    def forward(self, rgb, xyz, condition):
        condition = F.interpolate(
            condition, size=rgb.shape[-2:], mode='bilinear',
            align_corners=False)
        evidence = torch.cat((
            rgb, xyz, (rgb - xyz).abs(), rgb * xyz, condition), dim=1)
        return self.refine(self.project(evidence))


class BoundaryDetailAdapter(nn.Module):
    """Preserve RGB and geometric discontinuities before 4x downsampling."""

    input_channels = 11

    def __init__(self, channels, width=48):
        super().__init__()
        self.stem = ConvGNAct(
            self.input_channels, width, kernel_size=3, stride=2)
        self.low = ActiveMultiKernelMixer(
            width, expansion=1, dropout=0.03)
        self.down = ConvGNAct(
            width, width * 2, kernel_size=3, stride=2)
        self.project = nn.Sequential(
            ResidualDepthwiseBlock(width * 2, dropout=0.03),
            nn.Conv2d(width * 2, channels, 1, bias=False),
            nn.GroupNorm(_group_count(channels), channels))
        self.gate = nn.Sequential(
            nn.Conv2d(2, max(16, channels // 4), 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(16, channels // 4), channels, 1),
            nn.Sigmoid())
        self.mix_scale = nn.Parameter(torch.tensor(-1.4))

    @staticmethod
    def _edge(values):
        dx = F.pad(
            values[:, :, :, 1:] - values[:, :, :, :-1],
            (0, 1, 0, 0))
        dy = F.pad(
            values[:, :, 1:, :] - values[:, :, :-1, :],
            (0, 0, 0, 1))
        return torch.sqrt(
            dx.square().sum(dim=1, keepdim=True)
            + dy.square().sum(dim=1, keepdim=True) + 1e-8)

    def forward(self, features, rgb, xyz, valid, reliability):
        valid = valid.to(xyz.dtype).clamp(0.0, 1.0)
        reliability = reliability.to(xyz.dtype).clamp(0.0, 1.0) * valid
        gray = rgb.mean(dim=1, keepdim=True)
        rgb_edge = self._edge(gray).clamp_max(2.0)
        geometry_edge = self._edge(xyz * valid).clamp_max(2.0)
        depth_edge = self._edge(xyz[:, 2:3] * valid).clamp_max(2.0)
        inputs = torch.cat((
            rgb, xyz * valid, rgb_edge, geometry_edge, depth_edge,
            valid, reliability), dim=1)
        detail = self.project(self.down(self.low(self.stem(inputs))))
        if detail.shape[-2:] != features.shape[-2:]:
            detail = F.interpolate(
                detail, size=features.shape[-2:], mode='bilinear',
                align_corners=False)
        condition = F.interpolate(
            torch.cat((valid, reliability), dim=1),
            size=features.shape[-2:], mode='bilinear',
            align_corners=False)
        scale = 0.35 * torch.sigmoid(self.mix_scale)
        return features + scale * self.gate(condition) * detail


class AxialContextMixer(nn.Module):
    """Cheap large-receptive-field context without quadratic attention."""

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.horizontal = nn.Conv2d(
            channels, channels, (1, 9), padding=(0, 4),
            groups=channels, bias=False)
        self.vertical = nn.Conv2d(
            channels, channels, (9, 1), padding=(4, 0),
            groups=channels, bias=False)
        self.project = nn.Sequential(
            nn.Conv2d(channels * 2, channels * 2, 1, bias=False),
            nn.GroupNorm(_group_count(channels * 2), channels * 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels * 2, channels, 1, bias=False))
        self.scale = nn.Parameter(torch.tensor(-1.5))

    def forward(self, features):
        normalized = self.norm(features)
        context = self.project(torch.cat((
            self.horizontal(normalized), self.vertical(normalized)), dim=1))
        return features + 0.35 * torch.sigmoid(self.scale) * context


__all__ = [
    'ActiveCrossFusion', 'ActiveMultiKernelMixer', 'AxialContextMixer',
    'BoundaryDetailAdapter', 'EvidenceFusion', 'GeometryDescriptor',
    'ScratchStageEncoder',
]
