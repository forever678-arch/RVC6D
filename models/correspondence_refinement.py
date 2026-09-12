#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Functional group G: visible-surface correspondence and bounded refinement.

Dense soft assignment to learnable CAD shape anchors, inlier confidence
gating, and two shared bounded residual pose updates inside the network.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.sarr import matrix_to_sarr, sarr_to_matrix
from models.blocks import ConvGNAct, ResidualDepthwiseBlock
from models.feature_organization import ActiveMultiKernelMixer
from models.pose_hypotheses import ReliabilityPool, SARRTopologyProjector


class ShapeAnchorSelector(nn.Module):
    """Select order-invariant, shape-dependent CAD surface anchors."""

    def __init__(self):
        super().__init__()
        directions = []
        for axis in range(3):
            for sign in (-1.0, 1.0):
                direction = [0.0, 0.0, 0.0]
                direction[axis] = sign
                directions.append(direction)
        for x in (-1.0, 1.0):
            for y in (-1.0, 1.0):
                for z in (-1.0, 1.0):
                    directions.append([x, y, z])
        directions = F.normalize(
            torch.tensor(directions, dtype=torch.float32), dim=1)
        self.register_buffer('directions', directions, persistent=False)

    @property
    def count(self):
        return int(self.directions.size(0))

    def forward(self, model_xyz):
        points = model_xyz.transpose(1, 2)
        directions = self.directions.to(
            device=points.device, dtype=points.dtype)
        scores = torch.einsum('bnc,kc->bnk', points, directions)
        indices = scores.argmax(dim=1)
        return torch.gather(
            points, 1, indices[:, :, None].expand(-1, -1, 3))


class CADCorrespondenceFusion(nn.Module):
    """Fuse dense image evidence with a tiny object-specific CAD token set."""

    def __init__(self, channels, rank=64):
        super().__init__()
        self.rank = int(rank)
        self.selector = ShapeAnchorSelector()
        self.query = nn.Conv2d(channels, self.rank, 1, bias=False)
        self.anchor_encoder = nn.Sequential(
            nn.Linear(3, self.rank), nn.SiLU(),
            nn.Linear(self.rank, self.rank * 2))
        self.object_key = nn.Linear(channels, self.rank)
        update_channels = channels + self.rank + 5
        self.update = nn.Sequential(
            ConvGNAct(update_channels, channels, kernel_size=1),
            ActiveMultiKernelMixer(
                channels, expansion=1, dropout=0.04))
        self.inlier = nn.Sequential(
            ConvGNAct(channels + 2, channels // 2, kernel_size=1),
            nn.Conv2d(channels // 2, 1, 1))
        nn.init.constant_(self.inlier[-1].bias, -0.5)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(8.0)))
        self.mix_scale = nn.Parameter(torch.tensor(-1.2))

    def forward(self, features, object_token, model_xyz, observed_xyz,
                valid, reliability):
        height, width = features.shape[-2:]
        anchors = self.selector(model_xyz)
        encoded = self.anchor_encoder(anchors)
        keys, values = encoded.chunk(2, dim=2)
        keys = keys + self.object_key(object_token)[:, None]
        queries = F.normalize(self.query(features), dim=1, eps=1e-6)
        keys = F.normalize(keys, dim=2, eps=1e-6)
        scale = self.logit_scale.exp().clamp(1.0, 20.0)
        logits = scale * torch.einsum('bdhw,bkd->bkhw', queries, keys)
        attention = torch.softmax(logits, dim=1)
        context = torch.einsum('bkhw,bkd->bdhw', attention, values)
        expected_xyz = torch.einsum(
            'bkhw,bkc->bchw', attention, anchors)

        valid_small = F.interpolate(
            valid.float(), size=(height, width), mode='nearest')
        reliability_small = F.interpolate(
            reliability.float(), size=(height, width), mode='bilinear',
            align_corners=False).clamp(0.0, 1.0)
        inlier_logits = self.inlier(torch.cat((
            features, valid_small, reliability_small), dim=1))
        update = self.update(torch.cat((
            features, context, expected_xyz,
            valid_small, reliability_small), dim=1))
        confidence = (
            torch.sigmoid(inlier_logits)
            * valid_small * (0.25 + 0.75 * reliability_small))
        mix = 0.30 * torch.sigmoid(self.mix_scale)
        features = features + mix * confidence * update
        observed_small = F.interpolate(
            observed_xyz, size=(height, width), mode='nearest')
        return features, {
            'cad_attention_logits': logits,
            'cad_inlier_logits': inlier_logits,
            'cad_anchor_points': anchors,
            'cad_expected_xyz': expected_xyz,
            'cad_observed_xyz': observed_small,
            'cad_valid': valid_small,
            'cad_reliability': reliability_small,
            'cad_confidence': confidence,
        }


class KeypointResidualRefiner(nn.Module):
    """Two shared, bounded pose updates conditioned on CAD correspondences."""

    def __init__(self, channels, iterations=2):
        super().__init__()
        self.iterations = int(iterations)
        self.pool = ReliabilityPool()
        input_dim = channels * 3 + 20
        hidden = channels
        self.trunk = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, channels * 2), nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(channels * 2, hidden), nn.SiLU())
        self.delta = nn.Linear(hidden, 9)
        self.update_gate = nn.Linear(hidden, 1)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)
        nn.init.zeros_(self.update_gate.weight)
        nn.init.constant_(self.update_gate.bias, -2.0)
        self.projector = SARRTopologyProjector()

    @staticmethod
    def _observed_anchors(cad_info):
        logits = cad_info['cad_attention_logits'].flatten(2)
        raw_confidence = cad_info['cad_confidence'].flatten(2)
        confidence = raw_confidence.clamp_min(1e-6)
        spatial = torch.softmax(
            logits + confidence.log(), dim=2)
        observed = cad_info['cad_observed_xyz'].flatten(2).transpose(1, 2)
        observed_anchors = torch.einsum('bkn,bnc->bkc', spatial, observed)
        assignment = torch.softmax(logits, dim=1)
        support = torch.sum(
            assignment * raw_confidence, dim=2)
        support = support / raw_confidence.sum(dim=2).clamp_min(1e-6)
        has_support = raw_confidence.sum(dim=2) > 0.0
        return observed_anchors, support, has_support

    @staticmethod
    def _residual_statistics(rotation, center, anchors, observed, support):
        projected = torch.einsum(
            'bij,bkj->bki', rotation, anchors) + center[:, None]
        residual = observed - projected
        weights = support / support.sum(dim=1, keepdim=True).clamp_min(1e-6)
        mean = torch.sum(weights[:, :, None] * residual, dim=1)
        rms = torch.sqrt(torch.sum(
            weights[:, :, None] * residual.square(), dim=1).clamp_min(1e-8))
        maximum = residual.abs().amax(dim=1)
        support_stats = torch.stack((
            support.mean(dim=1), support.amax(dim=1)), dim=1)
        return torch.cat((mean, rms, maximum, support_stats), dim=1)

    def forward(self, features, reliability, object_token, base_sarr,
                base_center, cls_ids, cad_info):
        pooled = self.pool(features, reliability)
        anchors = cad_info['cad_anchor_points']
        observed, support, has_support = self._observed_anchors(cad_info)
        state_sarr = base_sarr
        state_center = base_center
        stage_sarr = []
        stage_center = []
        gates = []
        for _ in range(self.iterations):
            # SARR-to-matrix is used only to construct a geometric observation
            # for the next update.  Its inverse-trigonometric Jacobian is
            # singular at several exact symmetry representatives, so do not
            # backpropagate through this side path.  The state and every
            # predicted update remain directly supervised below.
            rotation = sarr_to_matrix(state_sarr.detach(), cls_ids)
            residual_stats = self._residual_statistics(
                rotation, state_center, anchors, observed, support)
            descriptor = self.trunk(torch.cat((
                pooled, object_token, residual_stats,
                state_sarr, state_center), dim=1))
            delta = torch.tanh(self.delta(descriptor))
            gate = (
                torch.sigmoid(self.update_gate(descriptor))
                * has_support.to(descriptor.dtype))
            # Express the topology projection itself as a residual.  This is
            # exactly a no-op when the zero-initialized delta is zero, even
            # for continuous-symmetry classes whose projector is otherwise
            # not perfectly idempotent in floating point.
            projected_state = self.projector(state_sarr, cls_ids)
            projected_update = self.projector(
                state_sarr + 0.12 * gate * delta[:, :6], cls_ids)
            state_sarr = F.normalize(
                state_sarr + projected_update - projected_state,
                dim=1, eps=1e-6)
            xy = state_center[:, :2] + 0.06 * gate * delta[:, 6:8]
            z = state_center[:, 2:3] * torch.exp(
                0.08 * gate * delta[:, 8:9])
            state_center = torch.cat((xy, z), dim=1)
            stage_sarr.append(state_sarr)
            stage_center.append(state_center)
            gates.append(gate.squeeze(1))
        return {
            'refined_sarr': state_sarr,
            'refined_center': state_center,
            'refinement_stage_sarr': torch.stack(stage_sarr, dim=1),
            'refinement_stage_center': torch.stack(stage_center, dim=1),
            'refinement_gate': torch.stack(gates, dim=1),
            'cad_anchor_support': support,
        }


__all__ = [
    'CADCorrespondenceFusion', 'KeypointResidualRefiner',
    'ShapeAnchorSelector',
]
