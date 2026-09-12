#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Functional group A: ambiguity-preserving pose hypotheses.

Object conditioning (class embedding + CAD template + symmetry topology),
support-weighted geometric anchors, K rotation hypotheses with hard
selection, M depth hypotheses with soft aggregation, and the detached
quality heads.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib import sarr as sarr_lib
from models.blocks import (
    ConvGNAct, ResidualDepthwiseBlock, _group_count, _safe_unit_vector,
    _symmetry_descriptor)


class ReliabilityPool(nn.Module):
    def forward(self, features, reliability):
        reliability = F.interpolate(
            reliability.float(), size=features.shape[-2:], mode='nearest')
        reliability = reliability.to(features.dtype).clamp(0.0, 1.0)
        mass = reliability.flatten(2).sum(dim=2)
        mean = (features * reliability).flatten(2).sum(dim=2)
        mean = mean / mass.clamp_min(1.0)
        variance = (
            (features - mean[:, :, None, None]).square() * reliability
        ).flatten(2).sum(dim=2) / mass.clamp_min(1.0)
        deviation = torch.sqrt(variance.clamp_min(1e-6))
        fallback_mean = features.flatten(2).mean(dim=2)
        fallback_deviation = features.flatten(2).std(
            dim=2, unbiased=False)
        has_depth = mass > 0.0
        mean = torch.where(has_depth, mean, fallback_mean)
        deviation = torch.where(
            has_depth, deviation, fallback_deviation)
        return torch.cat((mean, deviation), dim=1)


class SARRTopologyProjector(nn.Module):
    def forward(self, sarr, cls_ids):
        kappa = sarr_lib.tless_sarr_kappa(cls_ids, device=sarr.device)
        continuous = kappa >= 1000.0
        sine = torch.where(
            continuous, torch.zeros_like(sarr[:, :3]), sarr[:, :3])
        cosine = torch.where(
            continuous, torch.ones_like(sarr[:, 3:]), sarr[:, 3:])
        return F.normalize(
            torch.cat((sine, cosine), dim=1),
            p=2, dim=1, eps=1e-6)


class TemplateGeometryEncoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden = channels // 2
        self.points = nn.Sequential(
            nn.Conv1d(3, hidden, 1, bias=False),
            nn.GroupNorm(_group_count(hidden), hidden), nn.SiLU(),
            nn.Conv1d(hidden, channels, 1, bias=False),
            nn.GroupNorm(_group_count(channels), channels), nn.SiLU())
        self.project = nn.Sequential(
            nn.LayerNorm(channels * 2),
            nn.Linear(channels * 2, channels), nn.SiLU(),
            nn.Linear(channels, channels))

    def forward(self, model_xyz):
        encoded = self.points(model_xyz)
        pooled = torch.cat((
            encoded.mean(dim=2), encoded.amax(dim=2)), dim=1)
        return self.project(pooled)


class ObjectConditioner(nn.Module):
    def __init__(self, num_class, channels):
        super().__init__()
        self.class_embedding = nn.Embedding(num_class, channels)
        self.template_encoder = TemplateGeometryEncoder(channels)
        self.topology = nn.Sequential(
            nn.Linear(6, channels), nn.SiLU(),
            nn.Linear(channels, channels))
        self.fuse = nn.Sequential(
            nn.LayerNorm(channels * 3),
            nn.Linear(channels * 3, channels * 2), nn.SiLU(),
            nn.Linear(channels * 2, channels))
        self.film = nn.Linear(channels, channels * 2)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, features, cls_ids, model_xyz):
        topology = _symmetry_descriptor(
            cls_ids, features.dtype, features.device)
        object_token = self.fuse(torch.cat((
            self.class_embedding(cls_ids),
            self.template_encoder(model_xyz), self.topology(topology)), dim=1))
        scale, shift = self.film(object_token).chunk(2, dim=1)
        conditioned = features * (
            1.0 + 0.25 * torch.tanh(scale)[:, :, None, None])
        conditioned = conditioned + 0.25 * shift[:, :, None, None]
        return conditioned, object_token


class GeometricAnchorTokenizer(nn.Module):
    def __init__(self, channels, anchors):
        super().__init__()
        self.anchors = int(anchors)
        hidden = max(16, channels // 2)
        self.assignment = nn.Sequential(
            ConvGNAct(channels + 2, hidden, kernel_size=1),
            nn.Conv2d(hidden, self.anchors, 1))
        self.position = nn.Sequential(
            nn.Linear(3, channels), nn.SiLU(),
            nn.Linear(channels, channels))
        self.norm = nn.LayerNorm(channels)

    def forward(self, features, xyz, valid, reliability):
        height, width = features.shape[-2:]
        xyz = F.interpolate(xyz, size=(height, width), mode='nearest')
        valid = F.interpolate(valid, size=(height, width), mode='nearest')
        reliability = F.interpolate(
            reliability, size=(height, width),
            mode='bilinear', align_corners=False)
        support = valid * (0.25 + 0.75 * reliability)
        flat_support = support.flatten(2).squeeze(1)
        has_valid = flat_support.sum(dim=1, keepdim=True) > 0.0
        flat_support = torch.where(
            has_valid, flat_support, torch.ones_like(flat_support))
        logits = self.assignment(torch.cat((
            features, valid, reliability), dim=1)).flatten(2)
        weights = torch.softmax(
            logits + flat_support[:, None, :].clamp_min(1e-6).log(),
            dim=-1)
        tokens = torch.einsum(
            'bkn,bcn->bkc', weights, features.flatten(2))
        anchor_xyz = torch.einsum(
            'bkn,bpn->bkp', weights, xyz.flatten(2))
        tokens = self.norm(tokens + self.position(anchor_xyz))
        return tokens, weights.view(
            features.size(0), self.anchors, height, width), anchor_xyz


class RotationHypothesisHead(nn.Module):
    def __init__(self, channels, anchors, hypotheses, dropout=0.15):
        super().__init__()
        self.hypotheses = int(hypotheses)
        # The raw learned seeds are deliberately small at initialization.  If
        # they are simply added to the much larger pooled/object features,
        # every query sees effectively the same input and all hypotheses
        # collapse.  Normalizing before injection preserves a stable,
        # checkpoint-compatible query identity without adding parameters.
        self.query_seed_scale = 0.35
        self.ranking_temperature = 0.35
        self.pool = ReliabilityPool()
        self.pool_project = nn.Sequential(
            nn.LayerNorm(channels * 2),
            nn.Linear(channels * 2, channels), nn.SiLU(),
            nn.Linear(channels, channels))
        self.tokenizer = GeometricAnchorTokenizer(channels, anchors)
        self.query_seed = nn.Parameter(torch.empty(
            self.hypotheses, channels))
        nn.init.normal_(self.query_seed, std=0.02)
        heads = min(4, max(1, channels // 32))
        while channels % heads:
            heads -= 1
        self.cross_attention = nn.ModuleList([
            nn.MultiheadAttention(
                channels, heads, dropout=dropout, batch_first=True)
            for _ in range(2)])
        self.query_norms = nn.ModuleList([
            nn.LayerNorm(channels) for _ in range(2)])
        self.query_mlps = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(channels),
                nn.Linear(channels, channels * 2), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(channels * 2, channels))
            for _ in range(2)])
        self.coarse_head = nn.Sequential(
            nn.LayerNorm(channels * 2),
            nn.Linear(channels * 2, channels), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(channels, 6))
        self.hypothesis_head = nn.Sequential(
            nn.LayerNorm(channels), nn.Linear(channels, channels),
            nn.SiLU(), nn.Dropout(dropout), nn.Linear(channels, 6))
        self.hypothesis_score = nn.Sequential(
            nn.LayerNorm(channels), nn.Linear(channels, channels // 2),
            nn.SiLU(), nn.Linear(channels // 2, 1))
        self.sample_gate = nn.Sequential(
            nn.LayerNorm(channels * 2),
            nn.Linear(channels * 2, channels // 2), nn.SiLU(),
            nn.Linear(channels // 2, 1), nn.Sigmoid())
        self.refinement = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(channels * 3 + 6),
                nn.Linear(channels * 3 + 6, channels * 2), nn.SiLU(),
                nn.Dropout(dropout), nn.Linear(channels * 2, 6))
            for _ in range(2)])
        self.refinement_scale = nn.Parameter(
            torch.full((2,), -1.0))
        self.projector = SARRTopologyProjector()

    def _project_hypotheses(self, values, cls_ids):
        batch, hypotheses, _ = values.shape
        flat_ids = cls_ids[:, None].expand(
            batch, hypotheses).reshape(-1)
        return self.projector(
            values.reshape(-1, 6), flat_ids).reshape(
                batch, hypotheses, 6)

    def forward(self, features, xyz, valid, reliability, object_token,
                cls_ids):
        pooled = self.pool_project(self.pool(features, valid))
        anchors, anchor_maps, _ = self.tokenizer(
            features, xyz, valid, reliability)
        query_identity = F.layer_norm(
            self.query_seed, (self.query_seed.size(-1),))
        queries = (
            pooled[:, None] + object_token[:, None]
            + self.query_seed_scale * query_identity[None])
        attention = None
        for cross_attention, norm, mlp in zip(
                self.cross_attention, self.query_norms, self.query_mlps):
            update, attention = cross_attention(
                queries, anchors, anchors, need_weights=True,
                average_attn_weights=True)
            queries = norm(queries + update)
            queries = queries + mlp(queries)

        raw_hypotheses = self.hypothesis_head(queries)
        hypotheses = self._project_hypotheses(
            raw_hypotheses, cls_ids)
        hypothesis_logits = self.hypothesis_score(queries).squeeze(-1)
        hypothesis_probabilities = torch.softmax(
            hypothesis_logits / self.ranking_temperature, dim=1)
        best_indices = hypothesis_probabilities.argmax(dim=1)
        best_hypothesis = hypotheses[
            torch.arange(hypotheses.size(0), device=hypotheses.device),
            best_indices]
        # Do not average genuinely different rotations: the normalized average
        # can be a fifth, unsupported pose and its global regression gradient
        # pulls every branch back together.  Ranking is trained explicitly in
        # SARRPoseLoss, while the selected branch remains fully trainable.
        consensus = best_hypothesis

        coarse = self.projector(self.coarse_head(torch.cat((
            pooled, object_token), dim=1)), cls_ids)
        gate = self.sample_gate(torch.cat((
            pooled, object_token), dim=1))
        stage0_mix = (1.0 - gate) * coarse + gate * consensus
        stage0_fallback = torch.where(
            gate <= 0.5, coarse, consensus)
        stage0 = self.projector(_safe_unit_vector(
            stage0_mix, stage0_fallback), cls_ids)
        context = queries[
            torch.arange(queries.size(0), device=queries.device),
            best_indices]
        state = stage0
        stages = []
        for index, refinement in enumerate(self.refinement):
            delta = torch.tanh(refinement(torch.cat((
                pooled, object_token, context, state), dim=1)))
            scale = 0.25 * torch.sigmoid(
                self.refinement_scale[index])
            state = self.projector(state + scale * delta, cls_ids)
            stages.append(state)

        entropy = -torch.sum(
            hypothesis_probabilities
            * hypothesis_probabilities.clamp_min(1e-8).log(), dim=1)
        entropy_scale = max(math.log(float(self.hypotheses)), 1.0)
        concentration = (
            1.0 - entropy / entropy_scale
        ).clamp(0.0, 1.0)
        anchor_attention = torch.einsum(
            'bh,bha->ba', hypothesis_probabilities, attention)
        spatial_attention = torch.einsum(
            'ba,bahw->bhw', anchor_attention, anchor_maps).unsqueeze(1)
        return {
            'global_sarr': stages[-1],
            'coarse_sarr': coarse,
            'hypothesis_gate': gate.squeeze(1),
            'rotation_hypotheses': hypotheses,
            'rotation_hypothesis_logits': hypothesis_logits,
            'rotation_probabilities': hypothesis_probabilities,
            'selected_hypothesis': best_indices,
            'rotation_confidence': concentration,
            'anchor_entropy': entropy,
            'rotation_attention': spatial_attention,
            'quality_features': torch.cat((pooled, context), dim=1),
        }


class MultiDepthCenterHead(nn.Module):
    def __init__(self, channels, hypotheses=3):
        super().__init__()
        self.hypotheses = int(hypotheses)
        self.register_buffer(
            'quantiles', torch.tensor((0.2, 0.5, 0.8)), persistent=False)
        self.log_temperature = nn.Parameter(torch.tensor(-1.05))
        self.candidate_score = nn.Sequential(
            nn.LayerNorm(channels * 2 + 3),
            nn.Linear(channels * 2 + 3, channels), nn.SiLU(),
            nn.Linear(channels, 1))
        self.center_residual = nn.Sequential(
            nn.LayerNorm(channels * 2 + 3),
            nn.Linear(channels * 2 + 3, channels), nn.SiLU(),
            nn.Linear(channels, 3))
        self.dense_delta = nn.Sequential(
            ResidualDepthwiseBlock(channels, dropout=0.10),
            ConvGNAct(channels, channels // 2, kernel_size=1),
            nn.Conv2d(channels // 2, 3, 1))
        nn.init.zeros_(self.center_residual[-1].weight)
        nn.init.zeros_(self.center_residual[-1].bias)
        nn.init.zeros_(self.dense_delta[-1].weight)
        nn.init.zeros_(self.dense_delta[-1].bias)

    def _depth_seeds(self, depth, weights):
        sorted_depth, order = depth.sort(dim=1)
        sorted_weights = torch.gather(weights, 1, order)
        mass = sorted_weights.sum(dim=1, keepdim=True)
        fallback = mass <= 0.0
        sorted_weights = torch.where(
            fallback, torch.ones_like(sorted_weights), sorted_weights)
        cumulative = sorted_weights.cumsum(dim=1)
        cumulative = cumulative / cumulative[:, -1:].clamp_min(1e-6)
        quantiles = self.quantiles.to(
            device=depth.device, dtype=depth.dtype)[None].expand(
                depth.size(0), -1)
        indices = torch.searchsorted(
            cumulative.contiguous(), quantiles.contiguous(), right=False)
        indices = indices.clamp_max(depth.size(1) - 1)
        seeds = torch.gather(sorted_depth, 1, indices)
        return torch.where(fallback, torch.zeros_like(seeds), seeds)

    def forward(self, features, xyz, valid, reliability, object_token):
        height, width = features.shape[-2:]
        xyz = F.interpolate(xyz, size=(height, width), mode='nearest')
        valid = F.interpolate(valid, size=(height, width), mode='nearest')
        reliability = F.interpolate(
            reliability, size=(height, width),
            mode='bilinear', align_corners=False)
        flat_xyz = xyz.flatten(2)
        flat_features = features.flatten(2)
        flat_weights = (
            valid * (0.25 + 0.75 * reliability)).flatten(2).squeeze(1)
        seeds = self._depth_seeds(
            flat_xyz[:, 2].detach(), flat_weights.detach())
        temperature = F.softplus(self.log_temperature).clamp_min(0.08)
        distance = flat_xyz[:, None, 2] - seeds[:, :, None]
        memberships = torch.exp(
            -0.5 * (distance / temperature).square())
        memberships = memberships * flat_weights[:, None]
        mass = memberships.sum(dim=2, keepdim=True).clamp_min(1e-6)
        normalized = memberships / mass
        candidate_xyz = torch.einsum(
            'bkn,bpn->bkp', normalized, flat_xyz)
        candidate_features = torch.einsum(
            'bkn,bcn->bkc', normalized, flat_features)
        object_tokens = object_token[:, None].expand(
            -1, self.hypotheses, -1)
        candidate_inputs = torch.cat((
            candidate_features, object_tokens, candidate_xyz), dim=2)
        logits = self.candidate_score(candidate_inputs).squeeze(-1)
        logits = logits + mass.squeeze(-1).clamp_min(1e-6).log()
        probabilities = torch.softmax(logits, dim=1)
        selected_xyz = torch.sum(
            probabilities[:, :, None] * candidate_xyz, dim=1)
        selected_features = torch.sum(
            probabilities[:, :, None] * candidate_features, dim=1)
        residual = self.center_residual(torch.cat((
            selected_features, object_token, selected_xyz), dim=1))
        coarse_center = selected_xyz + residual
        dense_delta = self.dense_delta(features)
        pred_t = coarse_center[:, :, None, None] + dense_delta
        support = memberships.amax(dim=1).reshape(
            features.size(0), 1, height, width)
        support = support / support.flatten(2).amax(
            dim=2, keepdim=True).clamp_min(1e-6)[:, :, None]
        return {
            'pred_t': pred_t,
            'coarse_center': coarse_center,
            'center_hypotheses': candidate_xyz,
            'center_probabilities': probabilities,
            'center_confidence': probabilities.amax(dim=1),
            'center_features': selected_features,
            'center_support': support,
            'dense_delta': dense_delta,
        }


class DenseVoteQualityHead(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden = channels // 2
        self.head = nn.Sequential(
            ConvGNAct(channels + 6, hidden, kernel_size=1),
            ResidualDepthwiseBlock(hidden, dropout=0.05),
            nn.Conv2d(hidden, 1, 1))

    def forward(self, features, dense_delta, support, reliability):
        reliability = F.interpolate(
            reliability, size=features.shape[-2:],
            mode='bilinear', align_corners=False)
        delta_norm = torch.linalg.vector_norm(
            dense_delta, dim=1, keepdim=True)
        inputs = torch.cat((
            features.detach(), dense_delta.detach(), delta_norm.detach(),
            support.detach(), reliability.detach()), dim=1)
        return torch.sigmoid(self.head(inputs).squeeze(1))


class ObjectPoseQualityHead(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(channels * 4 + 3),
            nn.Linear(channels * 4 + 3, channels * 2), nn.SiLU(),
            nn.Dropout(0.10), nn.Linear(channels * 2, 1))

    def forward(self, rotation_features, center_features, object_token,
                rotation_confidence, center_confidence, dispersion):
        inputs = torch.cat((
            rotation_features.detach(), center_features.detach(),
            object_token.detach(), rotation_confidence[:, None].detach(),
            center_confidence[:, None].detach(), dispersion[:, None].detach()),
            dim=1)
        return torch.sigmoid(self.head(inputs).squeeze(1))


__all__ = [
    'DenseVoteQualityHead', 'GeometricAnchorTokenizer', 'MultiDepthCenterHead',
    'ObjectConditioner', 'ObjectPoseQualityHead', 'ReliabilityPool',
    'RotationHypothesisHead', 'SARRTopologyProjector',
    'TemplateGeometryEncoder',
]
