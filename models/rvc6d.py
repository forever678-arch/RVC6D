#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""RVC6D: real-time reliability-aware visible-surface correspondence.

Single flat model assembling the three published functional groups:

F: reliability-aware RGB-XYZ feature organization (feature_organization).
A: ambiguity-preserving rotation / depth pose hypotheses (pose_hypotheses).
G: visible-surface CAD correspondence and bounded residual refinement
   (correspondence_refinement).

B: the adaptive BMC regularizer (bmc) is attached for training only and
   contributes nothing at inference time.

The class hierarchy is intentionally flat: every module of the paper's
Fig. 1 pipeline is constructed and wired here in data-flow order.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.sarr import SARRPoseLoss, matrix_to_sarr
from models.bmc import AdaptiveBMCRegularizer
from models.blocks import _finite_tensor
from models.correspondence_refinement import (
    CADCorrespondenceFusion, KeypointResidualRefiner)
from models.feature_organization import (
    ActiveCrossFusion, ActiveMultiKernelMixer, AxialContextMixer,
    BoundaryDetailAdapter, EvidenceFusion, GeometryDescriptor,
    ScratchStageEncoder)
from models.pose_hypotheses import (
    DenseVoteQualityHead, MultiDepthCenterHead, ObjectConditioner,
    ObjectPoseQualityHead, RotationHypothesisHead)

WIDTH = 56
ROTATION_ANCHORS = 12
ROTATION_HYPOTHESES = 4
DEPTH_HYPOTHESES = 3
CORRESPONDENCE_RANK = 48
REFINEMENT_ITERATIONS = 2
PARAMETER_BUDGET = 2_600_000

BMC_OPTION_NAMES = {
    'projection_dim', 'temperature', 'shared_base_weight',
    'specific_base_weight', 'shared_target', 'specific_target',
    'ema_decay', 'warmup_steps', 'control_interval', 'deadband',
    'min_multiplier', 'max_multiplier', 'min_samples',
    'min_support_ratio', 'specific_variance_floor',
    'specific_variance_weight', 'pose_metric_target',
    'pose_ema_decay',
}


class RVC6D(nn.Module):
    """Compact RGB-D 6D pose estimator with a single-pass F-A-G pipeline."""

    parameter_budget = PARAMETER_BUDGET

    def __init__(self, num_class=30, width=WIDTH, **bmc_kwargs):
        super().__init__()
        width = int(width)
        if width != WIDTH:
            raise ValueError('RVC6D is fixed to width={}'.format(WIDTH))
        self.num_class = int(num_class)
        self.width = width
        self.channels = width * 2
        channels = self.channels

        # ---- F: reliability-aware dual-encoder feature organization ----
        self.geometry_descriptor = GeometryDescriptor()
        self.rgb_encoder = ScratchStageEncoder(3, width)
        self.xyz_encoder = ScratchStageEncoder(
            self.geometry_descriptor.output_channels, width)
        self.low_fusion = ActiveCrossFusion(width)
        self.high_fusion = ActiveCrossFusion(channels)
        self.final_fusion = EvidenceFusion(channels)
        self.context = ActiveMultiKernelMixer(channels, dropout=0.08)
        self.detail_adapter = BoundaryDetailAdapter(channels, width=32)

        # ---- A: ambiguity-preserving pose hypotheses ----
        self.object_conditioner = ObjectConditioner(self.num_class, channels)
        self.rotation_head = RotationHypothesisHead(
            channels, anchors=ROTATION_ANCHORS,
            hypotheses=ROTATION_HYPOTHESES, dropout=0.08)
        self.center_head = MultiDepthCenterHead(
            channels, hypotheses=DEPTH_HYPOTHESES)
        self.vote_quality_head = DenseVoteQualityHead(channels)
        self.pose_quality_head = ObjectPoseQualityHead(channels)

        # ---- G: visible-surface correspondence and bounded refinement ----
        self.cad_fusion = CADCorrespondenceFusion(
            channels, rank=CORRESPONDENCE_RANK)
        self.axial_context = AxialContextMixer(channels)
        self.pose_refiner = KeypointResidualRefiner(
            channels, iterations=REFINEMENT_ITERATIONS)

        # ---- B: training-only balanced modality constraint ----
        options = {
            key[4:]: value for key, value in bmc_kwargs.items()
            if key.startswith('bmc_') and key[4:] in BMC_OPTION_NAMES}
        self.adaptive_bmc = AdaptiveBMCRegularizer(
            channels=channels, **options)
        self._last_bmc = {}

    def forward_features(self, rgb, xyz, depth_valid, point_weight):
        """F: encode, fuse at two scales, and refine boundary detail.

        The reliability signal rho = point_weight * valid conditions every
        fusion and is passed through unchanged to the A/G groups.
        """
        valid = depth_valid.to(xyz.dtype).clamp(0.0, 1.0)
        reliability = point_weight.to(xyz.dtype).clamp(0.0, 1.0) * valid
        descriptor = self.geometry_descriptor(xyz, valid, reliability)
        condition = torch.cat((
            valid, reliability, descriptor[:, 6:7]), dim=1)

        appearance_low = self.rgb_encoder.encode_low(rgb)
        geometry_low = self.xyz_encoder.encode_low(descriptor)
        appearance_low, geometry_low = self.low_fusion(
            appearance_low, geometry_low, condition)
        appearance = self.rgb_encoder.encode_high(appearance_low)
        geometry = self.xyz_encoder.encode_high(geometry_low)
        appearance, geometry = self.high_fusion(
            appearance, geometry, condition)

        if self.training:
            self._last_bmc = self.adaptive_bmc(
                appearance, geometry, valid, reliability)
        else:
            self._last_bmc = {}
        fused = self.final_fusion(appearance, geometry, condition)
        features = self.context(fused)
        features = self.detail_adapter(features, rgb, xyz, valid, reliability)
        return features, valid, reliability

    def forward(self, rgb, xyz, depth, cls_ids, model_xyz=None,
                point_weight=None):
        if model_xyz is None:
            raise ValueError('RVC6D requires normalized CAD model_xyz')
        if point_weight is None:
            point_weight = depth
        cls_ids = cls_ids.view(rgb.size(0)).long()

        # F
        features, valid, reliability = self.forward_features(
            rgb, xyz, depth, point_weight)
        # A: object token conditioning shared by both hypothesis branches
        features, object_token = self.object_conditioner(
            features, cls_ids, model_xyz)
        # G: dense visible-surface CAD correspondence
        features, cad_info = self.cad_fusion(
            features, object_token, model_xyz, xyz, valid, reliability)
        features = self.axial_context(features)

        rotation = self.rotation_head(
            features, xyz, valid, reliability, object_token, cls_ids)
        center = self.center_head(
            features, xyz, valid, reliability, object_token)
        refinement = self.pose_refiner(
            features, reliability, object_token,
            rotation['global_sarr'], center['coarse_center'],
            cls_ids, cad_info)

        center_shift = (
            refinement['refined_center'] - center['coarse_center'])
        pred_t = center['pred_t'] + center_shift[:, :, None, None]
        pred_s = self.vote_quality_head(
            features, center['dense_delta'], center['center_support'],
            reliability)
        dispersion = (
            center['dense_delta'].square().sum(dim=1).sqrt()
            * F.interpolate(
                valid, size=pred_s.shape[-2:], mode='nearest').squeeze(1)
        ).flatten(1).sum(dim=1)
        valid_mass = F.interpolate(
            valid, size=pred_s.shape[-2:], mode='nearest'
        ).flatten(1).sum(dim=1).clamp_min(1.0)
        dispersion = dispersion / valid_mass
        pose_score = self.pose_quality_head(
            rotation['quality_features'], center['center_features'],
            object_token, rotation['rotation_confidence'],
            center['center_confidence'], dispersion)
        height, width = pred_t.shape[-2:]
        global_sarr = refinement['refined_sarr']
        predictions = {
            'pred_r': global_sarr[:, :, None, None].expand(
                -1, -1, height, width).contiguous(),
            'pred_t': pred_t.contiguous(),
            'pred_s': pred_s.contiguous(),
            'pose_score': pose_score.contiguous(),
            'cls_id': cls_ids.contiguous(),
            'global_sarr': global_sarr.contiguous(),
            'base_global_sarr': rotation['global_sarr'].contiguous(),
            'coarse_sarr': rotation['coarse_sarr'].contiguous(),
            'hypothesis_gate': rotation['hypothesis_gate'].contiguous(),
            'rotation_hypotheses': rotation[
                'rotation_hypotheses'].contiguous(),
            'rotation_hypothesis_logits': rotation[
                'rotation_hypothesis_logits'].contiguous(),
            'rotation_probabilities': rotation[
                'rotation_probabilities'].contiguous(),
            'selected_hypothesis': rotation[
                'selected_hypothesis'].contiguous(),
            'rotation_confidence': rotation[
                'rotation_confidence'].contiguous(),
            'coarse_center': refinement['refined_center'].contiguous(),
            'base_coarse_center': center['coarse_center'].contiguous(),
            'center_hypotheses': center[
                'center_hypotheses'].contiguous(),
            'center_probabilities': center[
                'center_probabilities'].contiguous(),
            'center_confidence': center[
                'center_confidence'].contiguous(),
            'anchor_entropy': rotation['anchor_entropy'].contiguous(),
            'rot_attention': rotation['rotation_attention'].contiguous(),
            'routing_probability': pred_s[:, None].contiguous(),
            'rvc6d_ambiguity_enabled': pred_s.new_tensor(1.0),
            'rvc6d_geometry_enabled': pred_s.new_tensor(1.0),
            'rvc6d_refinement_enabled': pred_s.new_tensor(1.0),
        }
        predictions.update(cad_info)
        predictions.update(refinement)
        predictions.update(self._last_bmc)
        self._last_bmc = {}
        return predictions

    def observe_bmc_pose_metric(self, value):
        self.adaptive_bmc.observe_pose_metric(value)

    def architecture_summary(self):
        return {
            'model': 'RVC6D',
            'width': self.width,
            'channels': self.channels,
            'rotation_hypotheses': ROTATION_HYPOTHESES,
            'depth_hypotheses': DEPTH_HYPOTHESES,
            'correspondence_rank': CORRESPONDENCE_RANK,
            'refinement_iterations': REFINEMENT_ITERATIONS,
            'parameters': sum(
                parameter.numel() for parameter in self.parameters()),
        }


class RVC6DLoss(SARRPoseLoss):
    """Full RVC6D training objective.

    SARR pose supervision with winner-takes-all hypothesis selection,
    ambiguity diversity / balance terms, correspondence and inlier
    supervision for G, three-stage refinement supervision (0.20 / 0.35 /
    0.45), and the training-only BMC regularizer.
    """

    def __init__(self, *args, correspondence_weight=0.16,
                 surface_inlier_weight=0.05,
                 refinement_rotation_weight=0.24,
                 refinement_center_weight=0.12, **kwargs):
        kwargs['aux_rotation_weight'] = 0.0
        kwargs['stage_rotation_weight'] = 0.0
        kwargs.setdefault('center_weight', 0.5)
        kwargs.setdefault('hypothesis_rotation_weight', 0.25)
        kwargs.setdefault('hypothesis_score_weight', 0.05)
        kwargs.setdefault('hypothesis_diversity_weight', 0.05)
        kwargs.setdefault('hypothesis_balance_weight', 0.005)
        kwargs.setdefault('pose_score_weight', 0.05)
        super().__init__(*args, **kwargs)
        self.correspondence_weight = float(correspondence_weight)
        self.surface_inlier_weight = float(surface_inlier_weight)
        self.refinement_rotation_weight = float(
            refinement_rotation_weight)
        self.refinement_center_weight = float(refinement_center_weight)

    @staticmethod
    def _sample_points(model_xyz, maximum=128):
        if model_xyz.ndim != 3:
            raise ValueError(
                'model_xyz must have shape (batch, 3, points), got {}'.format(
                    tuple(model_xyz.shape)))
        if model_xyz.shape[2] <= maximum:
            return model_xyz.transpose(1, 2)
        indices = torch.linspace(
            0, model_xyz.shape[2] - 1, steps=maximum,
            device=model_xyz.device).round().long()
        return model_xyz[:, :, indices].transpose(1, 2)

    @staticmethod
    def _coerce_model_xyz(model_xyz, reference):
        """Accept Tensor/NumPy/list CAD points and match loss device/dtype."""
        if not isinstance(model_xyz, torch.Tensor):
            model_xyz = torch.as_tensor(
                model_xyz, device=reference.device, dtype=reference.dtype)
        else:
            model_xyz = model_xyz.to(
                device=reference.device, dtype=reference.dtype)
        if model_xyz.ndim != 3 or model_xyz.shape[1] != 3:
            raise ValueError(
                'model_xyz must have shape (batch, 3, points), got {}'.format(
                    tuple(model_xyz.shape)))
        return model_xyz

    def _correspondence_losses(self, preds, gt_r, gt_t, model_xyz):
        logits = preds['cad_attention_logits']
        batch, anchors, height, width = logits.shape
        observed = preds['cad_observed_xyz'].flatten(2).transpose(1, 2)
        valid = preds['cad_valid'].flatten(1) > 0.5
        reliability = preds['cad_reliability'].flatten(1).clamp(0.0, 1.0)
        rotation = gt_r.view(batch, 3, 3)
        translation = gt_t.view(batch, 3)
        canonical = torch.einsum(
            'bni,bij->bnj', observed - translation[:, None], rotation)
        surface = self._sample_points(model_xyz)
        surface_distance = torch.cdist(canonical, surface).amin(dim=2)
        inlier = valid & (reliability > 0.05) & (surface_distance < 0.12)
        anchor_distance = torch.cdist(
            canonical, preds['cad_anchor_points'])
        labels = anchor_distance.argmin(dim=2)
        cross_entropy = F.cross_entropy(
            logits.permute(0, 2, 3, 1).reshape(-1, anchors),
            labels.reshape(-1), reduction='none', label_smoothing=0.02)
        weights = (reliability * inlier.to(reliability.dtype)).reshape(-1)
        correspondence_loss = torch.sum(cross_entropy * weights)
        correspondence_loss = correspondence_loss / weights.sum().clamp_min(1.0)

        inlier_logits = preds['cad_inlier_logits'].reshape(-1)
        target_inlier = inlier.reshape(-1).to(inlier_logits.dtype)
        positive_fraction = target_inlier.mean().detach()
        positive_weight = (
            (1.0 - positive_fraction) / positive_fraction.clamp_min(1e-3)
        ).clamp(1.0, 8.0)
        inlier_loss = F.binary_cross_entropy_with_logits(
            inlier_logits, target_inlier,
            pos_weight=positive_weight)
        with torch.no_grad():
            predicted = logits.argmax(dim=1).reshape(batch, -1)
            correct = predicted.eq(labels)
            accuracy = (
                (correct * inlier).sum().float()
                / inlier.sum().clamp_min(1).float())
            inlier_ratio = inlier.float().mean()
        return correspondence_loss, inlier_loss, accuracy, inlier_ratio

    def _stage_losses(self, preds, gt_r, gt_t, cls_ids):
        target_sarr = matrix_to_sarr(
            gt_r.view(-1, 3, 3), cls_ids.view(-1).long())
        stages = torch.cat((
            preds['base_global_sarr'][:, None],
            preds['refinement_stage_sarr'][:, :2]), dim=1)
        rotation_error = 1.0 - F.cosine_similarity(
            stages, target_sarr[:, None], dim=2, eps=1e-6)
        rotation_loss = (
            0.20 * rotation_error[:, 0]
            + 0.35 * rotation_error[:, 1]
            + 0.45 * rotation_error[:, 2]).mean()

        centers = torch.cat((
            preds['base_coarse_center'][:, None],
            preds['refinement_stage_center'][:, :2]), dim=1)
        center_error = F.smooth_l1_loss(
            centers, gt_t.view(-1, 1, 3).expand_as(centers),
            beta=0.05, reduction='none').mean(dim=2)
        center_loss = (
            0.20 * center_error[:, 0]
            + 0.35 * center_error[:, 1]
            + 0.45 * center_error[:, 2]).mean()
        return rotation_loss, center_loss

    def forward(self, preds, bbox_valid_mask, gt_r, gt_t, cls_ids,
                model_xyz=None, step=20, **kwargs):
        # SARR family pose supervision.
        loss, metrics = super().forward(
            preds, bbox_valid_mask, gt_r, gt_t, cls_ids,
            model_xyz=model_xyz, step=step, **kwargs)
        metrics.pop('aux_sarr_loss', None)
        metrics.pop('stage_sarr_loss', None)

        # Training-only BMC regularizer (inference contributes nothing).
        regularizer = preds.get('bmc_regularizer')
        pose_loss = loss
        applied = bool(self.train and regularizer is not None)
        if applied:
            if not bool(torch.isfinite(regularizer).all().item()):
                raise FloatingPointError(
                    'RVC6D BMC regularizer became non-finite')
            loss = loss + regularizer
            if not bool(torch.isfinite(loss).all().item()):
                raise FloatingPointError(
                    'RVC6D total loss became non-finite')
        metrics['pose_loss'] = float(pose_loss.detach().item())
        metrics['BMC_loss'] = float(
            regularizer.detach().item()) if applied else 0.0
        for key, value in preds.items():
            if key.startswith('bmc_') and key != 'bmc_regularizer':
                metrics[key] = float(value.detach().item())

        # G: correspondence, inlier and three-stage refinement supervision.
        if model_xyz is None:
            raise ValueError('RVC6D loss requires model_xyz')
        model_xyz = self._coerce_model_xyz(model_xyz, gt_r)
        correspondence, inlier, accuracy, inlier_ratio = (
            self._correspondence_losses(preds, gt_r, gt_t, model_xyz))
        stage_rotation, stage_center = self._stage_losses(
            preds, gt_r, gt_t, cls_ids)
        auxiliary = (
            self.correspondence_weight * correspondence
            + self.surface_inlier_weight * inlier
            + self.refinement_rotation_weight * stage_rotation
            + self.refinement_center_weight * stage_center)
        loss = loss + auxiliary
        _finite_tensor('RVC6D loss', loss)
        metrics.update({
            'geometry_auxiliary_loss': float(auxiliary.detach()),
            'correspondence_loss': float(correspondence.detach()),
            'surface_inlier_loss': float(inlier.detach()),
            'refinement_rotation_loss': float(stage_rotation.detach()),
            'refinement_center_loss': float(stage_center.detach()),
            'correspondence_accuracy': float(accuracy.detach()),
            'surface_inlier_ratio': float(inlier_ratio.detach()),
            'refinement_gate': float(
                preds['refinement_gate'].detach().mean()),
        })
        metrics['SARR_loss'] = float(loss.detach())
        return loss, metrics


get_loss = RVC6DLoss


def count_parameters(num_class=30):
    model = RVC6D(num_class=num_class)
    return sum(parameter.numel() for parameter in model.parameters())


__all__ = ['RVC6D', 'RVC6DLoss', 'count_parameters', 'get_loss']
