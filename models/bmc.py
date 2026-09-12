#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Training-only balanced modality constraints used by RVC6D.

Before late RGB/geometry fusion, each modality is split into equally sized
shared/specific halves.
The shared halves maximize a symmetric InfoNCE mutual-information lower bound,
while the specific halves minimize normalized linear cross-modal dependence.
Reliability-masked, shuffle-null-corrected CKA and fused rotation-loss EMAs
drive a slow controller that adapts both non-zero constraint weights after a
warmup period.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F



def _signed_deadzone(value, width):
    return torch.sign(value) * torch.relu(value.abs() - width)


class AdaptiveBMCRegularizer(nn.Module):
    """Training-only shared/specific cross-modal regularizer."""

    def __init__(
            self, channels=160, projection_dim=64, temperature=0.10,
            shared_base_weight=0.005, specific_base_weight=0.020,
            shared_target=0.35, specific_target=0.08,
            ema_decay=0.99, warmup_steps=500, control_interval=100,
            deadband=0.05, min_multiplier=0.25, max_multiplier=2.0,
            min_samples=32, min_support_ratio=0.05,
            specific_variance_floor=0.10,
            specific_variance_weight=1.0,
            pose_metric_target=0.10, pose_ema_decay=0.99):
        super().__init__()
        channels = int(channels)
        if channels % 2:
            raise ValueError('Adaptive BMC requires an even channel count')
        self.half_channels = channels // 2
        self.temperature = float(temperature)
        self.shared_base_weight = float(shared_base_weight)
        self.specific_base_weight = float(specific_base_weight)
        self.shared_target = float(shared_target)
        self.specific_target = float(specific_target)
        self.ema_decay = float(ema_decay)
        self.warmup_steps = int(warmup_steps)
        self.control_interval = int(control_interval)
        self.deadband = float(deadband)
        self.min_multiplier = float(min_multiplier)
        self.max_multiplier = float(max_multiplier)
        self.min_samples = int(min_samples)
        self.min_support_ratio = float(min_support_ratio)
        self.specific_variance_floor = float(specific_variance_floor)
        self.specific_variance_weight = float(specific_variance_weight)
        self.pose_metric_target = float(pose_metric_target)
        self.pose_ema_decay = float(pose_ema_decay)
        if self.temperature <= 0.0:
            raise ValueError('bmc temperature must be positive')
        if self.shared_base_weight < 0.0 or self.specific_base_weight < 0.0:
            raise ValueError('bmc base weights must be non-negative')
        if not 0.0 <= self.shared_target <= 1.0:
            raise ValueError('bmc shared target must be in [0, 1]')
        if not 0.0 <= self.specific_target <= 1.0:
            raise ValueError('bmc specific target must be in [0, 1]')
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError('bmc ema decay must be in [0, 1)')
        if self.warmup_steps < 0 or self.control_interval < 1:
            raise ValueError('invalid bmc controller schedule')
        if not 0.0 <= self.deadband < 1.0:
            raise ValueError('bmc deadband must be in [0, 1)')
        if not 0.0 < self.min_multiplier <= self.max_multiplier:
            raise ValueError('invalid bmc multiplier limits')
        if self.min_samples < 2:
            raise ValueError('bmc min_samples must be at least 2')
        if not 0.0 <= self.min_support_ratio <= 1.0:
            raise ValueError('bmc min_support_ratio must be in [0, 1]')
        if self.specific_variance_floor <= 0.0:
            raise ValueError('bmc specific_variance_floor must be positive')
        if self.specific_variance_weight < 0.0:
            raise ValueError(
                'bmc specific_variance_weight must be non-negative')
        if self.pose_metric_target <= 0.0:
            raise ValueError('bmc pose_metric_target must be positive')
        if not 0.0 <= self.pose_ema_decay < 1.0:
            raise ValueError('bmc pose_ema_decay must be in [0, 1)')

        projection_dim = int(projection_dim)
        if projection_dim < 2:
            raise ValueError('bmc projection_dim must be at least 2')
        self.shared_rgb_projector = nn.Sequential(
            nn.LayerNorm(self.half_channels),
            nn.Linear(self.half_channels, self.half_channels),
            nn.GELU(),
            nn.Linear(self.half_channels, projection_dim, bias=False),
        )
        self.shared_geometry_projector = nn.Sequential(
            nn.LayerNorm(self.half_channels),
            nn.Linear(self.half_channels, self.half_channels),
            nn.GELU(),
            nn.Linear(self.half_channels, projection_dim, bias=False),
        )

        self.register_buffer(
            'ema_shared_cka', torch.tensor(0.0), persistent=True)
        self.register_buffer(
            'ema_specific_cka', torch.tensor(0.0), persistent=True)
        self.register_buffer(
            'shared_multiplier', torch.tensor(1.0), persistent=True)
        self.register_buffer(
            'specific_multiplier', torch.tensor(1.0), persistent=True)
        self.register_buffer(
            'controller_updates', torch.tensor(0, dtype=torch.long),
            persistent=True)
        self.register_buffer(
            'ema_pose_metric', torch.tensor(0.0), persistent=True)
        self.register_buffer(
            'pose_metric_updates', torch.tensor(0, dtype=torch.long),
            persistent=True)

    @staticmethod
    def _weighted_pool(features, support):
        support = F.interpolate(
            support, size=features.shape[-2:], mode='bilinear',
            align_corners=False).detach().to(features.dtype)
        mass = support.flatten(2).sum(dim=2)
        pooled = (
            features * support).flatten(2).sum(dim=2) / mass.clamp_min(1e-6)
        return pooled, mass.squeeze(1)

    @staticmethod
    def _linear_cka(first, second, eps=1e-8):
        first = first.float() - first.float().mean(dim=0, keepdim=True)
        second = second.float() - second.float().mean(dim=0, keepdim=True)
        cross = first.transpose(0, 1) @ second
        first_cov = first.transpose(0, 1) @ first
        second_cov = second.transpose(0, 1) @ second
        numerator = cross.square().sum()
        denominator = torch.sqrt(
            first_cov.square().sum() * second_cov.square().sum()).clamp_min(
                eps)
        return (numerator / denominator).clamp(0.0, 1.0)

    @classmethod
    def _corrected_linear_cka(cls, first, second, eps=1e-6):
        """Remove finite-batch/high-dimensional CKA bias with a null pairing."""
        observed = cls._linear_cka(first, second)
        # A deterministic non-identity pairing avoids perturbing the training
        # RNG while estimating the CKA value caused by finite sample size.
        shift = max(1, first.size(0) // 2)
        null = cls._linear_cka(first, second.roll(shifts=shift, dims=0))
        corrected = (
            (observed - null) / (1.0 - null).clamp_min(eps)
        ).clamp(0.0, 1.0)
        return corrected, observed, null

    def _shared_loss(self, rgb, geometry):
        rgb = F.normalize(self.shared_rgb_projector(rgb).float(), dim=1)
        geometry = F.normalize(
            self.shared_geometry_projector(geometry).float(), dim=1)
        logits = rgb @ geometry.transpose(0, 1) / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        loss = 0.5 * (
            F.cross_entropy(logits, labels)
            + F.cross_entropy(logits.transpose(0, 1), labels))
        diagonal = torch.diagonal(logits * self.temperature)
        if logits.size(0) > 1:
            negative_sum = (
                (logits * self.temperature).sum() - diagonal.sum())
            negative_mean = negative_sum / (
                logits.numel() - diagonal.numel())
        else:
            negative_mean = diagonal.new_zeros(())
        retrieval = (
            logits.argmax(dim=1) == labels).float().mean()
        return loss, diagonal.mean(), negative_mean, retrieval

    def _specific_loss(self, rgb, geometry, eps=1e-6):
        rgb = rgb.float() - rgb.float().mean(dim=0, keepdim=True)
        geometry = (
            geometry.float() - geometry.float().mean(dim=0, keepdim=True))
        rgb_std = torch.sqrt(
            rgb.square().mean(dim=0).clamp_min(eps * eps))
        geometry_std = torch.sqrt(
            geometry.square().mean(dim=0).clamp_min(eps * eps))
        rgb = rgb / rgb_std.clamp_min(eps)
        geometry = geometry / geometry_std.clamp_min(eps)
        correlation = rgb.transpose(0, 1) @ geometry / max(1, rgb.size(0))
        # Divide by d, not d^2.  With d=80 this gives the specific branch
        # enough gradient to compete with the weighted shared InfoNCE term.
        decorrelation = (
            correlation.square().sum() / max(1, correlation.size(0)))
        variance_loss = 0.5 * (
            F.relu(self.specific_variance_floor - rgb_std).square().mean()
            + F.relu(
                self.specific_variance_floor
                - geometry_std).square().mean())
        loss = (
            decorrelation
            + self.specific_variance_weight * variance_loss)
        return (
            loss, decorrelation, variance_loss, correlation.abs().max(),
            rgb_std.mean(), geometry_std.mean(),
            rgb_std.min(), geometry_std.min())

    @torch.no_grad()
    def _update_controller(self, shared_cka, specific_cka):
        updates = int(self.controller_updates.item())
        if updates == 0:
            self.ema_shared_cka.copy_(shared_cka)
            self.ema_specific_cka.copy_(specific_cka)
        else:
            self.ema_shared_cka.mul_(self.ema_decay).add_(
                shared_cka, alpha=1.0 - self.ema_decay)
            self.ema_specific_cka.mul_(self.ema_decay).add_(
                specific_cka, alpha=1.0 - self.ema_decay)
        self.controller_updates.add_(1)
        updates = int(self.controller_updates.item())
        if (updates < self.warmup_steps
                or updates % self.control_interval != 0):
            return
        shared_error = _signed_deadzone(
            self.shared_target - self.ema_shared_cka, self.deadband)
        specific_error = _signed_deadzone(
            self.ema_specific_cka - self.specific_target, self.deadband)
        poor_gate = self._pose_poor_gate()
        # A low shared CKA only triggers extra alignment while the fused
        # rotation objective remains poor.  Relaxation above the target does
        # not need this gate.  The specific redundancy response is independent.
        shared_error = (
            torch.relu(shared_error) * poor_gate
            - torch.relu(-shared_error))
        self.shared_multiplier.copy_(
            (1.0 + shared_error).clamp(
                self.min_multiplier, self.max_multiplier))
        self.specific_multiplier.copy_(
            (1.0 + specific_error).clamp(
                self.min_multiplier, self.max_multiplier))

    @torch.no_grad()
    def observe_pose_metric(self, value):
        """Observe the fused rotation loss for the next controller update."""
        value = torch.as_tensor(
            value, device=self.ema_pose_metric.device,
            dtype=self.ema_pose_metric.dtype)
        if value.numel() != 1 or not bool(torch.isfinite(value).item()):
            return
        if int(self.pose_metric_updates.item()) == 0:
            self.ema_pose_metric.copy_(value)
        else:
            self.ema_pose_metric.mul_(self.pose_ema_decay).add_(
                value, alpha=1.0 - self.pose_ema_decay)
        self.pose_metric_updates.add_(1)

    @torch.no_grad()
    def _pose_poor_gate(self):
        if int(self.pose_metric_updates.item()) == 0:
            return self.ema_pose_metric.new_ones(())
        return (
            (self.ema_pose_metric - self.pose_metric_target)
            / self.pose_metric_target
        ).clamp(0.0, 1.0)

    def forward(self, appearance, geometry, valid, reliability):
        support = (
            valid.detach().float().clamp(0.0, 1.0)
            * (0.25 + 0.75 * reliability.detach().float().clamp(0.0, 1.0)))
        rgb_pooled, rgb_mass = self._weighted_pool(appearance, support)
        geometry_pooled, geometry_mass = self._weighted_pool(geometry, support)
        feature_pixels = float(appearance.shape[-2] * appearance.shape[-1])
        minimum_mass = max(4.0, self.min_support_ratio * feature_pixels)
        sample_valid = (
            (rgb_mass >= minimum_mass) & (geometry_mass >= minimum_mass))
        effective_samples = int(sample_valid.sum().item())
        zero = (appearance.sum() + geometry.sum()) * 0.0
        if effective_samples < self.min_samples:
            return {
                'bmc_regularizer': zero,
                'bmc_shared_loss': zero.detach(),
                'bmc_specific_loss': zero.detach(),
                'bmc_specific_decorrelation_loss': zero.detach(),
                'bmc_specific_variance_loss': zero.detach(),
                'bmc_shared_cka': self.ema_shared_cka.detach().clone(),
                'bmc_specific_cka': self.ema_specific_cka.detach().clone(),
                'bmc_shared_cka_raw': zero.detach(),
                'bmc_specific_cka_raw': zero.detach(),
                'bmc_shared_cka_null': zero.detach(),
                'bmc_specific_cka_null': zero.detach(),
                'bmc_cka_gap': (
                    self.ema_shared_cka - self.ema_specific_cka).detach(),
                'bmc_shared_weight': zero.detach(),
                'bmc_specific_weight': zero.detach(),
                'bmc_shared_multiplier': (
                    self.shared_multiplier.detach().clone()),
                'bmc_specific_multiplier': (
                    self.specific_multiplier.detach().clone()),
                'bmc_positive_cosine': zero.detach(),
                'bmc_negative_cosine': zero.detach(),
                'bmc_retrieval_top1': zero.detach(),
                'bmc_specific_max_correlation': zero.detach(),
                'bmc_rgb_specific_std': zero.detach(),
                'bmc_geometry_specific_std': zero.detach(),
                'bmc_rgb_specific_min_std': zero.detach(),
                'bmc_geometry_specific_min_std': zero.detach(),
                'bmc_ema_shared_cka': self.ema_shared_cka.detach().clone(),
                'bmc_ema_specific_cka': (
                    self.ema_specific_cka.detach().clone()),
                'bmc_controller_updates': zero.detach() + float(
                    self.controller_updates.item()),
                'bmc_pose_metric_ema': self.ema_pose_metric.detach().clone(),
                'bmc_pose_metric_updates': zero.detach() + float(
                    self.pose_metric_updates.item()),
                'bmc_shared_poor_gate': self._pose_poor_gate().detach(),
                'bmc_effective_samples': zero.detach() + float(
                    effective_samples),
                'bmc_skipped': zero.detach() + 1.0,
            }

        rgb_pooled = rgb_pooled[sample_valid]
        geometry_pooled = geometry_pooled[sample_valid]
        rgb_shared, rgb_specific = rgb_pooled.split(
            self.half_channels, dim=1)
        geometry_shared, geometry_specific = geometry_pooled.split(
            self.half_channels, dim=1)

        shared_cka, shared_cka_raw, shared_cka_null = (
            self._corrected_linear_cka(
                rgb_shared.detach(), geometry_shared.detach())
        )
        specific_cka, specific_cka_raw, specific_cka_null = (
            self._corrected_linear_cka(
                rgb_specific.detach(), geometry_specific.detach()))
        if self.training:
            self._update_controller(shared_cka, specific_cka)

        shared_loss, positive_cosine, negative_cosine, retrieval = (
            self._shared_loss(rgb_shared, geometry_shared))
        (
            specific_loss, specific_decorrelation_loss,
            specific_variance_loss, specific_max_correlation,
            rgb_specific_std, geometry_specific_std,
            rgb_specific_min_std, geometry_specific_min_std,
        ) = self._specific_loss(rgb_specific, geometry_specific)

        active = (
            int(self.controller_updates.item()) >= self.warmup_steps)
        shared_weight = (
            self.shared_base_weight * self.shared_multiplier
            if active else self.shared_multiplier.new_zeros(()))
        specific_weight = (
            self.specific_base_weight * self.specific_multiplier
            if active else self.specific_multiplier.new_zeros(()))
        regularizer = (
            shared_weight * shared_loss
            + specific_weight * specific_loss)
        return {
            'bmc_regularizer': regularizer,
            'bmc_shared_loss': shared_loss.detach(),
            'bmc_specific_loss': specific_loss.detach(),
            'bmc_shared_cka': shared_cka.detach(),
            'bmc_specific_cka': specific_cka.detach(),
            'bmc_shared_cka_raw': shared_cka_raw.detach(),
            'bmc_specific_cka_raw': specific_cka_raw.detach(),
            'bmc_shared_cka_null': shared_cka_null.detach(),
            'bmc_specific_cka_null': specific_cka_null.detach(),
            'bmc_cka_gap': (shared_cka - specific_cka).detach(),
            'bmc_shared_weight': shared_weight.detach(),
            'bmc_specific_weight': specific_weight.detach(),
            'bmc_shared_multiplier': self.shared_multiplier.detach().clone(),
            'bmc_specific_multiplier': (
                self.specific_multiplier.detach().clone()),
            'bmc_positive_cosine': positive_cosine.detach(),
            'bmc_negative_cosine': negative_cosine.detach(),
            'bmc_retrieval_top1': retrieval.detach(),
            'bmc_specific_max_correlation': (
                specific_max_correlation.detach()),
            'bmc_specific_decorrelation_loss': (
                specific_decorrelation_loss.detach()),
            'bmc_specific_variance_loss': (
                specific_variance_loss.detach()),
            'bmc_rgb_specific_std': rgb_specific_std.detach(),
            'bmc_geometry_specific_std': geometry_specific_std.detach(),
            'bmc_rgb_specific_min_std': rgb_specific_min_std.detach(),
            'bmc_geometry_specific_min_std': (
                geometry_specific_min_std.detach()),
            'bmc_ema_shared_cka': self.ema_shared_cka.detach().clone(),
            'bmc_ema_specific_cka': (
                self.ema_specific_cka.detach().clone()),
            'bmc_controller_updates': shared_cka.new_tensor(
                float(self.controller_updates.item())),
            'bmc_pose_metric_ema': self.ema_pose_metric.detach().clone(),
            'bmc_pose_metric_updates': shared_cka.new_tensor(
                float(self.pose_metric_updates.item())),
            'bmc_shared_poor_gate': self._pose_poor_gate().detach(),
            'bmc_effective_samples': shared_cka.new_tensor(
                float(effective_samples)),
            'bmc_skipped': shared_cka.new_zeros(()),
        }


__all__ = ['AdaptiveBMCRegularizer']
