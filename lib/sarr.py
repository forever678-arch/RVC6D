"""Differentiable SARR mapping for the T-LESS symmetry classes.

The representation follows Kriegler et al., "Towards Symmetry-sensitive
Pose Estimation" (IJCV 2026) and the accompanying SARR reference code.
Angles use the intrinsic XYZ convention used by scipy Rotation.as_euler.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


TLESS_SARR_KAPPA = torch.tensor([
    [1, 1, 1000], [1, 1, 1000], [1, 1, 1000], [1, 1, 1000],
    [1, 1, 2], [1, 1, 2], [1, 1, 2], [1, 1, 2], [1, 1, 2],
    [1, 1, 2], [1, 1, 2], [1, 1, 2],
    [1, 1, 1000], [1, 1, 1000], [1, 1, 1000], [1, 1, 1000],
    [1, 1, 1000], [1, 1, 1], [1, 2, 1], [1, 2, 1],
    [1, 1, 1], [1, 1, 1], [1, 2, 1], [1, 1, 1000],
    [1, 1, 2], [1, 1, 2], [1, 1, 4], [1, 1, 2], [1, 1, 2],
    [1, 1, 1000],
], dtype=torch.float32)

# LM-O: eight objects, six asymmetric [1,1,1] plus eggbox/glue with one
# discrete 180-degree fold [1,1,2] (faithful to the official models_info
# annotation).  Network class slots are the contiguous 1..8 mapping of the
# BOP ids {1,5,6,8,9,10,11,12}.
LMO_SARR_KAPPA = torch.tensor([
    [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1],
    [1, 1, 1], [1, 1, 2], [1, 1, 2], [1, 1, 1],
], dtype=torch.float32)

# Active per-class symmetry-fold table.  Defaults to T-LESS; call
# set_sarr_kappa_table(LMO_SARR_KAPPA) before building the model or loss
# for LM-O so that target encoding and SARR decoding share one
# canonicalisation.
SARR_KAPPA_TABLE = TLESS_SARR_KAPPA


def set_sarr_kappa_table(table):
    """Select the active SARR symmetry-fold table dataset-wide."""
    global SARR_KAPPA_TABLE
    SARR_KAPPA_TABLE = torch.as_tensor(table, dtype=torch.float32)



class SARRCosineLoss(nn.Module):
    """Cosine-distance regression loss for flattened six-coordinate SARR."""

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, prediction, target, weights=None):
        if prediction.shape[-1] != 6 or target.shape[-1] != 6:
            raise ValueError('SARRCosineLoss expects vectors with six coordinates')
        target = torch.broadcast_to(target, prediction.shape)
        losses = 1.0 - F.cosine_similarity(
            prediction, target, dim=-1, eps=self.eps)
        if weights is None:
            return losses.mean()
        weights = weights.detach().reshape(losses.shape).to(
            device=losses.device, dtype=losses.dtype)
        return torch.sum(losses * weights) / torch.sum(weights).clamp_min(self.eps)


class SARRPoseLoss(nn.Module):
    """SARR-first dense pose loss.

    Rotation follows the paper exactly: each six-coordinate SARR prediction
    and target is flattened and compared with one cosine distance.  In
    particular, do not normalize the three sine/cosine pairs before this loss:
    the relative magnitude in symmetry class V is part of the representation.
    Translation and confidence are auxiliary terms needed by the full BOP
    pipeline, not alternative rotation objectives.
    """

    def __init__(self, dataset=None, scoring_weight=0.01, loss_type='SARR',
                 train=True, rotation_weight=1.0, translation_weight=1.0,
                 score_weight=None, translation_beta=0.05,
                 primary_pose_loss='sarr_translation',
                 pose_distance_weight=5.0, pose_loss_points=128,
                 pose_loss_votes=16,
                 quality_translation_scale=0.3,
                 aux_rotation_weight=0.0, mask_weight=0.0,
                 stage_rotation_weight=0.0, attention_weight=0.0,
                 center_weight=0.0, hypothesis_rotation_weight=0.0,
                 hypothesis_score_weight=0.0,
                 hypothesis_diversity_weight=0.0,
                 hypothesis_balance_weight=0.0,
                 hypothesis_diversity_margin=0.20,
                 hypothesis_assignment_temperature=0.07,
                 pose_score_weight=0.0):
        super().__init__()
        self.rotation_weight = float(rotation_weight)
        self.translation_weight = float(translation_weight)
        self.score_weight = float(scoring_weight if score_weight is None else score_weight)
        self.translation_beta = float(translation_beta)
        aliases = {
            'sarr': 'sarr_translation',
            'sarr+translation': 'sarr_translation',
            'add_s': 'adds',
            'add(-s)': 'adds',
            'sym_pm': 'sym_pm_translation',
            'symmetry_aware_pm': 'sym_pm_translation',
            'pm_r_translation': 'sym_pm_translation',
        }
        primary_pose_loss = aliases.get(
            str(primary_pose_loss).strip().lower(),
            str(primary_pose_loss).strip().lower())
        if primary_pose_loss not in (
                'sarr_translation', 'adds', 'gadd',
                'sym_pm_translation'):
            raise ValueError(
                'primary_pose_loss must be one of: '
                'sarr_translation, adds, gadd, sym_pm_translation')
        self.primary_pose_loss = primary_pose_loss
        self.pose_distance_weight = float(pose_distance_weight)
        self.pose_loss_points = int(pose_loss_points)
        self.pose_loss_votes = int(pose_loss_votes)
        if self.pose_distance_weight < 0.0:
            raise ValueError('pose_distance_weight must be non-negative')
        if self.pose_loss_points < 1 or self.pose_loss_votes < 1:
            raise ValueError(
                'pose_loss_points and pose_loss_votes must be positive')
        self.sym_list = set(
            int(index) for index in (
                dataset.get_sym_list() if dataset is not None else ()))
        self.prim_groups = (
            getattr(dataset, 'prim_groups', None)
            if dataset is not None else None)
        self.symmetry_rotations = (
            getattr(dataset, 'symmetry_rotations', None)
            if dataset is not None else None)
        if (self.primary_pose_loss == 'gadd'
                and (not self.prim_groups or len(self.prim_groups) < 30)):
            raise ValueError(
                'GADD requires the 30-object T-LESS grouped primitives; '
                'ensure datasets/tless/tless_gp.json is available')
        if (self.primary_pose_loss == 'sym_pm_translation'
                and (not self.symmetry_rotations
                     or len(self.symmetry_rotations) < 30)):
            raise ValueError(
                'Symmetry-aware PM(R) requires BOP symmetry rotations for '
                'all 30 T-LESS objects')
        self.quality_translation_scale = float(quality_translation_scale)
        self.aux_rotation_weight = float(aux_rotation_weight)
        self.stage_rotation_weight = float(stage_rotation_weight)
        self.center_weight = float(center_weight)
        self.hypothesis_rotation_weight = float(
            hypothesis_rotation_weight)
        self.hypothesis_score_weight = float(hypothesis_score_weight)
        self.hypothesis_diversity_weight = float(
            hypothesis_diversity_weight)
        self.hypothesis_balance_weight = float(hypothesis_balance_weight)
        self.hypothesis_diversity_margin = float(
            hypothesis_diversity_margin)
        self.hypothesis_assignment_temperature = float(
            hypothesis_assignment_temperature)
        self.pose_score_weight = float(pose_score_weight)
        if self.hypothesis_diversity_margin < 0.0:
            raise ValueError('hypothesis_diversity_margin must be non-negative')
        if self.hypothesis_assignment_temperature <= 0.0:
            raise ValueError(
                'hypothesis_assignment_temperature must be positive')
        if float(mask_weight) != 0.0 or float(attention_weight) != 0.0:
            raise ValueError(
                'Segmentation-supervised mask/attention losses are disabled: '
                'segmentation annotations are metrics-only in bbox mode.')
        self.sarr_criterion = SARRCosineLoss()
        self.loss_type = loss_type
        self.train = train

    @staticmethod
    def _masked_mean(values, mask):
        mask = mask.to(dtype=values.dtype)
        counts = mask.flatten(1).sum(dim=1)
        totals = (values * mask).flatten(1).sum(dim=1)
        per_sample = totals / counts.clamp_min(1.0)
        valid = counts > 0
        if torch.any(valid):
            return per_sample[valid].mean()
        return values.sum() * 0.0

    @staticmethod
    def _coerce_model_xyz(model_xyz, reference):
        if model_xyz is None:
            raise ValueError(
                'point-matching losses require normalized CAD model_xyz')
        if not isinstance(model_xyz, torch.Tensor):
            model_xyz = torch.as_tensor(
                model_xyz, device=reference.device,
                dtype=reference.dtype)
        else:
            model_xyz = model_xyz.to(
                device=reference.device, dtype=reference.dtype)
        if model_xyz.ndim != 3 or model_xyz.shape[1] != 3:
            raise ValueError(
                'model_xyz must have shape (batch, 3, points), got {}'.format(
                    tuple(model_xyz.shape)))
        return model_xyz

    @staticmethod
    def _even_indices(length, maximum, device):
        count = min(int(length), int(maximum))
        if count == int(length):
            return torch.arange(count, device=device)
        # This is the deterministic equivalent of the fixed-step sampling in
        # the ES6D loss and avoids making the ablation depend on RNG state.
        return torch.div(
            torch.arange(count, device=device) * int(length),
            count, rounding_mode='floor')

    def _translation_votes(self, dense_translation, dense_mask, index):
        valid_indices = torch.nonzero(
            dense_mask[index].reshape(-1), as_tuple=False).flatten()
        if valid_indices.numel() == 0:
            return dense_translation[index].reshape(-1, 3)[:0]
        selection = self._even_indices(
            valid_indices.numel(), self.pose_loss_votes,
            valid_indices.device)
        selected = valid_indices[selection]
        return dense_translation[index].reshape(-1, 3)[selected]

    def _add_or_adds_loss(self, preds, dense_mask, gt_r, gt_t, cls_ids,
                          model_xyz):
        """Standard ADD for asymmetric and ADD-S for symmetric objects.

        RVC6D predicts one object rotation and dense translation votes.  The
        geometric definition follows DenseFusion/Uni6D; deterministic vote
        and CAD-point subsets keep the training objective practical.
        """
        model_xyz = self._coerce_model_xyz(model_xyz, gt_r)
        batch = gt_r.shape[0]
        if 'global_sarr' not in preds:
            raise ValueError(
                'ADD(-S) comparison requires preds[\'global_sarr\']')
        predicted_rotation = sarr_to_matrix_for_pose_loss(
            preds['global_sarr'], cls_ids.view(batch).long())
        dense_translation = preds['pred_t'].permute(0, 2, 3, 1)
        class_ids = cls_ids.detach().view(-1).cpu().tolist()
        sample_losses = []
        for index in range(batch):
            translations = self._translation_votes(
                dense_translation, dense_mask, index)
            if translations.numel() == 0:
                continue
            points = model_xyz[index]
            point_indices = self._even_indices(
                points.shape[1], self.pose_loss_points, points.device)
            points = points[:, point_indices]
            predicted = (
                torch.matmul(predicted_rotation[index], points)[None]
                + translations[:, :, None])
            target = (
                torch.matmul(gt_r[index].view(3, 3), points)
                + gt_t[index].view(3, 1))
            if int(class_ids[index]) in self.sym_list:
                # ADD-S: each predicted transformed point is matched to its
                # nearest target transformed model point.
                distances = torch.cdist(
                    predicted.transpose(1, 2),
                    target.transpose(0, 1)[None].expand(
                        predicted.shape[0], -1, -1))
                vote_loss = distances.amin(dim=2).mean(dim=1)
            else:
                # ADD: CAD point correspondences are retained.
                vote_loss = torch.linalg.vector_norm(
                    predicted - target[None], dim=1).mean(dim=1)
            sample_losses.append(vote_loss.mean())
        if sample_losses:
            return torch.stack(sample_losses).mean()
        return preds['pred_r'].sum() * 0.0

    def _symmetry_aware_pm_rotation_loss(
            self, preds, dense_mask, gt_r, cls_ids, model_xyz):
        """GDR-Net-style symmetry-aware rotation-only point matching.

        The closest symmetry-equivalent target rotation is selected from the
        BOP transformations using detached rotation error, exactly matching
        the target-selection semantics of GDR-Net's ``get_closest_rot_batch``.
        The selected rotation is then supervised with rotation-only L1 point
        matching.  Multiplication by three converts a coordinate-wise mean to
        the mean point-wise L1 distance used by the reference implementation.
        Translation remains the unchanged RVC6D Smooth-L1 term.
        """
        model_xyz = self._coerce_model_xyz(model_xyz, gt_r)
        batch = gt_r.shape[0]
        if 'global_sarr' not in preds:
            raise ValueError(
                'Symmetry-aware PM(R) requires preds[\'global_sarr\']')
        predicted_rotation = sarr_to_matrix_for_pose_loss(
            preds['global_sarr'], cls_ids.view(batch).long())
        class_ids = cls_ids.detach().view(-1).cpu().tolist()
        valid_samples = dense_mask.flatten(1).any(dim=1)
        sample_losses = []
        for index in range(batch):
            if not bool(valid_samples[index].item()):
                continue
            points = model_xyz[index]
            point_indices = self._even_indices(
                points.shape[1], self.pose_loss_points, points.device)
            points = points[:, point_indices]
            symmetries = torch.as_tensor(
                self.symmetry_rotations[int(class_ids[index])],
                device=gt_r.device, dtype=gt_r.dtype)
            if symmetries.ndim != 3 or symmetries.shape[-2:] != (3, 3):
                raise ValueError(
                    'symmetry rotations must have shape (K, 3, 3)')
            target_candidates = torch.matmul(
                gt_r[index].view(1, 3, 3), symmetries)
            # Rotation error is monotonic in the trace, so maximizing the trace
            # is equivalent to the reference implementation's minimum-angle
            # selection without evaluating an unstable acos.
            with torch.no_grad():
                relative = torch.matmul(
                    predicted_rotation[index].detach().transpose(0, 1)[None],
                    target_candidates)
                traces = torch.diagonal(
                    relative, dim1=-2, dim2=-1).sum(dim=-1)
                closest = int(traces.argmax().item())
            predicted_points = torch.matmul(
                predicted_rotation[index], points)
            target_points = torch.matmul(
                target_candidates[closest], points)
            sample_losses.append(
                3.0 * F.l1_loss(
                    predicted_points, target_points, reduction='mean'))
        if sample_losses:
            return torch.stack(sample_losses).mean()
        return preds['pred_r'].sum() * 0.0

    def _gadd_loss(self, preds, dense_mask, gt_r, gt_t, cls_ids):
        """ES6D grouped-primitives GADD geometric distance.

        The primitive-level mean/max rules reproduce the public ES6D
        implementation.  RVC6D's confidence and quality losses remain
        separate and unchanged, so only the geometric primary term varies.
        """
        batch = gt_r.shape[0]
        if 'global_sarr' not in preds:
            raise ValueError('GADD comparison requires preds[\'global_sarr\']')
        predicted_rotation = sarr_to_matrix_for_pose_loss(
            preds['global_sarr'], cls_ids.view(batch).long())
        dense_translation = preds['pred_t'].permute(0, 2, 3, 1)
        class_ids = cls_ids.detach().view(-1).cpu().tolist()
        sample_losses = []
        for index in range(batch):
            translations = self._translation_votes(
                dense_translation, dense_mask, index)
            if translations.numel() == 0:
                continue
            groups = self.prim_groups[int(class_ids[index])]
            if not groups:
                raise ValueError(
                    'missing GADD primitives for T-LESS object {}'.format(
                        int(class_ids[index]) + 1))
            group_losses = []
            for group_index, group in enumerate(groups):
                group = torch.as_tensor(
                    group, device=gt_r.device, dtype=gt_r.dtype)
                predicted = (
                    torch.matmul(predicted_rotation[index], group)[None]
                    + translations[:, :, None])
                target = (
                    torch.matmul(gt_r[index].view(3, 3), group)
                    + gt_t[index].view(3, 1))
                minimum = torch.cdist(
                    predicted.transpose(1, 2),
                    target.transpose(0, 1)[None].expand(
                        predicted.shape[0], -1, -1)
                ).amin(dim=2)
                if len(groups) == 3 and group_index == 2:
                    group_loss = minimum.amax(dim=1)
                else:
                    group_loss = minimum.mean(dim=1)
                group_losses.append(group_loss)
            grouped = torch.stack(group_losses, dim=0)
            if len(groups) == 3 and groups[2].shape[1] > 1:
                vote_loss = grouped.amax(dim=0)
            else:
                vote_loss = grouped.mean(dim=0)
            sample_losses.append(vote_loss.mean())
        if sample_losses:
            return torch.stack(sample_losses).mean()
        return preds['pred_r'].sum() * 0.0

    def forward(self, preds, bbox_valid_mask, gt_r, gt_t, cls_ids,
                model_xyz=None, step=20):
        """Compute pose objectives inside the detector bbox.

        ``bbox_valid_mask`` must be derived from the detector crop and
        inference-available depth validity.  There is intentionally no
        segmentation-mask argument in this differentiable API.
        """
        pred_sarr = preds['pred_r']
        if pred_sarr.size(1) != 6:
            raise ValueError('SARRPoseLoss expects six rotation channels')

        bs, _, h, w = pred_sarr.shape
        if bbox_valid_mask.dim() == 3:
            bbox_valid_mask = bbox_valid_mask.unsqueeze(1)
        bbox_valid_mask = F.interpolate(
            bbox_valid_mask.float(), size=(h, w), mode='nearest')
        dense_mask = bbox_valid_mask.squeeze(1) > 0.5
        valid_samples = dense_mask.flatten(1).any(dim=1)

        cls_ids = cls_ids.view(bs).long()
        target_sarr = matrix_to_sarr(gt_r.view(bs, 3, 3), cls_ids)
        if 'global_sarr' in preds:
            # The redesigned network predicts one rotation per object crop.
            # Supervise that Bx6 vector directly, as in SARR, rather than
            # manufacturing 16x16 independent copies of the same objective.
            global_sarr = preds['global_sarr']
            if global_sarr.shape != (bs, 6):
                raise ValueError('global_sarr must have shape (batch, 6)')
            global_error = 1.0 - F.cosine_similarity(
                global_sarr, target_sarr, dim=-1,
                eps=self.sarr_criterion.eps)
            if torch.any(valid_samples):
                sarr_loss = global_error[valid_samples].mean()
            else:
                sarr_loss = global_sarr.sum() * 0.0
            sarr_error = global_error[:, None, None].expand(bs, h, w)
        else:
            target_dense = target_sarr[:, None, None, :]
            pred_sarr = pred_sarr.permute(0, 2, 3, 1)
            sarr_error = 1.0 - F.cosine_similarity(
                pred_sarr, target_dense, dim=-1,
                eps=self.sarr_criterion.eps)
            sarr_loss = self._masked_mean(sarr_error, dense_mask)

        if 'global_sarr' in preds:
            sample_rotation_error = global_error
        else:
            sample_counts = dense_mask.flatten(1).sum(dim=1)
            sample_rotation_error = (
                sarr_error * dense_mask.to(sarr_error.dtype)
            ).flatten(1).sum(dim=1) / sample_counts.clamp_min(1)

        aux_sarr_loss = zero = preds['pred_r'].new_zeros(())
        if self.aux_rotation_weight != 0.0 and 'aux_sarr' in preds:
            aux_sarr = preds['aux_sarr']
            if aux_sarr.shape != (bs, 6):
                raise ValueError('aux_sarr must have shape (batch, 6)')
            aux_error = 1.0 - F.cosine_similarity(
                aux_sarr, target_sarr, dim=-1, eps=self.sarr_criterion.eps)
            if torch.any(valid_samples):
                aux_sarr_loss = aux_error[valid_samples].mean()
            else:
                aux_sarr_loss = aux_sarr.sum() * 0.0

        stage_sarr_loss = zero
        if (self.stage_rotation_weight != 0.0
                and 'stage0_sarr' in preds and 'stage1_sarr' in preds):
            stage0 = preds['stage0_sarr']
            stage1 = preds['stage1_sarr']
            if stage0.shape != (bs, 6) or stage1.shape != (bs, 6):
                raise ValueError('stage SARR predictions must have shape (batch, 6)')
            stage0_error = 1.0 - F.cosine_similarity(
                stage0, target_sarr, dim=-1, eps=self.sarr_criterion.eps)
            stage1_error = 1.0 - F.cosine_similarity(
                stage1, target_sarr, dim=-1, eps=self.sarr_criterion.eps)
            stage_error = (stage0_error + 2.0 * stage1_error) / 3.0
            if torch.any(valid_samples):
                stage_sarr_loss = stage_error[valid_samples].mean()
            else:
                stage_sarr_loss = (stage0.sum() + stage1.sum()) * 0.0

        hypothesis_rotation_loss = zero
        hypothesis_score_loss = zero
        hypothesis_diversity_loss = zero
        hypothesis_balance_loss = zero
        hypothesis_pairwise_distance = zero
        hypothesis_assignment_entropy = zero
        hypothesis_usage_entropy = zero
        hypothesis_winner_fraction_max = zero
        hypothesis_score_top1_accuracy = zero
        if (self.hypothesis_rotation_weight != 0.0
                and 'rotation_hypotheses' in preds):
            hypotheses = preds['rotation_hypotheses']
            if hypotheses.dim() != 3 or hypotheses.size(0) != bs or hypotheses.size(2) != 6:
                raise ValueError(
                    'rotation_hypotheses must have shape (batch, hypotheses, 6)')
            hypothesis_error = 1.0 - F.cosine_similarity(
                hypotheses, target_sarr[:, None, :], dim=-1,
                eps=self.sarr_criterion.eps)
            logits = preds.get('rotation_hypothesis_logits')
            if logits is None or logits.shape != hypothesis_error.shape:
                raise ValueError(
                    'rotation_hypothesis_logits must match rotation_hypotheses')
            # Winner-takes-all (variety) regression is essential here.  The
            # previous expected-error term sent every branch toward the same
            # target, so collapse was the exact optimum of the objective.
            best_indices = hypothesis_error.detach().argmin(dim=1)
            best_error = hypothesis_error.gather(
                1, best_indices[:, None]).squeeze(1)
            if torch.any(valid_samples):
                hypothesis_rotation_loss = best_error[
                    valid_samples].mean()
            else:
                hypothesis_rotation_loss = hypotheses.sum() * 0.0

            normalized_hypotheses = F.normalize(
                hypotheses, dim=-1, eps=self.sarr_criterion.eps)
            pairwise = torch.linalg.vector_norm(
                normalized_hypotheses[:, :, None, :]
                - normalized_hypotheses[:, None, :, :], dim=-1)
            pair_mask = torch.triu(
                torch.ones_like(pairwise, dtype=torch.bool), diagonal=1)
            pairwise_values = pairwise[pair_mask]
            if pairwise_values.numel() > 0:
                hypothesis_pairwise_distance = pairwise_values.mean()
                hypothesis_diversity_loss = F.relu(
                    self.hypothesis_diversity_margin
                    - pairwise_values).square().mean()

            # Maximize assignment mutual information: each sample should have
            # a clear winning query, while query usage remains balanced over a
            # batch.  This prevents one permanent winner without forcing all
            # four predictions to match every target.
            assignment = torch.softmax(
                -hypothesis_error
                / self.hypothesis_assignment_temperature, dim=1)
            log_assignment = assignment.clamp_min(1e-8).log()
            per_sample_entropy = -torch.sum(
                assignment * log_assignment, dim=1)
            mean_assignment = assignment.mean(dim=0)
            batch_entropy = -torch.sum(
                mean_assignment
                * mean_assignment.clamp_min(1e-8).log())
            log_hypotheses = max(math.log(float(hypotheses.size(1))), 1.0)
            hypothesis_assignment_entropy = (
                per_sample_entropy.mean() / log_hypotheses)
            hypothesis_usage_entropy = batch_entropy / log_hypotheses
            hypothesis_balance_loss = (
                1.0 + hypothesis_assignment_entropy
                - hypothesis_usage_entropy)
            if hypotheses.size(1) == 1:
                hypothesis_balance_loss = hypotheses.sum() * 0.0
            winner_counts = F.one_hot(
                best_indices, num_classes=hypotheses.size(1)
            ).float().mean(dim=0)
            hypothesis_winner_fraction_max = winner_counts.max()
            if torch.any(valid_samples):
                hypothesis_score_top1_accuracy = (
                    logits.argmax(dim=1)[valid_samples]
                    == best_indices[valid_samples]).float().mean()

            if self.hypothesis_score_weight != 0.0:
                if torch.any(valid_samples):
                    hypothesis_score_loss = F.cross_entropy(
                        logits[valid_samples], best_indices[valid_samples],
                        label_smoothing=0.02)
                else:
                    hypothesis_score_loss = logits.sum() * 0.0

        translation_loss = zero
        translation_error = None
        translation_distance = None
        if (self.translation_weight != 0.0 or self.score_weight != 0.0
                or self.pose_score_weight != 0.0
                or self.primary_pose_loss in (
                    'adds', 'gadd', 'sym_pm_translation')):
            target_t = gt_t.view(bs, 1, 1, 3)
            pred_t = preds['pred_t'].permute(0, 2, 3, 1)
            translation_error = F.smooth_l1_loss(
                pred_t, target_t.expand_as(pred_t), reduction='none',
                beta=self.translation_beta).mean(dim=-1)
            translation_distance = torch.linalg.vector_norm(
                pred_t - target_t, dim=-1)
            translation_loss = self._masked_mean(translation_error, dense_mask)

        center_loss = zero
        if self.center_weight != 0.0 and 'coarse_center' in preds:
            coarse_center = preds['coarse_center']
            if coarse_center.shape != (bs, 3):
                raise ValueError('coarse_center must have shape (batch, 3)')
            center_error = F.smooth_l1_loss(
                coarse_center, gt_t.view(bs, 3), reduction='none',
                beta=self.translation_beta).mean(dim=1)
            if torch.any(valid_samples):
                center_loss = center_error[valid_samples].mean()
            else:
                center_loss = coarse_center.sum() * 0.0

        score_loss = zero
        if self.score_weight != 0.0:
            quality = torch.exp(
                -translation_distance.detach()
                / self.quality_translation_scale
            ).clamp(0.0, 1.0)
            score_error = F.binary_cross_entropy(
                preds['pred_s'].clamp(1e-6, 1.0 - 1e-6),
                quality,
                reduction='none',
            )
            score_loss = self._masked_mean(score_error, dense_mask)

        pose_score_loss = zero
        if self.pose_score_weight != 0.0 and 'pose_score' in preds:
            vote_weights = (
                preds['pred_s'].detach() * dense_mask.to(preds['pred_s'].dtype))
            vote_weights = vote_weights / vote_weights.flatten(1).sum(
                dim=1).clamp_min(1e-6)[:, None, None]
            voted_translation = torch.sum(
                pred_t.detach() * vote_weights[:, :, :, None], dim=(1, 2))
            vote_error = torch.linalg.vector_norm(
                voted_translation - gt_t.view(bs, 3), dim=1)
            pose_quality = torch.exp(
                -sample_rotation_error.detach()
                -vote_error / self.quality_translation_scale
            ).clamp(0.0, 1.0)
            safe_pose_score = torch.nan_to_num(
                preds['pose_score'], nan=0.5, posinf=1.0, neginf=0.0
            ).clamp(1e-6, 1.0 - 1e-6)
            pose_score_error = F.binary_cross_entropy(
                safe_pose_score, pose_quality, reduction='none')
            if torch.any(valid_samples):
                pose_score_loss = pose_score_error[valid_samples].mean()
            else:
                pose_score_loss = preds['pose_score'].sum() * 0.0

        pose_distance_loss = zero
        if self.primary_pose_loss == 'sarr_translation':
            primary_pose_loss = (
                self.rotation_weight * sarr_loss
                + self.translation_weight * translation_loss)
        elif self.primary_pose_loss == 'adds':
            pose_distance_loss = self._add_or_adds_loss(
                preds, dense_mask, gt_r, gt_t, cls_ids, model_xyz)
            primary_pose_loss = (
                self.pose_distance_weight * pose_distance_loss)
        elif self.primary_pose_loss == 'gadd':
            pose_distance_loss = self._gadd_loss(
                preds, dense_mask, gt_r, gt_t, cls_ids)
            primary_pose_loss = (
                self.pose_distance_weight * pose_distance_loss)
        else:
            pose_distance_loss = self._symmetry_aware_pm_rotation_loss(
                preds, dense_mask, gt_r, cls_ids, model_xyz)
            primary_pose_loss = (
                self.pose_distance_weight * pose_distance_loss
                + self.translation_weight * translation_loss)

        total_loss = (
            primary_pose_loss
            + self.aux_rotation_weight * aux_sarr_loss
            + self.stage_rotation_weight * stage_sarr_loss
            + self.hypothesis_rotation_weight * hypothesis_rotation_loss
            + self.hypothesis_score_weight * hypothesis_score_loss
            + self.hypothesis_diversity_weight * hypothesis_diversity_loss
            + self.hypothesis_balance_weight * hypothesis_balance_loss
            + self.center_weight * center_loss
            + self.score_weight * score_loss
            + self.pose_score_weight * pose_score_loss
        )
        if not torch.isfinite(total_loss):
            raise FloatingPointError('Non-finite RVC6D pose loss')
        loss_dict = {
            'SARR_loss': total_loss.item(),
            'primary_pose_loss': primary_pose_loss.item(),
            'pose_distance_loss': pose_distance_loss.item(),
            'adds_loss': (
                pose_distance_loss.item()
                if self.primary_pose_loss == 'adds' else 0.0),
            'gadd_loss': (
                pose_distance_loss.item()
                if self.primary_pose_loss == 'gadd' else 0.0),
            'sym_pm_rotation_loss': (
                pose_distance_loss.item()
                if self.primary_pose_loss == 'sym_pm_translation' else 0.0),
            'primary_sarr_translation': float(
                self.primary_pose_loss == 'sarr_translation'),
            'primary_adds': float(self.primary_pose_loss == 'adds'),
            'primary_gadd': float(self.primary_pose_loss == 'gadd'),
            'primary_sym_pm_translation': float(
                self.primary_pose_loss == 'sym_pm_translation'),
            'sarr_cosine_loss': sarr_loss.item(),
            'rot_loss': sarr_loss.item(),
            'translation_loss': translation_loss.item(),
            'score_loss': score_loss.item(),
            'aux_sarr_loss': aux_sarr_loss.item(),
            'stage_sarr_loss': stage_sarr_loss.item(),
            'hypothesis_rotation_loss': hypothesis_rotation_loss.item(),
            'hypothesis_score_loss': hypothesis_score_loss.item(),
            'hypothesis_diversity_loss': hypothesis_diversity_loss.item(),
            'hypothesis_balance_loss': hypothesis_balance_loss.item(),
            'hypothesis_pairwise_distance': hypothesis_pairwise_distance.item(),
            'hypothesis_assignment_entropy': hypothesis_assignment_entropy.item(),
            'hypothesis_usage_entropy': hypothesis_usage_entropy.item(),
            'hypothesis_winner_fraction_max': hypothesis_winner_fraction_max.item(),
            'hypothesis_score_top1_accuracy':
                hypothesis_score_top1_accuracy.item(),
            'center_loss': center_loss.item(),
            'pose_score_loss': pose_score_loss.item(),
        }
        return total_loss, loss_dict


def tless_sarr_kappa(cls_ids, dtype=None, device=None):
    """Return the active SARR symmetry vector for zero-based class ids."""
    ids = torch.as_tensor(cls_ids, dtype=torch.long, device=device)
    table = SARR_KAPPA_TABLE
    if torch.any(ids < 0) or torch.any(ids >= table.size(0)):
        raise ValueError(
            'class ids must be in [0, {}] for the active SARR kappa table '
            '({} rows)'.format(table.size(0) - 1, table.size(0)))
    return table.to(device=ids.device, dtype=dtype or torch.float32)[ids]


def matrix_to_euler_xyz(matrix, eps=1e-6):
    """Convert rotation matrices to intrinsic XYZ Euler angles."""
    if matrix.shape[-2:] != (3, 3):
        raise ValueError('Expected rotation matrices with shape (..., 3, 3)')

    sin_beta = matrix[..., 0, 2].clamp(-1.0, 1.0)
    beta = torch.asin(sin_beta)
    cos_beta = torch.cos(beta)
    regular = cos_beta.abs() > eps

    alpha_regular = torch.atan2(-matrix[..., 1, 2], matrix[..., 2, 2])
    gamma_regular = torch.atan2(-matrix[..., 0, 1], matrix[..., 0, 0])

    # scipy sets gamma to zero at gimbal lock and folds it into alpha.
    alpha_gimbal = torch.atan2(
        torch.where(sin_beta >= 0, matrix[..., 1, 0], -matrix[..., 1, 0]),
        matrix[..., 1, 1],
    )
    alpha = torch.where(regular, alpha_regular, alpha_gimbal)
    gamma = torch.where(regular, gamma_regular, torch.zeros_like(gamma_regular))
    return torch.stack((alpha, beta, gamma), dim=-1)


def euler_xyz_to_matrix(euler):
    """Convert intrinsic XYZ Euler angles to rotation matrices."""
    if euler.shape[-1] != 3:
        raise ValueError('Expected Euler angles with shape (..., 3)')

    alpha, beta, gamma = euler.unbind(dim=-1)
    sx, cx = torch.sin(alpha), torch.cos(alpha)
    sy, cy = torch.sin(beta), torch.cos(beta)
    sz, cz = torch.sin(gamma), torch.cos(gamma)

    rows = (
        torch.stack((cy * cz, -cy * sz, sy), dim=-1),
        torch.stack((cx * sz + cz * sx * sy,
                     cx * cz - sx * sy * sz,
                     -cy * sx), dim=-1),
        torch.stack((sx * sz - cx * cz * sy,
                     cz * sx + cx * sy * sz,
                     cx * cy), dim=-1),
    )
    return torch.stack(rows, dim=-2)


def _broadcast_kappa(cls_ids, reference):
    kappa = tless_sarr_kappa(cls_ids, dtype=reference.dtype, device=reference.device)
    while kappa.dim() > reference.dim() and kappa.size(0) == 1:
        kappa = kappa.squeeze(0)
    while kappa.dim() < reference.dim():
        kappa = kappa.unsqueeze(-2)
    return kappa


def _is_class_v(kappa):
    return ((kappa[..., 0] == 1)
            & (kappa[..., 1] == 2)
            & (kappa[..., 2] == 1))


def matrix_to_sarr(matrix, cls_ids):
    """Map rotation matrices to SARR vectors ordered as [sa, sb, sg, ca, cb, cg]."""
    euler = matrix_to_euler_xyz(matrix)
    kappa = _broadcast_kappa(cls_ids, euler)
    alpha, beta, gamma = euler.unbind(dim=-1)
    class_v = _is_class_v(kappa)

    # Class V needs the alternate canonic mapping from Algorithm 1.
    alpha_mod = torch.remainder(alpha, 2.0 * math.pi)
    gamma_mod = torch.remainder(gamma, 2.0 * math.pi)
    flip = class_v & (alpha_mod > math.pi)
    alpha_v = torch.where(flip,
                          torch.remainder(alpha_mod - math.pi, 2.0 * math.pi),
                          alpha_mod)
    beta_v = torch.where(flip, -beta, beta)
    gamma_v = torch.where(flip,
                          torch.remainder(math.pi - gamma, 2.0 * math.pi),
                          gamma_mod)
    alpha = torch.where(class_v, alpha_v, alpha)
    beta = torch.where(class_v, beta_v, beta)
    gamma = torch.where(class_v, gamma_v, gamma)

    frequency = torch.remainder(kappa, 1000.0)
    nu_alpha = torch.where((kappa[..., 0] >= 2) & (kappa[..., 0] < 1000),
                           torch.cos(alpha), torch.ones_like(alpha))
    nu_beta = torch.where((kappa[..., 1] >= 2) & (kappa[..., 1] < 1000),
                          torch.cos(beta), torch.ones_like(beta))

    sin_alpha = torch.sin(frequency[..., 0] * alpha)
    cos_alpha = torch.cos(frequency[..., 0] * alpha)
    sin_beta = torch.sin(frequency[..., 1] * beta) * nu_alpha
    cos_beta = torch.cos(frequency[..., 1] * beta)
    sin_gamma = torch.sin(frequency[..., 2] * gamma) * nu_alpha * nu_beta
    cos_gamma = torch.cos(frequency[..., 2] * gamma)
    return torch.stack(
        (sin_alpha, sin_beta, sin_gamma, cos_alpha, cos_beta, cos_gamma),
        dim=-1,
    )


def normalize_sarr(sarr, cls_ids, eps=1e-6):
    """Remove cosine-loss scale ambiguity before the inverse SARR mapping."""
    if sarr.shape[-1] != 6:
        raise ValueError('Expected SARR vectors with shape (..., 6)')

    pairs = torch.stack((sarr[..., :3], sarr[..., 3:]), dim=-1)
    norms = torch.linalg.vector_norm(pairs, dim=-1, keepdim=True)
    unit_pairs = pairs / norms.clamp_min(eps)
    fallback = torch.zeros_like(unit_pairs)
    fallback[..., 1] = 1.0
    unit_pairs = torch.where(norms > eps, unit_pairs, fallback)
    column_normalized = torch.cat((unit_pairs[..., 0], unit_pairs[..., 1]), dim=-1)

    kappa = _broadcast_kappa(cls_ids, sarr)
    class_v = _is_class_v(kappa)
    # Class V's gamma sine is intentionally scaled by cos(beta). Preserve that
    # relative scale by estimating the global cosine-loss scale from columns 1-2.
    common_scale = 0.5 * (norms[..., 0, 0] + norms[..., 1, 0])
    class_v_sarr = sarr / common_scale.clamp_min(eps).unsqueeze(-1)
    class_v_sarr = torch.where(
        (common_scale > eps).unsqueeze(-1), class_v_sarr, column_normalized)
    return torch.where(class_v.unsqueeze(-1), class_v_sarr, column_normalized)


def _signed_acos(sine, cosine):
    angle = torch.acos(cosine.clamp(-1.0, 1.0))
    return torch.where(sine < 0, 2.0 * math.pi - angle, angle)


def sarr_to_euler_xyz(sarr, cls_ids):
    """Inverse-map SARR vectors to canonic intrinsic XYZ Euler angles."""
    sarr = normalize_sarr(sarr, cls_ids)
    kappa = _broadcast_kappa(cls_ids, sarr)
    sin_alpha, sin_beta, sin_gamma = sarr[..., :3].unbind(dim=-1)
    cos_alpha, cos_beta, cos_gamma = sarr[..., 3:].unbind(dim=-1)

    alpha = _signed_acos(sin_alpha, cos_alpha) / kappa[..., 0]
    beta = _signed_acos(sin_beta, cos_beta) / kappa[..., 1]
    gamma = _signed_acos(sin_gamma, cos_gamma) / kappa[..., 2]

    continuous = kappa >= 1000
    alpha = torch.where(continuous[..., 0], torch.zeros_like(alpha), alpha)
    beta = torch.where(continuous[..., 1], torch.zeros_like(beta), beta)
    gamma = torch.where(continuous[..., 2], torch.zeros_like(gamma), gamma)

    class_v = _is_class_v(kappa)
    class_v_alpha = _signed_acos(sin_alpha, cos_alpha)
    class_v_beta = _signed_acos(sin_beta, cos_beta) / 2.0
    class_v_gamma = _signed_acos(sin_gamma, cos_gamma)
    class_v_gamma = torch.where(sin_beta < 0, -class_v_gamma, class_v_gamma)
    alpha = torch.where(class_v, class_v_alpha, alpha)
    beta = torch.where(class_v, class_v_beta, beta)
    gamma = torch.where(class_v, class_v_gamma, gamma)
    return torch.stack((alpha, beta, gamma), dim=-1)


def sarr_to_matrix(sarr, cls_ids):
    return euler_xyz_to_matrix(sarr_to_euler_xyz(sarr, cls_ids))


def _signed_acos_for_pose_loss(sine, cosine, eps=1e-6):
    """Signed acos with a finite derivative for random initialization.

    The public SARR inverse uses acos on values in [-1, 1].  Exact endpoints
    have an infinite derivative, which is harmless for inference but can
    contaminate gradients when ADD(-S)/GADD backpropagates through the inverse
    mapping.  An interior clamp preserves the mapping up to a sub-degree
    numerical guard while keeping optimization finite.
    """
    cosine = cosine.clamp(-1.0 + eps, 1.0 - eps)
    angle = torch.acos(cosine)
    return torch.where(sine < 0, 2.0 * math.pi - angle, angle)


def sarr_to_euler_xyz_for_pose_loss(sarr, cls_ids):
    """Training-stable SARR inverse used only by point-distance losses."""
    sarr = normalize_sarr(sarr, cls_ids)
    kappa = _broadcast_kappa(cls_ids, sarr)
    sin_alpha, sin_beta, sin_gamma = sarr[..., :3].unbind(dim=-1)
    cos_alpha, cos_beta, cos_gamma = sarr[..., 3:].unbind(dim=-1)

    alpha = _signed_acos_for_pose_loss(
        sin_alpha, cos_alpha) / kappa[..., 0]
    beta = _signed_acos_for_pose_loss(
        sin_beta, cos_beta) / kappa[..., 1]
    gamma = _signed_acos_for_pose_loss(
        sin_gamma, cos_gamma) / kappa[..., 2]

    continuous = kappa >= 1000
    alpha = torch.where(
        continuous[..., 0], torch.zeros_like(alpha), alpha)
    beta = torch.where(
        continuous[..., 1], torch.zeros_like(beta), beta)
    gamma = torch.where(
        continuous[..., 2], torch.zeros_like(gamma), gamma)

    class_v = _is_class_v(kappa)
    class_v_alpha = _signed_acos_for_pose_loss(sin_alpha, cos_alpha)
    class_v_beta = _signed_acos_for_pose_loss(
        sin_beta, cos_beta) / 2.0
    class_v_gamma = _signed_acos_for_pose_loss(sin_gamma, cos_gamma)
    class_v_gamma = torch.where(
        sin_beta < 0, -class_v_gamma, class_v_gamma)
    alpha = torch.where(class_v, class_v_alpha, alpha)
    beta = torch.where(class_v, class_v_beta, beta)
    gamma = torch.where(class_v, class_v_gamma, gamma)
    return torch.stack((alpha, beta, gamma), dim=-1)


def sarr_to_matrix_for_pose_loss(sarr, cls_ids):
    """Convert SARR to rotation matrices with finite training gradients."""
    return euler_xyz_to_matrix(
        sarr_to_euler_xyz_for_pose_loss(sarr, cls_ids))


def canonical_rotation_error_degrees(prediction, target, cls_ids):
    """Geodesic error against the unique SARR-canonical target rotation."""
    if prediction.shape[-2:] != (3, 3) or target.shape[-2:] != (3, 3):
        raise ValueError('Expected rotation matrices with shape (..., 3, 3)')
    canonical_prediction = sarr_to_matrix(
        matrix_to_sarr(prediction, cls_ids), cls_ids)
    canonical_target = sarr_to_matrix(matrix_to_sarr(target, cls_ids), cls_ids)
    relative = canonical_prediction.transpose(-1, -2) @ canonical_target
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0) * 0.5)
    return torch.rad2deg(torch.acos(cosine.clamp(-1.0, 1.0)))


def sarr_to_quaternion(sarr, cls_ids, eps=1e-6):
    """Inverse-map SARR to scalar-first quaternions used by RVC6D."""
    alpha, beta, gamma = (sarr_to_euler_xyz(sarr, cls_ids) * 0.5).unbind(dim=-1)
    sx, cx = torch.sin(alpha), torch.cos(alpha)
    sy, cy = torch.sin(beta), torch.cos(beta)
    sz, cz = torch.sin(gamma), torch.cos(gamma)
    quat = torch.stack((
        cx * cy * cz - sx * sy * sz,
        sx * cy * cz + cx * sy * sz,
        cx * sy * cz - sx * cy * sz,
        cx * cy * sz + sx * sy * cz,
    ), dim=-1)
    return F.normalize(quat, dim=-1, eps=eps)
