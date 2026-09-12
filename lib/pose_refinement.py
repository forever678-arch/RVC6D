#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Visibility-aware, symmetry-stable depth refinement for RVC6D.

All geometry is expressed in the dataset's object-radius-normalized frame.
The implementation intentionally stays training-free and uses only PyTorch.
"""

import math

import torch
import torch.nn.functional as F


def _weighted_kabsch(source, target, weights, minimum_rank_ratio=1e-4):
    """Fit an absolute rigid transform and reject collinear degeneracy."""
    weights = weights / weights.sum().clamp_min(1e-6)
    source_center = torch.sum(source * weights[:, None], dim=0)
    target_center = torch.sum(target * weights[:, None], dim=0)
    source_centered = source - source_center
    target_centered = target - target_center
    covariance = source_centered.transpose(0, 1) @ (
        target_centered * weights[:, None])
    left, singular_values, right = torch.linalg.svd(covariance)
    if (not torch.isfinite(singular_values).all()
            or singular_values[0] <= 1e-8
            or singular_values[1] / singular_values[0]
            < float(minimum_rank_ratio)):
        return None
    rotation = right.transpose(0, 1) @ left.transpose(0, 1)
    if torch.det(rotation) < 0.0:
        right = right.clone()
        right[-1] *= -1.0
        rotation = right.transpose(0, 1) @ left.transpose(0, 1)
    translation = target_center - rotation @ source_center
    if not torch.isfinite(rotation).all() or not torch.isfinite(translation).all():
        return None
    return rotation, translation


def _rotation_delta_degrees(candidate, current):
    relative = candidate @ current.transpose(0, 1)
    cosine = ((torch.trace(relative) - 1.0) * 0.5).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cosine))


def _axis_angle_to_matrix(vector):
    angle = torch.linalg.vector_norm(vector)
    identity = torch.eye(3, device=vector.device, dtype=vector.dtype)
    if angle <= 1e-8:
        return identity
    axis = vector / angle
    x, y, z = axis
    zero = vector.new_zeros(())
    skew = torch.stack((
        torch.stack((zero, -z, y)),
        torch.stack((z, zero, -x)),
        torch.stack((-y, x, zero)),
    ))
    return (
        identity + torch.sin(angle) * skew
        + (1.0 - torch.cos(angle)) * (skew @ skew))


def _resize_crop_intrinsics(intrinsics, source_size, target_size):
    source_h, source_w = source_size
    target_h, target_w = target_size
    scale_x = float(target_w) / float(source_w)
    scale_y = float(target_h) / float(source_h)
    fx, fy, cx, cy = intrinsics.unbind()
    return torch.stack((
        fx * scale_x,
        fy * scale_y,
        (cx + 0.5) * scale_x - 0.5,
        (cy + 0.5) * scale_y - 0.5,
    ))


def _estimate_observed_normals(observed, valid):
    """Estimate organized point-cloud normals with central differences."""
    height, width = observed.shape[-2:]
    normals = observed.new_zeros(height, width, 3)
    normal_valid = torch.zeros(
        height, width, dtype=torch.bool, device=observed.device)
    if height < 3 or width < 3:
        return normals, normal_valid
    points = observed.permute(1, 2, 0)
    horizontal = points[1:-1, 2:] - points[1:-1, :-2]
    vertical = points[2:, 1:-1] - points[:-2, 1:-1]
    interior = torch.cross(horizontal, vertical, dim=-1)
    norm = torch.linalg.vector_norm(interior, dim=-1)
    support = (
        valid[1:-1, 1:-1]
        & valid[1:-1, 2:] & valid[1:-1, :-2]
        & valid[2:, 1:-1] & valid[:-2, 1:-1]
        & (norm > 1e-6))
    interior = interior / norm.clamp_min(1e-6)[..., None]
    normals[1:-1, 1:-1] = torch.where(
        support[..., None], interior, torch.zeros_like(interior))
    normal_valid[1:-1, 1:-1] = support
    return normals, normal_valid


def _project_visible_model(
        transformed, intrinsics, camera_origin, image_size,
        z_buffer_tolerance):
    """Project CAD points and keep the nearest model point in each pixel."""
    height, width = image_size
    camera_points = transformed - camera_origin[None]
    z = camera_points[:, 2]
    finite = torch.isfinite(camera_points).all(dim=1)
    positive = finite & (z > 1e-5)
    if not torch.any(positive):
        return None
    model_indices = positive.nonzero().flatten()
    camera_points = camera_points[model_indices]
    z = camera_points[:, 2]
    fx, fy, cx, cy = intrinsics.unbind()
    pixel_x = torch.round(fx * camera_points[:, 0] / z + cx).long()
    pixel_y = torch.round(fy * camera_points[:, 1] / z + cy).long()
    inside = (
        (pixel_x >= 0) & (pixel_x < width)
        & (pixel_y >= 0) & (pixel_y < height))
    if not torch.any(inside):
        return None
    model_indices = model_indices[inside]
    z = z[inside]
    pixel_x = pixel_x[inside]
    pixel_y = pixel_y[inside]
    pixels = pixel_y * width + pixel_x
    z_buffer = z.new_full((height * width,), float('inf'))
    z_buffer.scatter_reduce_(
        0, pixels, z, reduce='amin', include_self=True)
    visible = z <= (
        z_buffer[pixels] + float(z_buffer_tolerance))
    if not torch.any(visible):
        return None
    return model_indices[visible], pixels[visible], z[visible]


def _render_correspondences(
        canonical, transformed, observed, confidence, valid, normals,
        normal_valid, intrinsics, camera_origin, distance_threshold,
        occlusion_margin, z_buffer_tolerance):
    projection = _project_visible_model(
        transformed, intrinsics, camera_origin, observed.shape[-2:],
        z_buffer_tolerance)
    if projection is None:
        return None
    model_indices, pixels, model_z = projection
    observed_flat = observed.flatten(1).transpose(0, 1)
    confidence_flat = confidence.flatten()
    valid_flat = valid.flatten()
    normals_flat = normals.reshape(-1, 3)
    normal_valid_flat = normal_valid.flatten()
    target = observed_flat[pixels]
    observed_z = (target - camera_origin[None])[:, 2]
    known = (
        valid_flat[pixels] & torch.isfinite(target).all(dim=1)
        & (observed_z > 1e-5))
    # A closer observed surface means that the rendered model point is
    # externally occluded.  It is unknown, not a negative correspondence.
    externally_occluded = observed_z < (
        model_z - float(occlusion_margin))
    usable = known & ~externally_occluded
    source = transformed[model_indices]
    residual_vector = source - target
    distance = torch.linalg.vector_norm(residual_vector, dim=1)
    free_space = observed_z > (
        model_z + float(occlusion_margin))
    robust = (
        1.0 - (distance / float(distance_threshold)).square()
    ).clamp_min(0.0).square()
    weights = (
        confidence_flat[pixels].clamp_min(1e-4) * robust)
    inlier = (
        usable & (distance <= float(distance_threshold))
        & (weights > 0.0))
    return {
        'canonical': canonical[model_indices],
        'source': source,
        'target': target,
        'distance': distance,
        'depth_residual': (model_z - observed_z).abs(),
        'weights': weights,
        'inlier': inlier,
        'usable': usable,
        'free_space': free_space & known,
        'normal': normals_flat[pixels],
        'normal_valid': normal_valid_flat[pixels] & usable,
        'visible_count': model_indices.numel(),
    }


def _select_observed(observed, confidence, valid, max_points):
    valid_indices = valid.flatten().nonzero().flatten()
    if valid_indices.numel() == 0:
        return None
    scores = confidence.flatten()[valid_indices].clamp_min(1e-6)
    count = min(int(max_points), valid_indices.numel())
    order = torch.topk(scores, count, largest=True).indices
    indices = valid_indices[order]
    points = observed.flatten(1).transpose(0, 1)[indices]
    return points, scores[order]


def _observed_to_model(
        selected_observed, selected_scores, canonical, transformed,
        trim_fraction, distance_threshold, minimum_inliers):
    if selected_observed is None:
        return None
    distances = torch.cdist(
        selected_observed.unsqueeze(0),
        transformed.unsqueeze(0)).squeeze(0)
    nearest, nearest_indices = distances.min(dim=1)
    trim_count = max(
        int(minimum_inliers),
        int(math.ceil(nearest.numel() * float(trim_fraction))))
    trim_count = min(trim_count, nearest.numel())
    order = torch.argsort(nearest)[:trim_count]
    robust = (
        1.0 - (nearest[order] / float(distance_threshold)).square()
    ).clamp_min(0.0).square()
    weights = selected_scores[order] * robust
    inlier = (
        (nearest[order] <= float(distance_threshold))
        & (weights > 0.0))
    return {
        'canonical': canonical[nearest_indices[order]],
        'target': selected_observed[order],
        'distance': nearest[order],
        'weights': weights,
        'inlier': inlier,
        'all_distance': nearest,
    }


def _weighted_clipped_mean(values, weights, clip):
    values = values.clamp_max(float(clip))
    weights = weights.clamp_min(0.0)
    return torch.sum(values * weights) / weights.sum().clamp_min(1e-6)


def _evaluate_pose(
        pose, canonical, observed, confidence, valid, normals, normal_valid,
        selected_observed, selected_scores, intrinsics, camera_origin,
        distance_threshold, trim_fraction, minimum_inliers,
        occlusion_margin, z_buffer_tolerance):
    rotation = pose[:, :3]
    translation = pose[:, 3]
    transformed = canonical @ rotation.transpose(0, 1) + translation
    observed_pairs = _observed_to_model(
        selected_observed, selected_scores, canonical, transformed,
        trim_fraction, distance_threshold, minimum_inliers)
    observed_term = pose.new_tensor(float(distance_threshold))
    observed_inlier_ratio = pose.new_zeros(())
    if observed_pairs is not None:
        positive = observed_pairs['weights'] > 0.0
        if torch.any(positive):
            observed_term = _weighted_clipped_mean(
                observed_pairs['distance'][positive],
                observed_pairs['weights'][positive],
                distance_threshold)
        observed_inlier_ratio = (
            observed_pairs['all_distance'] <= float(distance_threshold)
        ).float().mean()

    render_pairs = None
    projection_available = (
        intrinsics is not None and camera_origin is not None
        and torch.isfinite(intrinsics).all()
        and torch.isfinite(camera_origin).all()
        and intrinsics[0] > 0.0 and intrinsics[1] > 0.0)
    if projection_available:
        render_pairs = _render_correspondences(
            canonical, transformed, observed, confidence, valid,
            normals, normal_valid, intrinsics, camera_origin,
            distance_threshold, occlusion_margin, z_buffer_tolerance)

    if render_pairs is None:
        objective = observed_term
        return objective, {
            'render_pairs': None,
            'observed_pairs': observed_pairs,
            'coverage': pose.new_zeros(()),
            'inlier_ratio': observed_inlier_ratio,
            'free_space_ratio': pose.new_zeros(()),
            'projection_available': False,
        }

    usable = render_pairs['usable']
    usable_weights = (
        confidence.flatten().new_ones(usable.shape)
        * render_pairs['weights'].clamp_min(1e-4))
    if torch.any(usable):
        render_term = _weighted_clipped_mean(
            render_pairs['distance'][usable], usable_weights[usable],
            distance_threshold)
        depth_term = _weighted_clipped_mean(
            render_pairs['depth_residual'][usable], usable_weights[usable],
            distance_threshold)
    else:
        render_term = pose.new_tensor(float(distance_threshold))
        depth_term = pose.new_tensor(float(distance_threshold))
    plane_mask = render_pairs['normal_valid'] & usable
    if torch.any(plane_mask):
        plane_residual = torch.abs(torch.sum(
            (render_pairs['source'] - render_pairs['target'])
            * render_pairs['normal'], dim=1))
        plane_term = _weighted_clipped_mean(
            plane_residual[plane_mask], usable_weights[plane_mask],
            distance_threshold)
    else:
        plane_term = render_term
    coverage = usable.float().sum() / max(
        1, int(render_pairs['visible_count']))
    inlier_ratio = render_pairs['inlier'].float().sum() / max(
        1, int(render_pairs['visible_count']))
    free_space_ratio = render_pairs['free_space'].float().sum() / max(
        1, int(render_pairs['visible_count']))
    coverage_penalty = float(distance_threshold) * (1.0 - coverage)
    free_space_penalty = float(distance_threshold) * free_space_ratio
    objective = (
        0.35 * render_term
        + 0.20 * depth_term
        + 0.15 * plane_term
        + 0.20 * observed_term
        + 0.07 * coverage_penalty
        + 0.03 * free_space_penalty)
    return objective, {
        'render_pairs': render_pairs,
        'observed_pairs': observed_pairs,
        'coverage': coverage,
        'inlier_ratio': inlier_ratio,
        'free_space_ratio': free_space_ratio,
        'projection_available': True,
    }


def _point_to_plane_candidate(
        pose, pairs, continuous_symmetry_axis, damping,
        max_rotation_degrees, max_translation, minimum_inliers):
    mask = pairs['inlier'] & pairs['normal_valid']
    if int(mask.sum()) < int(minimum_inliers):
        return None
    source = pairs['source'][mask]
    target = pairs['target'][mask]
    normal = pairs['normal'][mask]
    weights = pairs['weights'][mask].clamp_min(1e-6)
    residual = torch.sum((source - target) * normal, dim=1)
    jacobian = torch.cat((torch.cross(source, normal, dim=1), normal), dim=1)
    weighted_jacobian = jacobian * weights.sqrt()[:, None]
    weighted_residual = residual * weights.sqrt()
    normal_matrix = weighted_jacobian.transpose(0, 1) @ weighted_jacobian
    scale = normal_matrix.diagonal().mean().clamp_min(1e-6)
    normal_matrix = normal_matrix + (
        float(damping) * scale
        * torch.eye(6, device=pose.device, dtype=pose.dtype))
    if torch.linalg.matrix_rank(normal_matrix) < 5:
        return None
    rhs = -(weighted_jacobian.transpose(0, 1) @ weighted_residual)
    try:
        twist = torch.linalg.solve(normal_matrix, rhs)
    except RuntimeError:
        return None
    if not torch.isfinite(twist).all():
        return None
    omega, delta_translation = twist[:3], twist[3:]
    axis = continuous_symmetry_axis
    if axis is not None and torch.linalg.vector_norm(axis) > 1e-6:
        camera_axis = pose[:, :3] @ F.normalize(axis, dim=0)
        # Rotation about a continuous symmetry axis is unobservable.  Removing
        # that twist component prevents arbitrary self-spin.
        omega = omega - torch.dot(omega, camera_axis) * camera_axis
    max_angle = math.radians(float(max_rotation_degrees))
    omega_norm = torch.linalg.vector_norm(omega)
    if omega_norm > max_angle:
        omega = omega * (max_angle / omega_norm)
    translation_norm = torch.linalg.vector_norm(delta_translation)
    if translation_norm > float(max_translation):
        delta_translation = (
            delta_translation * float(max_translation) / translation_norm)
    delta_rotation = _axis_angle_to_matrix(omega)
    rotation = delta_rotation @ pose[:, :3]
    translation = delta_rotation @ pose[:, 3] + delta_translation
    return torch.cat((rotation, translation[:, None]), dim=1)


def _kabsch_candidate(
        pose, canonical, pairs, max_rotation_degrees, max_translation,
        minimum_inliers):
    if pairs is None:
        return None
    inlier = pairs['inlier']
    if int(inlier.sum()) < int(minimum_inliers):
        return None
    fitted = _weighted_kabsch(
        pairs['canonical'][inlier], pairs['target'][inlier],
        pairs['weights'][inlier])
    if fitted is None:
        return None
    rotation, translation = fitted
    if (_rotation_delta_degrees(rotation, pose[:, :3])
            > float(max_rotation_degrees)):
        return None
    if (torch.linalg.vector_norm(translation - pose[:, 3])
            > float(max_translation)):
        return None
    return torch.cat((rotation, translation[:, None]), dim=1)


@torch.no_grad()
def refine_pose_visible_depth(
        coarse_poses, model_xyz, observed_xyz, confidence,
        valid_mask=None, point_weight=None, crop_intrinsics=None,
        camera_origin=None, continuous_symmetry_axis=None,
        iterations=4, max_observed_points=384,
        trim_fraction=0.70, distance_threshold=0.20,
        final_distance_threshold=0.08, occlusion_margin=0.06,
        z_buffer_tolerance=0.02, max_rotation_degrees=8.0,
        max_translation=0.25, minimum_inliers=24,
        damping=1e-3, minimum_relative_improvement=0.01):
    """Refine one pose per sample with strict rollback and absolute fitness."""
    if coarse_poses.dim() != 3 or coarse_poses.shape[1:] != (3, 4):
        raise ValueError('coarse_poses must have shape (batch, 3, 4)')
    if model_xyz.dim() != 3 or model_xyz.size(1) != 3:
        raise ValueError('model_xyz must have shape (batch, 3, points)')
    if observed_xyz.dim() != 4 or observed_xyz.size(1) != 3:
        raise ValueError('observed_xyz must have shape (batch, 3, H, W)')
    if confidence.dim() != 3:
        raise ValueError('confidence must have shape (batch, H, W)')
    batch_size = coarse_poses.size(0)
    if model_xyz.size(0) != batch_size or observed_xyz.size(0) != batch_size:
        raise ValueError('all refinement inputs must share the same batch size')
    output = coarse_poses.clone()
    fitness = coarse_poses.new_zeros(batch_size)
    applied = torch.zeros(
        batch_size, device=coarse_poses.device, dtype=torch.bool)
    source_size = observed_xyz.shape[-2:]
    target_size = confidence.shape[-2:]
    observed_xyz = F.interpolate(
        observed_xyz, size=target_size, mode='nearest')
    if valid_mask is None:
        valid_mask = torch.linalg.vector_norm(
            observed_xyz, dim=1, keepdim=True) > 0.0
    else:
        valid_mask = F.interpolate(
            valid_mask.float(), size=target_size, mode='nearest') > 0.5
    if point_weight is None:
        point_weight = torch.ones_like(valid_mask, dtype=confidence.dtype)
    else:
        point_weight = F.interpolate(
            point_weight.float(), size=target_size,
            mode='bilinear', align_corners=False)
    combined_confidence = (
        confidence.clamp(0.0, 1.0)
        * point_weight[:, 0].clamp(0.0, 1.0).sqrt())
    valid_mask = (
        valid_mask
        & (point_weight > 1e-4)
        & torch.isfinite(observed_xyz).all(dim=1, keepdim=True))

    scaled_intrinsics = None
    if crop_intrinsics is not None:
        if crop_intrinsics.shape != (batch_size, 4):
            raise ValueError('crop_intrinsics must have shape (batch, 4)')
        scaled_intrinsics = torch.stack([
            _resize_crop_intrinsics(
                crop_intrinsics[index], source_size, target_size)
            for index in range(batch_size)
        ])
    if camera_origin is not None and camera_origin.shape != (batch_size, 3):
        raise ValueError('camera_origin must have shape (batch, 3)')
    if (continuous_symmetry_axis is not None
            and continuous_symmetry_axis.shape != (batch_size, 3)):
        raise ValueError(
            'continuous_symmetry_axis must have shape (batch, 3)')

    final_threshold = min(
        float(distance_threshold), float(final_distance_threshold))
    for batch_index in range(batch_size):
        valid = valid_mask[batch_index, 0]
        if int(valid.sum()) < int(minimum_inliers):
            continue
        observed = observed_xyz[batch_index]
        sample_confidence = combined_confidence[batch_index]
        canonical = model_xyz[batch_index].transpose(0, 1)
        if canonical.size(0) < int(minimum_inliers):
            continue
        normals, normal_valid = _estimate_observed_normals(observed, valid)
        selected = _select_observed(
            observed, sample_confidence, valid, max_observed_points)
        if selected is None or selected[0].size(0) < int(minimum_inliers):
            continue
        intrinsics = (
            scaled_intrinsics[batch_index]
            if scaled_intrinsics is not None else None)
        origin = (
            camera_origin[batch_index]
            if camera_origin is not None else None)
        symmetry_axis = (
            continuous_symmetry_axis[batch_index]
            if continuous_symmetry_axis is not None else None)
        pose = output[batch_index].clone()
        initial_pose = pose.clone()
        initial_objective, initial_details = _evaluate_pose(
            pose, canonical, observed, sample_confidence, valid,
            normals, normal_valid, selected[0], selected[1],
            intrinsics, origin, float(distance_threshold), trim_fraction,
            minimum_inliers, occlusion_margin, z_buffer_tolerance)
        current_objective = initial_objective
        current_details = initial_details

        for iteration in range(int(iterations)):
            if int(iterations) <= 1:
                fraction = 1.0
            else:
                fraction = float(iteration) / float(int(iterations) - 1)
            threshold = (
                float(distance_threshold)
                * (final_threshold / float(distance_threshold)) ** fraction)
            # Re-evaluate at the current coarse-to-fine threshold so that
            # correspondence weights and acceptance use the same support.
            current_objective, current_details = _evaluate_pose(
                pose, canonical, observed, sample_confidence, valid,
                normals, normal_valid, selected[0], selected[1],
                intrinsics, origin, threshold, trim_fraction,
                minimum_inliers, occlusion_margin, z_buffer_tolerance)
            candidates = []
            render_pairs = current_details['render_pairs']
            if render_pairs is not None:
                kabsch = _kabsch_candidate(
                    pose, canonical, render_pairs,
                    max_rotation_degrees, max_translation, minimum_inliers)
                if kabsch is not None:
                    candidates.append(kabsch)
                point_to_plane = _point_to_plane_candidate(
                    pose, render_pairs, symmetry_axis, damping,
                    max_rotation_degrees, max_translation, minimum_inliers)
                if point_to_plane is not None:
                    candidates.append(point_to_plane)
            fallback_kabsch = _kabsch_candidate(
                pose, canonical, current_details['observed_pairs'],
                max_rotation_degrees, max_translation, minimum_inliers)
            if fallback_kabsch is not None:
                candidates.append(fallback_kabsch)
            if not candidates:
                break

            best_pose = None
            best_objective = current_objective
            best_details = current_details
            for candidate in candidates:
                candidate_objective, candidate_details = _evaluate_pose(
                    candidate, canonical, observed, sample_confidence, valid,
                    normals, normal_valid, selected[0], selected[1],
                    intrinsics, origin, threshold, trim_fraction,
                    minimum_inliers, occlusion_margin, z_buffer_tolerance)
                if not torch.isfinite(candidate_objective):
                    continue
                coverage_ok = (
                    not current_details['projection_available']
                    or candidate_details['coverage'] + 0.02
                    >= current_details['coverage'])
                if coverage_ok and candidate_objective < best_objective:
                    best_pose = candidate
                    best_objective = candidate_objective
                    best_details = candidate_details
            required_gain = max(
                1e-5,
                float(minimum_relative_improvement)
                * max(float(current_objective), 0.1 * threshold))
            if (best_pose is None
                    or float(current_objective - best_objective)
                    <= required_gain):
                break
            pose = best_pose
            current_objective = best_objective
            current_details = best_details
            applied[batch_index] = True

        final_objective, final_details = _evaluate_pose(
            pose, canonical, observed, sample_confidence, valid,
            normals, normal_valid, selected[0], selected[1],
            intrinsics, origin, final_threshold, trim_fraction,
            minimum_inliers, occlusion_margin, z_buffer_tolerance)
        coarse_final_objective, _ = _evaluate_pose(
            initial_pose, canonical, observed, sample_confidence, valid,
            normals, normal_valid, selected[0], selected[1],
            intrinsics, origin, final_threshold, trim_fraction,
            minimum_inliers, occlusion_margin, z_buffer_tolerance)
        # Full rollback is mandatory: an intermediate acceptance must not leak
        # into the result unless it also wins under the final strict objective.
        if (not torch.isfinite(final_objective)
                or final_objective >= coarse_final_objective):
            pose = initial_pose
            final_objective = coarse_final_objective
            _, final_details = _evaluate_pose(
                pose, canonical, observed, sample_confidence, valid,
                normals, normal_valid, selected[0], selected[1],
                intrinsics, origin, final_threshold, trim_fraction,
                minimum_inliers, occlusion_margin, z_buffer_tolerance)
            applied[batch_index] = False
        output[batch_index] = pose
        support_quality = final_details['inlier_ratio'].clamp(0.0, 1.0)
        if final_details['projection_available']:
            support_quality = torch.sqrt(
                support_quality
                * final_details['coverage'].clamp(0.0, 1.0))
        fitness[batch_index] = (
            torch.exp(-final_objective / max(final_threshold, 1e-6))
            * support_quality).clamp(0.0, 1.0)
    return output, fitness, applied


def _expand_candidates(tensor, candidate_count):
    if tensor is None:
        return None
    return tensor[:, None].expand(
        tensor.size(0), candidate_count, *tensor.shape[1:]
    ).reshape(tensor.size(0) * candidate_count, *tensor.shape[1:])


def _gather_candidates(tensor, indices):
    if tensor is None:
        return None
    batch = torch.arange(tensor.size(0), device=tensor.device)[:, None]
    return tensor[batch, indices].reshape(
        tensor.size(0) * indices.size(1), *tensor.shape[2:])


def _fast_visible_candidate_scores(
        candidates, model_xyz, observed_xyz, confidence,
        valid_mask=None, point_weight=None, max_model_points=128,
        max_observed_points=192, inlier_threshold=0.15,
        clip_distance=0.30):
    """Vectorized depth score used for cheap per-epoch Pro decoding.

    The dominant term is observed-to-model distance, matching VSD's visible
    surface emphasis.  A lightly weighted best-half model-to-observation term
    prevents degenerate poses without penalizing legitimately occluded CAD
    surfaces.  All distances are in the radius-normalized object frame.
    """
    batch_size, candidate_count = candidates.shape[:2]
    target_size = confidence.shape[-2:]
    observed = F.interpolate(
        observed_xyz.float(), size=target_size, mode='nearest')
    if valid_mask is None:
        valid = torch.linalg.vector_norm(
            observed, dim=1, keepdim=True) > 0.0
    else:
        valid = F.interpolate(
            valid_mask.float(), size=target_size,
            mode='nearest') > 0.5
    if point_weight is None:
        reliability = valid.to(confidence.dtype)
    else:
        reliability = F.interpolate(
            point_weight.float(), size=target_size, mode='bilinear',
            align_corners=False).clamp(0.0, 1.0)
        reliability = reliability * valid.to(reliability.dtype)
    reliability = (
        reliability[:, 0]
        * confidence.float().clamp(0.0, 1.0).sqrt())

    observed = observed.flatten(2).transpose(1, 2)
    reliability = reliability.flatten(1)
    observed_count = min(int(max_observed_points), observed.size(1))
    selected_weight, selected_index = torch.topk(
        reliability, observed_count, dim=1, largest=True)
    observed = torch.gather(
        observed, 1, selected_index[:, :, None].expand(-1, -1, 3))
    selected_valid = selected_weight > 1e-5

    model = model_xyz.float().transpose(1, 2)
    if model.size(1) > int(max_model_points):
        model_index = torch.linspace(
            0, model.size(1) - 1, steps=int(max_model_points),
            device=model.device).round().long()
        model = model[:, model_index]
    transformed = torch.einsum(
        'bcij,bmj->bcmi', candidates[:, :, :, :3].float(), model)
    transformed = transformed + candidates[:, :, None, :, 3].float()

    model_count = transformed.size(2)
    expanded_observed = observed[:, None].expand(
        -1, candidate_count, -1, -1)
    pair_distance = torch.cdist(
        transformed.reshape(-1, model_count, 3),
        expanded_observed.reshape(
            -1, observed_count, 3)).reshape(
                batch_size, candidate_count, model_count, observed_count)

    observed_distance = pair_distance.amin(dim=2)
    observed_weight = selected_weight[:, None].expand(
        -1, candidate_count, -1)
    observed_mass = observed_weight.sum(dim=2).clamp_min(1e-6)
    visible_error = (
        observed_distance.clamp_max(float(clip_distance))
        * observed_weight).sum(dim=2) / observed_mass
    visible_inlier = (
        (observed_distance < float(inlier_threshold)).to(
            observed_weight.dtype) * observed_weight
    ).sum(dim=2) / observed_mass

    valid_pairs = selected_valid[:, None, None, :]
    model_distance = pair_distance.masked_fill(
        ~valid_pairs, float('inf')).amin(dim=3)
    model_distance = model_distance.clamp_max(float(clip_distance))
    visible_model_count = max(1, model_count // 2)
    visible_model_distance = torch.topk(
        model_distance, visible_model_count, dim=2,
        largest=False).values
    model_error = visible_model_distance.mean(dim=2)
    model_inlier = (
        visible_model_distance < float(inlier_threshold)
    ).float().mean(dim=2)

    error = 0.85 * visible_error + 0.15 * model_error
    support = torch.sqrt(visible_inlier.clamp(0.0, 1.0))
    support = support * (0.75 + 0.25 * model_inlier.clamp(0.0, 1.0))
    scores = torch.exp(
        -error / max(float(inlier_threshold), 1e-6)) * support
    enough_observed = (selected_valid.sum(dim=1) >= 8)[:, None]
    return torch.where(enough_observed, scores, torch.zeros_like(scores))


@torch.no_grad()
def decode_rvc6d(
        preds, model_xyz=None, observed_xyz=None, valid_mask=None,
        point_weight=None, crop_intrinsics=None, camera_origin=None,
        continuous_symmetry_axis=None, refine=True,
        refinement_kwargs=None):
    """Decode, geometrically screen Pro candidates, and optionally refine."""
    from lib.sarr import sarr_to_matrix
    from lib.utils import post_processing_sarr_wi_vote

    coarse = post_processing_sarr_wi_vote(preds)
    dense_score = preds['pred_s'].detach().flatten(1)
    vote_mask = preds.get('vote_valid_mask')
    if vote_mask is not None:
        vote_mask = F.interpolate(
            vote_mask.float(), size=preds['pred_s'].shape[-2:],
            mode='nearest').flatten(1) > 0.5
    vote_quality = []
    for batch_index in range(dense_score.size(0)):
        values = dense_score[batch_index]
        if vote_mask is not None and torch.any(vote_mask[batch_index]):
            values = values[vote_mask[batch_index]]
        top_values = torch.topk(
            values, min(32, values.numel()), largest=True).values
        vote_quality.append(top_values.mean())
    vote_quality = torch.stack(vote_quality)
    object_quality = preds.get('pose_score', vote_quality).detach()
    network_quality = (
        0.70 * object_quality + 0.30 * vote_quality).clamp(1e-6, 1.0)
    refinement_fitness = coarse.new_zeros(coarse.size(0))
    refinement_applied = torch.zeros(
        coarse.size(0), device=coarse.device, dtype=torch.bool)
    selected_candidate = torch.zeros(
        coarse.size(0), device=coarse.device, dtype=torch.long)
    poses = coarse

    has_pro_candidates = (
        'surface_pose' in preds or 'metric_pose' in preds)
    if has_pro_candidates and not refine:
        if model_xyz is None or observed_xyz is None:
            raise ValueError(
                'model_xyz and observed_xyz are required for Pro screening')
        hypotheses = preds.get('rotation_hypotheses')
        hypothesis_logits = preds.get('rotation_hypothesis_logits')
        if hypotheses is not None and hypothesis_logits is not None:
            batch_size, hypothesis_count, _ = hypotheses.shape
            flat_ids = preds['cls_id'][:, None].expand(
                batch_size, hypothesis_count).reshape(-1)
            hypothesis_rotation = sarr_to_matrix(
                hypotheses.detach().reshape(-1, 6), flat_ids
            ).reshape(batch_size, hypothesis_count, 3, 3)
            hypothesis_pose = coarse[:, None].expand(
                -1, hypothesis_count, -1, -1).clone()
            hypothesis_pose[:, :, :, :3] = hypothesis_rotation
            candidates = torch.cat((coarse[:, None], hypothesis_pose), dim=1)
            probability = torch.softmax(hypothesis_logits.detach(), dim=1)
            candidate_prior = torch.cat((
                probability.max(dim=1, keepdim=True).values,
                probability), dim=1)
        else:
            candidates = coarse[:, None]
            candidate_prior = coarse.new_ones(coarse.size(0), 1)
        if 'metric_pose' in preds:
            candidates = torch.cat((
                candidates, preds['metric_pose'].detach()[:, None]), dim=1)
            candidate_prior = torch.cat((candidate_prior, preds.get(
                'metric_pose_quality', network_quality
            ).detach()[:, None]), dim=1)
        if 'surface_pose' in preds:
            candidates = torch.cat((
                candidates, preds['surface_pose'].detach()[:, None]), dim=1)
            candidate_prior = torch.cat((candidate_prior, preds.get(
                'surface_pose_quality', network_quality
            ).detach()[:, None]), dim=1)
        geometry_scores = _fast_visible_candidate_scores(
            candidates, model_xyz, observed_xyz,
            dense_score.reshape(
                dense_score.size(0), *preds['pred_s'].shape[-2:]),
            valid_mask=valid_mask, point_weight=point_weight)
        combined_scores = geometry_scores + 0.03 * candidate_prior
        selected_candidate = combined_scores.argmax(dim=1)
        batch = torch.arange(coarse.size(0), device=coarse.device)
        poses = candidates[batch, selected_candidate]
        refinement_fitness = geometry_scores[batch, selected_candidate]

    if refine:
        if model_xyz is None or observed_xyz is None:
            raise ValueError(
                'model_xyz and observed_xyz are required for Pro screening/refinement')
        kwargs = dict(refinement_kwargs or {})
        requested_iterations = int(kwargs.pop('iterations', 4))
        refine_iterations = requested_iterations if refine else 0
        requested_topk = max(1, int(kwargs.pop('candidate_topk', 2)))
        candidate_topk = requested_topk if refine else 1
        candidate_prior_weight = float(
            kwargs.pop('candidate_prior_weight', 0.05))
        hypotheses = preds.get('rotation_hypotheses')
        hypothesis_logits = preds.get('rotation_hypothesis_logits')
        if hypotheses is not None and hypothesis_logits is not None:
            batch_size, hypothesis_count, _ = hypotheses.shape
            flat_ids = preds['cls_id'][:, None].expand(
                batch_size, hypothesis_count).reshape(-1)
            hypothesis_rotation = sarr_to_matrix(
                hypotheses.detach().reshape(-1, 6), flat_ids
            ).reshape(batch_size, hypothesis_count, 3, 3)
            hypothesis_pose = coarse[:, None].expand(
                -1, hypothesis_count, -1, -1).clone()
            hypothesis_pose[:, :, :, :3] = hypothesis_rotation
            candidates = torch.cat((coarse[:, None], hypothesis_pose), dim=1)
            probability = torch.softmax(hypothesis_logits.detach(), dim=1)
            candidate_prior = torch.cat((
                probability.max(dim=1, keepdim=True).values,
                probability), dim=1)
        else:
            candidates = coarse[:, None]
            candidate_prior = coarse.new_ones(coarse.size(0), 1)

        extra_candidates = []
        extra_priors = []
        if 'metric_pose' in preds:
            extra_candidates.append(preds['metric_pose'].detach()[:, None])
            extra_priors.append(preds.get(
                'metric_pose_quality', network_quality
            ).detach()[:, None])
        if 'surface_pose' in preds:
            extra_candidates.append(preds['surface_pose'].detach()[:, None])
            extra_priors.append(preds.get(
                'surface_pose_quality', network_quality
            ).detach()[:, None])
        if extra_candidates:
            candidates = torch.cat([candidates] + extra_candidates, dim=1)
            candidate_prior = torch.cat(
                [candidate_prior] + extra_priors, dim=1)
        candidate_count = candidates.size(1)
        candidate_topk = min(candidate_topk, candidate_count)

        expanded = candidates.reshape(-1, 3, 4)
        expanded_model = _expand_candidates(model_xyz, candidate_count)
        expanded_observed = _expand_candidates(observed_xyz, candidate_count)
        expanded_confidence = _expand_candidates(
            dense_score.reshape(
                dense_score.size(0), *preds['pred_s'].shape[-2:]),
            candidate_count)
        expanded_valid = _expand_candidates(valid_mask, candidate_count)
        expanded_weight = _expand_candidates(point_weight, candidate_count)
        expanded_intrinsics = _expand_candidates(
            crop_intrinsics, candidate_count)
        expanded_origin = _expand_candidates(camera_origin, candidate_count)
        expanded_axis = _expand_candidates(
            continuous_symmetry_axis, candidate_count)
        # Stage 1 evaluates every mode under the same absolute geometric
        # objective.  Only the strongest modes pay the iterative refinement
        # cost, so four-hypothesis inference remains practical.
        _, prescore, _ = refine_pose_visible_depth(
            expanded, expanded_model, expanded_observed,
            expanded_confidence, valid_mask=expanded_valid,
            point_weight=expanded_weight,
            crop_intrinsics=expanded_intrinsics,
            camera_origin=expanded_origin,
            continuous_symmetry_axis=expanded_axis,
            iterations=0, **kwargs)
        prescore = prescore.reshape(coarse.size(0), candidate_count)
        screening_score = (
            prescore + candidate_prior_weight * candidate_prior)
        top_indices = torch.topk(
            screening_score, candidate_topk, dim=1).indices
        selected_coarse = _gather_candidates(candidates, top_indices)
        selected_prior = _gather_candidates(
            candidate_prior[:, :, None], top_indices).reshape(
                coarse.size(0), candidate_topk)
        batch = torch.arange(coarse.size(0), device=coarse.device)[:, None]
        selected_model = model_xyz[batch.expand_as(top_indices)].reshape(
            coarse.size(0) * candidate_topk, *model_xyz.shape[1:])
        selected_observed = observed_xyz[
            batch.expand_as(top_indices)].reshape(
                coarse.size(0) * candidate_topk, *observed_xyz.shape[1:])

        def repeat_selected(tensor):
            if tensor is None:
                return None
            return tensor[batch.expand_as(top_indices)].reshape(
                coarse.size(0) * candidate_topk, *tensor.shape[1:])

        selected_confidence = repeat_selected(
            dense_score.reshape(
                dense_score.size(0), *preds['pred_s'].shape[-2:]))
        refined, final_fitness, final_applied = refine_pose_visible_depth(
            selected_coarse, selected_model, selected_observed,
            selected_confidence, valid_mask=repeat_selected(valid_mask),
            point_weight=repeat_selected(point_weight),
            crop_intrinsics=repeat_selected(crop_intrinsics),
            camera_origin=repeat_selected(camera_origin),
            continuous_symmetry_axis=repeat_selected(
                continuous_symmetry_axis),
            iterations=refine_iterations, **kwargs)
        refined = refined.reshape(coarse.size(0), candidate_topk, 3, 4)
        final_fitness = final_fitness.reshape(
            coarse.size(0), candidate_topk)
        final_applied = final_applied.reshape(
            coarse.size(0), candidate_topk)
        final_score = (
            final_fitness + candidate_prior_weight * selected_prior)
        winner = final_score.argmax(dim=1)
        poses = refined[
            torch.arange(coarse.size(0), device=coarse.device), winner]
        refinement_fitness = final_fitness[
            torch.arange(coarse.size(0), device=coarse.device), winner]
        refinement_applied = final_applied[
            torch.arange(coarse.size(0), device=coarse.device), winner]
        selected_candidate = top_indices[
            torch.arange(coarse.size(0), device=coarse.device), winner]

    geometry_valid = refinement_fitness > 0.0
    pose_quality = torch.where(
        geometry_valid,
        0.80 * network_quality + 0.20 * refinement_fitness,
        network_quality).clamp(1e-6, 1.0)
    return poses, pose_quality, {
        'fitness': refinement_fitness,
        'applied': refinement_applied,
        'selected_candidate': selected_candidate,
        'vote_quality': vote_quality,
        'object_quality': object_quality,
    }
