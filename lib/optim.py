#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Optimizer utilities for RVC6D scratch training."""

import math

import torch


def _is_no_decay_parameter(name, parameter):
    lower = name.lower()
    return (
        parameter.ndim <= 1
        or lower.endswith('.bias')
        or 'embedding' in lower
        or 'query_seed' in lower
        or lower.endswith('.gamma')
        or lower.endswith('_scale')
        or '.scale' in lower)


def build_rvc6d_adamw(
        model, learning_rate=3e-4, weight_decay=5e-4,
        betas=(0.9, 0.999), eps=1e-8):
    """Build the uniform-learning-rate AdamW optimizer used by RVC6D."""
    learning_rate = float(learning_rate)
    weight_decay = float(weight_decay)
    if learning_rate <= 0.0:
        raise ValueError('RVC6D learning rate must be positive')
    if weight_decay < 0.0:
        raise ValueError('weight_decay must be non-negative')
    grouped = {'scratch_decay': [], 'scratch_no_decay': []}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        suffix = (
            'no_decay' if _is_no_decay_parameter(name, parameter)
            else 'decay')
        grouped['scratch_{}'.format(suffix)].append(parameter)
    parameter_groups = []
    for group_name in ('scratch_decay', 'scratch_no_decay'):
        parameters = grouped[group_name]
        if not parameters:
            continue
        parameter_groups.append({
            'params': parameters,
            'lr': learning_rate,
            'base_lr': learning_rate,
            'initial_lr': learning_rate,
            'weight_decay': (
                0.0 if group_name.endswith('no_decay') else weight_decay),
            'group_name': group_name,
        })
    return torch.optim.AdamW(
        parameter_groups, betas=tuple(betas), eps=float(eps))


def apply_warmup_cosine(
        optimizer, step, warmup_steps, total_steps,
        minimum_ratio=0.01, warmup_start_ratio=0.1):
    step = max(0, int(step))
    warmup_steps = max(0, int(warmup_steps))
    total_steps = max(warmup_steps + 1, int(total_steps))
    minimum_ratio = float(minimum_ratio)
    warmup_start_ratio = float(warmup_start_ratio)
    if not 0.0 < minimum_ratio <= 1.0:
        raise ValueError('minimum_ratio must be in (0, 1]')
    if not 0.0 < warmup_start_ratio <= 1.0:
        raise ValueError('warmup_start_ratio must be in (0, 1]')

    if warmup_steps > 0 and step <= warmup_steps:
        progress = step / float(warmup_steps)
        factor = warmup_start_ratio + (
            1.0 - warmup_start_ratio) * progress
    else:
        progress = (
            (step - warmup_steps)
            / float(max(1, total_steps - warmup_steps)))
        progress = min(1.0, max(0.0, progress))
        factor = minimum_ratio + 0.5 * (
            1.0 - minimum_ratio) * (1.0 + math.cos(math.pi * progress))

    learning_rates = {}
    for index, group in enumerate(optimizer.param_groups):
        base_lr = float(group.get('base_lr', group.get('initial_lr', group['lr'])))
        group['base_lr'] = base_lr
        group['lr'] = base_lr * factor
        name = group.get('group_name', 'group_{}'.format(index))
        learning_rates[name] = group['lr']
    return learning_rates


def optimizer_group_summary(optimizer):
    summary = []
    for index, group in enumerate(optimizer.param_groups):
        summary.append({
            'name': group.get('group_name', 'group_{}'.format(index)),
            'parameters': sum(parameter.numel() for parameter in group['params']),
            'base_lr': float(group.get('base_lr', group['lr'])),
            'weight_decay': float(group['weight_decay']),
        })
    return summary


def load_optimizer_state_name_matched(optimizer, checkpoint, model):
    """Resume AdamW state by parameter name when module order changed.

    Checkpoints saved by older versions of this codebase keep optimizer
    momentum keyed to the old module registration order, while torch maps
    optimizer state onto live parameters positionally.  Registration order
    changed in the rename, so a positional load silently pairs every
    momentum slot with the wrong parameter.  The checkpoint state_dict
    still preserves the old traversal order, so rebuild the per-group
    parameter sequence by name and hand torch the saved slot ids in live
    parameter order.  Returns True when the remap was applied.
    """
    saved = checkpoint.get('optimizer')
    if not saved or not saved.get('param_groups') or not saved.get('state'):
        return False
    state_source = checkpoint.get(
        'train_state_dict', checkpoint.get('state_dict'))
    if not state_source:
        return False
    named = dict(model.named_parameters())
    name_by_param = {
        id(parameter): name for name, parameter in model.named_parameters()}
    old_order = [key for key in state_source.keys() if key in named]
    suffix_groups = {'decay': [], 'no_decay': []}
    for name in old_order:
        suffix = (
            'no_decay' if _is_no_decay_parameter(name, named[name])
            else 'decay')
        suffix_groups[suffix].append(name)

    def _suffix(group, fallback):
        group_name = str(group.get('group_name', ''))
        if group_name.endswith('no_decay'):
            return 'no_decay'
        if group_name.endswith('decay'):
            return 'decay'
        return fallback

    saved_groups = list(saved['param_groups'])
    current_groups = list(optimizer.param_groups)
    if len(saved_groups) != len(current_groups):
        return False
    saved_id_by_name = {}
    for index, (group, current) in enumerate(zip(saved_groups, current_groups)):
        names = suffix_groups[_suffix(
            group, 'decay' if index == 0 else 'no_decay')]
        if len(names) != len(group['params']):
            return False
        for saved_id, name in zip(group['params'], names):
            if saved_id not in saved['state']:
                return False
            saved_id_by_name[name] = saved_id
    if len(saved_id_by_name) != len(old_order):
        return False
    flat_current = [
        parameter for group in current_groups
        for parameter in group['params']]
    try:
        flat_saved_ids = [
            saved_id_by_name[name_by_param[id(parameter)]]
            for parameter in flat_current]
    except KeyError:
        return False
    remapped_groups = []
    offset = 0
    for group in current_groups:
        fixed = {
            key: value for key, value in group.items() if key != 'params'}
        fixed['params'] = flat_saved_ids[offset:offset + len(group['params'])]
        offset += len(group['params'])
        remapped_groups.append(fixed)
    optimizer.load_state_dict({
        'state': saved['state'], 'param_groups': remapped_groups})
    return True
