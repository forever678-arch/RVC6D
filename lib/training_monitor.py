"""Persistent training metrics, overfitting diagnostics, EMA, and curve plots."""

import copy
import csv
import hashlib
import json
import math
import os
import time

import torch


def _finite_float(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().float().mean().item()
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


class WeightedMetricAccumulator:
    """Sample-weighted aggregation, including an average batch size sanity check."""

    def __init__(self):
        self.totals = {}
        self.weights = {}
        self.samples = 0

    def update(self, metrics, weight=1):
        weight = int(weight)
        if weight <= 0:
            return
        self.samples += weight
        for key, value in metrics.items():
            value = _finite_float(value)
            if value is None:
                continue
            self.totals[key] = self.totals.get(key, 0.0) + value * weight
            self.weights[key] = self.weights.get(key, 0) + weight

    def average(self):
        return {
            key: self.totals[key] / max(1, self.weights[key])
            for key in sorted(self.totals)
        }


class ModelEMA:
    """Evaluation-only exponential moving average with checkpoint support."""

    def __init__(self, model, decay=0.999):
        self.decay = float(decay)
        self.module = copy.deepcopy(model).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)
        self.updates = 0

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        decay = min(self.decay, (1.0 + self.updates) / (10.0 + self.updates))
        source = model.state_dict()
        for key, ema_value in self.module.state_dict().items():
            model_value = source[key].detach()
            if ema_value.is_floating_point():
                ema_value.mul_(decay).add_(model_value, alpha=1.0 - decay)
            else:
                ema_value.copy_(model_value)

    def state_dict(self):
        return {
            'state_dict': self.module.state_dict(),
            'updates': self.updates,
            'decay': self.decay,
        }

    def load_state_dict(self, state):
        if not state:
            return
        model_state = state.get('state_dict', state)
        self.module.load_state_dict(model_state, strict=True)
        self.updates = int(state.get('updates', self.updates)) if isinstance(state, dict) else 0
        if isinstance(state, dict):
            self.decay = float(state.get('decay', self.decay))


class TrainingMonitor:
    """Write one canonical row per epoch and diagnose validation divergence."""

    def __init__(self, output_dir, tensorboard_writer=None, config=None,
                 overfit_patience=3, overfit_relative_delta=0.03,
                 plot_curves=True):
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        self.csv_path = os.path.join(self.output_dir, 'metrics.csv')
        self.jsonl_path = os.path.join(self.output_dir, 'metrics.jsonl')
        self.plot_path = os.path.join(self.output_dir, 'training_curves.png')
        self.writer = tensorboard_writer
        self.overfit_patience = int(overfit_patience)
        self.overfit_relative_delta = float(overfit_relative_delta)
        self.plot_curves = bool(plot_curves)
        self.rows = self._read_rows()
        config_json = json.dumps(config or {}, sort_keys=True, default=str)
        self.config_hash = hashlib.sha256(config_json.encode('utf-8')).hexdigest()[:12]
        self.best_val_rot = float('inf')
        self.best_epoch = -1
        self.bad_epochs = 0
        self.last_status = 'initializing'
        self._rebuild_detector_state()

    def _read_rows(self):
        if not os.path.isfile(self.jsonl_path):
            return []
        rows = {}
        with open(self.jsonl_path, 'r', encoding='utf-8') as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                    rows[int(row['epoch'])] = row
                except (ValueError, KeyError, TypeError):
                    continue
        return [rows[key] for key in sorted(rows)]

    def _rebuild_detector_state(self):
        self.best_val_rot = float('inf')
        self.best_epoch = -1
        self.bad_epochs = 0
        for row in self.rows:
            value = _finite_float(row.get('val/rot_loss'))
            if value is None:
                continue
            if value < self.best_val_rot:
                self.best_val_rot = value
                self.best_epoch = int(row['epoch'])
                self.bad_epochs = 0
            else:
                self.bad_epochs += 1
        if self.rows:
            self.last_status = self.rows[-1].get('status', 'restored')

    @staticmethod
    def _prefix(metrics, prefix):
        return {
            '{}/{}'.format(prefix, key): float(value)
            for key, value in metrics.items()
            if _finite_float(value) is not None
        }

    def _diagnose(self, row):
        val_rot = _finite_float(row.get('val/rot_loss'))
        probe_rot = _finite_float(row.get('train_probe/rot_loss'))
        val_arc = _finite_float(row.get('val/sarr_AR_C'))
        probe_arc = _finite_float(row.get('train_probe/sarr_AR_C'))
        previous_arc = None
        previous_probe_rot = None
        if self.rows:
            previous_arc = _finite_float(self.rows[-1].get('val/sarr_AR_C'))
            previous_probe_rot = _finite_float(
                self.rows[-1].get('train_probe/rot_loss'))

        if val_rot is None:
            return 'train_only'

        # Generalization gaps are recorded for every comparable epoch,
        # including healthy/new-best epochs.  This makes the curve complete.
        gap = None
        if probe_rot is not None:
            gap = val_rot - probe_rot
            row['gap/rot_loss'] = gap
            row['gap/rot_ratio'] = val_rot / max(abs(probe_rot), 1e-8)
        if val_arc is not None and probe_arc is not None:
            row['gap/sarr_AR_C'] = probe_arc - val_arc

        if val_rot < self.best_val_rot:
            self.best_val_rot = val_rot
            self.best_epoch = int(row['epoch'])
            self.bad_epochs = 0
            return 'healthy'

        self.bad_epochs += 1
        relative_degradation = (
            (val_rot - self.best_val_rot) / max(abs(self.best_val_rot), 1e-8))
        if previous_arc is not None and val_arc is not None:
            if relative_degradation >= self.overfit_relative_delta and val_arc > previous_arc:
                return 'loss_metric_divergence'

        if gap is None:
            return 'plateau' if self.bad_epochs < self.overfit_patience else 'overfit_warning'

        prior_gaps = [
            _finite_float(old.get('gap/rot_loss')) for old in self.rows[-self.overfit_patience:]]
        prior_gaps = [value for value in prior_gaps if value is not None]
        gap_expanding = bool(prior_gaps) and gap > prior_gaps[-1] + 1e-8
        probe_still_improving = (
            previous_probe_rot is None
            or probe_rot <= previous_probe_rot * 1.001)
        if (self.bad_epochs >= self.overfit_patience
                and relative_degradation >= self.overfit_relative_delta
                and gap_expanding and probe_still_improving):
            return 'overfit_confirmed'
        if relative_degradation >= self.overfit_relative_delta:
            return 'overfit_warning'
        return 'plateau'

    def record(self, epoch, global_step, train_metrics, val_metrics=None,
               train_probe_metrics=None, bop_metrics=None, lr=None, grad_norm=None,
               samples_per_second=None, elapsed_seconds=None,
               extra_metrics=None):
        row = {
            'epoch': int(epoch),
            'global_step': int(global_step),
            'config_hash': self.config_hash,
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        }
        row.update(self._prefix(train_metrics or {}, 'train'))
        row.update(self._prefix(train_probe_metrics or {}, 'train_probe'))
        row.update(self._prefix(val_metrics or {}, 'val'))
        row.update(self._prefix(bop_metrics or {}, 'bop_test'))
        row.update(self._prefix(extra_metrics or {}, 'system'))
        optional = {
            'optimizer/lr': lr,
            'system/grad_norm': grad_norm,
            'system/samples_per_second': samples_per_second,
            'system/elapsed_seconds': elapsed_seconds,
        }
        for key, value in optional.items():
            value = _finite_float(value)
            if value is not None:
                row[key] = value

        # Replace a resumed epoch rather than creating duplicate history.
        self.rows = [old for old in self.rows if int(old['epoch']) != int(epoch)]
        self.rows.sort(key=lambda old: int(old['epoch']))
        self._rebuild_detector_state()
        row['status'] = self._diagnose(row)
        self.last_status = row['status']
        row['best_epoch'] = self.best_epoch
        row['best_val_rot_loss'] = (
            self.best_val_rot if math.isfinite(self.best_val_rot) else None)
        self.rows.append(row)
        self.rows.sort(key=lambda old: int(old['epoch']))

        self._write_history()
        self._write_tensorboard(row)
        if self.plot_curves:
            self._plot()
        return row

    def _write_history(self):
        json_temp = self.jsonl_path + '.tmp'
        with open(json_temp, 'w', encoding='utf-8') as handle:
            for row in self.rows:
                handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + '\n')
        os.replace(json_temp, self.jsonl_path)

        fieldnames = sorted({key for row in self.rows for key in row})
        csv_temp = self.csv_path + '.tmp'
        with open(csv_temp, 'w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)
        os.replace(csv_temp, self.csv_path)

    def _write_tensorboard(self, row):
        if self.writer in (None, 0):
            return
        epoch = int(row['epoch'])
        for key, value in row.items():
            value = _finite_float(value)
            if value is not None and key not in ('epoch', 'global_step', 'best_epoch'):
                self.writer.add_scalar('epoch/' + key, value, epoch)
        self.writer.flush()

    def _plot(self):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except Exception as error:
            print('Warning: training curves disabled: {}'.format(error))
            self.plot_curves = False
            return
        epochs = [int(row['epoch']) for row in self.rows]
        if not epochs:
            return

        def series(key):
            return [float(row.get(key, float('nan'))) for row in self.rows]

        figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        axis = axes[0, 0]
        axis.plot(epochs, series('train/rot_loss'), 'o-', label='train online')
        axis.plot(epochs, series('train_probe/rot_loss'), 's-', label='train probe')
        axis.plot(epochs, series('val/rot_loss'), '^-', label='validation')
        axis.set(title='SARR cosine loss', xlabel='epoch', ylabel='loss')
        axis.grid(alpha=0.25)
        axis.legend()

        axis = axes[0, 1]
        axis.plot(epochs, series('train/aux_sarr_loss'), 'o-', label='train geometry SARR')
        axis.plot(epochs, series('val/aux_sarr_loss'), '^-', label='val geometry SARR')
        axis.plot(epochs, series('train/stage_sarr_loss'), 'o--', label='train refinement')
        axis.plot(epochs, series('val/stage_sarr_loss'), '^--', label='val refinement')
        axis.plot(epochs, series('val/translation_loss'), 's-', label='val translation')
        axis.plot(epochs, series('val/score_loss'), 'd-', label='val confidence')
        axis.set(title='Pose auxiliary objectives', xlabel='epoch', ylabel='loss')
        axis.grid(alpha=0.25)
        axis.legend()

        axis = axes[1, 0]
        axis.plot(epochs, series('train_probe/sarr_AR_C'), 's-', label='probe SARR AR_C (%)')
        axis.plot(epochs, series('val/sarr_AR_C'), 'o-', label='val SARR AR_C (%)')
        axis.plot(epochs, series('bop_test/sarr_AR_C'), 'd--', label='BOP-test SARR AR_C (%)')
        axis.plot(epochs, series('bop_test/add_auc'), '^--', label='BOP-test ADD AUC (%)')
        axis.plot(epochs, series('bop_test/refined_sarr_AR_C'),
                  'd-', label='BOP refined SARR AR_C (%)')
        axis.plot(epochs, series('bop_test/refined_add_auc'),
                  '^-', label='BOP refined ADD AUC (%)')
        route_iou = [100.0 * value for value in series('bop_test/routing_iou_metric')]
        axis.plot(epochs, route_iou, ':', color='0.45', label='routing IoU diagnostic (%)')
        axis.set(title='Pose accuracy (segmentation is diagnostic only)',
                 xlabel='epoch', ylabel='percent')
        axis.grid(alpha=0.25)
        axis.legend()

        axis = axes[1, 1]
        axis.plot(epochs, series('gap/rot_loss'), 'o-', label='rotation gap')
        axis.set(title='Generalization and optimization', xlabel='epoch', ylabel='val - probe')
        axis.grid(alpha=0.25)
        learning_rate_axis = axis.twinx()
        learning_rate_axis.plot(
            epochs, series('optimizer/lr'), color='tab:orange', linestyle='--', label='learning rate')
        learning_rate_axis.set_ylabel('learning rate')
        positive_lrs = [value for value in series('optimizer/lr') if value > 0]
        if positive_lrs:
            learning_rate_axis.set_yscale('log')
        handles, labels = axis.get_legend_handles_labels()
        handles2, labels2 = learning_rate_axis.get_legend_handles_labels()
        axis.legend(handles + handles2, labels + labels2)

        temp_path = self.plot_path + '.tmp.png'
        figure.savefig(temp_path, dpi=150)
        plt.close(figure)
        os.replace(temp_path, self.plot_path)

    def should_early_stop(self, patience):
        patience = int(patience)
        return (
            patience > 0
            and self.bad_epochs >= patience
            and self.last_status in ('overfit_confirmed', 'plateau'))

    def state_dict(self):
        return {
            'best_val_rot': self.best_val_rot,
            'best_epoch': self.best_epoch,
            'bad_epochs': self.bad_epochs,
            'last_status': self.last_status,
            'config_hash': self.config_hash,
        }

    def load_state_dict(self, state):
        if not state:
            return
        self.best_val_rot = float(state.get('best_val_rot', self.best_val_rot))
        self.best_epoch = int(state.get('best_epoch', self.best_epoch))
        self.bad_epochs = int(state.get('bad_epochs', self.bad_epochs))
        self.last_status = state.get('last_status', self.last_status)
