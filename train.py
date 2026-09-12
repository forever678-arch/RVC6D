#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import argparse
import os
import importlib
# Single GPU only (GPU 0); multi-GPU paths are kept commented out
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # single-GPU setup
# Distributed backend settings are not needed for single-GPU training
# os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"

import warnings
warnings.filterwarnings("ignore")
import random
import time
from copy import copy
import gc
import shutil, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
# Multiprocessing launch is not needed for single-GPU training
# import torch.multiprocessing as mp
import torch.utils.data
# Distributed data loading is not needed for single-GPU training
# import torch.utils.data.distributed
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None
from torch.utils.data import ConcatDataset, Subset, WeightedRandomSampler

# Dataset and model imports
# from datasets.tless.tless_datasets import PoseDataset as pose_dataset
# from datasets.tless.tless_dataset_bbox import PoseDataset as pose_dataset
from datasets.tless.dataset_all import PoseDataset as pose_dataset
# from datasets.tless.dataset_bbox import PoseDataset as pose_dataset
# from models import ES6D as pose_net
# model module is selected dynamically by --model
from lib.tless_pose_evaluator import TLessPoseEvaluator
from lib.utils import close_logger, setup_logger
# from lib.utils import post_processing_ycb_quaternion as post_processing
from lib.utils import post_processing_ycb_quaternion_wi_vote as post_processing
from lib.utils import post_processing_sarr_wi_vote
from lib.sarr import canonical_rotation_error_degrees, sarr_to_matrix
from lib.bbox_pipeline import bbox_depth_valid_mask
from lib.bbox_diagnostics import routing_mask_diagnostics
from lib.pose_refinement import decode_rvc6d
from lib.training_monitor import ModelEMA, TrainingMonitor, WeightedMetricAccumulator
from lib.data_loading import (
    build_loader_kwargs, process_tree_pss_mb, process_tree_rss_mb,
    shutdown_data_loader)
from lib.optim import (
    apply_warmup_cosine, build_rvc6d_adamw,
    load_optimizer_state_name_matched, optimizer_group_summary)
from models.rvc6d import RVC6D

st_time = time.time()
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TLESS_ROOT = os.environ.get(
    'TLESS_ROOT', os.path.join(PROJECT_ROOT, 'data', 'tless'))
RVC6D_MODEL = 'RVC6D'

def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in ('yes', 'true', 't', '1', 'y'):
        return True
    if value in ('no', 'false', 'f', '0', 'n'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')

def load_pose_module(model_name):
    if model_name != RVC6D_MODEL:
        raise ValueError('This clean project contains only RVC6D')
    module = importlib.import_module('models.rvc6d')
    return module, module.RVC6D


class _FallbackSummaryWriter:
    """No-op writer for environments without the optional TensorBoard package."""

    def __init__(self, *args, **kwargs):
        self.log_dir = args[0] if args else kwargs.get('log_dir', '')

    def add_scalar(self, *args, **kwargs):
        return None

    def add_text(self, *args, **kwargs):
        return None

    def close(self):
        return None


if SummaryWriter is None:
    SummaryWriter = _FallbackSummaryWriter


RVC6D_MODELS = (RVC6D_MODEL,)
BMC_MODELS = RVC6D_MODELS
CAD_CONDITIONED_MODELS = RVC6D_MODELS
MODERN_POSE_MODELS = RVC6D_MODELS
MODEL_PARAMETER_BUDGETS = {RVC6D_MODEL: RVC6D.parameter_budget}


def is_sarr_model(model_name):
    return model_name in MODERN_POSE_MODELS


def is_bbox_only_model(model_name):
    return False


def uses_cad_condition(model_name):
    return model_name in CAD_CONDITIONED_MODELS


def is_rvc6d_model(model_name):
    """Return whether the model uses the RVC6D heads, loss, and decoder."""
    return model_name in MODERN_POSE_MODELS


def is_pro_model(model_name):
    return False


def pro_dataset_kwargs(opt):
    if not (is_pro_model(opt.model) or opt.model in RVC6D_MODELS):
        return {}
    return {
        'depth_augmentation': opt.depth_augmentation,
        'depth_dropout': opt.depth_dropout,
        'depth_noise_std': opt.depth_noise_std,
        'depth_hole_probability': opt.depth_hole_probability,
        'bbox_jitter': (
            None if opt.bbox_jitter < 0.0 else opt.bbox_jitter),
    }


def has_model_scale(model_name):
    return False


def effective_score_weight(opt):
    return opt.sarr_score_weight


def bmc_model_kwargs(opt):
    if opt.model not in BMC_MODELS:
        return {}
    return {
        'bmc_projection_dim': opt.bmc_projection_dim,
        'bmc_temperature': opt.bmc_temperature,
        'bmc_shared_base_weight': opt.bmc_shared_base_weight,
        'bmc_specific_base_weight': opt.bmc_specific_base_weight,
        'bmc_shared_target': opt.bmc_shared_target,
        'bmc_specific_target': opt.bmc_specific_target,
        'bmc_ema_decay': opt.bmc_ema_decay,
        'bmc_warmup_steps': opt.bmc_warmup_steps,
        'bmc_control_interval': opt.bmc_control_interval,
        'bmc_deadband': opt.bmc_deadband,
        'bmc_min_multiplier': opt.bmc_min_multiplier,
        'bmc_max_multiplier': opt.bmc_max_multiplier,
        'bmc_min_samples': opt.bmc_min_samples,
        'bmc_min_support_ratio': opt.bmc_min_support_ratio,
        'bmc_specific_variance_floor': opt.bmc_specific_variance_floor,
        'bmc_specific_variance_weight': opt.bmc_specific_variance_weight,
        'bmc_pose_metric_target': opt.bmc_pose_metric_target,
        'bmc_pose_ema_decay': opt.bmc_pose_ema_decay,
    }

def resolve_existing_path(path):
    if not path or os.path.isabs(path):
        return path
    candidates = [
        os.path.abspath(path),
        os.path.join('/root', path),
        os.path.join(PROJECT_ROOT, path),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return path


# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default=RVC6D_MODEL,
                    choices=[RVC6D_MODEL], help='pose model')
parser.add_argument('--rot_loss_weight', type=float, default=0.1, help='symmetry-aware rotation regularization weight')
parser.add_argument('--sarr_loss_weight', type=float, default=1.0, help='SARR cosine loss weight')
parser.add_argument('--sarr_translation_weight', type=float, default=1.0, help='auxiliary translation loss weight; SARR rotation remains primary')
parser.add_argument(
    '--primary_pose_loss', type=str, default='sarr_translation',
    choices=[
        'sarr_translation', 'adds', 'gadd', 'sym_pm_translation'],
    help=(
        'primary pose objective: SARR cosine plus translation Smooth-L1, '
        'standard ADD(-S), ES6D grouped-primitives GADD, or '
        'GDR-Net-style symmetry-aware PM(R) plus translation Smooth-L1'))
parser.add_argument(
    '--pose_distance_weight', type=float, default=5.0,
    help='weight applied to ADD(-S) or GADD when it replaces the two SARR primary terms')
parser.add_argument(
    '--pose_loss_points', type=int, default=128,
    help='deterministic CAD-point sample count used by the training-time ADD(-S) objective')
parser.add_argument(
    '--pose_loss_votes', type=int, default=16,
    help='maximum deterministic valid translation votes per instance for ADD(-S)/GADD')
parser.add_argument('--sarr_score_weight', type=float, default=0.001, help='dense vote confidence loss weight')
parser.add_argument('--center_weight', type=float, default=0.50,
                    help='object-level center supervision weight')
parser.add_argument('--hypothesis_weight', type=float, default=0.25,
                    help='multi-hypothesis rotation weight')
parser.add_argument('--hypothesis_score_weight', type=float, default=0.05,
                    help='rotation-hypothesis ranking weight')
parser.add_argument('--hypothesis_diversity_weight', type=float, default=0.05,
                    help='anti-collapse margin loss for distinct rotation hypotheses')
parser.add_argument('--hypothesis_balance_weight', type=float, default=0.005,
                    help='batch-level specialization/load-balance weight for rotation hypotheses')
parser.add_argument('--hypothesis_diversity_margin', type=float, default=0.20,
                    help='minimum normalized SARR distance between rotation hypotheses')
parser.add_argument('--hypothesis_temperature', type=float, default=0.07,
                    help='soft assignment temperature used by hypothesis ranking/specialization')
parser.add_argument('--pose_score_weight', type=float, default=0.05,
                    help='object-level pose-quality weight')
parser.add_argument('--correspondence_weight', type=float, default=0.12,
                    help='visible CAD-correspondence classification weight')
parser.add_argument('--surface_inlier_weight', type=float, default=0.04,
                    help='visible-surface inlier weight')
parser.add_argument('--refinement_rotation_weight', type=float, default=0.20,
                    help='residual-stage rotation supervision')
parser.add_argument('--refinement_center_weight', type=float, default=0.10,
                    help='residual-stage center supervision')
# The local_rank argument was removed; single-GPU training does not need it
# parser.add_argument('--local_rank', type=int, default=0, help='gpu number')

parser.add_argument('--experiment', type=str, default="train", help='brief description about experiment setting: train, test')
parser.add_argument('--loss_type', type=str, default="SARR", help='output namespace for the SARR training and evaluation pipeline')
parser.add_argument('--dataset', type=str, default='tless', help='tless, lmo')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--workers', type=int, default=40, help='number of data loading workers')
parser.add_argument('--persistent_workers', type=str2bool, default=False,
                    help='keep loader workers between epochs; false prevents per-worker dataset RSS growth')
parser.add_argument('--prefetch_factor', type=int, default=1,
                    help='batches prefetched by each worker')
parser.add_argument('--pin_memory', type=str2bool, default=False,
                    help='use page-locked loader buffers; disabled because transfers are synchronous')
parser.add_argument('--memory_sample_interval', type=int, default=500,
                    help='training batches between process-tree RSS samples')
parser.add_argument('--learning_rate', type=float, default=3e-4,
                    help='uniform RVC6D scratch learning rate')
parser.add_argument('--min_lr_ratio', type=float, default=0.01,
                    help='cosine final/base learning-rate ratio')
parser.add_argument('--warmup_start_ratio', type=float, default=0.10,
                    help='warmup starting/base learning-rate ratio')
parser.add_argument('--bmc_projection_dim', type=int, default=64,
                    help='BMC shared projection dimension')
parser.add_argument('--bmc_temperature', type=float, default=0.10,
                    help='BMC symmetric InfoNCE temperature')
parser.add_argument('--bmc_shared_base_weight', type=float, default=0.005,
                    help='base weight for the BMC shared constraint')
parser.add_argument('--bmc_specific_base_weight', type=float, default=0.020,
                    help='base weight for the BMC specific constraint')
parser.add_argument('--bmc_shared_target', type=float, default=0.35,
                    help='target null-corrected CKA of the shared halves')
parser.add_argument('--bmc_specific_target', type=float, default=0.08,
                    help='target upper null-corrected CKA of specific halves')
parser.add_argument('--bmc_ema_decay', type=float, default=0.99,
                    help='EMA decay for detached BMC CKA statistics')
parser.add_argument('--bmc_warmup_steps', type=int, default=500,
                    help='steps before activating adaptive BMC losses')
parser.add_argument('--bmc_control_interval', type=int, default=100,
                    help='steps between BMC multiplier updates')
parser.add_argument('--bmc_deadband', type=float, default=0.05,
                    help='CKA controller deadband')
parser.add_argument('--bmc_min_multiplier', type=float, default=0.25,
                    help='minimum adaptive BMC loss multiplier')
parser.add_argument('--bmc_max_multiplier', type=float, default=2.0,
                    help='maximum adaptive BMC loss multiplier')
parser.add_argument('--bmc_min_samples', type=int, default=32,
                    help='minimum valid samples required for a BMC update')
parser.add_argument('--bmc_min_support_ratio', type=float, default=0.05,
                    help='minimum weighted feature-grid support per sample')
parser.add_argument('--bmc_specific_variance_floor', type=float, default=0.10,
                    help='anti-collapse standard-deviation floor')
parser.add_argument('--bmc_specific_variance_weight', type=float, default=1.0,
                    help='anti-collapse term inside the specific objective')
parser.add_argument('--bmc_pose_metric_target', type=float, default=0.10,
                    help='rotation-loss target gating extra shared alignment')
parser.add_argument('--bmc_pose_ema_decay', type=float, default=0.99,
                    help='EMA decay of the BMC fused-pose performance gate')
parser.add_argument('--warnup_iters', type=int, default=1000, help='linear warmup iterations')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='optimizer weight decay')
parser.add_argument('--grad_clip', type=float, default=5.0, help='maximum gradient norm; <=0 disables clipping')
parser.add_argument('--noise_trans', type=float, default=0.005, help='range of the random noise of translation added to the training data')
parser.add_argument('--augmentation', type=str2bool, default=True, help='train tless with color and crop augmentation')
parser.add_argument('--depth_augmentation', type=str2bool, default=False,
                    help='apply bbox-only sensor noise/dropout to training depth')
parser.add_argument('--depth_dropout', type=float, default=0.0,
                    help='independent valid-depth dropout probability')
parser.add_argument('--depth_noise_std', type=float, default=0.0,
                    help='training depth Gaussian noise scale in metres')
parser.add_argument('--depth_hole_probability', type=float, default=0.0,
                    help='probability of one or two rectangular missing-depth regions')
parser.add_argument('--bbox_jitter', type=float, default=-1.0,
                    help='training bbox jitter ratio; negative keeps dataset default')


parser.add_argument('--train_split', type=str, default='train_pbr',
                    choices=['auto', 'train_primesense', 'train_pbr'],
                    help='prefer the real train_primesense split used by SARR when available')
parser.add_argument('--train_splits', type=str, nargs='+', default=['train_pbr', 'train_primesense'],
                    choices=['train_primesense', 'train_pbr', 'train'],
                    help='training splits to combine; overrides --train_split')
parser.add_argument('--real_sampling_ratio', type=float, default=0.05,
                    help='real-sample fraction in each mixed epoch (default: 5%% real / 95%% PBR)')
parser.add_argument('--class_balance_power', type=float, default=0.0,
                    help='per-split inverse class-frequency exponent; 0 keeps natural sampling')
parser.add_argument('--epoch_samples', type=int, default=0,
                    help='samples drawn per epoch after balancing; 0 keeps each PBR sample once on average')
parser.add_argument('--point_filter', type=str, default='mad_soft',
                    choices=['valid', 'diameter_hard', 'mad_soft'],
                    help='bbox point filtering: validity only, diameter hard gate, or MAD soft reliability')


parser.add_argument('--nepoch', type=int, default=40, help='maximum training epochs')
parser.add_argument('--save_every', type=int, default=1, help='save checkpoint every N epochs')
parser.add_argument('--test_every', type=int, default=1, help='run validation every N epochs')
parser.add_argument('--selection_metric', type=str, default='rot_loss',
                    choices=['rot_loss', 'add_auc'],
                    help='test-set diagnostic used to save checkpoint_best')
parser.add_argument('--resume', type=str, default='', help='checkpoint path; empty starts a clean run')
parser.add_argument('--outf_override', type=str, default='',
                    help='override checkpoint output folder (e.g. continue a pre-rename experiment in place)')
parser.add_argument('--log_dir_override', type=str, default='',
                    help='override tensorboard/log folder (e.g. continue a pre-rename experiment in place)')
parser.add_argument('--warm_start', type=str, default='',
                    help='model/EMA weights only; resets epoch, optimizer, EMA and monitor state')
parser.add_argument('--allow_resume_config_mismatch', type=str2bool, default=False,
                    help='explicitly allow loss/model settings to differ from a resumed checkpoint')
parser.add_argument('--start_epoch', type=int, default=0, help='which epoch to start')



parser.add_argument('--seed', type=int, default=2026, help='reproducible random seed')
parser.add_argument('--use_ema', type=str2bool, default=True,
                    help='validate and save an exponential moving average model')
parser.add_argument('--ema_decay', type=float, default=0.999,
                    help='EMA decay after the startup correction')
parser.add_argument('--train_probe_size', type=int, default=900,
                    help='fixed no-augmentation train subset used for a valid generalization gap')
parser.add_argument('--val_size', type=int, default=0,
                    help='deprecated and ignored: all training samples are used')
parser.add_argument('--bop_refine', type=str2bool, default=True,
                    help='also run visible-depth iterative refinement at BOP diagnostics')
parser.add_argument('--bop_refine_iterations', type=int, default=4,
                    help='visible-depth refinement iterations per BOP diagnostic')
parser.add_argument('--bop_refine_points', type=int, default=384,
                    help='maximum observed depth points used by BOP refinement')
parser.add_argument('--bop_refine_candidates', type=int, default=2,
                    help='number of geometrically screened pose modes refined per proposal')
parser.add_argument('--bop_refine_distance', type=float, default=0.20,
                    help='initial refinement correspondence threshold in object radii')
parser.add_argument('--bop_refine_final_distance', type=float, default=0.08,
                    help='final refinement correspondence threshold in object radii')
parser.add_argument('--bop_refine_occlusion_margin', type=float, default=0.06,
                    help='depth margin in object radii for external-occlusion rejection')
parser.add_argument('--overfit_patience', type=int, default=3,
                    help='consecutive validation degradations before overfit confirmation')
parser.add_argument('--early_stop_patience', type=int, default=5,
                    help='stop after this many non-improving validations; 0 only warns')
parser.add_argument('--plot_curves', type=str2bool, default=True,
                    help='write metrics.csv/jsonl and training_curves.png every epoch')
parser.add_argument('--max_train_batches', type=int, default=0,
                    help='debug only: stop each train epoch after N batches; 0 uses all')
parser.add_argument('--max_eval_batches', type=int, default=0,
                    help='debug only: stop probe/validation after N batches; 0 uses all')

opt = parser.parse_args()
if opt.model in RVC6D_MODELS and not is_pro_model(opt.model):
    # RVC6D uses the compact coarse decoder during routine evaluation.
    opt.bop_refine = False
if opt.train_splits is None:
    opt.train_splits = [opt.train_split]
if not 0.0 <= opt.real_sampling_ratio <= 1.0:
    parser.error('--real_sampling_ratio must be in [0, 1]')
if opt.epoch_samples < 0:
    parser.error('--epoch_samples must be non-negative')
if opt.workers < 0:
    parser.error('--workers must be non-negative')
if opt.prefetch_factor < 1:
    parser.error('--prefetch_factor must be positive')
if opt.memory_sample_interval < 1:
    parser.error('--memory_sample_interval must be positive')
if opt.learning_rate <= 0.0:
    parser.error('--learning_rate must be positive')
if not 0.0 < opt.min_lr_ratio <= 1.0:
    parser.error('--min_lr_ratio must be in (0, 1]')
if not 0.0 < opt.warmup_start_ratio <= 1.0:
    parser.error('--warmup_start_ratio must be in (0, 1]')
if opt.bmc_projection_dim < 2:
    parser.error('--bmc_projection_dim must be at least 2')
if opt.bmc_temperature <= 0.0:
    parser.error('--bmc_temperature must be positive')
if opt.bmc_shared_base_weight < 0.0 or opt.bmc_specific_base_weight < 0.0:
    parser.error('BMC base weights must be non-negative')
if not 0.0 <= opt.bmc_shared_target <= 1.0:
    parser.error('--bmc_shared_target must be in [0, 1]')
if not 0.0 <= opt.bmc_specific_target <= 1.0:
    parser.error('--bmc_specific_target must be in [0, 1]')
if not 0.0 <= opt.bmc_ema_decay < 1.0:
    parser.error('--bmc_ema_decay must be in [0, 1)')
if opt.bmc_warmup_steps < 0 or opt.bmc_control_interval < 1:
    parser.error('invalid BMC controller schedule')
if not 0.0 <= opt.bmc_deadband < 1.0:
    parser.error('--bmc_deadband must be in [0, 1)')
if not 0.0 < opt.bmc_min_multiplier <= opt.bmc_max_multiplier:
    parser.error('invalid BMC multiplier limits')
if opt.bmc_min_samples < 2:
    parser.error('--bmc_min_samples must be at least 2')
if not 0.0 <= opt.bmc_min_support_ratio <= 1.0:
    parser.error('--bmc_min_support_ratio must be in [0, 1]')
if opt.bmc_specific_variance_floor <= 0.0:
    parser.error('--bmc_specific_variance_floor must be positive')
if opt.bmc_specific_variance_weight < 0.0:
    parser.error('--bmc_specific_variance_weight must be non-negative')
if opt.bmc_pose_metric_target <= 0.0:
    parser.error('--bmc_pose_metric_target must be positive')
if not 0.0 <= opt.bmc_pose_ema_decay < 1.0:
    parser.error('--bmc_pose_ema_decay must be in [0, 1)')
if any(weight < 0.0 for weight in (
        opt.correspondence_weight, opt.surface_inlier_weight,
        opt.refinement_rotation_weight, opt.refinement_center_weight)):
    parser.error('all correspondence loss weights must be non-negative')
if not 0.0 <= opt.depth_dropout < 1.0:
    parser.error('--depth_dropout must be in [0, 1)')
if opt.depth_noise_std < 0.0:
    parser.error('--depth_noise_std must be non-negative')
if not 0.0 <= opt.depth_hole_probability <= 1.0:
    parser.error('--depth_hole_probability must be in [0, 1]')
if opt.bbox_jitter >= 0.5:
    parser.error('--bbox_jitter must be below 0.5')
if opt.resume and opt.warm_start:
    parser.error('--resume and --warm_start are mutually exclusive')
if not 0.0 <= opt.class_balance_power <= 1.0:
    parser.error('--class_balance_power must be in [0, 1]')
opt.effective_sarr_score_weight = effective_score_weight(opt)

def main():
    # Presets
    global opt

    # This server's current CUDA/cuDNN combination can fail during cuDNN
    # initialization.  Keep the known-good fallback by default while allowing
    # cuDNN to be opted in explicitly on compatible machines.
    torch.backends.cudnn.enabled = os.environ.get('RVC6D_ENABLE_CUDNN', '0') == '1'
    opt.manualSeed = opt.seed
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(opt.manualSeed)

    # Distributed environment variables are not needed for single-GPU training
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '1235'

    # Single-GPU setup
    opt.gpu_number = 1  # fixed to one GPU instead of torch.cuda.device_count()

    # Dataset-specific configuration
    if opt.dataset == 'tless':
        opt.num_objects = 30
        opt.num_points = 1024
        opt.dataset_root = DEFAULT_TLESS_ROOT

        opt.outf = os.path.join(
            PROJECT_ROOT, 'experiments', 'tless', opt.loss_type,
            opt.experiment, 'model_RVC6D')
        opt.log_dir = os.path.join(
            PROJECT_ROOT, 'experiments', 'tless', opt.loss_type,
            opt.experiment, 'log_RVC6D')

    elif opt.dataset == 'lmo':
        # LM-O: 8 objects, contiguous class slots 1..8.  The SARR kappa
        # table must be switched before the model or loss encodes or
        # decodes any rotation.
        opt.num_objects = 8
        opt.num_points = 1024
        opt.dataset_root = os.environ.get(
            'LMO_ROOT', os.path.join(PROJECT_ROOT, 'data', 'lmo'))
        from lib import sarr as _sarr_module
        _sarr_module.set_sarr_kappa_table(_sarr_module.LMO_SARR_KAPPA)

        opt.outf = os.path.join(
            PROJECT_ROOT, 'experiments', 'lmo', opt.loss_type,
            opt.experiment, 'model_RVC6D')
        opt.log_dir = os.path.join(
            PROJECT_ROOT, 'experiments', 'lmo', opt.loss_type,
            opt.experiment, 'log_RVC6D')

    if not os.path.isdir(opt.outf):
        os.makedirs(opt.outf)
    if not os.path.isdir(opt.log_dir):
        os.makedirs(opt.log_dir)

    # Optional overrides to continue a previous experiment in place, so
    # checkpoints and tensorboard curves land in the original directories
    if opt.outf_override:
        opt.outf = opt.outf_override
        if not os.path.isdir(opt.outf):
            os.makedirs(opt.outf)
    if opt.log_dir_override:
        opt.log_dir = opt.log_dir_override
        if not os.path.isdir(opt.log_dir):
            os.makedirs(opt.log_dir)

    # Run the single-GPU worker directly (no multiprocessing spawn)
    # mp.spawn(per_processor, nprocs=opt.gpu_number, args=(opt,))
    per_processor(0, opt)  # single-GPU worker on GPU 0

def predict(data, estimator, lossor, opt, mode='train', refine=False):
    """Forward pass and loss computation for one batch."""
    # Move the batch to the GPU
    cls_ids = data['class_id'].to(opt.gpu)
    rgb = data['rgb'].to(opt.gpu)
    xyz = data[
        'xyz_all' if uses_cad_condition(opt.model) else 'xyz'].to(opt.gpu)
    raw_depth = data['depth_map'].to(opt.gpu)
    point_weight = data.get('point_weight')
    point_weight = (
        point_weight.to(opt.gpu) if point_weight is not None
        else (raw_depth > 0.0).to(raw_depth.dtype))
    if uses_cad_condition(opt.model):
        depth = data['depth_valid_mask'].to(opt.gpu)
    else:
        depth = point_weight if is_bbox_only_model(opt.model) else raw_depth
    if is_bbox_only_model(opt.model) and 'bbox_mask' not in data:
        raise KeyError('bbox-only models require data[\'bbox_mask\']; segmentation fallback is forbidden')
    bbox_mask = data['bbox_mask'].to(opt.gpu) if 'bbox_mask' in data else data['mask'].to(opt.gpu)
    pose_mask = data['pose_mask'].to(opt.gpu) if 'pose_mask' in data else bbox_mask
    loss_mask = pose_mask
    gt_r = data['target_r'].to(opt.gpu)
    gt_t = data['target_t'].to(opt.gpu)

    model_xyz_tensor = data['model_xyz'].to(opt.gpu)
    model_xyz = model_xyz_tensor.cpu().numpy()

    pose_forward_seconds = 0.0
    if mode == 'test':
        if rgb.device.type == 'cuda':
            torch.cuda.synchronize(rgb.device)
        pose_forward_start = time.perf_counter()

    # Forward pass
    if uses_cad_condition(opt.model):
        preds = estimator(
            rgb, xyz, depth, cls_ids, model_xyz=model_xyz_tensor,
            point_weight=point_weight)
    else:
        preds = estimator(rgb, xyz, depth, cls_ids)
    if mode == 'test':
        if rgb.device.type == 'cuda':
            torch.cuda.synchronize(rgb.device)
        pose_forward_seconds = time.perf_counter() - pose_forward_start

    # Loss computation
    if is_pro_model(opt.model):
        loss, loss_dict = lossor(
            preds, loss_mask, gt_r, gt_t, cls_ids,
            model_xyz=model_xyz_tensor,
            crop_intrinsics=(
                data['crop_intrinsics'].to(opt.gpu)
                if 'crop_intrinsics' in data else None),
            camera_origin=(
                data['camera_origin'].to(opt.gpu)
                if 'camera_origin' in data else None))
    elif is_sarr_model(opt.model):
        loss, loss_dict = lossor(
            preds, loss_mask, gt_r, gt_t, cls_ids, model_xyz_tensor)
    else:
        loss, loss_dict = lossor(
            preds, loss_mask, gt_r, gt_t, cls_ids, model_xyz)
    if (mode == 'train'
            and hasattr(estimator, 'observe_bmc_pose_metric')):
        pose_metric = loss_dict.get(
            'rot_loss', loss_dict.get('SARR_loss'))
        if pose_metric is not None:
            estimator.observe_bmc_pose_metric(pose_metric)

    if 'global_sarr' in preds:
        loss_dict['sarr_norm'] = float(
            preds['global_sarr'].detach().norm(dim=1).mean().item())
    if 'anchor_entropy' in preds:
        loss_dict['anchor_entropy'] = float(
            preds['anchor_entropy'].detach().mean().item())
    if 'hypothesis_gate' in preds:
        loss_dict['hypothesis_gate'] = float(
            preds['hypothesis_gate'].detach().mean().item())
    if 'low_reliability' in preds:
        loss_dict['low_geometry_reliability'] = float(
            preds['low_reliability'].detach().mean().item())
        loss_dict['high_geometry_reliability'] = float(
            preds['high_reliability'].detach().mean().item())
    if is_bbox_only_model(opt.model):
        with torch.no_grad():
            loss_dict['point_support_ratio'] = float(
                (point_weight > 0.0).float().mean().item())
            loss_dict['point_weight_mean'] = float(
                point_weight.float().mean().item())
    if ('routing_probability' in preds
            and 'metric_mask_visib' in data):
        # Segmentation is diagnostic-only: this entire block is detached and
        # runs after loss construction, so it cannot influence gradients.
        with torch.no_grad():
            metric_mask_visib = data['metric_mask_visib'].to(opt.gpu)
            loss_dict.update(routing_mask_diagnostics(
                preds['routing_probability'], metric_mask_visib))

    if mode == 'train':
        return loss, loss_dict

    if mode == 'probe':
        if 'global_sarr' in preds:
            with torch.no_grad():
                predicted_rotation = sarr_to_matrix(
                    preds['global_sarr'], cls_ids.view(-1))
                errors = canonical_rotation_error_degrees(
                    predicted_rotation, gt_r.view(-1, 3, 3), cls_ids.view(-1))
                loss_dict['rot_deg_mean'] = float(errors.mean().item())
                thresholds = errors.new_tensor((2.0, 5.0, 10.0, 15.0, 25.0, 40.0))
                loss_dict['sarr_AR_C'] = float(
                    100.0 * (errors[:, None] < thresholds[None, :]).float().mean().item())
        return loss, loss_dict

    if mode == 'test':
        # Test-mode post-processing
        mean_xyz = data['mean_xyz'].cpu().numpy()

        preds['xyz'] = xyz
        preds['bbox_mask'] = bbox_mask
        preds['pose_mask'] = loss_mask

        if rgb.device.type == 'cuda':
            torch.cuda.synchronize(rgb.device)
        pose_decode_start = time.perf_counter()
        refinement_info = None
        pose_quality = None
        if refine:
            if not uses_cad_condition(opt.model):
                raise ValueError(
                    'iterative BOP refinement requires CAD-conditioned RVC6D')
            res_T, pose_quality, refinement_info = decode_rvc6d(
                preds, model_xyz=model_xyz_tensor, observed_xyz=xyz,
                valid_mask=depth,
                point_weight=point_weight,
                crop_intrinsics=(
                    data['crop_intrinsics'].to(opt.gpu)
                    if 'crop_intrinsics' in data else None),
                camera_origin=(
                    data['camera_origin'].to(opt.gpu)
                    if 'camera_origin' in data else None),
                continuous_symmetry_axis=(
                    data['continuous_symmetry_axis'].to(opt.gpu)
                    if 'continuous_symmetry_axis' in data else None),
                refine=True,
                refinement_kwargs={
                    'iterations': opt.bop_refine_iterations,
                    'max_observed_points': opt.bop_refine_points,
                    'candidate_topk': opt.bop_refine_candidates,
                    'distance_threshold': opt.bop_refine_distance,
                    'final_distance_threshold':
                        opt.bop_refine_final_distance,
                    'occlusion_margin': opt.bop_refine_occlusion_margin,
                })
            loss_dict['refinement_applied_fraction'] = float(
                refinement_info['applied'].float().mean().item())
            loss_dict['refinement_fitness'] = float(
                refinement_info['fitness'].mean().item())
        elif is_pro_model(opt.model):
            # Pro candidate selection uses one cheap vectorized visible-depth
            # score in this same pass; no second validation inference is run.
            res_T, pose_quality, refinement_info = decode_rvc6d(
                preds, model_xyz=model_xyz_tensor, observed_xyz=xyz,
                valid_mask=depth, point_weight=point_weight,
                crop_intrinsics=(
                    data['crop_intrinsics'].to(opt.gpu)
                    if 'crop_intrinsics' in data else None),
                camera_origin=(
                    data['camera_origin'].to(opt.gpu)
                    if 'camera_origin' in data else None),
                continuous_symmetry_axis=(
                    data['continuous_symmetry_axis'].to(opt.gpu)
                    if 'continuous_symmetry_axis' in data else None),
                refine=False)
            loss_dict['pro_geometry_fitness'] = float(
                refinement_info['fitness'].mean().item())
            loss_dict['pro_noncoarse_selection'] = float(
                (refinement_info['selected_candidate'] > 0).float(
                ).mean().item())
        elif uses_cad_condition(opt.model):
            # This preserves the existing coarse pose decoder while exposing
            # the proposal-level network quality used as the BOP score.
            res_T, pose_quality, _ = decode_rvc6d(
                preds, refine=False)
        elif is_sarr_model(opt.model):
            res_T = post_processing_sarr_wi_vote(preds, opt.sym_list)
        else:
            res_T = post_processing(preds, opt.sym_list)
        if pose_quality is None:
            dense_quality = preds['pred_s'].detach().flatten(1)
            pose_quality = torch.stack([
                torch.topk(
                    values, min(32, values.numel()), largest=True
                ).values.mean()
                for values in dense_quality
            ]).clamp(1e-6, 1.0)
        if rgb.device.type == 'cuda':
            torch.cuda.synchronize(rgb.device)
        pose_elapsed_seconds = (
            pose_forward_seconds + time.perf_counter() - pose_decode_start)
        bs, _, _ = res_T.size()
        rotation_errors = canonical_rotation_error_degrees(
            res_T[:, :, :3], gt_r.view(bs, 3, 3), cls_ids.view(bs))

        res_T = res_T.cpu().numpy()
        tar_T = torch.cat([gt_r, gt_t.unsqueeze(dim=2)], dim=2)
        tar_T = tar_T.cpu().numpy()

        gt_cls = data['class_id'].cpu().numpy().astype(np.int32)
        instance_id = data['instance_id'].cpu().numpy().astype(np.int32)

        rt_list = []
        gt_rt_list = []
        gt_cls_list = []
        model_list = []
        instance_eval_rt_list = []
        instance_id_list = []
        pose_score_list = []

        pred = res_T.copy()
        pose_quality = pose_quality.detach().cpu().numpy()

        for i in range(bs):
            scale = opt.obj_radius[int(gt_cls[i][0])]
            # instance_mean_xyz = mean_xyz[i][0, 0, :]
            instance_mean_xyz = mean_xyz[i]

            pred[i, :, 3] *= scale
            pred[i, :, 3] += instance_mean_xyz

            res_T[i, :, 3] *= scale
            tar_T[i, :, 3] *= scale
            model_xyz[i] *= scale



            instance_id_list.append([instance_id[i]])
            rt_list.append(res_T[i])
            gt_rt_list.append(tar_T[i])
            gt_cls_list.append(gt_cls[i] + 1)
            model_list.append(model_xyz[i])
            instance_eval_rt_list.append(pred[i])
            pose_score_list.append(float(pose_quality[i]))

        return (loss, loss_dict, rt_list, gt_rt_list, gt_cls_list,
                model_list, instance_id_list, instance_eval_rt_list,
                rotation_errors.cpu().tolist(), pose_score_list,
                pose_elapsed_seconds)

def _stratified_probe_indices(dataset, requested_size, seed):
    """Select a deterministic, approximately class-balanced training probe."""
    requested_size = min(int(requested_size), len(dataset))
    if requested_size <= 0:
        return []
    groups = {}
    for index, item in enumerate(_dataset_data_items(dataset)):
        class_id = int(item['gt_info']['obj_id'])
        groups.setdefault(class_id, []).append(index)
    generator = random.Random(int(seed))
    for indices in groups.values():
        generator.shuffle(indices)
    selected = []
    active_classes = sorted(groups)
    cursor = 0
    while len(selected) < requested_size and active_classes:
        next_classes = []
        for class_id in active_classes:
            if cursor < len(groups[class_id]):
                selected.append(groups[class_id][cursor])
                next_classes.append(class_id)
                if len(selected) >= requested_size:
                    break
        active_classes = next_classes
        cursor += 1
    return selected



def _dataset_data_items(dataset):
    """Expose BOP metadata from a dataset or a concatenation of datasets."""
    if isinstance(dataset, ConcatDataset):
        return [item for child in dataset.datasets for item in child.data_list]
    return dataset.data_list


def _make_evaluation_copy(dataset):
    """Share immutable paths while disabling every training augmentation."""
    evaluation_dataset = copy(dataset)
    evaluation_dataset.add_noise = False
    evaluation_dataset.data_augmentation = False
    evaluation_dataset.depth_augmentation = False
    evaluation_dataset.bbox_jitter = 0.0
    return evaluation_dataset


def _build_mixed_sampler(train_datasets, training_indices, real_sampling_ratio,
                         epoch_samples, class_balance_power=0.0):
    """Balance source domains and optionally soften object imbalance."""
    class_balance_power = float(class_balance_power)
    if len(train_datasets) == 1 and class_balance_power == 0.0:
        return None, len(training_indices), None

    boundaries = []
    total = 0
    for dataset in train_datasets:
        total += len(dataset)
        boundaries.append(total)

    source_ids = []
    local_indices = []
    source_index = 0
    for index in training_indices:
        while index >= boundaries[source_index]:
            source_index += 1
        source_ids.append(source_index)
        previous_boundary = (
            boundaries[source_index - 1] if source_index > 0 else 0)
        local_indices.append(index - previous_boundary)

    counts = [source_ids.count(index) for index in range(len(train_datasets))]
    if any(count == 0 for count in counts):
        raise ValueError('A requested training split has no samples after validation holdout')

    real_sources = [
        index for index, dataset in enumerate(train_datasets)
        if dataset.train_split == 'train_primesense']
    other_sources = [
        index for index in range(len(train_datasets)) if index not in real_sources]
    if real_sources and other_sources:
        shares = [0.0] * len(train_datasets)
        for index in real_sources:
            shares[index] = real_sampling_ratio / len(real_sources)
        for index in other_sources:
            shares[index] = (1.0 - real_sampling_ratio) / len(other_sources)
    else:
        shares = [1.0 / len(train_datasets)] * len(train_datasets)

    if any(share == 0.0 for share in shares):
        active_sources = [index for index, share in enumerate(shares) if share > 0.0]
        samples = epoch_samples or sum(counts[index] for index in active_sources)
    else:
        samples = epoch_samples or int(max(
            counts[index] / shares[index] for index in range(len(train_datasets))))

    object_ids = [
        int(train_datasets[source].data_list[local]['gt_info']['obj_id'])
        for source, local in zip(source_ids, local_indices)]
    class_counts = [{} for _ in train_datasets]
    for source, object_id in zip(source_ids, object_ids):
        class_counts[source][object_id] = (
            class_counts[source].get(object_id, 0) + 1)
    source_normalizers = []
    for source, histogram in enumerate(class_counts):
        normalizer = sum(
            count ** (1.0 - class_balance_power)
            for count in histogram.values())
        source_normalizers.append(max(float(normalizer), 1.0))
    weights = [
        (shares[source]
         * class_counts[source][object_id] ** (-class_balance_power)
         / source_normalizers[source])
        if shares[source] > 0.0 else 0.0
        for source, object_id in zip(source_ids, object_ids)]

    sampler = WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double), samples, replacement=True)
    summary = ', '.join(
        '{}: {} samples, {:.1%} draw probability'.format(
            train_datasets[index].train_split, counts[index], shares[index])
        for index in range(len(train_datasets)))
    if class_balance_power > 0.0:
        summary += ', class balance power {:.2f}'.format(
            class_balance_power)
    return sampler, samples, summary


def evaluate_probe(loader, estimator, lossor, opt):
    estimator.eval()
    metrics = WeightedMetricAccumulator()
    with torch.no_grad():
        for batch_index, data in enumerate(loader, start=1):
            _, values = predict(data, estimator, lossor, opt, mode='probe')
            metrics.update(values, weight=data['rgb'].size(0))
            if opt.max_eval_batches > 0 and batch_index >= opt.max_eval_batches:
                break
    return metrics.average()


def _checkpoint_payload(epoch, estimator, optimizer, scheduler, ema, monitor,
                        global_step, best_rot_loss, train_metrics,
                        val_metrics, opt):
    return {
        'epoch': epoch + 1,
        'global_step': int(global_step),
        'state_dict': estimator.state_dict(),
        'ema_state_dict': ema.state_dict() if ema is not None else None,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler is not None else None,
        'monitor': monitor.state_dict(),
        'best_rot_loss': best_rot_loss,
        'best_selection_value': best_rot_loss,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'args': vars(opt),
    }


def per_processor(gpu, opt):
    """Single-GPU training with EMA validation and persistent diagnostics."""
    opt.gpu = gpu
    print('init gpu:{}'.format(gpu))
    torch.cuda.set_device(gpu)

    pose_net, estimator_cls = load_pose_module(opt.model)
    estimator_kwargs = {'num_class': opt.num_objects}
    estimator_kwargs.update(bmc_model_kwargs(opt))
    estimator = estimator_cls(**estimator_kwargs).to(gpu)
    param_count = sum(parameter.numel() for parameter in estimator.parameters())
    try:
        from lib.complexity import measure_model_complexity
        complexity = measure_model_complexity(
            estimator, num_points=opt.num_points, input_size=128,
            device=gpu)
        print(
            'model: {} | params: {:.3f}M | GFLOPS: {:.3f} '
            '(torch profiler, {}x{} crop, {} CAD points)'.format(
                opt.model, complexity['parameters_M'], complexity['gflops'],
                complexity['input_size'], complexity['input_size'],
                complexity['num_points']))
    except Exception as error:  # noqa: BLE001 - profiling must not block training
        print('model: {} | params: {:.3f}M | complexity unavailable: {}'.format(
            opt.model, param_count / 1e6, error))
    parameter_budget = MODEL_PARAMETER_BUDGETS.get(opt.model, 5_500_000)
    if ((is_pro_model(opt.model) or opt.model in RVC6D_MODELS)
            and param_count > parameter_budget):
        raise RuntimeError(
            'model parameter budget exceeded: {:,} > {:,}'.format(
                param_count, parameter_budget))

    optimizer = build_rvc6d_adamw(
        estimator, learning_rate=opt.learning_rate,
        weight_decay=opt.weight_decay,
        betas=(0.9, 0.999), eps=1e-8)
    scheduler = None
    for group in optimizer_group_summary(optimizer):
        print(
            'optimizer group: {name} | params={parameters} '
            'lr={base_lr:.2e} wd={weight_decay:.2e}'.format(**group))
    ema = ModelEMA(estimator, decay=opt.ema_decay) if opt.use_ema else None
    best_rot_loss = float('inf')
    global_step = 0
    checkpoint = None

    if opt.warm_start:
        warm_start_path = resolve_existing_path(opt.warm_start)
        warm_checkpoint = torch.load(
            warm_start_path, map_location='cuda:{}'.format(gpu),
            weights_only=False)
        warm_state = warm_checkpoint.get(
            'ema_state_dict', warm_checkpoint.get(
                'train_state_dict', warm_checkpoint.get('state_dict')))
        if (isinstance(warm_state, dict)
                and isinstance(warm_state.get('state_dict'), dict)):
            warm_state = warm_state['state_dict']
        if warm_state is None:
            raise KeyError(
                'warm-start checkpoint has no EMA/model state: {}'.format(
                    warm_start_path))
        cleaned = {
            (key[7:] if key.startswith('module.') else key): value
            for key, value in warm_state.items()}
        model_state = estimator.state_dict()
        compatible = {
            key: value for key, value in cleaned.items()
            if key in model_state and model_state[key].shape == value.shape}
        if not compatible:
            raise ValueError(
                'warm-start checkpoint has no compatible tensors: {}'.format(
                    warm_start_path))
        model_state.update(compatible)
        estimator.load_state_dict(model_state)
        skipped = len(cleaned) - len(compatible)
        # A warm start is deliberately a new experiment.  Recreate EMA from
        # the loaded model and keep fresh optimizer/global-step/monitor state.
        ema = ModelEMA(estimator, decay=opt.ema_decay) if opt.use_ema else None
        print(
            "warm-started '{}' from {} tensors ({} skipped); "
            'optimizer/EMA/schedule reset'.format(
                warm_start_path, len(compatible), skipped))

    if opt.resume:
        if opt.warm_start:
            raise ValueError('--resume and --warm_start are mutually exclusive')
        resume_path = resolve_existing_path(opt.resume)
        checkpoint = torch.load(
            resume_path, map_location='cuda:{}'.format(gpu),weights_only=False)
        saved_args = checkpoint.get('args', {})
        if isinstance(saved_args, argparse.Namespace):
            saved_args = vars(saved_args)
        experiment_keys = (
            'model', 'sarr_loss_weight',
            'sarr_translation_weight', 'primary_pose_loss',
            'pose_distance_weight', 'pose_loss_points', 'pose_loss_votes',
            'sarr_score_weight', 'point_filter',
            'center_weight',
            'hypothesis_weight', 'hypothesis_score_weight',
            'hypothesis_diversity_weight',
            'hypothesis_balance_weight',
            'hypothesis_diversity_margin',
            'hypothesis_temperature',
            'pose_score_weight', 'selection_metric')
        if opt.model in RVC6D_MODELS:
            experiment_keys += (
                'batch_size', 'depth_augmentation',
                'depth_dropout', 'depth_noise_std',
                'depth_hole_probability', 'bbox_jitter',
                'real_sampling_ratio', 'class_balance_power')
        if opt.model in RVC6D_MODELS:
            experiment_keys += (
                'correspondence_weight', 'surface_inlier_weight',
                'refinement_rotation_weight', 'refinement_center_weight')
        if opt.model in BMC_MODELS:
            experiment_keys += (
                'batch_size', 'bmc_projection_dim',
                'bmc_temperature', 'bmc_shared_base_weight',
                'bmc_specific_base_weight', 'bmc_shared_target',
                'bmc_specific_target', 'bmc_ema_decay',
                'bmc_warmup_steps', 'bmc_control_interval',
                'bmc_deadband', 'bmc_min_multiplier',
                'bmc_max_multiplier', 'bmc_min_samples',
                'bmc_min_support_ratio',
                'bmc_specific_variance_floor',
                'bmc_specific_variance_weight',
                'bmc_pose_metric_target', 'bmc_pose_ema_decay')
        optimizer_keys = (
            'learning_rate',
            'min_lr_ratio', 'warmup_start_ratio',
            'warnup_iters', 'weight_decay')
        def config_mismatches(keys):
            return [
                '{}: checkpoint={} current={}'.format(
                    key, saved_args[key], getattr(opt, key))
                for key in keys
                if isinstance(saved_args, dict) and key in saved_args
                and hasattr(opt, key)
                and saved_args[key] != getattr(opt, key)]

        experiment_mismatches = config_mismatches(experiment_keys)
        optimizer_mismatches = config_mismatches(optimizer_keys)
        has_optimizer_state = bool(checkpoint.get('optimizer'))
        blocking_mismatches = list(experiment_mismatches)
        if has_optimizer_state:
            blocking_mismatches.extend(optimizer_mismatches)
        if blocking_mismatches and not opt.allow_resume_config_mismatch:
            raise ValueError(
                'Resume configuration mismatch would make loss curves '
                'incomparable:\n  ' + '\n  '.join(blocking_mismatches)
                + '\nPass --allow_resume_config_mismatch true only for an intentional ablation.')
        if optimizer_mismatches and not has_optimizer_state:
            print(
                'model-only warm-start: using current optimizer settings:\n  '
                + '\n  '.join(optimizer_mismatches))
        checkpoint_state = checkpoint.get('train_state_dict', checkpoint['state_dict'])
        if is_bbox_only_model(opt.model):
            estimator.load_state_dict(checkpoint_state, strict=True)
            skipped = []
        else:
            model_state = estimator.state_dict()
            compatible = {
                key: value for key, value in checkpoint_state.items()
                if key in model_state and model_state[key].shape == value.shape}
            skipped = [key for key in checkpoint_state if key not in compatible]
            model_state.update(compatible)
            estimator.load_state_dict(model_state)
        if skipped:
            print("partially loaded checkpoint '{}': {} tensors loaded, {} skipped".format(
                resume_path, len(compatible), len(skipped)))
        else:
            opt.start_epoch = int(checkpoint.get('epoch', opt.start_epoch))
            if checkpoint.get('optimizer'):
                if load_optimizer_state_name_matched(
                        optimizer, checkpoint, estimator):
                    mismatched = [
                        (name, tuple(parameter.shape), tuple(
                            optimizer.state[parameter]['exp_avg'].shape))
                        for name, parameter in estimator.named_parameters()
                        if parameter in optimizer.state
                        and optimizer.state[parameter].get('exp_avg')
                        is not None
                        and optimizer.state[parameter]['exp_avg'].shape
                        != parameter.shape]
                    if mismatched:
                        raise RuntimeError(
                            'optimizer state shape mismatch after '
                            'name-matched resume: {}'.format(mismatched[:3]))
                    print(
                        'optimizer state remapped by parameter name '
                        '(module order changed since checkpoint)')
                else:
                    optimizer.load_state_dict(checkpoint['optimizer'])
            if scheduler is not None and checkpoint.get('scheduler'):
                scheduler.load_state_dict(checkpoint['scheduler'])
            best_rot_loss = float(checkpoint.get(
                'best_selection_value', checkpoint.get(
                    'best_rot_loss', best_rot_loss)))
            global_step = int(checkpoint.get('global_step', 0))
            if ema is not None:
                ema_state = checkpoint.get('ema_state_dict')
                if ema_state:
                    ema.load_state_dict(ema_state)
                else:
                    ema = ModelEMA(estimator, decay=opt.ema_decay)
            print("strictly resumed checkpoint '{}' at epoch {}".format(
                resume_path, opt.start_epoch))

    tensorboard_writer = (
        SummaryWriter(opt.log_dir, purge_step=global_step) if gpu == 0 else 0)
    monitor = TrainingMonitor(
        opt.log_dir, tensorboard_writer=tensorboard_writer,
        config=vars(opt), overfit_patience=opt.overfit_patience,
        plot_curves=opt.plot_curves)
    if checkpoint is not None:
        monitor.load_state_dict(checkpoint.get('monitor'))

    loader_kwargs = build_loader_kwargs(
        opt.workers, persistent_workers=opt.persistent_workers,
        prefetch_factor=opt.prefetch_factor, pin_memory=opt.pin_memory)
    test_set = pose_dataset(
        'test', opt.num_points, opt.dataset_root, False, opt.noise_trans,
        augmentation=False, point_filter=opt.point_filter,
        dataset=opt.dataset)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=opt.batch_size, shuffle=False, **loader_kwargs)

    dataset = None
    train_datasets = None
    trainloader = None
    probe_loader = None
    train_datasets = [
        pose_dataset(
            'train', opt.num_points, opt.dataset_root, True, opt.noise_trans,
            augmentation=opt.augmentation, train_split=train_split,
            point_filter=opt.point_filter, dataset=opt.dataset,
            **pro_dataset_kwargs(opt))
        for train_split in opt.train_splits]
    dataset = (train_datasets[0] if len(train_datasets) == 1
               else ConcatDataset(train_datasets))
    print('training splits: {}'.format(', '.join(
        '{}={}'.format(item.train_split, len(item)) for item in train_datasets)))
    # Validation is performed on the official test split.  Do not remove
    # samples from training: the former 3,000-item holdout is included in
    # every training epoch.
    training_indices = list(range(len(dataset)))
    train_subset = Subset(dataset, training_indices)
    sampler, samples_per_epoch, sampling_summary = _build_mixed_sampler(
        train_datasets, training_indices, opt.real_sampling_ratio,
        opt.epoch_samples, opt.class_balance_power)
    trainloader = torch.utils.data.DataLoader(
        train_subset, batch_size=opt.batch_size,
        shuffle=sampler is None, sampler=sampler, **loader_kwargs)
    evaluation_workers = min(4, opt.workers)
    evaluation_kwargs = build_loader_kwargs(
        evaluation_workers,
        persistent_workers=opt.persistent_workers,
        prefetch_factor=opt.prefetch_factor,
        pin_memory=opt.pin_memory)
    requested_probe = min(max(0, int(opt.train_probe_size)), len(dataset))
    probe_indices = []
    if requested_probe:
        # This diagnostic subset remains in training; it is never a
        # holdout and only supports the train/test gap chart.
        evaluation_datasets = [
            _make_evaluation_copy(item) for item in train_datasets]
        evaluation_dataset = (
            evaluation_datasets[0] if len(evaluation_datasets) == 1
            else ConcatDataset(evaluation_datasets))
        probe_indices = _stratified_probe_indices(
            evaluation_dataset, requested_probe, opt.seed)
        probe_loader = torch.utils.data.DataLoader(
            Subset(evaluation_dataset, probe_indices),
            batch_size=opt.batch_size, shuffle=False, **evaluation_kwargs)
    print('split disabled: train={} test_validation={} train_probe={}'.format(
        len(training_indices), len(test_set), len(probe_indices)))
    if sampling_summary is not None:
        print('mixed sampling: {}; {} draws/epoch'.format(
            sampling_summary, samples_per_epoch))

    opt.steps_per_epoch = (
        min(len(trainloader), opt.max_train_batches)
        if opt.max_train_batches > 0 else len(trainloader))
    opt.total_train_steps = max(
        opt.warnup_iters + 1,
        opt.steps_per_epoch * max(1, opt.nepoch))
    print('optimizer schedule: {} steps/epoch, {} total steps'.format(
        opt.steps_per_epoch, opt.total_train_steps))

    metadata_dataset = (
        train_datasets[0] if train_datasets is not None else test_set)
    opt.sym_list = metadata_dataset.get_sym_list()
    opt.num_points_mesh = metadata_dataset.get_num_points_mesh()
    opt.obj_radius = metadata_dataset.obj_radius
    opt.raw_prim_groups = metadata_dataset.raw_prim_groups

    rotation_weight = opt.sarr_loss_weight
    loss_kwargs = {
        'dataset': metadata_dataset,
        'loss_type': opt.loss_type,
        'rotation_weight': rotation_weight,
    }
    if is_sarr_model(opt.model):
        loss_kwargs.update({
            'translation_weight': opt.sarr_translation_weight,
            'score_weight': effective_score_weight(opt),
            'primary_pose_loss': opt.primary_pose_loss,
            'pose_distance_weight': opt.pose_distance_weight,
            'pose_loss_points': opt.pose_loss_points,
            'pose_loss_votes': opt.pose_loss_votes,
        })
    if is_rvc6d_model(opt.model):
        loss_kwargs.update({
            'center_weight': opt.center_weight,
            'hypothesis_rotation_weight': opt.hypothesis_weight,
            'hypothesis_score_weight': opt.hypothesis_score_weight,
            'hypothesis_diversity_weight':
                opt.hypothesis_diversity_weight,
            'hypothesis_balance_weight': opt.hypothesis_balance_weight,
            'hypothesis_diversity_margin':
                opt.hypothesis_diversity_margin,
            'hypothesis_assignment_temperature':
                opt.hypothesis_temperature,
            'pose_score_weight': opt.pose_score_weight,
        })
    if opt.model in RVC6D_MODELS:
        loss_kwargs.update({
            'correspondence_weight': opt.correspondence_weight,
            'surface_inlier_weight': opt.surface_inlier_weight,
            'refinement_rotation_weight': opt.refinement_rotation_weight,
            'refinement_center_weight': opt.refinement_center_weight,
        })
    train_loss = pose_net.get_loss(train=True, **loss_kwargs).to(gpu)
    test_loss = pose_net.get_loss(train=False, **loss_kwargs).to(gpu)

    for epoch in range(opt.start_epoch, opt.nepoch):
            opt.cur_epoch = epoch
            epoch_start = time.time()
            print('>>>>>>>>>>>train>>>>>>>>>>>')
            train_metrics, global_step, train_stats = train(
                trainloader, estimator, train_loss, optimizer, epoch,
                tensorboard_writer, global_step, opt, ema=ema)
            evaluation_model = ema.module if ema is not None else estimator
            probe_metrics = (
                evaluate_probe(probe_loader, evaluation_model, test_loss, opt)
                if probe_loader is not None else None)
            torch.cuda.empty_cache()

            val_metrics = None
            improved = False
            if epoch % opt.test_every == 0:
                print('>>>>>>>>>>>test-set validation>>>>>>>>>>>')
                val_metrics = test(
                    test_loader, evaluation_model, test_loss, epoch,
                    tensorboard_writer, [], opt)
                if scheduler is not None:
                    scheduler.step(val_metrics['rot_loss'])
                selection_value = (
                    -float(val_metrics['add_auc'])
                    if opt.selection_metric == 'add_auc'
                    else float(val_metrics['rot_loss']))
                improved = selection_value < best_rot_loss
                if improved:
                    best_rot_loss = selection_value

            current_lr = max(
                float(group['lr']) for group in optimizer.param_groups)
            monitor_row = monitor.record(
                epoch, global_step, train_metrics, val_metrics,
                train_probe_metrics=probe_metrics,
                lr=current_lr,
                grad_norm=train_stats['grad_norm'],
                samples_per_second=train_stats['samples_per_second'],
                elapsed_seconds=time.time() - epoch_start,
                extra_metrics={
                    'gpu_peak_mb': train_stats['gpu_peak_mb'],
                    'cpu_tree_pss_peak_mb': train_stats['cpu_tree_pss_peak_mb'],
                    'cpu_tree_pss_end_mb': train_stats['cpu_tree_pss_end_mb'],
                    'cpu_tree_rss_end_mb': train_stats['cpu_tree_rss_end_mb'],
                })
            print('monitor status: {} | best epoch: {}'.format(
                monitor_row['status'], monitor.best_epoch))

            payload = _checkpoint_payload(
                epoch, estimator, optimizer, scheduler, ema, monitor,
                global_step, best_rot_loss, train_metrics, val_metrics,
                opt)
            if epoch % opt.save_every == 0:
                torch.save(
                    payload,
                    os.path.join(
                        opt.outf, 'checkpoint_{:04d}.pth.tar'.format(epoch)))
            if improved:
                torch.save(
                    payload, os.path.join(opt.outf, 'checkpoint_best.pth.tar'))
                selected_metric = (
                    val_metrics[opt.selection_metric]
                    if val_metrics is not None else float('nan'))
                print('new best validation {}: {:.6f}'.format(
                    opt.selection_metric, selected_metric))
            torch.cuda.empty_cache()
            if monitor.should_early_stop(opt.early_stop_patience):
                print('early stopping after {} non-improving validations'.format(
                    monitor.bad_epochs))
                break
    # post-training cleanup
    for loader in (trainloader, probe_loader, test_loader):
        shutdown_data_loader(loader)
    gc.collect()
    torch.cuda.empty_cache()
    if tensorboard_writer not in (None, 0):
        tensorboard_writer.flush()
        tensorboard_writer.close()


def train(train_loader, estimator, lossor, optimizer, epoch,
          tensorboard_writer, global_step, opt, ema=None):
    """Train one epoch and return sample-weighted metrics and system stats."""
    logger = setup_logger(
        'epoch%d' % epoch,
        os.path.join(opt.log_dir, 'epoch_%d_log.txt' % epoch))
    for key, value in sorted(vars(opt).items()):
        if key == 'raw_prim_groups':
            logger.info('raw_prim_groups: {} objects'.format(len(value)))
        elif key == 'obj_radius':
            logger.info('obj_radius: {} values'.format(len(value)))
        else:
            logger.info('{}: {}'.format(key, value))
    logger.info('total train number : {}'.format(len(train_loader)))

    estimator.train()
    optimizer.zero_grad(set_to_none=True)
    accumulator = WeightedMetricAccumulator()
    grad_norm_total = 0.0
    batches = 0
    samples = 0
    start = time.time()
    cpu_tree_pss_peak_mb = process_tree_pss_mb()
    torch.cuda.reset_peak_memory_stats(opt.gpu)

    for batch_index, data in enumerate(train_loader, start=1):
        global_step += 1
        current_lr = adjust_learning_rate(
            optimizer, epoch, global_step, opt)
        loss, loss_dict = predict(
            data, estimator, lossor, opt, mode='train')
        loss.backward()
        if opt.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                estimator.parameters(), opt.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                estimator.parameters(), float('inf'))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if ema is not None:
            ema.update(estimator)

        batch_size = data['rgb'].size(0)
        accumulator.update(loss_dict, weight=batch_size)
        samples += batch_size
        batches += 1
        grad_norm_total += float(grad_norm.detach().item())
        if (batch_index == 1
                or batch_index % opt.memory_sample_interval == 0):
            cpu_tree_pss_peak_mb = max(
                cpu_tree_pss_peak_mb, process_tree_pss_mb())
        log_function([loss_dict], logger, epoch, batch_index, current_lr)
        if tensorboard_writer not in (None, 0) and global_step % 50 == 0:
            for key, value in loss_dict.items():
                tensorboard_writer.add_scalar(
                    'batch/train/' + key, value, global_step)
            tensorboard_writer.add_scalar(
                'batch/optimizer/lr', current_lr, global_step)
            for name, learning_rate in getattr(
                    opt, 'current_group_lrs', {}).items():
                tensorboard_writer.add_scalar(
                    'batch/optimizer_lr/' + name,
                    learning_rate, global_step)
        if opt.max_train_batches > 0 and batch_index >= opt.max_train_batches:
            break

    epoch_metrics = accumulator.average()
    logger.info('TRAIN ENDING: ' + ' '.join(
        '{}:{:.4f}'.format(key, value)
        for key, value in epoch_metrics.items()))
    elapsed = max(time.time() - start, 1e-6)
    cpu_tree_pss_end_mb = process_tree_pss_mb()
    cpu_tree_rss_end_mb = process_tree_rss_mb()
    stats = {
        'grad_norm': grad_norm_total / max(1, batches),
        'samples_per_second': samples / elapsed,
        'gpu_peak_mb': torch.cuda.max_memory_allocated(opt.gpu) / (1024.0 ** 2),
        'cpu_tree_pss_peak_mb': max(
            cpu_tree_pss_peak_mb, cpu_tree_pss_end_mb),
        'cpu_tree_pss_end_mb': cpu_tree_pss_end_mb,
        'cpu_tree_rss_end_mb': cpu_tree_rss_end_mb,
    }
    close_logger(logger)
    return epoch_metrics, global_step, stats


def test(test_loader, estimator, lossor, epoch, tensorboard_writer,
         tensorboard_test_list, opt):
    """Run internal test-split validation (ADD-AUC and rotation stats)."""
    test_accumulator = WeightedMetricAccumulator()
    result_tag = 'coarse'
    logger = setup_logger(
        'test_{}_epoch{}'.format(result_tag, epoch),
        os.path.join(opt.log_dir, 'test_{}_{}_log.txt'.format(
            result_tag, epoch)))
    logger.info('total test number : {}'.format(len(test_loader)))

    # Internal ADD-AUC validation; no submission CSV is produced here
    # (use tools/evaluate_bop_detections.py for the detection-track
    # end-to-end evaluation).
    diameters_list = (test_loader.dataset.diameters
                      if hasattr(test_loader.dataset, 'diameters')
                      else None)
    tless_add_evaluator = TLessPoseEvaluator(diameters=diameters_list)
    

    # Switch the model to evaluation mode
    estimator.eval()
    with torch.no_grad():
        i = 0
        total_gt_list = []
        total_cls_list = []
        total_instance_list = []
        total_RT_list = []
        total_rotation_errors = []

        for data in test_loader:
            i += 1
            (_, test_loss_dict, rt_list, gt_rt_list, gt_cls_list,
             model_list, instance_id_list, instance_eval_rt_list,
             rotation_errors, pose_score_list, pose_elapsed) = predict(
                data, estimator, lossor, opt, mode='test', refine=False)

            total_gt_list += gt_rt_list
            total_cls_list += gt_cls_list
            total_instance_list += instance_id_list
            total_RT_list += instance_eval_rt_list
            total_rotation_errors += rotation_errors

            # Internal ADD stays in the centred relative frame.
            tless_add_evaluator.eval_pose_parallel(
                rt_list, gt_cls_list, gt_rt_list, gt_cls_list, model_list)

            # tless_gadd_evaluator.eval_pose_parallel(rt_list, gt_cls_list, gt_rt_list, gt_cls_list, model_list)

            # Record validation losses
            test_accumulator.update(
                test_loss_dict, weight=data['rgb'].size(0))
            log_function(
                [test_loss_dict], logger, epoch, i, opt.learning_rate)
            if opt.max_eval_batches > 0 and i >= opt.max_eval_batches:
                break

        # Plot validation losses
        l = test_accumulator.average()

        # AUC over the 0.1*diameter threshold sweep.
        add_cur_eval_info_dict = tless_add_evaluator.cal_auc()

        # gadd_cur_eval_info_dict = tless_gadd_evaluator.cal_auc()

        l['add_auc'] = add_cur_eval_info_dict['auc']
        rotation_errors = np.asarray(total_rotation_errors, dtype=np.float64)
        l['rot_deg_mean'] = float(rotation_errors.mean())
        l['rot_deg_median'] = float(np.median(rotation_errors))
        thresholds = (2.0, 5.0, 10.0, 15.0, 25.0, 40.0)
        l['sarr_AR_C'] = 100.0 * float(np.mean([
            np.mean(rotation_errors < threshold) for threshold in thresholds]))
        # l['gadd_auc'] = gadd_cur_eval_info_dict['auc']

    if tensorboard_writer not in (None, 0):
        for key, value in l.items():
            tensorboard_writer.add_scalar(
                'bop_{}/'.format(result_tag) + key, value, epoch)

    # Report the validation results
    log_tmp = 'TEST {} ENDING: '.format(result_tag.upper())
    for key in l:
        log_tmp = log_tmp + ' {}:{:.4f}'.format(key, l[key])
    logger.info(log_tmp)
    close_logger(logger)
    return l

def adjust_learning_rate(optimizer, epoch, iter, opt):
    """Adjust the learning rate."""
    if is_rvc6d_model(opt.model):
        learning_rates = apply_warmup_cosine(
            optimizer, step=iter, warmup_steps=opt.warnup_iters,
            total_steps=opt.total_train_steps,
            minimum_ratio=opt.min_lr_ratio,
            warmup_start_ratio=opt.warmup_start_ratio)
        opt.current_group_lrs = learning_rates
        return max(learning_rates.values())
    raise ValueError('adjust_learning_rate only supports RVC6D')

def log_function(loss_list, logger, epoch, batch, lr):
    """Logging helper."""
    l = loss_list[-1]
    tmp = 'time{} E{} B{} lr:{:.9f}'.format(
        time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), epoch, batch, lr)
    for key in l:
        tmp = tmp + ' {}:{:.4f}'.format(key, l[key])
    logger.info(tmp)

def draw_loss_list(phase, loss_list, tensorboard_writer):
    """Plot the loss curves."""
    loss = loss_list[-1]
    for key in loss:
        tensorboard_writer.add_scalar(phase+'/'+key, loss[key], len(loss_list))

if __name__ == '__main__':
    main()
