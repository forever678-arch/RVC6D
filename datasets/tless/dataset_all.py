import os
import json
import math
import random
import cv2
import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
import torch.nn.functional as F
import torchvision.transforms as transforms
import open3d as o3d
from lib.point_filter import POINT_FILTER_CHOICES, compute_point_filter


def _axis_angle_rotation(axis, angle):
    """Rodrigues rotation used by the BOP symmetry discretization."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    x, y, z = axis
    skew = np.asarray(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=np.float64)
    identity = np.eye(3, dtype=np.float64)
    return (
        identity + math.sin(angle) * skew
        + (1.0 - math.cos(angle)) * np.matmul(skew, skew))


def _bop_symmetry_rotations(model_info, max_sym_disc_step=0.01):
    """Return BOP/GDR-Net-compatible model-to-model symmetry rotations.

    This mirrors ``bop_toolkit_lib.misc.get_symmetry_transformations`` and
    retains the rotation component required by GDR-Net's rotation-only PM
    objective.  Continuous symmetries use the same default discretization
    step (0.01 object diameters) used by the official GDR-Net data pipeline.
    """
    discrete = [np.eye(3, dtype=np.float64)]
    for symmetry in model_info.get('symmetries_discrete') or ():
        discrete.append(
            np.asarray(symmetry, dtype=np.float64).reshape(4, 4)[:3, :3])

    continuous = []
    for symmetry in model_info.get('symmetries_continuous') or ():
        step_count = int(math.ceil(math.pi / float(max_sym_disc_step)))
        step = 2.0 * math.pi / step_count
        axis = symmetry.get('axis', [0.0, 0.0, 1.0])
        continuous.extend(
            _axis_angle_rotation(axis, index * step)
            for index in range(step_count))

    if continuous:
        rotations = [
            np.matmul(rotation_continuous, rotation_discrete)
            for rotation_discrete in discrete
            for rotation_continuous in continuous]
    else:
        rotations = discrete
    return torch.as_tensor(
        np.stack(rotations), dtype=torch.float32).contiguous()

class PoseDataset(data.Dataset):
    # LM-O: BOP object ids are a non-contiguous subset of LineMOD.
    # Network classes are the contiguous 1..8 slots defined here.
    LMO_BOP_TO_CLS = {1: 1, 5: 2, 6: 3, 8: 4, 9: 5, 10: 6, 11: 7, 12: 8}

    def __init__(self, mode, num_pt, root, add_noise=False, noise_trans=0.03,
                 augmentation=True, train_split='auto',
                 point_filter='mad_soft', depth_augmentation=False,
                 depth_dropout=0.0, depth_noise_std=0.0,
                 depth_hole_probability=0.0, bbox_jitter=None,
                 dataset='tless'):
        self.dataset = dataset
        self.mode = mode
        self.num_pt = num_pt
        self.root = root
        self.add_noise = add_noise
        self.noise_trans = noise_trans
        if point_filter not in POINT_FILTER_CHOICES:
            raise ValueError(
                'point_filter must be one of {}'.format(POINT_FILTER_CHOICES))
        self.point_filter = point_filter
        if train_split not in (
                'auto', 'train_primesense', 'train_pbr', 'train'):
            raise ValueError(
                "train_split must be 'auto', 'train_primesense', 'train_pbr' "
                "or 'train' (BOP real split)")
        self.train_split = train_split
        if self.mode == 'train' and self.train_split == 'auto':
            primesense_path = os.path.join(self.root, 'train_primesense')
            self.train_split = (
                'train_primesense' if os.path.isdir(primesense_path)
                else 'train_pbr')
            if self.train_split == 'train_pbr':
                print(
                    'Warning: train_primesense is unavailable; falling back '
                    'to train_pbr (this differs from the SARR paper).')
        # BBox-only pose mode: crops simulate detector output.  The visible
        # instance mask is loaded only as ``metric_mask_visib`` for detached
        # diagnostics and never determines a crop, feature, target, or loss.
        self.bbox_source = 'bbox_visib'
        self.bbox_pad_ratio = 0.05
        default_bbox_jitter = 0.05 if (
            self.mode == 'train' and self.add_noise) else 0.0
        self.bbox_jitter = (
            default_bbox_jitter if bbox_jitter is None
            else float(bbox_jitter) if self.mode == 'train' else 0.0)
        self.depth_augmentation = bool(
            depth_augmentation) and self.mode == 'train'
        self.depth_dropout = float(depth_dropout)
        self.depth_noise_std = float(depth_noise_std)
        self.depth_hole_probability = float(depth_hole_probability)
        if not 0.0 <= self.bbox_jitter < 0.5:
            raise ValueError('bbox_jitter must be in [0, 0.5)')
        if not 0.0 <= self.depth_dropout < 1.0:
            raise ValueError('depth_dropout must be in [0, 1)')
        if self.depth_noise_std < 0.0:
            raise ValueError('depth_noise_std must be non-negative')
        if not 0.0 <= self.depth_hole_probability <= 1.0:
            raise ValueError('depth_hole_probability must be in [0, 1]')

        # ======== 1. Read and parse object diameters from the models JSON ========
        self.diameters = [0.0] * 31  
        self.obj_radius = [0.10] * 30  
        self.continuous_sym_axes = [None] * 30
        self.discrete_sym_rots = [[np.eye(3, dtype=np.float32)] for _ in range(30)]
        self.symmetry_rotations = [
            torch.eye(3, dtype=torch.float32).unsqueeze(0)
            for _ in range(30)]
        
        # BOP layouts ship the models under ``models_eval``; the T-LESS
        # release used ``model_eval``.  Accept either without a symlink.
        model_dir = 'model_eval'
        if not os.path.isdir(os.path.join(self.root, model_dir)):
            if os.path.isdir(os.path.join(self.root, 'models_eval')):
                model_dir = 'models_eval'
        self.model_dir = model_dir
        models_info_path = os.path.join(
            self.root, model_dir, 'models_info.json')
        if os.path.exists(models_info_path):
            print(f"Loading object diameters from: {models_info_path}")
            with open(models_info_path, 'r') as f:
                models_info = json.load(f)
                for obj_id_str, info in models_info.items():
                    obj_id = int(obj_id_str)
                    if self.dataset == 'lmo':
                        if obj_id not in self.LMO_BOP_TO_CLS:
                            continue
                        slot = self.LMO_BOP_TO_CLS[obj_id]
                    else:
                        slot = obj_id
                    if 'diameter' in info:
                        diam_m = info['diameter'] * 0.001 
                        while len(self.diameters) <= slot:
                            self.diameters.append(0.0)
                        while len(self.obj_radius) < slot:
                            self.obj_radius.append(0.10)

                        self.diameters[slot] = diam_m
                        self.obj_radius[slot - 1] = diam_m / 2.0

                    if 1 <= slot <= 30:
                        self.symmetry_rotations[slot - 1] = (
                            _bop_symmetry_rotations(info))
                        cont_syms = info.get('symmetries_continuous') or []
                        if cont_syms:
                            axis = np.asarray(cont_syms[0].get('axis', [0.0, 0.0, 1.0]), dtype=np.float32)
                            norm = np.linalg.norm(axis)
                            if norm > 1e-6:
                                self.continuous_sym_axes[slot - 1] = axis / norm

                        disc_syms = info.get('symmetries_discrete') or []
                        if disc_syms:
                            rots = [np.eye(3, dtype=np.float32)]
                            for sym in disc_syms:
                                mat = np.asarray(sym, dtype=np.float32).reshape(4, 4)
                                rots.append(mat[:3, :3])
                            self.discrete_sym_rots[slot - 1] = rots
        else:
            print(f"Warning: Models info file not found at {models_info_path}. Using default 10cm radius.")

        # --- 2. Scan the raw data ---
        test_folder = 'test' if self.dataset == 'lmo' else 'test_primesense'
        split_name = self.train_split if self.mode == 'train' else test_folder
        print(
            f"Start scanning {split_name} dataset. "
            "No intermediate files will be saved.")
        self.data_list = self._build_data_list()
        self.length = len(self.data_list)
        print(f"Total valid data number: {self.length}")

        # --- 3. Load CAD template point clouds ---
        class_file_path = os.path.join(self.root, 'classes.txt')
        self.cld = []
        if os.path.exists(class_file_path):
            with open(class_file_path, 'r') as class_file:
                while 1:
                    class_input = class_file.readline()
                    if not class_input:
                        break
                    try:
                        class_id = int(class_input.strip())
                        file_name = f'obj_{class_id:06d}.ply'
                        file_path = os.path.join(
                            self.root, self.model_dir, file_name)
                        
                        input_cloud = o3d.io.read_point_cloud(file_path)
                        if not input_cloud.has_points():
                            print(f"Warning: Point cloud empty: {file_path}")
                        
                        raw_xyz = torch.tensor(np.asarray(input_cloud.points).reshape((1, -1, 3)), dtype=torch.float32)
                        xyz_ids = farthest_point_sample(raw_xyz, num_pt).cpu().numpy()
                        raw_xyz = np.asarray(input_cloud.points).astype(np.float32) * 0.001
                        self.cld.append(raw_xyz[xyz_ids[0, :], :])
                        
                    except Exception as e:
                        print(f"Error reading point cloud {file_path}: {e}")
        else:
            print("Warning: classes.txt not found. Cannot load template point clouds.")

        # --- 4. Initialize remaining parameters ---
        self.data_augmentation = bool(augmentation) and self.mode == 'train'
        self.prim_groups = []
        self.raw_prim_groups = []
        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.resize_img_width = 128
        self.noise_img_loc = 0.0
        self.noise_img_scale = 7.0
        self.minimum_num_pt = 50
        self.norm = transforms.Normalize(mean=[0.485*255.0, 0.456*255.0, 0.406*255.0], 
                                         std=[0.229*255.0, 0.224*255.0, 0.225*255.0])

        self.symmetry_obj_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 22, 23, 24, 25, 26, 27, 28, 29]
        if self.dataset == 'lmo':
            # Symmetric objects: BOP ids 10 and 11 -> network slots 6, 7.
            self.symmetry_obj_idx = sorted(
                self.LMO_BOP_TO_CLS[bop] - 1
                for bop in (10, 11))
        # Mapping from network class slot back to the BOP object id used
        # in detection files and submission CSVs.
        self.cls_to_bop = (
            [0] + sorted(self.LMO_BOP_TO_CLS)
            if self.dataset == 'lmo' else list(range(31)))

        # GADD uses the grouped geometric primitives released with the
        # public ES6D dataset.  Prefer dataset-local metadata, but keep a
        # bundled copy so GADD runs work out of the box.
        prim_candidates = (
            os.path.join(self.root, 'tless_gp.json'),
            os.path.join(os.path.dirname(__file__), 'tless_gp.json'),
        )
        prim_path = next(
            (path for path in prim_candidates if os.path.exists(path)), None) \
            if self.dataset == 'tless' else None
        if prim_path is not None:
            with open(prim_path, 'r') as f:
                prim_groups = json.load(f)
                for i, prim in enumerate(prim_groups):
                    tmp = []
                    raw = []
                    for grp in prim['groups']:
                        tmp.append(torch.tensor(grp, dtype=torch.float).permute(1, 0).contiguous() / self.obj_radius[i])
                        raw.append(torch.tensor(grp, dtype=torch.float).permute(1, 0).contiguous())
                    self.prim_groups.append(tmp)
                    self.raw_prim_groups.append(raw)

    def _build_data_list(self):
        data_package = []
        test_folder = 'test' if self.dataset == 'lmo' else 'test_primesense'
        folder_name = self.train_split if self.mode == 'train' else test_folder
        target_path = os.path.join(self.root, folder_name)
        if not os.path.isdir(target_path):
            raise FileNotFoundError(
                '{} split not found: {}'.format(self.dataset, target_path))

        if not os.path.exists(target_path):
            return data_package

        for dir_name in sorted(os.listdir(target_path)):
            scene_path = os.path.join(target_path, dir_name)
            if not os.path.isdir(scene_path):
                continue

            gt_path = os.path.join(scene_path, 'scene_gt.json')
            gt_info_path = os.path.join(scene_path, 'scene_gt_info.json')
            cam_path = os.path.join(scene_path, 'scene_camera.json')
            if not (os.path.exists(gt_path) and os.path.exists(gt_info_path) and os.path.exists(cam_path)):
                continue

            with open(gt_path, 'r') as f:
                scene_gt = json.load(f)
            with open(gt_info_path, 'r') as f:
                scene_gt_info = json.load(f)
            with open(cam_path, 'r') as f:
                camera_info = json.load(f)

            # Visible masks are retained only for detached diagnostics.
            mask_dir = os.path.join(scene_path, 'mask_visib')
            if not os.path.isdir(mask_dir):
                mask_dir = os.path.join(scene_path, 'mask')
            rgb_dir = os.path.join(scene_path, 'rgb')
            depth_dir = os.path.join(scene_path, 'depth')
            # PBR uses JPEG while the real PrimeSense splits use PNG.
            rgb_ext = '.jpg' if any(
                entry.name.lower().endswith('.jpg')
                for entry in os.scandir(rgb_dir)) else '.png'

            for im_id in sorted(scene_gt.keys(), key=lambda x: int(x)):
                if im_id not in scene_gt_info or im_id not in camera_info:
                    continue
                im_id_int = int(im_id)
                for obj_idx, gt_item in enumerate(scene_gt[im_id]):
                    if obj_idx >= len(scene_gt_info[im_id]):
                        continue
                    info_item = scene_gt_info[im_id][obj_idx]
                    bbox = self._select_bbox(info_item)
                    if bbox is None:
                        continue

                    mask_name = f'{im_id_int:06d}_{obj_idx:06d}.png'
                    if self.dataset == 'lmo':
                        bop_obj_id = int(gt_item['obj_id'])
                        if bop_obj_id not in self.LMO_BOP_TO_CLS:
                            continue
                        gt_info_entry = dict(
                            gt_item, obj_id=self.LMO_BOP_TO_CLS[bop_obj_id])
                    else:
                        gt_info_entry = gt_item
                    data_package.append({
                        'rgb_path': os.path.join(rgb_dir, f'{im_id_int:06d}{rgb_ext}'),
                        'depth_path': os.path.join(depth_dir, f'{im_id_int:06d}.png'),
                        'mask_path': os.path.join(mask_dir, mask_name),
                        'gt_info': gt_info_entry,
                        'gt_bbox_info': info_item,
                        'bbox': bbox,
                        'cam_info': camera_info[im_id],
                        'instance_name': f"{dir_name}_{im_id_int:06d}_{obj_idx:06d}",
                        'img_id': im_id,
                        'scene_id': int(dir_name)
                    })
        return data_package

    def _select_bbox(self, info_item):
        bbox = info_item.get(self.bbox_source)
        if bbox is None or len(bbox) != 4:
            return None
        x, y, w, h = [float(v) for v in bbox]
        if w <= 1 or h <= 1:
            return None
        return [x, y, w, h]

    def _bbox_to_crop(self, bbox, img_height, img_width):
        x, y, w, h = bbox
        cx = x + w * 0.5
        cy = y + h * 0.5
        side = max(w, h)

        if self.bbox_jitter > 0:
            shift = self.bbox_jitter * side
            cx += random.uniform(-shift, shift)
            cy += random.uniform(-shift, shift)
            side *= random.uniform(1.0 - self.bbox_jitter, 1.0 + self.bbox_jitter)

        side *= (1.0 + 2.0 * self.bbox_pad_ratio)
        side = max(side, 24.0)

        rmin = int(math.floor(cy - side * 0.5))
        rmax = int(math.ceil(cy + side * 0.5))
        cmin = int(math.floor(cx - side * 0.5))
        cmax = int(math.ceil(cx + side * 0.5))

        rmin = max(0, min(img_height - 1, rmin))
        cmin = max(0, min(img_width - 1, cmin))
        rmax = max(rmin + 1, min(img_height, rmax))
        cmax = max(cmin + 1, min(img_width, cmax))
        return rmin, rmax, cmin, cmax

    def _handle_missing_file(self, data_info):
        """Last-resort fallback for unreadable files."""
        if self.mode == 'train':
            return self.__getitem__(random.randint(0, self.length - 1))
        else:
            cls_id = int(data_info['gt_info']['obj_id'])
            diam = self.diameters[cls_id] if len(self.diameters) > cls_id and self.diameters[cls_id] > 0 else 0.20
            parts = data_info['instance_name'].split('_')
            instance_id = torch.tensor([int(parts[0]), int(parts[1]), int(parts[2])], dtype=torch.long) if len(parts) >= 3 else torch.tensor([0, 0, 0], dtype=torch.long)

            return {
                'rgb': torch.zeros((3, self.resize_img_width, self.resize_img_width), dtype=torch.float32),
                'xyz': torch.zeros((3, self.resize_img_width, self.resize_img_width), dtype=torch.float32),
                'xyz_all': torch.zeros((3, self.resize_img_width, self.resize_img_width), dtype=torch.float32),
                'depth_valid_mask': torch.zeros((1, self.resize_img_width, self.resize_img_width), dtype=torch.float32),
                  'mask': torch.ones((1, self.resize_img_width, self.resize_img_width), dtype=torch.float32),
                  'bbox_mask': torch.ones((1, self.resize_img_width, self.resize_img_width), dtype=torch.float32),
                  'pose_mask': torch.ones((1, self.resize_img_width, self.resize_img_width), dtype=torch.float32),
                  'point_weight': torch.zeros((1, self.resize_img_width, self.resize_img_width), dtype=torch.float32),
                  'crop_intrinsics': torch.zeros(4, dtype=torch.float32),
                  'camera_origin': torch.zeros(3, dtype=torch.float32),
                  'continuous_symmetry_axis': torch.zeros(
                      3, dtype=torch.float32),
                  'metric_mask_visib': torch.zeros((1, self.resize_img_width, self.resize_img_width), dtype=torch.float32),
                'target_r': torch.eye(3, dtype=torch.float32),
                'target_t': torch.zeros(3, dtype=torch.float32),
                'depth_map': torch.zeros((1, self.resize_img_width, self.resize_img_width), dtype=torch.float32),
                'model_xyz': torch.zeros((3, self.num_pt), dtype=torch.float32),
                'class_id': torch.tensor([cls_id - 1], dtype=torch.long),
                'target_xyz': torch.zeros((3, self.num_pt), dtype=torch.float32),
                'instance_id': instance_id,
                'mean_xyz': torch.zeros(3, dtype=torch.float32),
                'img_id': torch.tensor(int(data_info['img_id']), dtype=torch.int32),
                'scene_id': torch.tensor(int(data_info['scene_id']), dtype=torch.int32),
                'diameter': torch.tensor([diam], dtype=torch.float32)
            }

    def _augment_depth_crop(self, depth_crop):
        """Apply sensor-like noise/dropout without using an object mask."""
        if not self.depth_augmentation:
            return depth_crop
        output = depth_crop.copy()
        depth = output[..., 0]
        valid = depth > 0.0
        if self.depth_noise_std > 0.0 and np.any(valid):
            # A small depth-dependent component captures the growth in active
            # sensor uncertainty while keeping the CLI value interpretable in
            # metres near the usual T-LESS working distance.
            sigma = self.depth_noise_std * (0.75 + 0.25 * depth)
            noise = np.random.normal(0.0, 1.0, depth.shape).astype(
                np.float32) * sigma
            depth[valid] = np.maximum(
                depth[valid] + noise[valid], np.float32(1e-5))
        if self.depth_dropout > 0.0 and np.any(valid):
            random_drop = np.random.random(depth.shape) < self.depth_dropout
            depth[valid & random_drop] = 0.0
        if (self.depth_hole_probability > 0.0
                and random.random() < self.depth_hole_probability):
            height, width = depth.shape
            hole_count = random.randint(1, 2)
            for _ in range(hole_count):
                hole_height = random.randint(
                    max(1, height // 24), max(1, height // 8))
                hole_width = random.randint(
                    max(1, width // 24), max(1, width // 8))
                top = random.randint(0, max(0, height - hole_height))
                left = random.randint(0, max(0, width - hole_width))
                depth[top:top + hole_height, left:left + hole_width] = 0.0
        output[..., 0] = depth
        return output

    def __getitem__(self, index):
        data_info = self.data_list[index]
        cam_scale = data_info['cam_info']['depth_scale']
        
        # 1. Physical file read
        try:
            img = Image.open(data_info['rgb_path'])
            depth = np.array(Image.open(data_info['depth_path'])).astype(np.float32) * cam_scale * 0.001
        except Exception:
            return self._handle_missing_file(data_info)

        img_width, img_length = depth.shape
        cam_cx, cam_cy = data_info['cam_info']['cam_K'][2], data_info['cam_info']['cam_K'][5]
        cam_fx, cam_fy = data_info['cam_info']['cam_K'][0], data_info['cam_info']['cam_K'][4]
        obj_id = data_info['gt_info']['obj_id']
        cls_id = int(obj_id)

        # 2. BBox crop: this is the only spatial prior used by the network.
        # The bbox comes from BOP scene_gt_info, which simulates detector output.
        if 'bbox' not in data_info:
            raise RuntimeError(
                'BBox-only mode requires a detector/annotation bbox; '
                'segmentation-derived crop fallback is forbidden.')
        rmin, rmax, cmin, cmax = self._bbox_to_crop(
            data_info['bbox'], img_width, img_length)
        resize_side = float(self.resize_img_width)
        crop_scale_x = resize_side / float(max(1, cmax - cmin))
        crop_scale_y = resize_side / float(max(1, rmax - rmin))
        crop_intrinsics = np.asarray([
            cam_fx * crop_scale_x,
            cam_fy * crop_scale_y,
            (cam_cx - cmin + 0.5) * crop_scale_x - 0.5,
            (cam_cy - rmin + 0.5) * crop_scale_y - 0.5,
        ], dtype=np.float32)

        img_crop = np.array(img)[:, :, :3][rmin:rmax, cmin:cmax, :]
        depth_crop = depth[rmin:rmax, cmin:cmax, np.newaxis].astype(np.float32)
        depth_crop = self._augment_depth_crop(depth_crop)
        depth_crop_for_output = depth_crop.copy()

        bbox_mask_crop = np.ones((rmax - rmin, cmax - cmin), dtype=np.float32)
        mask_path = data_info.get('mask_path', '')
        metric_mask_visib_full = cv2.imread(mask_path, 0) if mask_path else None
        if metric_mask_visib_full is None:
            metric_mask_visib_full = np.zeros(depth.shape, dtype=np.uint8)
        metric_mask_visib_crop = (
            metric_mask_visib_full[rmin:rmax, cmin:cmax].astype(np.float32)
            / 255.0)
        mask_crop = bbox_mask_crop
        x_indices, y_indices = np.meshgrid(np.arange(cmin, cmax), np.arange(rmin, rmax))
        
        pt2 = depth_crop
        pt0 = (x_indices[:, :, np.newaxis].astype(np.float32) - cam_cx) * pt2 / cam_fx
        pt1 = (y_indices[:, :, np.newaxis].astype(np.float32) - cam_cy) * pt2 / cam_fy
        depth_xyz = np.concatenate((pt0, pt1, pt2), axis=2)
        
        valid_points = depth_xyz[depth_xyz[:, :, 2] > 0].reshape(-1, 3)
        
        diam = self.diameters[cls_id] if len(self.diameters) > cls_id and self.diameters[cls_id] > 0 else 0.20
        dynamic_threshold = diam * 0.75

        # # ======== 3. Degraded-depth fallback (kept for reference) ========
        # if valid_points.shape[0] < 10:
        #     # Strategy: assume a fixed 0.8 m depth when depth fails completely.
        #     # The RGB branch keeps valid 2D semantics from the crop.
        #     mean_xyz = np.array([[[0.0, 0.0, 0.8]]], dtype=np.float32)
        # else:
        #     median_z = np.median(valid_points[:, 2])
        #     core_mask = np.abs(valid_points[:, 2] - median_z) < dynamic_threshold
        #     core_points = valid_points[core_mask]
            
        #     # Strategy: fall back to the mean of all valid points if filtering is too aggressive
        #     if core_points.shape[0] < 5: 
        #         mean_xyz = valid_points.mean(axis=0).reshape((1, 1, 3))
        #     else:
        #         mean_xyz = core_points.mean(axis=0).reshape((1, 1, 3))
        # ======== 3. Fallback for large-scale depth loss ========
        if valid_points.shape[0] < 10:
            # When depth fails completely, recover the 3D center from the 2D bbox and camera intrinsics
            # 1. Estimate the object size in pixels (undo the bbox padding)
            bbox_w = cmax - cmin
            bbox_h = rmax - rmin
            # pixel_size = max(bbox_w, bbox_h) / 1.2  
            pixel_size = max(bbox_w, bbox_h)
            if pixel_size < 10.0: 
                pixel_size = 50.0 # guard against division by zero
            
            # 2. Recover the depth from the similar-triangle relation
            focal_length = (cam_fx + cam_fy) / 2.0
            estimated_z = (diam * focal_length) / pixel_size
            
            # 3. Recover X/Y from the 2D bbox center
            center_u = (cmin + cmax) / 2.0
            center_v = (rmin + rmax) / 2.0
            estimated_x = (center_u - cam_cx) * estimated_z / cam_fx
            estimated_y = (center_v - cam_cy) * estimated_z / cam_fy
            
            # Build the dynamic initial center
            mean_xyz = np.array([[[estimated_x, estimated_y, estimated_z]]], dtype=np.float32)
            
        else:
            median_z = np.median(valid_points[:, 2])
            core_mask = np.abs(valid_points[:, 2] - median_z) < dynamic_threshold
            core_points = valid_points[core_mask]
            
            # Fall back to the mean of all valid points if the filter is too aggressive
            if core_points.shape[0] < 5: 
                mean_xyz = valid_points.mean(axis=0).reshape((1, 1, 3))
            else:
                mean_xyz = core_points.mean(axis=0).reshape((1, 1, 3))

        depth_xyz = (depth_xyz - mean_xyz) * mask_crop[:, :, np.newaxis]

        # 4. Target translation handling
        R = np.array(data_info['gt_info']["cam_R_m2c"]).reshape(3, 3)
        T = np.array(data_info['gt_info']["cam_t_m2c"]).reshape(3, 1)
        pose = np.concatenate((R, T), axis=1)
        pose[:3, 3] = pose[:3, 3] * 0.001 - mean_xyz.flatten()

        rgb = img_crop.astype(np.float32)
        mask = mask_crop.astype(np.float32)
        metric_mask_visib = metric_mask_visib_crop.astype(np.float32)
        xyz = depth_xyz.astype(np.float32)
        target = pose.astype(np.float32)

        # 5. Sanitize NaN/Inf values
        rgb = np.nan_to_num(rgb, nan=0.0, posinf=255.0, neginf=0.0)
        xyz = np.nan_to_num(xyz, nan=0.0)
        target = np.nan_to_num(target, nan=0.0)

        if self.mode == 'train' and self.add_noise:
            # Add noise instead of dropping corrupted samples, so the network learns robustness
            noise_xyz = np.random.uniform(-0.001, 0.001, xyz.shape)
            xyz += noise_xyz
            # xyz = xyz * mask[:, :, np.newaxis]

        rgb, xyz, mask = resize(rgb, xyz, mask, self.resize_img_width, self.resize_img_width)
        metric_mask_visib = resize_mask(
            metric_mask_visib, self.resize_img_width, self.resize_img_width)
        
        depth_resized = F.interpolate(
            torch.from_numpy(np.nan_to_num(depth_crop_for_output, nan=0.0)).permute(2,0,1).unsqueeze(0), 
            size=(self.resize_img_width, self.resize_img_width), mode='bilinear', align_corners=False
        ).squeeze(0).squeeze(0).numpy()  
        depth_valid_mask = F.interpolate(
            torch.from_numpy(
                (depth_crop_for_output[..., 0] > 0.0).astype(np.float32)
            )[None, None],
            size=(self.resize_img_width, self.resize_img_width),
            mode='nearest',
        ).squeeze(0).squeeze(0).numpy()

        target_r = pose[:, 0:3].astype(np.float32)
        target_t = np.array([pose[:, 3:4].flatten()]).reshape(3, -1).astype(np.float32)

        if len(self.cld) >= cls_id: model_points = self.cld[cls_id - 1].T
        else: model_points = np.zeros((3, self.num_pt), dtype=np.float32)

        # Selectable bbox-only point filtering. ``point_weight`` can be soft,
        # while ``geometry_mask`` is its support for loss/voting and for
        # keeping invalid XYZ values at zero. No segmentation is consulted.
        point_weight, point_support = compute_point_filter(
            depth_resized, xyz, diam, mode=self.point_filter,
            minimum_support=32)
        xyz_all = xyz.copy() * depth_valid_mask[:, :, np.newaxis]
        geometry_mask = point_support.astype(np.float32)
        xyz = xyz * geometry_mask[:, :, np.newaxis]
        depth_resized = depth_resized * geometry_mask

        if self.mode == 'train' and self.add_noise:
            noise_t = np.asarray([np.random.uniform(-self.noise_trans, self.noise_trans) for i in range(3)]).astype(np.float32)
            # Simulate an imperfect crop center without turning zeroed
            # background pixels into a constant, highly predictive plane.
            xyz_all = (xyz_all + noise_t) * depth_valid_mask[:, :, np.newaxis]
            xyz = (xyz + noise_t) * geometry_mask[:, :, np.newaxis]
            target_t += noise_t.reshape((3, 1))

        if self.mode == 'train' and self.data_augmentation:
            rgb = np.asarray(
                self.trancolor(Image.fromarray(rgb.astype('uint8')))
            ).astype(np.float32)

        # Normalization
        norm_radius = diam / 2.0
        normalization_center = mean_xyz.astype(np.float32).reshape(3).copy()
        camera_origin = -normalization_center / max(norm_radius, 1e-6)
        continuous_symmetry_axis = self.continuous_sym_axes[cls_id - 1]
        if continuous_symmetry_axis is None:
            continuous_symmetry_axis = np.zeros(3, dtype=np.float32)
        else:
            continuous_symmetry_axis = np.asarray(
                continuous_symmetry_axis, dtype=np.float32)
        xyz_all = np.nan_to_num(xyz_all / norm_radius, nan=0.0)
        xyz = np.nan_to_num(xyz / norm_radius, nan=0.0)
        target_t = np.nan_to_num(target_t / norm_radius, nan=0.0)
        model_points = np.nan_to_num(model_points / norm_radius, nan=0.0)

        target_xyz = target_r @ model_points + target_t

        rgb = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
        xyz_all = torch.from_numpy(xyz_all).permute(2, 0, 1).contiguous()
        xyz = torch.from_numpy(xyz).permute(2, 0, 1).contiguous()

        if mask.sum() == 0.0: mask = np.ones(mask.shape, dtype=np.float32)

        mask = torch.from_numpy(mask).unsqueeze(dim=0)
        rgb = self.norm(rgb)

        if self.mode == 'test':
            parts = data_info['instance_name'].split('_')
            instance_id = torch.tensor([int(parts[0]), int(parts[1]), int(parts[2])]) if len(parts) >= 3 else torch.tensor([0, 0, 0])
        else:
            mean_xyz = np.array([0.0])
            instance_id = torch.tensor([0])

        return {
            'rgb': rgb,
            'xyz': xyz,
            'xyz_all': xyz_all,
            'depth_valid_mask': torch.from_numpy(
                depth_valid_mask).unsqueeze(dim=0),
            'mask': mask,
              'bbox_mask': mask,
              'pose_mask': torch.from_numpy(geometry_mask).unsqueeze(dim=0),
              'point_weight': torch.from_numpy(
                  point_weight.astype(np.float32)).unsqueeze(dim=0),
              # Inference-available projection metadata for organized-depth
              # z-buffer refinement. Coordinates refer to the resized crop and
              # to the same radius-normalized frame as xyz/model_xyz.
              'crop_intrinsics': torch.from_numpy(crop_intrinsics),
              'camera_origin': torch.from_numpy(
                  camera_origin.astype(np.float32)),
              'continuous_symmetry_axis': torch.from_numpy(
                  continuous_symmetry_axis),
              # Ground-truth segmentation is diagnostic-only.  Training and
              # inference code may read this key only inside ``torch.no_grad``.
              'metric_mask_visib': torch.from_numpy(
                  metric_mask_visib.astype(np.float32)).unsqueeze(dim=0),
            'target_r': torch.from_numpy(target_r).view(3, 3),
            'target_t': torch.from_numpy(target_t).view(3),
            'depth_map': torch.from_numpy(depth_resized).unsqueeze(0),
            'model_xyz': torch.from_numpy(model_points),
            'class_id': torch.LongTensor([cls_id - 1]),
            'target_xyz': torch.from_numpy(target_xyz.astype(np.float32)),
            'instance_id': instance_id,
            'mean_xyz': torch.from_numpy(mean_xyz.astype(np.float32).flatten()),
            'img_id': torch.tensor(int(data_info['img_id']), dtype=torch.int32),
            'scene_id': torch.tensor(int(data_info['scene_id']), dtype=torch.int32),
            'diameter': torch.tensor([diam], dtype=torch.float32)
        }

    def __len__(self): return self.length
    def get_sym_list(self): return self.symmetry_obj_idx
    def get_num_points_mesh(self): return self.num_pt

# ================= Helper functions =================
def farthest_point_sample(xyz, npoint):
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def resize(rgb, xyz, mask, width, height):
    rgb = torch.from_numpy(rgb.astype(np.float32)).unsqueeze(dim=0).permute(0, 3, 1, 2).contiguous()
    xyz = torch.from_numpy(xyz.astype(np.float32)).unsqueeze(dim=0).permute(0, 3, 1, 2).contiguous()
    mask = torch.from_numpy(mask.astype(np.float32)).unsqueeze(dim=0).unsqueeze(dim=0)

    rgb = F.interpolate(rgb, size=(height, width), mode='bilinear').squeeze(dim=0).permute(1, 2, 0).contiguous()
    xyz = F.interpolate(xyz, size=(height, width), mode='nearest').squeeze(dim=0).permute(1, 2, 0).contiguous()
    mask = F.interpolate(mask, size=(height, width), mode='nearest').squeeze(dim=0).squeeze(dim=0)
    return rgb.cpu().numpy(), xyz.cpu().numpy(), mask.cpu().numpy()

def resize_mask(mask, width, height):
    mask = torch.from_numpy(mask.astype(np.float32)).unsqueeze(dim=0).unsqueeze(dim=0)
    mask = F.interpolate(mask, size=(height, width), mode='nearest').squeeze(dim=0).squeeze(dim=0)
    return (mask.cpu().numpy() > 0.5).astype(np.float32)

def random_rotation_translation(rgb, xyz, mask, degree_range, trans_range):
    h, w, c = rgb.shape
    rgb = torch.from_numpy(rgb.astype(np.float32)).unsqueeze(dim=0).permute(0, 3, 1, 2).contiguous()
    xyz = torch.from_numpy(xyz.astype(np.float32)).unsqueeze(dim=0).permute(0, 3, 1, 2).contiguous()
    mask = torch.from_numpy(mask.astype(np.float32)).unsqueeze(dim=0).unsqueeze(dim=0)

    angle = float(random.uniform(-degree_range, degree_range)) * math.pi / 180.0
    trans1 = random.choice([float(random.uniform(trans_range[0], trans_range[1])), -float(random.uniform(trans_range[0], trans_range[1]))])
    trans2 = random.choice([float(random.uniform(trans_range[0], trans_range[1])), -float(random.uniform(trans_range[0], trans_range[1]))])

    theta = torch.tensor([
        [math.cos(angle), math.sin(-angle), trans1],
        [math.sin(angle), math.cos(angle), trans2]
    ], dtype=torch.float)

    grid = F.affine_grid(theta.unsqueeze(0), rgb.size())
    rgb = F.grid_sample(rgb, grid).squeeze(dim=0).permute(1, 2, 0).contiguous()
    xyz = F.grid_sample(xyz, grid).squeeze(dim=0).permute(1, 2, 0).contiguous()
    mask = F.grid_sample(mask, grid, mode='nearest').squeeze(dim=0).squeeze(dim=0)

    return rgb.cpu().numpy(), xyz.cpu().numpy(), mask.cpu().numpy()

def paste_two_objects(f_rgb, f_xyz, f_mask, b_rgb, b_xyz, b_mask):
    mask = b_mask - b_mask * f_mask
    rgb = b_rgb * (1 - f_mask[:, :, np.newaxis]) + f_rgb * f_mask[:, :, np.newaxis]
    xyz = b_xyz * (1 - f_mask[:, :, np.newaxis]) + f_xyz * f_mask[:, :, np.newaxis]
    return rgb, xyz, mask
