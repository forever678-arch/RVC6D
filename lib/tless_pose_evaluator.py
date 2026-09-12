#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import os
import torch
import torch.nn.parallel
import concurrent.futures
import numpy as np
import torch.utils.data
import csv
import time


cls_num = 31
cls_lst = [i for i in range(cls_num)]  # cls[0] means all classes
cls_name = ['1', '2', '3', '4', '5',
            '6', '7', '8', '9', '10',
            '11', '12', '13', '14', '15', '16',
            '17', '18', '19', '20', '21',
            '22', '23', '24', '25', '26',
            '27', '28', '29', '30']
class TLessPoseEvaluator:

    def __init__(self, csv_output_path=None, diameters=None):
        """
        diameters: dict or list, indexed by object id (1..30) with the diameter in metres
                   e.g. diameters[1] = 0.12 means object 1 has a 12 cm diameter
        """
        n_cls = cls_num
        
        self.n_cls = cls_num
        self.cls_add_dis = [list() for i in range(n_cls)]
        self.cls_adds_dis = [list() for i in range(n_cls)]
        self.cls_add_s_dis = [list() for i in range(n_cls)]
        sym_cls_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 22, 23, 24, 25, 26, 27, 28, 29]
        self.sym_cls_ids = [i+1 for i in sym_cls_ids]
        # Diameter lookup
        self.diameters = diameters  # indexed by class id; index 0 is unused

        # BOP-format result buffer
        self.bop_results = []
        self.csv_output_path = csv_output_path
        self.current_scene_id = 0
        self.current_im_id = 0

    def add_bop_results(self, absolute_RT_lst, pred_clsID_lst, scene_ids,
                        img_ids, scores, time_vals):
        """Add absolute metric poses to the BOP export buffer.

        ``absolute_RT_lst`` translations are expressed in metres.  BOP CSV
        translations are millimetres, so conversion happens exactly once
        here.  ``time_vals`` is per-proposal pose time; ``generate_bop_csv``
        later sums it per image and writes the same image time on every row.
        """
        fields = {
            'pred_clsID_lst': pred_clsID_lst,
            'scene_ids': scene_ids,
            'img_ids': img_ids,
            'scores': scores,
            'time_vals': time_vals,
        }
        count = len(absolute_RT_lst)
        for name, values in fields.items():
            if len(values) != count:
                raise ValueError(
                    'Length of {} ({}) must equal pose count ({})'.format(
                        name, len(values), count))

        for pose, cls_id, scene_id, im_id, score, time_val in zip(
                absolute_RT_lst, pred_clsID_lst, scene_ids, img_ids,
                scores, time_vals):
            pose = np.asarray(pose, dtype=np.float32)
            if pose.shape != (3, 4):
                raise ValueError(
                    'Absolute BOP pose must have shape (3, 4), got {}'.format(
                        pose.shape))
            score = float(score)
            time_val = float(time_val)
            if not np.isfinite(score):
                raise ValueError('BOP score must be finite')
            if not np.isfinite(time_val) or time_val < 0.0:
                raise ValueError(
                    'BOP pose time must be finite and non-negative')
            self.bop_results.append({
                'scene_id': int(scene_id),
                'im_id': int(im_id),
                'obj_id': int(np.asarray(cls_id).reshape(-1)[0]),
                'R': pose[:, :3].copy(),
                't': (pose[:, 3] * 1000.0).copy(),
                'time': time_val,
                'score': score,
            })

    def generate_bop_csv(self, results_list=None, csv_path=None):
        if results_list is None:
            results_list = self.bop_results

        # # Clear previous results to avoid duplicated writes
        # self.bop_results.clear()
    
        if not results_list:
            print("Warning: No BOP results to save")
            return
    
        if csv_path is None:
            csv_path = self.csv_output_path
    
        if not csv_path:
            print("No CSV output path specified, skipping CSV generation")
            return
    
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        # Group by (scene_id, im_id) and unify the time
        from collections import defaultdict
        grouped = defaultdict(list)
        for source in results_list:
            # Do not deduplicate by pose.  Multiple instances of the same
            # object may legitimately receive identical estimates.
            res = dict(source)
            key = (res['scene_id'], res['im_id'])
            grouped[key].append(res)
    
        # unified_results = []
        # for key, group in grouped.items():
        #     # Use the first (or the average) time per image
        #     uniform_time = group[0].get('time', -1.0)
        #     for res in group:
        #         res['time'] = uniform_time
        #         unified_results.append(res)
        unified_results = []
        for key, group in grouped.items():
            # Sum the times of all detections in the image
            total_time = sum(res.get('time', 0.0) for res in group)
            for res in group:
                res['time'] = total_time   # one shared image time
                unified_results.append(res)
    
        with open(csv_path, 'w', newline='') as f:
            # BOP results format consumed by eval_bop19_pose (bop_toolkit
            # load_bop_results, version="bop19"): seven comma-separated
            # columns scene_id,im_id,obj_id,score,R,t,time.  R holds the 9
            # row-major rotation values and t the 3 millimetre
            # translations, each space-separated within one cell.  time is
            # the per-image processing time in seconds under the BOP
            # timing protocol (forward + decoding, CUDA-synchronized;
            # dataset reading and one-time initialisation excluded).
            writer = csv.writer(f)
            writer.writerow(
                ['scene_id', 'im_id', 'obj_id', 'score', 'R', 't', 'time'])

            for res in unified_results:
                R_flat = res['R'].flatten()
                R_str = ' '.join(f'{v:.6f}' for v in R_flat)
                t_str = ' '.join(f'{v:.6f}' for v in res['t'])
                writer.writerow([
                    res['scene_id'],
                    res['im_id'],
                    res['obj_id'],
                    f"{res.get('score', 1.0):.6f}",
                    R_str,
                    t_str,
                    f"{res.get('time', -1.0):.6f}"
                ])
    
        print(f"BOP CSV file saved to: {csv_path}")
        print(f"Total predictions: {len(unified_results)}")
        if unified_results:
            example = unified_results[0]
            print(f"Example: scene_id={example['scene_id']}, im_id={example['im_id']}, "
                  f"obj_id={example['obj_id']}, score={example.get('score', 1.0):.4f}, time={example.get('time', -1.0):.2f}")

    def cal_auc(self):
        # Rebuild the aggregate each time so repeated reporting is idempotent.
        self.cls_add_s_dis[0] = []
        # Merge symmetric and asymmetric distances into add_s_dis
        for cls_id in range(1, self.n_cls):
            if cls_id in self.sym_cls_ids:
                self.cls_add_s_dis[cls_id] = self.cls_adds_dis[cls_id]
            else:
                self.cls_add_s_dis[cls_id] = self.cls_add_dis[cls_id]
            self.cls_add_s_dis[0] += self.cls_add_s_dis[cls_id]

        add_auc_lst = []
        adds_auc_lst = []
        add_s_auc_lst = []
        add_2cm_lst = []
        adds_2cm_lst = []
        add_s_2cm_lst = []
        # Track the 0.1*diam accuracy
        add_s_diam_acc_lst = []  # per-class accuracy in percent
        diam_coeff = 0.1

        for i in range(self.n_cls):
            add_auc, add_2cm = cal_auc(self.cls_add_dis[i], max_dis=0.05, thr_m=0.01)
            adds_auc, adds_2cm = cal_auc(self.cls_adds_dis[i], max_dis=0.05, thr_m=0.01)
            add_s_auc, add_s_2cm = cal_auc(self.cls_add_s_dis[i], max_dis=0.05, thr_m=0.01)
            add_auc_lst.append(add_auc)
            adds_auc_lst.append(adds_auc)
            add_s_auc_lst.append(add_s_auc)
            add_2cm_lst.append(add_2cm)
            adds_2cm_lst.append(adds_2cm)
            add_s_2cm_lst.append(add_s_2cm)

            # ---- ADD(-S) < 0.1*diam accuracy ----
            diam_acc = -1.0  # -1 marks not computed
            if hasattr(self, 'diameters') and self.diameters is not None and i > 0:
                diam = self.diameters[i]
                
                if diam > 0 and len(self.cls_add_s_dis[i]) > 0:
                    thr = diam_coeff * diam
                    arr = np.array(self.cls_add_s_dis[i])
                    correct = np.sum(arr < thr) / len(arr)
                    diam_acc = correct * 100
            add_s_diam_acc_lst.append(diam_acc)

            if i == 0:
                continue
            if len(self.cls_add_s_dis[i]) == 0:
                continue
            print(cls_name[i-1])
            print("**** add: {:.2f}, adds: {:.2f}, add(-s): {:.2f}".format(add_auc, adds_auc, add_s_auc))
            print("<2cm add: {:.2f}, adds: {:.2f}, add(-s): {:.2f}".format(add_2cm, adds_2cm, add_s_2cm))
            if diam_acc >= 0:
                print("ADD(-S) < 0.1*diam: {:.2f}%".format(diam_acc))

        # Average over classes with a valid diameter only
        valid_diam = [v for v in add_s_diam_acc_lst[1:] if v >= 0]
        avg_add_s_diam = np.mean(valid_diam) if valid_diam else -1.0

        print("Average of all object:")
        print("**** add: {:.2f}, adds: {:.2f}, add(-s): {:.2f}".format(
            np.nanmean(add_auc_lst[1:]), np.nanmean(adds_auc_lst[1:]),
            np.nanmean(add_s_auc_lst[1:])))
        print("<2cm add: {:.2f}, adds: {:.2f}, add(-s): {:.2f}".format(
            np.nanmean(add_2cm_lst[1:]), np.nanmean(adds_2cm_lst[1:]),
            np.nanmean(add_s_2cm_lst[1:])))
        if avg_add_s_diam >= 0:
            print("ADD(-S) < 0.1*diam (avg): {:.2f}%".format(avg_add_s_diam))

        print("All object (following PoseCNN):")
        print("**** add: {:.2f}, adds: {:.2f}, add(-s): {:.2f}".format(
            add_auc_lst[0], adds_auc_lst[0], add_s_auc_lst[0]))
        print("<2cm add: {:.2f}, adds: {:.2f}, add(-s): {:.2f}".format(
            add_2cm_lst[0], adds_2cm_lst[0], add_s_2cm_lst[0]))
        if add_s_diam_acc_lst[0] >= 0:
            print("ADD(-S) < 0.1*diam (all): {:.2f}%".format(add_s_diam_acc_lst[0]))

        # if self.csv_output_path and self.bop_results:
        # # Copy and clear to avoid duplicated writes
        #     results_copy = self.bop_results[:]
        #     self.bop_results.clear()
        #     self.generate_bop_csv(results_list=results_copy)

        # if self.csv_output_path and self.bop_results:
        #     self.generate_bop_csv()

        return {'auc': add_s_auc_lst[0]}

    def eval_pose_parallel(self, pred_RT_lst, pred_clsID_lst, gt_RT_lst, gt_clsID_lst, models_pts_lst,
                           scene_id=None, im_id_start=0, time_val=-1.0,
                           img_ids=None, scores=None):
        bs = len(pred_clsID_lst)

        if isinstance(time_val, (int, float)):
            time_vals = [time_val] * bs
        else:
            time_vals = time_val
            if len(time_vals) != bs:
                time_vals = [time_val[0]] * bs

        if img_ids is None:
            img_ids = [im_id_start + i for i in range(bs)]
        else:
            if len(img_ids) != bs:
                raise ValueError(f"Length of img_ids ({len(img_ids)}) must equal batch size ({bs})")

        if scores is None:
            scores = [1.0] * bs
        else:
            if len(scores) != bs:
                raise ValueError(f"Length of scores ({len(scores)}) must equal batch size ({bs})")

        if scene_id is None:
            scene_id = [scene_id] * bs
        else:
            if len(scene_id) != bs:
                raise ValueError(f"Length of scene_ids ({len(scene_id)}) must equal batch size ({bs})")

        self.current_scene_id = scene_id

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(bs, 8)) as executor:
            futures = []
            for i in range(bs):
                future = executor.submit(
                    self.eval_metric_with_bop,
                    pred_RT_lst[i], pred_clsID_lst[i], gt_RT_lst[i], gt_clsID_lst[i],
                    models_pts_lst[i], scene_id[i], img_ids[i], time_vals[i], scores[i]
                )
                futures.append(future)

            for future in futures:
                cls_add_dis_lst, cls_adds_dis_lst = future.result()
                self.cls_add_dis = self.merge_lst(self.cls_add_dis, cls_add_dis_lst)
                self.cls_adds_dis = self.merge_lst(self.cls_adds_dis, cls_adds_dis_lst)

        return max(img_ids) + 1 if img_ids else im_id_start + bs

    def eval_metric_with_bop(self, pred_RT, pred_cls_id, gt_RT, gt_cls_id, models_pts,
                             scene_id=0, im_id=0, time_val=-1.0, score=1.0):
        n_cls = cls_num
        cls_add_dis = [list() for i in range(n_cls)]
        cls_adds_dis = [list() for i in range(n_cls)]

        if pred_cls_id[0] != gt_cls_id[0]:
            return cls_add_dis, cls_adds_dis

        pred_RT_np = pred_RT.astype(np.float32)

        # Compatibility export for older callers which already provide an
        # absolute metric pose together with real scene/image ids.
        if self.csv_output_path and scene_id is not None and im_id is not None:
            self.add_bop_results(
                [pred_RT_np], [pred_cls_id], [scene_id], [im_id],
                [score], [time_val])

        pred_RT = torch.from_numpy(pred_RT.astype(np.float32)).cuda()
        gt_RT = torch.from_numpy(gt_RT.astype(np.float32)).cuda()
        mesh_pts = torch.from_numpy(models_pts.astype(np.float32)).cuda()
        add = cal_add_cuda(pred_RT, gt_RT, mesh_pts)
        adds = cal_adds_cuda(pred_RT, gt_RT, mesh_pts)
        cls_add_dis[pred_cls_id[0]].append(add.item())
        cls_adds_dis[pred_cls_id[0]].append(adds.item())
        cls_add_dis[0].append(add.item())
        cls_adds_dis[0].append(adds.item())

        return cls_add_dis, cls_adds_dis

    def merge_lst(self, targ, src):
        for i in range(len(targ)):
            targ[i] += src[i]
        return targ


# ==================== Helper functions ====================
def eval_metric(pred_RT, pred_cls_id, gt_RT, gt_cls_id, models_pts):
    n_cls = cls_num
    cls_add_dis = [list() for i in range(n_cls)]
    cls_adds_dis = [list() for i in range(n_cls)]
    if pred_cls_id[0] == 0 or pred_cls_id[0] != gt_cls_id[0]:
        return cls_add_dis, cls_adds_dis

    pred_RT = torch.from_numpy(pred_RT.astype(np.float32)).cuda()
    gt_RT = torch.from_numpy(gt_RT.astype(np.float32)).cuda()
    mesh_pts = torch.from_numpy(models_pts.astype(np.float32)).cuda()
    add = cal_add_cuda(pred_RT, gt_RT, mesh_pts)
    adds = cal_adds_cuda(pred_RT, gt_RT, mesh_pts)
    cls_add_dis[pred_cls_id[0]].append(add.item())
    cls_adds_dis[pred_cls_id[0]].append(adds.item())
    cls_add_dis[0].append(add.item())
    cls_adds_dis[0].append(adds.item())

    return cls_add_dis, cls_adds_dis


def cal_add_cuda(pred_RT, gt_RT, p3ds):
    _, N = p3ds.size()
    pred_p3ds = torch.mm(pred_RT[:, :3], p3ds) + pred_RT[:, 3].view(3, 1).repeat(1, N)
    gt_p3ds = torch.mm(gt_RT[:, :3], p3ds) + gt_RT[:, 3].view(3, 1).repeat(1, N)
    dis = torch.norm(pred_p3ds - gt_p3ds, dim=0)
    return torch.mean(dis)


def cal_adds_cuda(pred_RT, gt_RT, p3ds):
    _, N = p3ds.size()
    pd = torch.mm(pred_RT[:, :3], p3ds) + pred_RT[:, 3].view(3, 1).repeat(1, N)
    pd = pd.view(1, 3, N).repeat(N, 1, 1).permute(2, 1, 0)
    gt = torch.mm(gt_RT[:, :3], p3ds) + gt_RT[:, 3].view(3, 1).repeat(1, N)
    gt = gt.view(1, 3, N).repeat(N, 1, 1)
    dis = torch.norm(pd - gt, dim=1)
    mdis = torch.min(dis, dim=1)[0]
    return torch.mean(mdis)


def cal_auc(add_dis, max_dis=0.1, thr_m=0.02):
    # A partial validation/probe can legitimately omit classes.  NaN marks
    # them as unobserved so callers can use nanmean instead of treating an
    # absent class as either a failure or a perfect score.
    if len(add_dis) == 0:
        return float('nan'), float('nan')
    D = np.array(add_dis)
    D[np.where(D > max_dis)] = np.inf
    D = np.sort(D)
    n = len(add_dis)
    acc = np.cumsum(np.ones((1, n)), dtype=np.float32) / n
    aps = VOCap(D, acc)
    add_t_cm = np.where(D < thr_m)[0].size / D.size
    return aps * 100, add_t_cm * 100


def VOCap(rec, prec):
    idx = np.where(rec != np.inf)
    if len(idx[0]) == 0:
        return 0
    rec = rec[idx]
    prec = prec[idx]
    mrec = np.array([0.0] + list(rec) + [0.1])
    mpre = np.array([0.0] + list(prec) + [prec[-1]])
    for i in range(1, prec.shape[0]):
        mpre[i] = max(mpre[i], mpre[i-1])
    i = np.where(mrec[1:] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[i] - mrec[i-1]) * mpre[i]) * 10
    return ap

