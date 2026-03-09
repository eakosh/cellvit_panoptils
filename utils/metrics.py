import numpy as np
from typing import Dict, List, Tuple
from scipy.optimize import linear_sum_assignment


def _get_instance_bboxes(instance_map: np.ndarray, ids: np.ndarray):
    H, W = instance_map.shape
    max_id = int(ids.max()) if len(ids) > 0 else 0
    row_min = np.full(max_id + 1, H, dtype=np.int32)
    row_max = np.full(max_id + 1, -1, dtype=np.int32)
    col_min = np.full(max_id + 1, W, dtype=np.int32)
    col_max = np.full(max_id + 1, -1, dtype=np.int32)

    rows, cols = np.where(instance_map > 0)
    vals = instance_map[rows, cols]
    np.minimum.at(row_min, vals, rows)
    np.maximum.at(row_max, vals, rows)
    np.minimum.at(col_min, vals, cols)
    np.maximum.at(col_max, vals, cols)

    return row_min, row_max, col_min, col_max

def _bbox_overlap(r1_min, r1_max, c1_min, c1_max, r2_min, r2_max, c2_min, c2_max):
    return not (r1_max < r2_min or r2_max < r1_min or c1_max < c2_min or c2_max < c1_min)


def _compute_centroids(instance_map: np.ndarray, ids: np.ndarray) -> Dict[int, np.ndarray]:
    centroids = {}
    for inst_id in ids:
        rows, cols = np.where(instance_map == inst_id)
        if len(rows) > 0:
            centroids[int(inst_id)] = np.array([rows.mean(), cols.mean()])
    return centroids


def _get_instance_types(
    instance_map: np.ndarray,
    type_map: np.ndarray,
    ids: np.ndarray,
    exclude_labels: Tuple[int, ...] = (0,),
) -> Dict[int, int]:
    types = {}
    for inst_id in ids:
        mask = instance_map == inst_id
        if mask.any():
            vals = type_map[mask].astype(np.int64)
            valid = vals[~np.isin(vals, exclude_labels)]
            if len(valid) > 0:
                types[int(inst_id)] = int(np.bincount(valid).argmax())
            else:
                types[int(inst_id)] = int(np.bincount(vals).argmax())
    return types


def match_instances_by_centroid(
    pred_instances: np.ndarray,
    gt_instances: np.ndarray,
    radius: float = 12.0,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Args:
        pred_instances: predicted instance map [H, W]
        gt_instances:   ground-truth instance map [H, W]
        radius:         maximum centroid distance for a valid match

    Returns:
        matched_pairs:  list of (pred_id, gt_id) tuples
        unmatched_pred: list of unmatched predicted instance ids (FP)
        unmatched_gt:   list of unmatched GT instance ids (FN)
    """
    pred_ids = np.unique(pred_instances[pred_instances > 0])
    gt_ids   = np.unique(gt_instances[gt_instances > 0])

    if len(pred_ids) == 0 and len(gt_ids) == 0:
        return [], [], []
    if len(pred_ids) == 0:
        return [], [], list(gt_ids)
    if len(gt_ids) == 0:
        return [], list(pred_ids), []

    pred_cents = _compute_centroids(pred_instances, pred_ids)
    gt_cents   = _compute_centroids(gt_instances,   gt_ids)

    # distance matrix [n_pred × n_gt]
    dist = np.full((len(pred_ids), len(gt_ids)), np.inf)
    for i, pid in enumerate(pred_ids):
        if pid not in pred_cents:
            continue
        for j, gid in enumerate(gt_ids):
            if gid not in gt_cents:
                continue
            dist[i, j] = np.linalg.norm(pred_cents[pid] - gt_cents[gid])

    pred_idx, gt_idx = linear_sum_assignment(dist)

    matched_pairs = []
    matched_pred_set: set = set()
    matched_gt_set: set = set()
    for pi, gi in zip(pred_idx, gt_idx):
        if dist[pi, gi] < radius:
            matched_pairs.append((int(pred_ids[pi]), int(gt_ids[gi])))
            matched_pred_set.add(int(pred_ids[pi]))
            matched_gt_set.add(int(gt_ids[gi]))

    unmatched_pred = [int(p) for p in pred_ids if p not in matched_pred_set]
    unmatched_gt   = [int(g) for g in gt_ids   if g not in matched_gt_set]

    return matched_pairs, unmatched_pred, unmatched_gt


def match_instances(
    pred_instances: np.ndarray,
    gt_instances: np.ndarray,
    iou_threshold: float = 0.5
) -> Tuple[List[Tuple[int, int]], List[int], List[int], List[float]]:
    """
    Args:
        pred_instances: predicted instance map [H, W]
        gt_instances: ground truth instance map [H, W]
        iou_threshold: IoU threshold for matching

    Returns:
        matched_pairs: List of (pred_id, gt_id) tuples
        unmatched_pred: List of unmatched predicted instance IDs
        unmatched_gt: List of unmatched ground truth instance IDs
        matched_ious: List of IoU values for each matched pair
    """
    pred_ids = np.unique(pred_instances)
    pred_ids = pred_ids[pred_ids != 0]

    gt_ids = np.unique(gt_instances)
    gt_ids = gt_ids[gt_ids != 0]

    if len(pred_ids) == 0 or len(gt_ids) == 0:
        return [], list(pred_ids), list(gt_ids), []

    # precompute bounding boxes
    p_rmin, p_rmax, p_cmin, p_cmax = _get_instance_bboxes(pred_instances, pred_ids)
    g_rmin, g_rmax, g_cmin, g_cmax = _get_instance_bboxes(gt_instances, gt_ids)

    # compute IoU matrix
    iou_matrix = np.zeros((len(pred_ids), len(gt_ids)))
    for i, pred_id in enumerate(pred_ids):
        pr1, pr2, pc1, pc2 = p_rmin[pred_id], p_rmax[pred_id], p_cmin[pred_id], p_cmax[pred_id]
        for j, gt_id in enumerate(gt_ids):
            if not _bbox_overlap(pr1, pr2, pc1, pc2,
                                 g_rmin[gt_id], g_rmax[gt_id], g_cmin[gt_id], g_cmax[gt_id]):
                continue

            r1 = min(pr1, g_rmin[gt_id])
            r2 = max(pr2, g_rmax[gt_id]) + 1
            c1 = min(pc1, g_cmin[gt_id])
            c2 = max(pc2, g_cmax[gt_id]) + 1
            p_crop = pred_instances[r1:r2, c1:c2] == pred_id
            g_crop = gt_instances[r1:r2, c1:c2] == gt_id
            intersection = np.logical_and(p_crop, g_crop).sum()
            union = np.logical_or(p_crop, g_crop).sum()
            iou_matrix[i, j] = intersection / union if union > 0 else 0.0

    if iou_threshold >= 0.5:
        # for iou >= 0.5, matched pairs are provably unique
        iou_matrix_filtered = iou_matrix.copy()
        iou_matrix_filtered[iou_matrix_filtered <= iou_threshold] = 0.0
        pred_indices, gt_indices = np.nonzero(iou_matrix_filtered)
        matched_ious = iou_matrix[pred_indices, gt_indices].tolist()
    else:
        # for iou < 0.5, pairs may not be unique — Hungarian matching
        pred_indices, gt_indices = linear_sum_assignment(-iou_matrix)
        iou_vals = iou_matrix[pred_indices, gt_indices]
        valid = iou_vals >= iou_threshold
        pred_indices = pred_indices[valid]
        gt_indices = gt_indices[valid]
        matched_ious = iou_vals[valid].tolist()

    matched_pairs = [(pred_ids[pi], gt_ids[gi]) for pi, gi in zip(pred_indices, gt_indices)]
    matched_pred_set = {pred_ids[pi] for pi in pred_indices}
    matched_gt_set = {gt_ids[gi] for gi in gt_indices}

    unmatched_pred = [pid for pid in pred_ids if pid not in matched_pred_set]
    unmatched_gt = [gid for gid in gt_ids if gid not in matched_gt_set]

    return matched_pairs, unmatched_pred, unmatched_gt, matched_ious


def compute_pq(
    pred_instances: np.ndarray,
    gt_instances: np.ndarray,
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute Panoptic Quality (PQ) metric

    PQ = SQ * DQ, where:
        - SQ (Segmentation Quality) = average IoU of matched pairs
        - DQ (Detection Quality) = TP / (TP + 0.5*FP + 0.5*FN)

    Args:
        pred_instances: predicted instance map [H, W]
        gt_instances: ground truth instance map [H, W]
        iou_threshold: IoU threshold for matching

    Returns:
        Dictionary with pq, sq, rq (dq), f1, precision, recall
    """
    matched_pairs, unmatched_pred, unmatched_gt, matched_ious = match_instances(
        pred_instances, gt_instances, iou_threshold
    )

    TP = len(matched_pairs)
    FP = len(unmatched_pred)
    FN = len(unmatched_gt)

    if TP == 0:
        return {
            'pq': 0.0,
            'sq': 0.0,
            'rq': 0.0,
            'f1': 0.0,
            'precision': 0.0,
            'recall': 0.0
        }

    iou_sum = sum(matched_ious)

    sq = iou_sum / TP
    rq = TP / (TP + 0.5 * FP + 0.5 * FN)
    pq = sq * rq

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0.0

    return {
        'pq': pq,
        'sq': sq,
        'rq': rq,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def compute_dice_coefficient(pred: np.ndarray, gt: np.ndarray) -> float:
    intersection = np.logical_and(pred, gt).sum()
    pred_sum = pred.sum()
    gt_sum = gt.sum()

    if pred_sum + gt_sum == 0:
        return 1.0

    dice = (2.0 * intersection) / (pred_sum + gt_sum)
    return dice


class MetricsAggregator:
    """
    Aggregator for validation metrics across batches

    Tracks:
    - PQ / SQ / DQ  
    - per-class PQ  
    - det_f1 / det_precision / det_recall 
    - per-class cls_f1  
    """
    # weighting vector [w_FPc, w_FNc, w_FPd, w_FNd] 
    _W = [2, 2, 1, 1]

    def __init__(
        self,
        num_classes: int = 0,
        unlabeled_class: int = None,
        background_class: int = None,
        centroid_radius: float = 12.0,
    ):
        self.num_classes = num_classes
        self.unlabeled_class = unlabeled_class
        self.background_class = background_class
        self.centroid_radius = centroid_radius
        self.reset()

    def reset(self):
        self.pq_scores = []
        self.sq_scores = []
        self.rq_scores = []
        self.f1_scores = []
        self.precision_scores = []
        self.recall_scores = []
        self.dice_scores = []

        _skip = {c for c in [self.unlabeled_class, self.background_class] if c is not None}

        self.per_class_pq: Dict[int, List[float]] = {
            c: [] for c in range(1, self.num_classes) if c not in _skip
        } if self.num_classes > 0 else {}

        self.det_f1_scores: List[float] = []
        self.det_precision_scores: List[float] = []
        self.det_recall_scores: List[float] = []

        self.per_class_det: Dict[int, List[float]] = {
            c: [] for c in range(1, self.num_classes) if c not in _skip
        } if self.num_classes > 0 else {}

    def update(
        self,
        pred_instances: np.ndarray,
        gt_instances: np.ndarray,
        pred_binary: np.ndarray = None,
        gt_binary: np.ndarray = None,
        pred_type_map: np.ndarray = None,
        gt_type_map: np.ndarray = None,
    ):
        pq_metrics = compute_pq(pred_instances, gt_instances)
        self.pq_scores.append(pq_metrics['pq'])
        self.sq_scores.append(pq_metrics['sq'])
        self.rq_scores.append(pq_metrics['rq'])
        self.f1_scores.append(pq_metrics['f1'])
        self.precision_scores.append(pq_metrics['precision'])
        self.recall_scores.append(pq_metrics['recall'])

        if self.num_classes > 0 and pred_type_map is not None and gt_type_map is not None:
            for c in self.per_class_pq:
                gt_class_instances = gt_instances.copy()
                gt_class_instances[gt_type_map != c] = 0

                if not np.any(gt_class_instances > 0):
                    continue

                pred_class_instances = pred_instances.copy()
                pred_class_instances[pred_type_map != c] = 0

                class_metrics = compute_pq(pred_class_instances, gt_class_instances)
                self.per_class_pq[c].append(class_metrics['pq'])

        if pred_binary is not None and gt_binary is not None:
            self.dice_scores.append(compute_dice_coefficient(pred_binary, gt_binary))

        if pred_type_map is not None and gt_type_map is not None:
            matched_pairs, unmatched_pred, unmatched_gt = match_instances_by_centroid(
                pred_instances, gt_instances, self.centroid_radius
            )

            TP_d = len(matched_pairs)
            FP_d = len(unmatched_pred)
            FN_d = len(unmatched_gt)

            det_f1 = (
                2 * TP_d / (2 * TP_d + FP_d + FN_d)
                if (2 * TP_d + FP_d + FN_d) > 0 else 0.0
            )
            det_prec = TP_d / (TP_d + FP_d) if (TP_d + FP_d) > 0 else 0.0
            det_rec  = TP_d / (TP_d + FN_d) if (TP_d + FN_d) > 0 else 0.0
            self.det_f1_scores.append(det_f1)
            self.det_precision_scores.append(det_prec)
            self.det_recall_scores.append(det_rec)

            if self.per_class_det:
                pred_ids_all = np.unique(pred_instances[pred_instances > 0])
                gt_ids_all   = np.unique(gt_instances[gt_instances > 0])

                _excl = tuple(c for c in [self.unlabeled_class, self.background_class] if c is not None)
                pred_types = _get_instance_types(pred_instances, pred_type_map, pred_ids_all, exclude_labels=_excl)
                gt_types   = _get_instance_types(gt_instances,   gt_type_map,   gt_ids_all,   exclude_labels=_excl)

                paired_pred_t = np.array(
                    [pred_types.get(p, 0) for p, _ in matched_pairs], dtype=np.int32
                )
                paired_gt_t = np.array(
                    [gt_types.get(g, 0) for _, g in matched_pairs], dtype=np.int32
                )

                unpaired_pred_t = np.array(
                    [pred_types.get(p, 0) for p in unmatched_pred], dtype=np.int32
                )
                unpaired_gt_t = np.array(
                    [gt_types.get(g, 0) for g in unmatched_gt], dtype=np.int32
                )

                for c in self.per_class_det:
                    if not np.any(gt_type_map == c):
                        continue

                    if TP_d > 0:
                        sel = (paired_gt_t == c) | (paired_pred_t == c)
                        pt = paired_gt_t[sel]
                        pp = paired_pred_t[sel]
                        tp_c = int(((pt == c) & (pp == c)).sum())
                        fp_c = int(((pt != c) & (pp == c)).sum())  # wrong cls in matched
                        fn_c = int(((pt == c) & (pp != c)).sum())  # missed cls in matched
                    else:
                        tp_c = fp_c = fn_c = 0

                    # unmatched instances of class c
                    fp_d = int((unpaired_pred_t == c).sum()) if len(unpaired_pred_t) > 0 else 0
                    fn_d = int((unpaired_gt_t   == c).sum()) if len(unpaired_gt_t)   > 0 else 0

                    w = self._W
                    denom = 2 * tp_c + w[0] * fp_c + w[1] * fn_c + w[2] * fp_d + w[3] * fn_d
                    if denom == 0:
                        continue
                    self.per_class_det[c].append(2 * tp_c / denom)

    def compute(self) -> Dict[str, float]:
        metrics = {
            'pq': np.mean(self.pq_scores) if self.pq_scores else 0.0,
            'sq': np.mean(self.sq_scores) if self.sq_scores else 0.0,
            'rq': np.mean(self.rq_scores) if self.rq_scores else 0.0,
            'f1': np.mean(self.f1_scores) if self.f1_scores else 0.0,
            'precision': np.mean(self.precision_scores) if self.precision_scores else 0.0,
            'recall': np.mean(self.recall_scores) if self.recall_scores else 0.0,
        }

        if self.dice_scores:
            metrics['dice'] = np.mean(self.dice_scores)

        if self.per_class_pq:
            class_pqs = []
            for c, scores in self.per_class_pq.items():
                if scores:
                    avg = np.mean(scores)
                    metrics[f'pq_class_{c}'] = avg
                    class_pqs.append(avg)
            if class_pqs:
                metrics['pq_class_avg'] = np.mean(class_pqs)

        if self.det_f1_scores:
            metrics['det_f1'] = np.mean(self.det_f1_scores)
            metrics['det_precision'] = np.mean(self.det_precision_scores)
            metrics['det_recall'] = np.mean(self.det_recall_scores)

        if self.per_class_det:
            cls_f1s = []
            for c, scores in self.per_class_det.items():
                if scores:
                    avg = np.mean(scores)
                    metrics[f'cls_f1_class_{c}'] = avg
                    cls_f1s.append(avg)
            if cls_f1s:
                metrics['cls_f1_avg'] = np.mean(cls_f1s)

        return metrics

    def __repr__(self):
        m = self.compute()
        parts = [
            f"bPQ={m['pq']:.4f}  SQ={m['sq']:.4f}  DQ={m['rq']:.4f}",
        ]
        if 'pq_class_avg' in m:
            parts.append(f"mPQ={m['pq_class_avg']:.4f}")
        if 'det_f1' in m:
            parts.append(
                f"F1_d={m['det_f1']:.4f}  P_d={m['det_precision']:.4f}  R_d={m['det_recall']:.4f}"
            )
        if 'cls_f1_avg' in m:
            parts.append(f"F1_c_avg={m['cls_f1_avg']:.4f}")
        if 'dice' in m:
            parts.append(f"Dice={m['dice']:.4f}")
        return "  |  ".join(parts)
