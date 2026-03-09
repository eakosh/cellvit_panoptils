import numpy as np


def gen_instance_hv_map(inst_map: np.ndarray) -> np.ndarray:
    """generate horizontal-vertical (HV) maps from an instance segmentation map
    [-1; 1]
    """
    orig = inst_map.astype(np.int32)
    H, W = orig.shape
    x_map = np.zeros((H, W), dtype=np.float32)
    y_map = np.zeros((H, W), dtype=np.float32)

    rows, cols = np.where(orig > 0)
    if len(rows) == 0:
        return np.stack([x_map, y_map], axis=0)

    ids = orig[rows, cols]
    max_id = int(ids.max())

    # compute per-instance centroids 
    counts = np.bincount(ids, minlength=max_id + 1).astype(np.float64)
    sum_rows = np.bincount(ids, weights=rows.astype(np.float64), minlength=max_id + 1)
    sum_cols = np.bincount(ids, weights=cols.astype(np.float64), minlength=max_id + 1)

    counts_safe = np.where(counts > 0, counts, 1.0)
    cy = (sum_rows / counts_safe)[ids].astype(np.float32)
    cx = (sum_cols / counts_safe)[ids].astype(np.float32)

    # relative coordinates for each pixel
    xr = cols.astype(np.float32) - cx
    yr = rows.astype(np.float32) - cy

    
    xr_neg_scale = np.zeros(max_id + 1, dtype=np.float32)
    xr_pos_scale = np.zeros(max_id + 1, dtype=np.float32)
    yr_neg_scale = np.zeros(max_id + 1, dtype=np.float32)
    yr_pos_scale = np.zeros(max_id + 1, dtype=np.float32)

    neg_x = xr < 0
    pos_x = xr > 0
    neg_y = yr < 0
    pos_y = yr > 0

    if np.any(neg_x):
        np.maximum.at(xr_neg_scale, ids[neg_x], -xr[neg_x])
    if np.any(pos_x):
        np.maximum.at(xr_pos_scale, ids[pos_x], xr[pos_x])
    if np.any(neg_y):
        np.maximum.at(yr_neg_scale, ids[neg_y], -yr[neg_y])
    if np.any(pos_y):
        np.maximum.at(yr_pos_scale, ids[pos_y], yr[pos_y])

    # normalize to [-1, 1]
    xr_norm = xr.copy()
    if np.any(neg_x):
        scale = xr_neg_scale[ids[neg_x]]
        xr_norm[neg_x] = np.where(scale > 0, xr[neg_x] / scale, 0.0)
    if np.any(pos_x):
        scale = xr_pos_scale[ids[pos_x]]
        xr_norm[pos_x] = np.where(scale > 0, xr[pos_x] / scale, 0.0)
    
    yr_norm = yr.copy()
    if np.any(neg_y):
        scale = yr_neg_scale[ids[neg_y]]
        yr_norm[neg_y] = np.where(scale > 0, yr[neg_y] / scale, 0.0)
    if np.any(pos_y):
        scale = yr_pos_scale[ids[pos_y]]
        yr_norm[pos_y] = np.where(scale > 0, yr[pos_y] / scale, 0.0)

    x_map[rows, cols] = xr_norm
    y_map[rows, cols] = yr_norm

    return np.stack([x_map, y_map], axis=0)
