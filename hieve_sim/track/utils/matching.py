"""Minimal matching utilities for ByteTrack (no ultralytics dependency).

The original Ultralytics implementation includes additional metrics and LAPJV support.
For this simulation we only need:
  - iou_distance(tracks, detections)  -> cost matrix (1 - IoU)
  - fuse_score(cost, detections)      -> optional cost adjustment
  - linear_assignment(cost, thresh)   -> Hungarian + threshold
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

def _bbox_iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    """IoU between two boxes in xyxy."""
    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[2], b[2])
    yB = min(a[3], b[3])
    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0

def iou_distance(tracks, detections) -> np.ndarray:
    """Return cost matrix = 1 - IoU (smaller is better)."""
    if len(tracks) == 0 or len(detections) == 0:
        return np.zeros((len(tracks), len(detections)), dtype=np.float32)

    cost = np.ones((len(tracks), len(detections)), dtype=np.float32)
    for i, t in enumerate(tracks):
        ta = np.asarray(t.xyxy, dtype=np.float32)
        for j, d in enumerate(detections):
            db = np.asarray(d.xyxy, dtype=np.float32)
            iou = _bbox_iou_xyxy(ta, db)
            cost[i, j] = 1.0 - iou
    return cost

def fuse_score(cost: np.ndarray, detections) -> np.ndarray:
    """Optionally fuse detection score into cost.

    In Ultralytics, fusing score helps prefer high-confidence detections.
    Here we implement a light version: cost = cost * (2 - score).
    """
    if cost.size == 0:
        return cost
    scores = np.asarray([float(d.score) for d in detections], dtype=np.float32)
    # scale in [1,2] when score in [1,0]
    scale = (2.0 - scores).reshape(1, -1)
    return cost * scale

def linear_assignment(cost_matrix: np.ndarray, thresh: float, use_lap: bool = False):
    """Hungarian assignment with threshold on cost.

    Args:
        cost_matrix: shape (N,M), smaller is better.
        thresh: accept match if cost <= thresh
    Returns:
        matches: np.ndarray shape (K,2) of (row,col)
        unmatched_a: np.ndarray rows not matched
        unmatched_b: np.ndarray cols not matched
    """
    n, m = cost_matrix.shape
    if n == 0 or m == 0:
        return np.zeros((0, 2), dtype=int), np.arange(n, dtype=int), np.arange(m, dtype=int)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches = []
    matched_rows = set()
    matched_cols = set()
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] <= thresh:
            matches.append((r, c))
            matched_rows.add(r)
            matched_cols.add(c)

    unmatched_a = np.array([i for i in range(n) if i not in matched_rows], dtype=int)
    unmatched_b = np.array([j for j in range(m) if j not in matched_cols], dtype=int)
    return np.asarray(matches, dtype=int), unmatched_a, unmatched_b
