from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

# Each per-frame list: (id, ltwh)
BoxList = List[Tuple[int, np.ndarray]]


def _iou_ltwh(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, aw, ah = map(float, a)
    bx1, by1, bw, bh = map(float, b)
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    xA, yA = max(ax1, bx1), max(ay1, by1)
    xB, yB = min(ax2, bx2), min(ay2, by2)
    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(0.0, aw) * max(0.0, ah)
    area_b = max(0.0, bw) * max(0.0, bh)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _match_frame(gt: BoxList, pr: BoxList, iou_thr: float) -> Tuple[List[Tuple[int, int, float]], float]:
    """Match GT and Pred in one frame with Hungarian maximizing IoU."""
    if not gt or not pr:
        return [], 0.0
    iou_mat = np.zeros((len(gt), len(pr)), dtype=np.float32)
    for i, (_, gbox) in enumerate(gt):
        for j, (_, pbox) in enumerate(pr):
            iou_mat[i, j] = _iou_ltwh(gbox, pbox)

    # Hungarian on max IoU => min cost = -IoU
    row_ind, col_ind = linear_sum_assignment(-iou_mat)
    matches = []
    sum_iou = 0.0
    for r, c in zip(row_ind, col_ind):
        iou = float(iou_mat[r, c])
        if iou >= iou_thr:
            matches.append((r, c, iou))
            sum_iou += iou
    return matches, sum_iou


@dataclass
class MOTMetrics:
    # MOTP is mean IoU over matched pairs. None means "no matched pairs" (undefined).
    motp: Optional[float]
    # IDF1/IDP/IDR are None when both GT and Pred are empty (undefined but 'perfect empty').
    idf1: Optional[float]
    idp: Optional[float]
    idr: Optional[float]
    idtp: int
    idfp: int
    idfn: int
    total_gt: int
    total_pred: int
    total_matches: int


def compute_motp_idf1_simple(
    gt_frames: List[BoxList],
    pred_frames: List[BoxList],
    *,
    iou_thr: float = 0.5,
) -> MOTMetrics:
    """Compute MOTP (avg IoU over matched pairs) and IDF1.

    - Frame-wise matching uses Hungarian maximizing IoU with threshold iou_thr.
    - IDF1 uses global one-to-one mapping between GT IDs and Pred IDs maximizing matched-frame counts.

    Notes on edge cases:
    - If there are no matched pairs at all, MOTP is undefined -> motp=None.
    - If both GT and Pred are completely empty over all frames, IDF1/IDP/IDR are undefined -> None.
    - If only one side is empty, IDF1 is 0.0 (all FN or all FP).
    """
    n = min(len(gt_frames), len(pred_frames))
    gt_frames = gt_frames[:n]
    pred_frames = pred_frames[:n]

    total_matches = 0
    sum_iou = 0.0

    # Collect co-occurrence counts between (gt_id, pred_id)
    gt_ids = set()
    pr_ids = set()
    gt_count: Dict[int, int] = {}
    pr_count: Dict[int, int] = {}
    pair_count: Dict[Tuple[int, int], int] = {}

    for t in range(n):
        gt = gt_frames[t]
        pr = pred_frames[t]
        for gid, _ in gt:
            gt_ids.add(gid)
            gt_count[gid] = gt_count.get(gid, 0) + 1
        for pid, _ in pr:
            pr_ids.add(pid)
            pr_count[pid] = pr_count.get(pid, 0) + 1

        matches, frame_sum_iou = _match_frame(gt, pr, iou_thr)
        sum_iou += frame_sum_iou
        total_matches += len(matches)

        for r, c, _iou in matches:
            gid = gt[r][0]
            pid = pr[c][0]
            pair_count[(gid, pid)] = pair_count.get((gid, pid), 0) + 1

    total_gt = sum(gt_count.values())
    total_pred = sum(pr_count.values())

    motp: Optional[float] = (sum_iou / total_matches) if total_matches > 0 else None

    # Handle empty cases for ID metrics
    if total_gt == 0 and total_pred == 0:
        return MOTMetrics(
            motp=motp,
            idf1=None, idp=None, idr=None,
            idtp=0, idfp=0, idfn=0,
            total_gt=0, total_pred=0, total_matches=total_matches,
        )
    if total_gt == 0 and total_pred > 0:
        return MOTMetrics(
            motp=motp,
            idf1=0.0, idp=0.0, idr=None,
            idtp=0, idfp=total_pred, idfn=0,
            total_gt=0, total_pred=total_pred, total_matches=total_matches,
        )
    if total_gt > 0 and total_pred == 0:
        return MOTMetrics(
            motp=motp,
            idf1=0.0, idp=None, idr=0.0,
            idtp=0, idfp=0, idfn=total_gt,
            total_gt=total_gt, total_pred=0, total_matches=total_matches,
        )

    # IDF1: max assignment on pair_count
    gt_list = sorted(list(gt_ids))
    pr_list = sorted(list(pr_ids))

    M = np.zeros((len(gt_list), len(pr_list)), dtype=np.int32)
    for i, gid in enumerate(gt_list):
        for j, pid in enumerate(pr_list):
            M[i, j] = pair_count.get((gid, pid), 0)

    # Maximize sum M => minimize cost = -M
    row_ind, col_ind = linear_sum_assignment(-M)
    idtp = int(sum(M[r, c] for r, c in zip(row_ind, col_ind)))

    # Using standard IDF1 count-based definition:
    # IDFP = total_pred - IDTP; IDFN = total_gt - IDTP
    idfp = int(total_pred - idtp)
    idfn = int(total_gt - idtp)

    denom_f1 = (2 * idtp + idfp + idfn)
    idf1 = float((2 * idtp) / denom_f1) if denom_f1 > 0 else None
    idp = float(idtp / (idtp + idfp)) if (idtp + idfp) > 0 else None
    idr = float(idtp / (idtp + idfn)) if (idtp + idfn) > 0 else None

    return MOTMetrics(
        motp=motp,
        idf1=idf1,
        idp=idp,
        idr=idr,
        idtp=idtp,
        idfp=idfp,
        idfn=idfn,
        total_gt=total_gt,
        total_pred=total_pred,
        total_matches=total_matches,
    )


def pretty_float(v: Optional[float], fmt: str = ".4f") -> str:
    """Helper: format Optional[float] for logs."""
    if v is None:
        return "NA"
    return format(v, fmt)



def compute_motp_idf1(
    gt_frames: List[BoxList],
    pred_frames: List[BoxList],
    *,
    iou_thr: float = 0.5,
    backend: str = "simple",
) -> MOTMetrics:
    """Compute MOTP/IDF1/IDP/IDR for a (GT, pred) sequence.

    Backends:
      - simple: built-in implementation (Hungarian IoU matching per frame + global ID mapping).
      - motmetrics: use the 3rd-party `motmetrics` library (if installed). This is useful for
        external validation / authority in papers. If the library is missing, this will raise
        a clear ImportError with installation hints.

    Notes:
      - For IoU-based evaluation, we report MOTP as mean IoU over matched pairs (MOTChallenge-style).
        motmetrics internally reports 'motp' as average *distance*; with IoU distance = 1 - IoU,
        we convert back via (1 - motp_distance).
    """
    backend = str(backend or "simple").strip().lower()
    if backend in {"simple", "builtin"}:
        return compute_motp_idf1_simple(gt_frames, pred_frames, iou_thr=iou_thr)

    if backend in {"motmetrics", "mm"}:
        try:
            import numpy as np
            import motmetrics as mm  # type: ignore
        except Exception as e:
            raise ImportError(
                "metrics-backend=motmetrics requested, but 'motmetrics' is not installed. "
                "Install it on your machine with:  pip install motmetrics"
            ) from e

        acc = mm.MOTAccumulator(auto_id=True)
        iou_thr_f = float(iou_thr)

        # motmetrics uses distances; for IoU, distance = 1 - IoU. Allow assignments only if IoU >= iou_thr.
        max_iou_dist = 1.0 - iou_thr_f

        n = min(len(gt_frames), len(pred_frames))
        for t in range(n):
            gt = gt_frames[t]
            pr = pred_frames[t]
            gt_ids = [int(gid) for gid, _ in gt]
            pr_ids = [int(pid) for pid, _ in pr]
            gt_boxes = np.array([ltwh for _, ltwh in gt], dtype=float) if gt else np.zeros((0, 4), dtype=float)
            pr_boxes = np.array([ltwh for _, ltwh in pr], dtype=float) if pr else np.zeros((0, 4), dtype=float)

            dist = mm.distances.iou_matrix(gt_boxes, pr_boxes, max_iou=max_iou_dist)  # expects ltwh
            acc.update(gt_ids, pr_ids, dist)

        mh = mm.metrics.create()
        res = mh.compute(
            acc,
            metrics=[
                "motp", "idf1", "idp", "idr",
                "num_matches", "num_objects", "num_predictions",
                "idtp", "idfp", "idfn",
            ],
            name="seq",
        )

        motp_dist = float(res.loc["seq", "motp"]) if not np.isnan(res.loc["seq", "motp"]) else None
        motp_iou = (1.0 - motp_dist) if motp_dist is not None else None

        def _f(key: str) -> Optional[float]:
            v = res.loc["seq", key]
            return float(v) if not np.isnan(v) else None

        def _i(key: str) -> int:
            v = res.loc["seq", key]
            return int(v) if not np.isnan(v) else 0

        return MOTMetrics(
            motp=motp_iou,
            idf1=_f("idf1"),
            idp=_f("idp"),
            idr=_f("idr"),
            idtp=_i("idtp"),
            idfp=_i("idfp"),
            idfn=_i("idfn"),
            total_gt=_i("num_objects"),
            total_pred=_i("num_predictions"),
            total_matches=_i("num_matches"),
        )

    raise ValueError(f"Unknown metrics backend: {backend!r}. Expected one of: simple, motmetrics")
