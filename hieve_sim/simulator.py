from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Tuple

from collections import Counter

import numpy as np
import random
import statistics
import math
import json

from scipy.optimize import linear_sum_assignment

from .calc_engine import AISimEngine, CallbackContext
from .hieve_reader import MOTLabelSequence, MOTDetSequence, FrameDet
from .metrics import BoxList, compute_motp_idf1, pretty_float
from .track import BYTETracker
from .track.basetrack import BaseTrack
from .utils import LOGGER


class SimpleResults:
    """A tiny detection container compatible with our BYTETracker implementation."""

    def __init__(self, xywh: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self.xywh = xywh.astype(np.float32, copy=False)  # center-based
        self.conf = conf.astype(np.float32, copy=False)
        self.cls = cls.astype(np.float32, copy=False)

    def __len__(self):
        return int(self.xywh.shape[0])

    def __getitem__(self, item):
        # item can be boolean mask or indices
        return SimpleResults(self.xywh[item], self.conf[item], self.cls[item])


def _empty_results() -> SimpleResults:
    return SimpleResults(
        xywh=np.zeros((0, 4), dtype=np.float32),
        conf=np.zeros((0,), dtype=np.float32),
        cls=np.zeros((0,), dtype=np.float32),
    )


def _det_to_results(frame_det: FrameDet) -> SimpleResults:
    """Convert one-frame detections (score, ltwh) -> SimpleResults."""
    if not frame_det:
        return _empty_results()
    xywh = []
    conf = []
    for score, ltwh in frame_det:
        x, y, w, h = map(float, ltwh)
        xywh.append([x + w / 2.0, y + h / 2.0, w, h])
        conf.append(float(score))
    xywh = np.asarray(xywh, dtype=np.float32)
    conf = np.asarray(conf, dtype=np.float32)
    cls = np.zeros((len(frame_det),), dtype=np.float32)
    return SimpleResults(xywh=xywh, conf=conf, cls=cls)


@dataclass(frozen=True)
class SourceSpec:
    """One simulated video/channel.

    Important project rule (as requested by user):
      - **All detections MUST come from the DET file.**
      - We DO NOT allow using GT as a detection source.

    A channel is considered "active" only when both GT and DET are provided and num_frames>0.
    """
    name: str
    fps: float
    num_frames: int
    gt: Optional[MOTLabelSequence] = None
    det: Optional[MOTDetSequence] = None
    is_empty: bool = False


@dataclass
class ChannelSimResult:
    name: str
    gt_path: Optional[str]
    det_path: Optional[str]
    fps: float
    tick_ms: int
    num_frames: int
    # per frame: list of (pred_id, ltwh)
    pred_frames: List[BoxList]
    # per frame: list of (gt_id, ltwh)
    gt_frames: List[BoxList]
    metrics: dict


@dataclass
class OverallSimResult:
    metrics: dict


def _init_tracker(fps: float, tracker_cfg: Optional[dict]) -> BYTETracker:
    if tracker_cfg is None:
        tracker_cfg = {}
    default_args = dict(
        track_high_thresh=0.6,
        track_low_thresh=0.1,
        new_track_thresh=0.6,
        track_buffer=90,
        match_thresh=0.8,  # cost threshold on 1-IoU
        fuse_score=False,
    )
    default_args.update(tracker_cfg)
    args = SimpleNamespace(**default_args)
    return BYTETracker(args=args, frame_rate=max(1, int(round(fps))))


def _as_gt_frames(gt_seq: Optional[MOTLabelSequence], num_frames: int) -> List[BoxList]:
    out: List[BoxList] = [[] for _ in range(max(0, int(num_frames)))]
    if gt_seq is None:
        return out
    # clamp to min length
    n = min(num_frames, gt_seq.num_frames)
    for i in range(n):
        out[i] = [(gid, ltwh.copy()) for gid, ltwh in gt_seq.frames[i]]
    return out


def run_simulation_sources(
    sources: List[SourceSpec],
    *,
    tick_ms: int = 100,
    # Legacy single IoU threshold. If metric_iou_thr/gate_iou_thr are not provided,
    # they default to this value.
    iou_thr: float = 0.5,
    # IoU threshold used for evaluation metrics (MOTP/IDF1).
    metric_iou_thr: Optional[float] = None,
    # IoU threshold used for dynamic-weight gating (boost/decay/retro).
    gate_iou_thr: Optional[float] = None,
    # Evaluator backend for MOTP/IDF1.
    metrics_backend: str = "simple",
    iou_det_score_thr: Optional[float] = None,
    iou_min_top_pct: float = 0.95,
    tracker_cfg: Optional[dict] = None,
    seed: int = 0,
    verbose_engine: bool = False,
    dynamic_weight: bool = False,
    weight_log_path: Optional[str] = None,
    id_log_path: Optional[str] = None,
    weight_adjust_log_path: Optional[str] = None,
    iou_id_log_path: Optional[str] = None,
    evolution_steps: int = 10,
    retro_fill: bool = False,
    retro_max_gap_ms: int = 200,
    run_mode: Optional[str] = None,
) -> Tuple[List[ChannelSimResult], OverallSimResult]:
    """Run multi-channel sparse scheduling simulation + ByteTrack fill.

    Run modes:
      - static: fixed equal weights, single tracker per channel, no retro-fill.
      - dyn: dynamic weights (IoU feedback), no retro-fill (each progressed callback costs 1 tick).
      - dyn-retro: dynamic weights + retro-fill (extra mid-point DET update on effective boost; costs 2 ticks).

    Implements:
      - Random weighted scheduling P(select ch_i) = weight_i / sum(weight).
      - Per-channel two trackers:
          * full-rate: always uses DET on callback.
          * half-rate: alternates DET usage per callback (read on 1st, skip on 2nd, ...).
            When skipping, we call BYTETracker.PREDICT(n) to advance KF state and output predictions.
      - Dynamic weight adjustment (if enabled) using IoU + Hungarian matching:
          * full minIoU < 0.5  => weight *= 2
          * half minIoU > 0.5  => weight //= 2
        Only when current DET has >0 boxes and the source is not an "empty" channel.
      - Weight change logs (CSV) and metric evolution over time.
      - Optional eval tracker retro-fill (BYTE TRACK C): when a boost is triggered and the
        weight actually increases (i.e. not saturated at max_weight), we inject an extra
        mid-point DET update for the eval tracker, and charge 2 ticks for that callback.

    Returns:
      (per_channel_results, overall_result)
    """

    # Effective thresholds (separate evaluation vs dynamic-gate).
    metric_iou_thr_eff = float(iou_thr if metric_iou_thr is None else metric_iou_thr)
    gate_iou_thr_eff = float(iou_thr if gate_iou_thr is None else gate_iou_thr)

    def _empty_overall() -> OverallSimResult:
        return OverallSimResult(metrics={
            "motp": None, "idf1": None, "idp": None, "idr": None,
            "idtp": 0, "idfp": 0, "idfn": 0,
            "total_gt": 0, "total_pred": 0, "total_matches": 0,
        })

    if not sources:
        return [], _empty_overall()

    np.random.seed(seed)
    random.seed(seed)

    retro_max_gap_ms = int(retro_max_gap_ms) if retro_max_gap_ms is not None else 200
    if retro_max_gap_ms < 1:
        retro_max_gap_ms = 1
    BaseTrack.reset_id()

    # Resolve effective run mode.
    if run_mode is None:
        mode = "dyn-retro" if (dynamic_weight and retro_fill) else ("dyn" if dynamic_weight else "static")
    else:
        mode = str(run_mode).strip().lower()
    if mode not in {"static", "dyn", "dyn-retro"}:
        raise ValueError(f"Invalid run_mode: {run_mode!r}. Expected one of: static, dyn, dyn-retro")
    # Mode is authoritative for these toggles.
    dynamic_weight = (mode != "static")
    retro_fill = (mode == "dyn-retro")

    # A channel is "active" iff it has a time axis (num_frames>0) and has GT+DET sequences.
    # Empty channels are represented by synthetic GT/DET sequences with path starting with "<empty:".
    active_indices = [i for i, s in enumerate(sources) if (s.gt is not None and s.det is not None and s.num_frames > 0)]
    active_sources = [sources[i] for i in active_indices]

    if not active_sources:
        # Nothing to simulate. Still return per-channel stubs + empty overall.
        stubs: List[ChannelSimResult] = []
        for s in sources:
            stubs.append(ChannelSimResult(
                name=s.name,
                gt_path=(s.gt.path if s.gt is not None else None),
                det_path=(s.det.path if s.det is not None else None),
                fps=s.fps,
                tick_ms=tick_ms,
                num_frames=max(0, int(s.num_frames)),
                pred_frames=[[] for _ in range(max(0, int(s.num_frames)))],
                gt_frames=_as_gt_frames(s.gt, max(0, int(s.num_frames))),
                metrics={
                    "motp": None, "idf1": None, "idp": None, "idr": None,
                    "idtp": 0, "idfp": 0, "idfn": 0,
                    "total_gt": 0, "total_pred": 0, "total_matches": 0,
                },
            ))
        return stubs, _empty_overall()

    # Tracker instances.
    #   - static: only an eval tracker per channel.
    #   - dyn/dyn-retro: three trackers per channel:
    #       A full-rate (signal), B half-rate (signal), C eval-output (optional retro-fill).
    if mode == "static":
        full_trackers = []
        half_trackers = []
        eval_trackers = [_init_tracker(src.fps, tracker_cfg) for src in active_sources]
    else:
        full_trackers = [_init_tracker(src.fps, tracker_cfg) for src in active_sources]
        half_trackers = [_init_tracker(src.fps, tracker_cfg) for src in active_sources]
        eval_trackers = [_init_tracker(src.fps, tracker_cfg) for src in active_sources]

    # Initial weights: 1 for all channels (empty channels are forced to stay at 1)
    engine = AISimEngine(
        # Engine only needs one tracker object per channel for bookkeeping; for static we
        # provide the eval trackers, for dynamic we provide the full-rate trackers.
        channel_bytetracks=(eval_trackers if mode == "static" else full_trackers),
        channel_weights=[1] * len(active_sources),
        channel_fps=[src.fps for src in active_sources],
        channel_num_frames=[int(src.num_frames) for src in active_sources],
    )

    # Per-channel state
    last_frame_done = [-1] * len(active_sources)
    half_phase = [0] * len(active_sources)  # 0=>read det, 1=>skip det

    sampled_frames_idx: List[List[int]] = [[] for _ in range(len(active_sources))]

    # Prepare per-channel outputs for ALL sources (including inactive ones)
    pred_frames_all: List[List[BoxList]] = [[[] for _ in range(max(0, int(src.num_frames)))] for src in sources]
    gt_frames_all: List[List[BoxList]] = [_as_gt_frames(src.gt, max(0, int(src.num_frames))) for src in sources]

    # Shortcuts for active storage (views into *_all)
    pred_frames = [pred_frames_all[i] for i in active_indices]
    gt_frames = [gt_frames_all[i] for i in active_indices]

    # total virtual time = **min** duration among channels (project rule)
    total_ms = int(min(((max(0, src.num_frames) / max(1e-6, src.fps)) * 1000.0) for src in active_sources)) if active_sources else 0

    # Weight log rows
    weight_rows: List[dict] = []

    # New GT-ID event log rows (JSONL)
    id_rows: List[dict] = []
    seen_gt_ids: List[set[int]] = [set() for _ in range(len(active_sources))]

    # Detailed weight-adjustment events (JSONL; only when weights attempt to change)
    # NOTE: This can be very large; we stream JSONL to disk to avoid a long "hang" at the end
    # when dumping a huge in-memory list.
    weight_adjust_fp = None
    weight_adjust_cnt = 0

    # Per-channel minIoU timeline with embedded new-id events (JSONL)
    iou_id_fp = None
    iou_id_cnt = 0

    if weight_adjust_log_path:
        p = Path(weight_adjust_log_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        weight_adjust_fp = p.open("w", encoding="utf-8")

    if iou_id_log_path:
        p = Path(iou_id_log_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        iou_id_fp = p.open("w", encoding="utf-8")

    def _scan_new_gt_ids(ch: int, start_f: int, end_f: int) -> List[dict]:
        """Scan and (optionally) record events when a new GT id first appears.

        - We detect first-appearance GT IDs in [start_f, end_f] (inclusive).
        - We ALWAYS update `seen_gt_ids` so later scans won't duplicate events.
        - If `id_log_path` is provided, we also append the events to `id_rows` for the existing new-id log.
        - Returns the list of event dicts (may be empty).
        """
        events: List[dict] = []
        if start_f > end_f:
            return events
        src = active_sources[ch]
        fps = float(src.fps)
        sset = seen_gt_ids[ch]
        lo = max(0, int(start_f))
        hi = min(int(end_f), int(src.num_frames) - 1)
        if hi < lo:
            return events

        for fr in range(lo, hi + 1):
            g = gt_frames[ch][fr] if 0 <= fr < len(gt_frames[ch]) else []
            if not g:
                continue
            new_ids: List[int] = []
            for gid, _box in g:
                gid_i = int(gid)
                if gid_i not in sset:
                    new_ids.append(gid_i)
            if new_ids:
                for nid in new_ids:
                    sset.add(int(nid))
                t_ms = int(round((fr * 1000.0) / max(1e-6, fps)))
                ev = {
                    "t_ms": int(t_ms),
                    "ch": int(ch),
                    "name": src.name,
                    "frame": int(fr),
                    "new_ids": new_ids,
                    "new_cnt": int(len(new_ids)),
                    "gt_box_cnt": int(len(g)),
                }
                events.append(ev)
                if id_log_path is not None:
                    id_rows.append(ev)

        return events

    def _ltwh_to_xyxy(ltwh) -> np.ndarray:
        x, y, w, h = map(float, ltwh)
        return np.asarray([x, y, x + w, y + h], dtype=np.float32)

    def _hungarian_min_iou(
        pred_boxes_xyxy: np.ndarray,
        det_list: FrameDet,
        top_pct: float = 0.95,
    ) -> float:
        """Return a DET-side *robust* min IoU using Hungarian matching.

        Motivation:
          - In dense scenes, 1~2 DET boxes can flicker or appear/disappear between callbacks.
          - The strict min IoU (and especially the `n < m => 0` rule) is too harsh and
            causes persistent boosting even when tracking is generally OK.

        Definition:
          - We compute a one-to-one Hungarian assignment between predictions and DET boxes
            using IoU as affinity (cost = 1 - IoU).
          - We then build a DET-side IoU list of length `m` (number of DETs):
              * matched DET gets its assigned IoU
              * unmatched DET gets IoU = 0
          - Instead of taking the strict minimum, we take the minimum among the TOP `top_pct`
            fraction (default 0.95). Equivalently, we discard the worst `floor((1-top_pct)*m)`
            DET IoUs and take the smallest of the remaining.
          - When the rank is non-integer, we follow the user's rule:
              * e.g. (1-top_pct)*m = 10.3  => discard 10 => choose the 11th smallest (1-based).
            This corresponds to using `floor` on the discard count.

        Args:
          top_pct: in (0, 1]. 1.0 reduces to strict DET-side min (with unmatched=0),
                   0.95 ignores up to ~5% worst DETs.

        Returns:
          A float in [0, 1]. 1 means perfect overlap, 0 means at least ~5% of DETs are
          completely unmatched / non-overlapping.
        """
        m = len(det_list)
        if m <= 0:
            return 1.0

        # Clamp/validate top_pct.
        try:
            tp = float(top_pct)
        except Exception:
            tp = 0.95
        if tp <= 0.0:
            tp = 0.95
        if tp > 1.0:
            tp = 1.0

        if pred_boxes_xyxy is None or len(pred_boxes_xyxy) == 0:
            return 0.0

        det_xyxy = np.stack([_ltwh_to_xyxy(ltwh) for _score, ltwh in det_list], axis=0)
        pred_xyxy = np.asarray(pred_boxes_xyxy, dtype=np.float32)
        if pred_xyxy.ndim != 2 or pred_xyxy.shape[1] < 4:
            return 0.0
        pred_xyxy = pred_xyxy[:, :4]

        n = int(pred_xyxy.shape[0])
        if n <= 0:
            return 0.0

        # IoU matrix (n x m)
        ious = np.zeros((n, m), dtype=np.float32)
        for i in range(n):
            ax1, ay1, ax2, ay2 = map(float, pred_xyxy[i])
            a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
            for j in range(m):
                bx1, by1, bx2, by2 = map(float, det_xyxy[j])
                xA = max(ax1, bx1)
                yA = max(ay1, by1)
                xB = min(ax2, bx2)
                yB = min(ay2, by2)
                inter_w = max(0.0, xB - xA)
                inter_h = max(0.0, yB - yA)
                inter = inter_w * inter_h
                if inter <= 0.0:
                    continue
                b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
                union = a_area + b_area - inter
                ious[i, j] = float(inter / union) if union > 0.0 else 0.0

        # Hungarian assignment
        cost = 1.0 - ious
        row_ind, col_ind = linear_sum_assignment(cost)

        # DET-side IoU list, unmatched DET as 0
        det_ious = [0.0] * m
        for r, c in zip(row_ind, col_ind):
            if 0 <= c < m and 0 <= r < n:
                det_ious[int(c)] = float(ious[int(r), int(c)])

        det_ious.sort()  # ascending

        # Discard worst floor((1 - tp) * m) and take the next (1-based) => idx = discard_count (0-based)
        discard = int(math.floor((1.0 - tp) * m))
        if discard < 0:
            discard = 0
        if discard >= m:
            discard = m - 1
        return float(det_ious[discard])


    def _pred_arr_to_boxes(pred_arr) -> List[dict]:
        """Convert tracker prediction arrays (Nx8-ish) into JSON-friendly dicts."""
        if pred_arr is None:
            return []
        arr = np.asarray(pred_arr)
        if arr.size == 0:
            return []
        if arr.ndim != 2:
            return []
        out: List[dict] = []
        for row in arr:
            row = np.asarray(row).tolist()
            if len(row) < 4:
                continue
            d = {
                "x1": float(row[0]),
                "y1": float(row[1]),
                "x2": float(row[2]),
                "y2": float(row[3]),
            }
            if len(row) >= 5:
                try:
                    d["tid"] = int(row[4])
                except Exception:
                    d["tid"] = row[4]
            if len(row) >= 6:
                d["score"] = float(row[5])
            if len(row) >= 7:
                try:
                    d["cls"] = int(row[6])
                except Exception:
                    d["cls"] = row[6]
            if len(row) >= 8:
                try:
                    d["det_idx"] = int(row[7])
                except Exception:
                    d["det_idx"] = row[7]
            out.append(d)
        return out

    def _det_list_to_boxes(det_list: FrameDet) -> List[dict]:
        """Convert one-frame DET list into JSON-friendly dicts."""
        out: List[dict] = []
        if not det_list:
            return out
        for score, ltwh in det_list:
            x, y, w, h = map(float, ltwh)
            out.append({
                "score": float(score),
                "x": float(x),
                "y": float(y),
                "w": float(w),
                "h": float(h),
                "x1": float(x),
                "y1": float(y),
                "x2": float(x + w),
                "y2": float(y + h),
            })
        return out

    def _tracks_to_boxes(tracks) -> List[dict]:
        """Convert BYTETrack STrack objects into JSON-friendly dicts."""
        out: List[dict] = []
        if tracks is None:
            return out
        # Avoid numpy's ambiguous truth value errors.
        try:
            if hasattr(tracks, "shape"):
                if int(tracks.shape[0]) == 0:
                    return out
            else:
                if len(tracks) == 0:
                    return out
        except Exception:
            pass
        for t in tracks:
            try:
                x1, y1, x2, y2 = map(float, getattr(t, "tlbr"))
            except Exception:
                continue
            d = {
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
            }
            # Best-effort add ids/scores if present.
            if hasattr(t, "track_id"):
                try:
                    d["tid"] = int(getattr(t, "track_id"))
                except Exception:
                    d["tid"] = getattr(t, "track_id")
            if hasattr(t, "score"):
                try:
                    d["score"] = float(getattr(t, "score"))
                except Exception:
                    pass
            if hasattr(t, "cls"):
                try:
                    d["cls"] = int(getattr(t, "cls"))
                except Exception:
                    d["cls"] = getattr(t, "cls")
            if hasattr(t, "det_idx"):
                try:
                    d["det_idx"] = int(getattr(t, "det_idx"))
                except Exception:
                    d["det_idx"] = getattr(t, "det_idx")
            out.append(d)
        return out

    def callback_static(ctx: CallbackContext) -> int:
        nonlocal iou_id_cnt
        """Static mode: single tracker per channel, fixed equal weights."""
        ch = ctx.ch
        src = active_sources[ch]
        fps = src.fps
        num_frames = src.num_frames

        frame = int((ctx.now_ms * fps) / 1000.0)
        if ctx.prev_ms is None:
            pre_frame = -1
        else:
            pre_frame = int((ctx.prev_ms * fps) / 1000.0)

        if frame == pre_frame:
            return 0
        if frame >= num_frames:
            return 0
        if frame <= last_frame_done[ch]:
            return 0

        sampled_frames_idx[ch].append(frame)
        delta = frame - last_frame_done[ch]
        new_id_events = _scan_new_gt_ids(ch, last_frame_done[ch] + 1, frame)


        if src.det is None or frame >= src.det.num_frames:
            frame_det = []
        else:
            frame_det = src.det.frames[frame]

        dets = _det_to_results(frame_det)

        weight_before = int(ctx.weight)
        action = "none"
        cost_units = 1

        # Fast-path: if no detections and tracker has no state, just emit empties.
        if len(dets) == 0 and (not eval_trackers[ch].tracked_stracks and not eval_trackers[ch].lost_stracks):
            for fr_idx in range(last_frame_done[ch] + 1, frame + 1):
                if 0 <= fr_idx < num_frames:
                    pred_frames[ch][fr_idx] = []
            last_frame_done[ch] = frame
            weight_rows.append({
                "now_ms": int(ctx.now_ms),
                "ch": int(ch),
                "name": src.name,
                "frame": int(frame),
                "delta": int(delta),
                "det_len": int(len(frame_det)),
                "half_read_det": True,
                "min_iou_full": "",
                "min_iou_half": "",
                "weight_before": int(weight_before),
                "weight_after": int(engine.get_weight(ch)),
                "action": action,
                "retro": False,
                "retro_mid_ms": "",
                "retro_mid_frame": "",
                "retro_mid2_ms": "",
                "retro_mid2_frame": "",
                "retro_mid3_ms": "",
                "retro_mid3_frame": "",
                "cost_units": int(cost_units),
                "total_weight": int(engine.total_weight),
            })

            if iou_id_fp is not None and (new_id_events or len(frame_det) > 0):
                row = {
                    "now_ms": int(ctx.now_ms),
                    "ch": int(ch),
                    "name": src.name,
                    "frame": int(frame),
                    "det_len": int(len(frame_det)),
                    "min_iou_full": None,
                    "min_iou_half": None,
                    "pred_full_box_cnt": 0,
                    "pred_half_box_cnt": 0,
                    "half_read_det": True,
                    "weight": int(engine.get_weight(ch)),
                    "action": str(action),
                    "retro": False,
                    "new_id_events": new_id_events,
                    "metric_iou_thr": float(metric_iou_thr_eff),
                    "gate_iou_thr": float(gate_iou_thr_eff),
                    "iou_min_top_pct": float(iou_min_top_pct),
                }
                iou_id_fp.write(json.dumps(row, ensure_ascii=False) + "\n")
                iou_id_cnt += 1
            return 1

        eval_trackers[ch].update(dets, n=delta)
        _ = eval_trackers[ch].pop_pred_box_history()
        hist_eval = eval_trackers[ch].pop_box_history()

        start_f = last_frame_done[ch] + 1
        for k, boxes in enumerate(hist_eval):
            fr_idx = start_f + k
            if fr_idx >= num_frames:
                break
            out_list: BoxList = []
            if boxes is not None and len(boxes):
                for row in boxes:
                    x1, y1, x2, y2, tid, _score, _cls_id, _det_idx = row.tolist()
                    ltwh = np.asarray([x1, y1, x2 - x1, y2 - y1], dtype=np.float32)
                    out_list.append((int(tid), ltwh))
            pred_frames[ch][fr_idx] = out_list

        last_frame_done[ch] = frame
        weight_rows.append({
            "now_ms": int(ctx.now_ms),
            "ch": int(ch),
            "name": src.name,
            "frame": int(frame),
            "delta": int(delta),
            "det_len": int(len(frame_det)),
            "half_read_det": True,
            "min_iou_full": "",
            "min_iou_half": "",
            "weight_before": int(weight_before),
            "weight_after": int(engine.get_weight(ch)),
            "action": action,
            "retro": False,
            "retro_mid_ms": "",
            "retro_mid_frame": "",
                "retro_mid2_ms": "",
                "retro_mid2_frame": "",
                "retro_mid3_ms": "",
                "retro_mid3_frame": "",
            "cost_units": int(cost_units),
            "total_weight": int(engine.total_weight),
        })

        if iou_id_fp is not None and (new_id_events or len(frame_det) > 0):
            row = {
                "now_ms": int(ctx.now_ms),
                "ch": int(ch),
                "name": src.name,
                "frame": int(frame),
                "det_len": int(len(frame_det)),
                "min_iou_full": None,
                "min_iou_half": None,
                "pred_full_box_cnt": int(len(hist_eval[-1]) if (hist_eval and hist_eval[-1] is not None) else 0),
                "pred_half_box_cnt": 0,
                "half_read_det": True,
                "weight": int(engine.get_weight(ch)),
                "action": str(action),
                "retro": False,
                "new_id_events": new_id_events,
                    "metric_iou_thr": float(metric_iou_thr_eff),
                    "gate_iou_thr": float(gate_iou_thr_eff),
                    "iou_min_top_pct": float(iou_min_top_pct),
            }
            iou_id_fp.write(json.dumps(row, ensure_ascii=False) + "\n")
            iou_id_cnt += 1
        return 1

    def callback_dyn(ctx: CallbackContext) -> int:
        nonlocal iou_id_cnt, weight_adjust_cnt
        ch = ctx.ch
        src = active_sources[ch]
        fps = src.fps
        num_frames = src.num_frames

        # map virtual time to frame index (0-based)
        frame = int((ctx.now_ms * fps) / 1000.0)
        if ctx.prev_ms is None:
            pre_frame = -1
        else:
            pre_frame = int((ctx.prev_ms * fps) / 1000.0)

        if frame == pre_frame:
            return 0
        if frame >= num_frames:
            return 0
        if frame <= last_frame_done[ch]:
            return 0

        sampled_frames_idx[ch].append(frame)

        delta = frame - last_frame_done[ch]
        new_id_events = _scan_new_gt_ids(ch, last_frame_done[ch] + 1, frame)


        # DET must come from DET file
        if src.det is None or frame >= src.det.num_frames:
            frame_det = []
        else:
            frame_det = src.det.frames[frame]

        if verbose_engine and (src.name.startswith("empty") or (src.gt is not None and str(src.gt.path).startswith("<empty:"))):
            print(
                f"[cb] {src.name} now_ms={ctx.now_ms} frame={frame} prev_ms={ctx.prev_ms} pre_frame={pre_frame} det_len={len(frame_det)}"
            )

        dets = _det_to_results(frame_det)

        # Half-rate: read det on phase==0, skip on phase==1
        half_read_det = (half_phase[ch] % 2 == 0)

        weight_before = int(ctx.weight)
        action = "none"
        min_iou_full: float | None = None
        min_iou_half: float | None = None
        retro_trigger = False
        retro_mid_ms: int | None = None
        retro_mid_frame: int | None = None
        retro_mid2_ms: int | None = None
        retro_mid2_frame: int | None = None
        retro_mid3_ms: int | None = None
        retro_mid3_frame: int | None = None
        cost_units = 1

        # FOR DEBUG Fast-path: if no detections and all trackers empty, avoid update.
        if len(dets) == 0 \
            and (not full_trackers[ch].tracked_stracks and not full_trackers[ch].lost_stracks) \
            and (not half_trackers[ch].tracked_stracks and not half_trackers[ch].lost_stracks) \
            and (not eval_trackers[ch].tracked_stracks and not eval_trackers[ch].lost_stracks):
            for fr_idx in range(last_frame_done[ch] + 1, frame + 1):
                if 0 <= fr_idx < num_frames:
                    pred_frames[ch][fr_idx] = []
            last_frame_done[ch] = frame

            # Decay weight on empty-DET no-track frames (before logging), so logs reflect reality.
            new_w = int(ctx.decay_weight())
            weight_after = int(engine.get_weight(ch))
            action = "decay_empty" if new_w < weight_before else "decay_empty_sat"

            weight_rows.append({
                "now_ms": int(ctx.now_ms),
                "ch": int(ch),
                "name": src.name,
                "frame": int(frame),
                "delta": int(delta),
                "det_len": int(len(frame_det)),
                "half_read_det": bool(half_read_det),
                "min_iou_full": "",
                "min_iou_half": "",
                "weight_before": int(weight_before),
                "weight_after": int(weight_after),
                "action": action,
                "retro": False,
                "retro_mid_ms": "",
                "retro_mid_frame": "",
                "retro_mid2_ms": "",
                "retro_mid2_frame": "",
                "retro_mid3_ms": "",
                "retro_mid3_frame": "",
                "cost_units": int(cost_units),
                "total_weight": int(engine.total_weight),
            })


            

            # Also record into weight-adjust log for transparency.
            if weight_adjust_fp is not None and (not src.is_empty):
                row = {
                    "now_ms": int(ctx.now_ms),
                    "ch": int(ch),
                    "name": src.name,
                    "frame": int(frame),
                    "delta": int(delta),
                    "half_read_det": bool(half_read_det),
                    "weight_before": int(weight_before),
                    "weight_after": int(weight_after),
                    "action": str(action),
                    "reason": "empty_det_and_no_tracks",
                    "trigger": "empty_det",
                    "trigger_iou": None,
                    "metric_iou_thr": float(metric_iou_thr_eff),
                    "iou_gate_thr": float(gate_iou_thr_eff),
                    "iou_min_top_pct": float(iou_min_top_pct),
                    "min_iou_full": None,
                    "min_iou_half": None,
                    "pred_full_seq_len": 0,
                    "pred_half_seq_len": 0,
                    "pred_full_step_cnt": 0,
                    "pred_half_step_cnt": 0,
                    "pred_full_seq_box_cnt": 0,
                    "pred_half_seq_box_cnt": 0,
                    "pred_full_box_cnt": 0,
                    "pred_half_box_cnt": 0,
                    "pred_full_post_box_cnt": 0,
                    "pred_half_post_box_cnt": 0,
                    "hieve_det_box_cnt": 0,
                    "hieve_det_box_cnt_raw": int(len(frame_det)),
                    "iou_det_score_thr": float(iou_det_score_thr) if iou_det_score_thr is not None else None,
                    "pred_full_boxes": [],
                    "pred_half_boxes": [],
                    "pred_full_post_boxes": [],
                    "pred_half_post_boxes": [],
                    "det_boxes": _det_list_to_boxes(frame_det),
                    "new_id_events": new_id_events,
                }
                weight_adjust_fp.write(json.dumps(row, ensure_ascii=False) + "\n")
                weight_adjust_cnt += 1

            if iou_id_fp is not None and (new_id_events or len(frame_det) > 0):
                row = {
                    "now_ms": int(ctx.now_ms),
                    "ch": int(ch),
                    "name": src.name,
                    "frame": int(frame),
                    "det_len": int(len(frame_det)),
                    "min_iou_full": None,
                    "min_iou_half": None,
                    "pred_full_box_cnt": 0,
                    "pred_half_box_cnt": 0,
                    "half_read_det": bool(half_read_det),
                    "weight": int(weight_after),
                    "action": str(action),
                    "retro": False,
                    "new_id_events": new_id_events,
                    "metric_iou_thr": float(metric_iou_thr_eff),
                    "gate_iou_thr": float(gate_iou_thr_eff),
                    "iou_min_top_pct": float(iou_min_top_pct),
                }
                iou_id_fp.write(json.dumps(row, ensure_ascii=False) + "\n")
                iou_id_cnt += 1

            half_phase[ch] += 1
            return 1

        # --- Tracker A (full-rate, always update with DET) ---
        full_fused_tracks, _ = full_trackers[ch].update(dets, n=delta)
        pred_hist_full = full_trackers[ch].pop_pred_box_history()
        _ = full_trackers[ch].pop_box_history()  # A is not used for evaluation output

        # --- Tracker B (half-rate) ---
        if half_read_det:
            half_fused_tracks, _ = half_trackers[ch].update(dets, n=delta)
        else:
            half_fused_tracks, _ = half_trackers[ch].PREDICT(n=delta)
        pred_hist_half = half_trackers[ch].pop_pred_box_history()
        _ = half_trackers[ch].pop_box_history()  # clear

        # --- Dynamic weighting (only if det>0 and not empty channel) ---
        det_len_raw = int(len(frame_det))
        # For the IoU gate we optionally filter DETs by a score threshold that is consistent
        # with ByteTrack's "high/new" threshold. This prevents low-confidence DETs (that
        # cannot initialize/drive tracks) from forcing minIoU to 0 and spuriously boosting.
        if iou_det_score_thr is not None:
            iou_det_score_thr_eff = float(iou_det_score_thr)
        else:
            iou_det_score_thr_eff = float(max(full_trackers[ch].args.track_high_thresh, full_trackers[ch].args.new_track_thresh))
        frame_det_iou: FrameDet = [(s, b) for (s, b) in frame_det if float(s) >= iou_det_score_thr_eff]
        det_len_iou = int(len(frame_det_iou))
        weight_after = int(engine.get_weight(ch))
        iou_gate_thr = float(gate_iou_thr_eff)
        adjust_reason = ""
        adjust_trigger = ""
        adjust_trigger_iou: float | None = None

        # Keep the last-frame prediction arrays around for logging/debug.
        pf = np.zeros((0, 8), dtype=np.float32)
        ph = np.zeros((0, 8), dtype=np.float32)

        if det_len_iou > 0:
            pf = pred_hist_full[-1] if pred_hist_full else np.zeros((0, 8), dtype=np.float32)
            ph = pred_hist_half[-1] if pred_hist_half else np.zeros((0, 8), dtype=np.float32)
            min_iou_full = _hungarian_min_iou(pf, frame_det_iou, iou_min_top_pct)
            min_iou_half = _hungarian_min_iou(ph, frame_det_iou, iou_min_top_pct)

            if dynamic_weight and (not src.is_empty):
                if min_iou_full < iou_gate_thr:
                    new_w = int(ctx.boost_weight())
                    weight_after = new_w
                    action = "boost" if new_w > weight_before else "boost_sat"
                    adjust_reason = "full_min_iou_below_thr"
                    adjust_trigger = "full"
                    adjust_trigger_iou = float(min_iou_full)
                elif min_iou_half > iou_gate_thr:
                    new_w = int(ctx.decay_weight())
                    weight_after = new_w
                    action = "decay" if new_w < weight_before else "decay_sat"
                    adjust_reason = "half_min_iou_above_thr"
                    adjust_trigger = "half"
                    adjust_trigger_iou = float(min_iou_half)

            # Detailed weight-adjustment log (JSONL).
            # Only record when we attempted to change weights (including saturation cases).
            if weight_adjust_fp is not None and action != "none" and (not src.is_empty):
                row = {
                    "now_ms": int(ctx.now_ms),
                    "ch": int(ch),
                    "name": src.name,
                    "frame": int(frame),
                    "delta": int(delta),
                    "half_read_det": bool(half_read_det),
                    "weight_before": int(weight_before),
                    "weight_after": int(engine.get_weight(ch)),
                    "action": str(action),
                    "reason": str(adjust_reason),
                    "trigger": str(adjust_trigger),
                    "trigger_iou": (None if adjust_trigger_iou is None else float(adjust_trigger_iou)),
                    "metric_iou_thr": float(metric_iou_thr_eff),
                    "iou_gate_thr": float(iou_gate_thr),
                    "metric_iou_thr": float(metric_iou_thr_eff),
                    "gate_iou_thr": float(gate_iou_thr_eff),
                    "iou_min_top_pct": float(iou_min_top_pct),
                    "min_iou_full": (None if min_iou_full is None else float(min_iou_full)),
                    "min_iou_half": (None if min_iou_half is None else float(min_iou_half)),
                    "pred_full_seq_len": int(len(pred_hist_full)),
                    "pred_half_seq_len": int(len(pred_hist_half)),
                    "pred_full_step_cnt": int(len(pred_hist_full)),
                    "pred_half_step_cnt": int(len(pred_hist_half)),
                    "pred_full_seq_box_cnt": int(sum(int(h.shape[0]) for h in (pred_hist_full or []))),
                    "pred_half_seq_box_cnt": int(sum(int(h.shape[0]) for h in (pred_hist_half or []))),
                    "pred_full_box_cnt": int(pf.shape[0]) if hasattr(pf, "shape") else int(len(pf)),
                    "pred_half_box_cnt": int(ph.shape[0]) if hasattr(ph, "shape") else int(len(ph)),
                    "pred_full_post_box_cnt": int(len(full_fused_tracks)) if full_fused_tracks is not None else 0,
                    "pred_half_post_box_cnt": int(len(half_fused_tracks)) if half_fused_tracks is not None else 0,
                    "hieve_det_box_cnt": int(det_len_iou),
                    "hieve_det_box_cnt_raw": int(len(frame_det)),
                    "iou_det_score_thr": float(iou_det_score_thr_eff),
                    "pred_full_boxes": _pred_arr_to_boxes(pf),
                    "pred_half_boxes": _pred_arr_to_boxes(ph),
                    "pred_full_post_boxes": _tracks_to_boxes(full_fused_tracks),
                    "pred_half_post_boxes": (
                        _tracks_to_boxes(half_fused_tracks) if half_read_det else _pred_arr_to_boxes(half_fused_tracks)
                    ),
                    "det_boxes": _det_list_to_boxes(frame_det),
                    "new_id_events": new_id_events,
                    "metric_iou_thr": float(metric_iou_thr_eff),
                    "gate_iou_thr": float(gate_iou_thr_eff),
                    "iou_min_top_pct": float(iou_min_top_pct),
                }
                weight_adjust_fp.write(json.dumps(row, ensure_ascii=False) + "\n")
                weight_adjust_cnt += 1

        # Enforce rule: empty/no-det channels stay at weight 1
        #if src.is_empty and engine.get_weight(ch) != 1:
        #    engine._set_weight(ch, 1)
        #    weight_after = 1
        #    action = "force1"

        # --- Tracker C (eval output) ---
        # Only retro-fill when:
        #   - enabled by config/CLI
        #   - this callback had DETs (det_len>0)
        #   - we detected "insufficient" via full-rate minIoU
        #   - and the boost is EFFECTIVE (weight increases; not saturated at max_weight)
        # This prevents the pathological case: already at max weight but still paying 2x cost.
        if (
            retro_fill
            and (not src.is_empty)
            and det_len_iou > 0
            and (min_iou_full is not None and min_iou_full < iou_gate_thr)
            and (action == "boost")
            and (weight_after > weight_before)
            and (ctx.prev_ms is not None)
        ):
            pre_ms = int(ctx.prev_ms)
            now_ms = int(ctx.now_ms)
            mid_ms = int((pre_ms + now_ms) // 2)
            retro_gap_ms = now_ms - mid_ms
            if retro_gap_ms > 50:
                mid_frame = int((mid_ms * fps) / 1000.0)
                # Clamp to a valid in-between frame.
                mid_frame = max(last_frame_done[ch] + 1, min(frame - 1, mid_frame))
                if last_frame_done[ch] < mid_frame < frame:
                    # Get midpoint DET
                    if src.det is None or mid_frame >= src.det.num_frames:
                        mid_det = []
                    else:
                        mid_det = src.det.frames[mid_frame]
                    mid_dets = _det_to_results(mid_det)
                    # If the midpoint is still far away from "now", insert extra midpoints
                    # between (mid_ms, now_ms) so the last supplementary DET is not too far
                    # from the current time.
                    # Rule (user-requested):
                    #   - if (now_ms - mid_ms) > retro_max_gap_ms  => add MID2
                    #   - if (now_ms - mid2_ms) > retro_max_gap_ms => add MID3
                    mid2_ms: int | None = None
                    mid2_frame: int | None = None
                    mid3_ms: int | None = None
                    mid3_frame: int | None = None

                    if (now_ms - mid_ms) > int(retro_max_gap_ms):
                        mid2_ms = int((mid_ms + now_ms) // 2)
                        if mid2_ms <= mid_ms:
                            mid2_ms = mid_ms + 1
                        mid2_frame = int((mid2_ms * fps) / 1000.0)
                        # Clamp to a valid in-between frame, strictly after mid_frame.
                        mid2_frame = max(int(mid_frame) + 1, min(int(frame) - 1, int(mid2_frame)))
                        if not (int(mid_frame) < int(mid2_frame) < int(frame)):
                            mid2_ms = None
                            mid2_frame = None

                    if mid2_frame is not None and mid2_ms is not None and (now_ms - int(mid2_ms)) > int(retro_max_gap_ms):
                        mid3_ms = int((int(mid2_ms) + now_ms) // 2)
                        if mid3_ms <= int(mid2_ms):
                            mid3_ms = int(mid2_ms) + 1
                        mid3_frame = int((mid3_ms * fps) / 1000.0)
                        # Clamp to a valid in-between frame, strictly after mid2_frame.
                        mid3_frame = max(int(mid2_frame) + 1, min(int(frame) - 1, int(mid3_frame)))
                        if not (int(mid2_frame) < int(mid3_frame) < int(frame)):
                            mid3_ms = None
                            mid3_frame = None

                    if mid2_frame is None:
                        n1 = mid_frame - last_frame_done[ch]
                        n2 = frame - mid_frame
                        # Two-step update: midpoint then now
                        eval_trackers[ch].update(mid_dets, n=n1)
                        _ = eval_trackers[ch].pop_pred_box_history()
                        hist1 = eval_trackers[ch].pop_box_history()

                        eval_trackers[ch].update(dets, n=n2)
                        _ = eval_trackers[ch].pop_pred_box_history()
                        hist2 = eval_trackers[ch].pop_box_history()

                        hist_eval = list(hist1) + list(hist2)
                        retro_trigger = True
                        retro_mid_frame = int(mid_frame)
                        retro_mid_ms = int(mid_ms)
                        retro_mid2_frame = None
                        retro_mid2_ms = None
                        retro_mid3_frame = None
                        retro_mid3_ms = None
                        cost_units = 2
                    else:
                        # Get 2nd midpoint DET
                        if src.det is None or mid2_frame >= src.det.num_frames:
                            mid2_det = []
                        else:
                            mid2_det = src.det.frames[mid2_frame]
                        mid2_dets = _det_to_results(mid2_det)

                        if mid3_frame is None:
                            n1 = int(mid_frame) - int(last_frame_done[ch])
                            n2 = int(mid2_frame) - int(mid_frame)
                            n3 = int(frame) - int(mid2_frame)
                            # Three-step update: midpoint -> midpoint2 -> now
                            eval_trackers[ch].update(mid_dets, n=n1)
                            _ = eval_trackers[ch].pop_pred_box_history()
                            hist1 = eval_trackers[ch].pop_box_history()

                            eval_trackers[ch].update(mid2_dets, n=n2)
                            _ = eval_trackers[ch].pop_pred_box_history()
                            hist2 = eval_trackers[ch].pop_box_history()

                            eval_trackers[ch].update(dets, n=n3)
                            _ = eval_trackers[ch].pop_pred_box_history()
                            hist3 = eval_trackers[ch].pop_box_history()

                            hist_eval = list(hist1) + list(hist2) + list(hist3)
                            retro_trigger = True
                            retro_mid_frame = int(mid_frame)
                            retro_mid_ms = int(mid_ms)
                            retro_mid2_frame = int(mid2_frame)
                            retro_mid2_ms = int(mid2_ms) if mid2_ms is not None else None
                            retro_mid3_frame = None
                            retro_mid3_ms = None
                            cost_units = 3
                        else:
                            # Get 3rd midpoint DET
                            if src.det is None or mid3_frame >= src.det.num_frames:
                                mid3_det = []
                            else:
                                mid3_det = src.det.frames[mid3_frame]
                            mid3_dets = _det_to_results(mid3_det)

                            n1 = int(mid_frame) - int(last_frame_done[ch])
                            n2 = int(mid2_frame) - int(mid_frame)
                            n3 = int(mid3_frame) - int(mid2_frame)
                            n4 = int(frame) - int(mid3_frame)
                            # Four-step update: midpoint -> midpoint2 -> midpoint3 -> now
                            eval_trackers[ch].update(mid_dets, n=n1)
                            _ = eval_trackers[ch].pop_pred_box_history()
                            hist1 = eval_trackers[ch].pop_box_history()

                            eval_trackers[ch].update(mid2_dets, n=n2)
                            _ = eval_trackers[ch].pop_pred_box_history()
                            hist2 = eval_trackers[ch].pop_box_history()

                            eval_trackers[ch].update(mid3_dets, n=n3)
                            _ = eval_trackers[ch].pop_pred_box_history()
                            hist3 = eval_trackers[ch].pop_box_history()

                            eval_trackers[ch].update(dets, n=n4)
                            _ = eval_trackers[ch].pop_pred_box_history()
                            hist4 = eval_trackers[ch].pop_box_history()

                            hist_eval = list(hist1) + list(hist2) + list(hist3) + list(hist4)
                            retro_trigger = True
                            retro_mid_frame = int(mid_frame)
                            retro_mid_ms = int(mid_ms)
                            retro_mid2_frame = int(mid2_frame)
                            retro_mid2_ms = int(mid2_ms) if mid2_ms is not None else None
                            retro_mid3_frame = int(mid3_frame)
                            retro_mid3_ms = int(mid3_ms) if mid3_ms is not None else None
                            cost_units = 4
                else:
                    retro_trigger = False
            else:
                # Not enough gap to insert a midpoint frame; fall back to normal update.
                retro_trigger = False

        if not retro_trigger:
            eval_trackers[ch].update(dets, n=delta)
            _ = eval_trackers[ch].pop_pred_box_history()
            hist_eval = eval_trackers[ch].pop_box_history()

        # Write outputs (EVAL tracker) for frames last+1 ... frame
        start_f = last_frame_done[ch] + 1
        for k, boxes in enumerate(hist_eval):
            fr_idx = start_f + k
            if fr_idx >= num_frames:
                break
            out_list: BoxList = []
            if boxes is not None and len(boxes):
                for row in boxes:
                    x1, y1, x2, y2, tid, _score, _cls_id, _det_idx = row.tolist()
                    ltwh = np.asarray([x1, y1, x2 - x1, y2 - y1], dtype=np.float32)
                    out_list.append((int(tid), ltwh))
            pred_frames[ch][fr_idx] = out_list

        last_frame_done[ch] = frame

        # log row
        weight_rows.append({
            "now_ms": int(ctx.now_ms),
            "ch": int(ch),
            "name": src.name,
            "frame": int(frame),
            "delta": int(delta),
            "det_len": int(det_len_raw),
            "det_len_iou": int(det_len_iou),
            "iou_det_score_thr": float(iou_det_score_thr_eff),
            "half_read_det": bool(half_read_det),
            "min_iou_full": ("" if min_iou_full is None else f"{min_iou_full:.6f}"),
            "min_iou_half": ("" if min_iou_half is None else f"{min_iou_half:.6f}"),
            "weight_before": int(weight_before),
            "weight_after": int(engine.get_weight(ch)),
            "action": action,
            "retro": bool(retro_trigger),
            "retro_mid_ms": ("" if retro_mid_ms is None else int(retro_mid_ms)),
            "retro_mid_frame": ("" if retro_mid_frame is None else int(retro_mid_frame)),
            "retro_mid2_ms": ("" if retro_mid2_ms is None else int(retro_mid2_ms)),
            "retro_mid2_frame": ("" if retro_mid2_frame is None else int(retro_mid2_frame)),
            "retro_mid3_ms": ("" if retro_mid3_ms is None else int(retro_mid3_ms)),
            "retro_mid3_frame": ("" if retro_mid3_frame is None else int(retro_mid3_frame)),
            "cost_units": int(cost_units),
            "total_weight": int(engine.total_weight),
        })

        if iou_id_fp is not None and (new_id_events or det_len_raw > 0):
            new_ids_flat: List[int] = []
            if new_id_events:
                _s = set()
                for _ev in new_id_events:
                    for _nid in _ev.get("new_ids", []):
                        _s.add(int(_nid))
                new_ids_flat = sorted(_s)

            row = {
                "now_ms": int(ctx.now_ms),
                "ch": int(ch),
                "name": src.name,
                "frame": int(frame),
                "det_len": int(det_len_raw),
                "det_len_iou": int(det_len_iou),
                "iou_det_score_thr": float(iou_det_score_thr_eff),
                "min_iou_full": (None if min_iou_full is None else float(min_iou_full)),
                "min_iou_half": (None if min_iou_half is None else float(min_iou_half)),
                "pred_full_box_cnt": int(pf.shape[0]) if 'pf' in locals() and hasattr(pf, "shape") else 0,
                "pred_half_box_cnt": int(ph.shape[0]) if 'ph' in locals() and hasattr(ph, "shape") else 0,
                "pred_full_post_box_cnt": int(len(full_fused_tracks)) if full_fused_tracks is not None else 0,
                "pred_half_post_box_cnt": int(len(half_fused_tracks)) if half_fused_tracks is not None else 0,
                "half_read_det": bool(half_read_det),
                "weight": int(engine.get_weight(ch)),
                "action": str(action),
                "retro": bool(retro_trigger),
                "retro_mid_ms": (None if retro_mid_ms is None else int(retro_mid_ms)),
                "retro_mid_frame": (None if retro_mid_frame is None else int(retro_mid_frame)),
                "retro_mid2_ms": (None if retro_mid2_ms is None else int(retro_mid2_ms)),
                "retro_mid2_frame": (None if retro_mid2_frame is None else int(retro_mid2_frame)),
                "retro_mid3_ms": (None if retro_mid3_ms is None else int(retro_mid3_ms)),
                "retro_mid3_frame": (None if retro_mid3_frame is None else int(retro_mid3_frame)),
                "new_ids": new_ids_flat,
                "new_id_events": new_id_events,
                    "metric_iou_thr": float(metric_iou_thr_eff),
                    "gate_iou_thr": float(gate_iou_thr_eff),
                    "iou_min_top_pct": float(iou_min_top_pct),
            }
            iou_id_fp.write(json.dumps(row, ensure_ascii=False) + "\n")
            iou_id_cnt += 1

        half_phase[ch] += 1
        return int(cost_units)

    callback = callback_static if mode == "static" else callback_dyn

    engine.set_callback(callback)
    engine.run_blocking(tick_ms=tick_ms, total_ms=total_ms, virtual_time=True, verbose=verbose_engine)

    # Flush each channel to the end
    for ch, src in enumerate(active_sources):
        num_frames = src.num_frames
        remain = (num_frames - 1) - last_frame_done[ch]
        if remain > 0:
            # Flush EVAL tracker outputs to the end (used for metrics)
            eval_trackers[ch].update(_empty_results(), n=remain)
            _ = eval_trackers[ch].pop_pred_box_history()
            hist = eval_trackers[ch].pop_box_history()
            start_f = last_frame_done[ch] + 1
            for k, boxes in enumerate(hist):
                fr_idx = start_f + k
                if fr_idx >= num_frames:
                    break
                out_list: BoxList = []
                if boxes is not None and len(boxes):
                    for row in boxes:
                        x1, y1, x2, y2, tid, _score, _cls_id, _det_idx = row.tolist()
                        ltwh = np.asarray([x1, y1, x2 - x1, y2 - y1], dtype=np.float32)
                        out_list.append((int(tid), ltwh))
                pred_frames[ch][fr_idx] = out_list

            # Clear A/B trackers (not used for metrics) only when they exist.
            if mode != "static":
                full_trackers[ch].update(_empty_results(), n=remain)
                _ = full_trackers[ch].pop_pred_box_history()
                _ = full_trackers[ch].pop_box_history()

                half_trackers[ch].PREDICT(n=remain)
                _ = half_trackers[ch].pop_pred_box_history()
                _ = half_trackers[ch].pop_box_history()

            last_frame_done[ch] = num_frames - 1

    # Metrics per channel
    results: List[ChannelSimResult] = []
    active_pos = {orig_idx: pos for pos, orig_idx in enumerate(active_indices)}

    # Evolution checkpoints in ms
    evo_steps = max(0, int(evolution_steps))
    evo_steps = 0 if evo_steps < 2 else evo_steps
    evo_ms = [int(total_ms * (i + 1) / evo_steps) for i in range(evo_steps)] if evo_steps else []

    for orig_idx, src in enumerate(sources):
        if orig_idx in active_pos:
            pos = active_pos[orig_idx]
            m = compute_motp_idf1(gt_frames[pos], pred_frames[pos], iou_thr=metric_iou_thr_eff, backend=metrics_backend)
            metrics = {
                "motp": m.motp,
                "idf1": m.idf1,
                "idp": m.idp,
                "idr": m.idr,
                "idtp": m.idtp,
                "idfp": m.idfp,
                "idfn": m.idfn,
                "total_gt": m.total_gt,
                "total_pred": m.total_pred,
                "total_matches": m.total_matches,
            }

            # Sampling stats
            sf = sorted(set(sampled_frames_idx[pos]))
            max_frame = src.num_frames - 1
            gaps = [sf[i] - sf[i - 1] for i in range(1, len(sf))]
            metrics.update({
                "sampled_count": len(sf),
                "gt_frames": max_frame + 1,
                "sample_ratio": float(len(sf) / (max_frame + 1)) if max_frame >= 0 else 0.0,
                "gap_mean": float(statistics.mean(gaps)) if gaps else 0.0,
                "gap_p95": float(statistics.quantiles(gaps, n=20)[18]) if len(gaps) >= 20 else (float(max(gaps)) if gaps else 0.0),
                "gap_max": float(max(gaps)) if gaps else 0.0,
                "final_weight": int(engine.get_weight(pos)),
            })

            # Per-channel evolution (prefix metrics)
            if evo_ms:
                evo_list = []
                for tms in evo_ms:
                    f_end = int((tms * float(src.fps)) / 1000.0)
                    f_end = min(src.num_frames, max(0, f_end))
                    sub_gt = gt_frames[pos][:f_end]
                    sub_pr = pred_frames[pos][:f_end]
                    sm = compute_motp_idf1(sub_gt, sub_pr, iou_thr=metric_iou_thr_eff, backend=metrics_backend) if f_end > 0 else None
                    evo_list.append({
                        "t_ms": int(tms),
                        "frames": int(f_end),
                        "idf1": (sm.idf1 if sm else None),
                        "motp": (sm.motp if sm else None),
                    })
                metrics["evolution"] = evo_list

            results.append(ChannelSimResult(
                name=src.name,
                gt_path=(src.gt.path if src.gt is not None else None),
                det_path=(src.det.path if src.det is not None else None),
                fps=src.fps,
                tick_ms=tick_ms,
                num_frames=src.num_frames,
                pred_frames=pred_frames_all[orig_idx],
                gt_frames=gt_frames_all[orig_idx],
                metrics=metrics,
            ))
        else:
            results.append(ChannelSimResult(
                name=src.name,
                gt_path=(src.gt.path if src.gt is not None else None),
                det_path=(src.det.path if src.det is not None else None),
                fps=src.fps,
                tick_ms=tick_ms,
                num_frames=max(0, int(src.num_frames)),
                pred_frames=pred_frames_all[orig_idx],
                gt_frames=gt_frames_all[orig_idx],
                metrics={
                    "motp": None, "idf1": None, "idp": None, "idr": None,
                    "idtp": 0, "idfp": 0, "idfn": 0,
                    "total_gt": 0, "total_pred": 0, "total_matches": 0,
                },
            ))

    # Overall metrics across ACTIVE channels only.
    global_gt: List[BoxList] = []
    global_pr: List[BoxList] = []
    for aidx, src in enumerate(active_sources):
        id_off = aidx * 1_000_000_000
        for t in range(src.num_frames):
            g = [(id_off + gid, box) for (gid, box) in gt_frames[aidx][t]]
            p = [(id_off + pid, box) for (pid, box) in pred_frames[aidx][t]]
            global_gt.append(g)
            global_pr.append(p)

    gm = compute_motp_idf1(global_gt, global_pr, iou_thr=metric_iou_thr_eff, backend=metrics_backend)
    overall_metrics = {
        "motp": gm.motp,
        "idf1": gm.idf1,
        "idp": gm.idp,
        "idr": gm.idr,
        "idtp": gm.idtp,
        "idfp": gm.idfp,
        "idfn": gm.idfn,
        "total_gt": gm.total_gt,
        "total_pred": gm.total_pred,
        "total_matches": gm.total_matches,
    }

    # Overall evolution
    if evo_ms:
        evo_list = []
        for tms in evo_ms:
            sub_gt_all: List[BoxList] = []
            sub_pr_all: List[BoxList] = []
            for aidx, src in enumerate(active_sources):
                id_off = aidx * 1_000_000_000
                f_end = int((tms * float(src.fps)) / 1000.0)
                f_end = min(src.num_frames, max(0, f_end))
                for t in range(f_end):
                    sub_gt_all.append([(id_off + gid, box) for (gid, box) in gt_frames[aidx][t]])
                    sub_pr_all.append([(id_off + pid, box) for (pid, box) in pred_frames[aidx][t]])
            sm = compute_motp_idf1(sub_gt_all, sub_pr_all, iou_thr=metric_iou_thr_eff, backend=metrics_backend) if sub_gt_all else None
            evo_list.append({
                "t_ms": int(tms),
                "idf1": (sm.idf1 if sm else None),
                "motp": (sm.motp if sm else None),
            })
        overall_metrics["evolution"] = evo_list

    overall = OverallSimResult(metrics=overall_metrics)

    # Write weight log CSV
    if weight_log_path:
        p = Path(weight_log_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        cols = [
            "now_ms", "ch", "name", "frame", "delta", "det_len", "det_len_iou", "iou_det_score_thr",
            "half_read_det", "min_iou_full", "min_iou_half",
            "weight_before", "weight_after", "action",
            "retro", "retro_mid_ms", "retro_mid_frame", "retro_mid2_ms",
            "retro_mid2_frame", "retro_mid3_ms", "retro_mid3_frame",
            "cost_units",
            "total_weight",
        ]
        lines = [",".join(cols)]
        for r in weight_rows:
            lines.append(",".join(str(r.get(c, "")) for c in cols)
            )
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")

        # Extra debug summary: how many progressed callbacks each channel actually received.
        if active_sources:
            cnt = Counter(r.get("name") for r in weight_rows)
            ordered_names = [s.name for s in active_sources]
            parts = [f"{nm}:{int(cnt.get(nm, 0))}" for nm in ordered_names]
            LOGGER.info(f"[SCHED] progressed callbacks per channel (total={len(weight_rows)}): " + " ".join(parts))
        LOGGER.info(f"[WEIGHT] wrote weight log CSV: {str(p)} ({len(weight_rows)} rows)")


    # Write new GT-ID event log (JSONL)
    if id_log_path:
        p = Path(id_log_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            for r in id_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        LOGGER.info(f"[ID] wrote new-id log: {str(p)} ({len(id_rows)} events)")


    # Close streamed JSONL logs
    if weight_adjust_fp is not None:
        weight_adjust_fp.close()
        LOGGER.info(f"[WEIGHT-ADJUST] wrote detail log: {str(Path(weight_adjust_log_path))} ({weight_adjust_cnt} events)")

    if iou_id_fp is not None:
        iou_id_fp.close()
        LOGGER.info(f"[IOU-ID] wrote integrated log: {str(Path(iou_id_log_path))} ({iou_id_cnt} rows)")

    return results, overall



def run_simulation(

    sequences: List[MOTLabelSequence],
    *,
    fps: float = 30.0,
    tick_ms: int = 100,
    iou_thr: float = 0.5,
    tracker_cfg: Optional[dict] = None,
    seed: int = 0,
    verbose_engine: bool = False,
) -> Tuple[List[ChannelSimResult], OverallSimResult]:
    """Deprecated.

    Project rule: detections must come from a DET file, not from GT.
    Use JSON config mode with per-source GT+DET instead.
    """
    raise RuntimeError("run_simulation() is disabled: detections must come from DET files. Use --config with gt+det.")


def save_mot_predictions(out_path: str, pred_frames: List[BoxList]) -> None:
    """Save predictions as MOTChallenge-style CSV.

    Output format (common in MOTChallenge):
      frame, id, x, y, w, h, score, -1, -1, -1

    We output score=1.0 for all predicted boxes.
    """
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for fr, boxes in enumerate(pred_frames):
        for tid, ltwh in boxes:
            x, y, w, h = map(float, ltwh)
            lines.append(f"{fr+1},{int(tid)},{x:.3f},{y:.3f},{w:.3f},{h:.3f},1,-1,-1,-1")
    p.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
