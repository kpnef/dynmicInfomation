from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Tuple

from collections import Counter

import numpy as np
import random
import statistics

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
    iou_thr: float = 0.5,
    tracker_cfg: Optional[dict] = None,
    seed: int = 0,
    verbose_engine: bool = False,
    dynamic_weight: bool = False,
    weight_log_path: Optional[str] = None,
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

    def _ltwh_to_xyxy(ltwh) -> np.ndarray:
        x, y, w, h = map(float, ltwh)
        return np.asarray([x, y, x + w, y + h], dtype=np.float32)

    def _hungarian_min_iou(pred_boxes_xyxy: np.ndarray, det_list: FrameDet) -> float:
        """Return the min IoU across Hungarian matches.

        "Unmatched counts as 0" is applied **from the DET side**:
        - If a DET cannot be matched to any predicted box (i.e., #pred < #det), we treat minIoU as 0.
        - Extra predicted boxes (i.e., #pred > #det) do **not** force minIoU to 0.

        This matches the intent of using DET as the observation: if all observed objects can be
        matched well, tracking is considered "good enough" even if there are extra lingering tracks.
        """
        det_n = len(det_list)
        if det_n <= 0:
            return 1
        if pred_boxes_xyxy is None or len(pred_boxes_xyxy) == 0:
            return 0.0

        det_xyxy = np.stack([_ltwh_to_xyxy(ltwh) for _score, ltwh in det_list], axis=0)
        pred_xyxy = np.asarray(pred_boxes_xyxy, dtype=np.float32)
        if pred_xyxy.ndim != 2 or pred_xyxy.shape[1] < 4:
            return 0.0
        pred_xyxy = pred_xyxy[:, :4]
        n, m = pred_xyxy.shape[0], det_xyxy.shape[0]
        if n == 0 or m == 0:
            return 0.0

        # IoU matrix
        ious = np.zeros((n, m), dtype=np.float32)
        for i in range(n):
            a = pred_xyxy[i]
            ax1, ay1, ax2, ay2 = map(float, a)
            a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
            for j in range(m):
                b = det_xyxy[j]
                bx1, by1, bx2, by2 = map(float, b)
                xA = max(ax1, bx1)
                yA = max(ay1, by1)
                xB = min(ax2, bx2)
                yB = min(ay2, by2)
                inter_w = max(0.0, xB - xA)
                inter_h = max(0.0, yB - yA)
                inter = inter_w * inter_h
                if inter <= 0.0:
                    ious[i, j] = 0.0
                    continue
                b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
                union = a_area + b_area - inter
                ious[i, j] = float(inter / union) if union > 0 else 0.0

        cost = 1.0 - ious
        row_ind, col_ind = linear_sum_assignment(cost)
        matched_ious = [float(ious[r, c]) for r, c in zip(row_ind, col_ind)]
        if not matched_ious:
            return 0.0
        min_iou = min(matched_ious)

        # "Unmatched as 0" is applied on the DET side only.
        # If there are more DETs than predictions, at least one DET is unmatched.
        if n < m:
            min_iou = 0.0
        return float(min_iou)

    def callback_static(ctx: CallbackContext) -> int:
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
                "cost_units": int(cost_units),
                "total_weight": int(engine.total_weight),
            })
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
            "cost_units": int(cost_units),
            "total_weight": int(engine.total_weight),
        })
        return 1

    def callback_dyn(ctx: CallbackContext) -> int:
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
        cost_units = 1

        # Fast-path: if no detections and all trackers empty, avoid update.
        if len(dets) == 0 \
            and (not full_trackers[ch].tracked_stracks and not full_trackers[ch].lost_stracks) \
            and (not half_trackers[ch].tracked_stracks and not half_trackers[ch].lost_stracks) \
            and (not eval_trackers[ch].tracked_stracks and not eval_trackers[ch].lost_stracks):
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
                "half_read_det": bool(half_read_det),
                "min_iou_full": "",
                "min_iou_half": "",
                "weight_before": int(weight_before),
                "weight_after": int(engine.get_weight(ch)),
                "action": action,
                "retro": False,
                "retro_mid_ms": "",
                "retro_mid_frame": "",
                "cost_units": int(cost_units),
                "total_weight": int(engine.total_weight),
            })

            half_phase[ch] += 1
            return 1

        # --- Tracker A (full-rate, always update with DET) ---
        full_trackers[ch].update(dets, n=delta)
        pred_hist_full = full_trackers[ch].pop_pred_box_history()
        _ = full_trackers[ch].pop_box_history()  # A is not used for evaluation output

        # --- Tracker B (half-rate) ---
        if half_read_det:
            half_trackers[ch].update(dets, n=delta)
        else:
            half_trackers[ch].PREDICT(n=delta)
        pred_hist_half = half_trackers[ch].pop_pred_box_history()
        _ = half_trackers[ch].pop_box_history()  # clear

        # --- Dynamic weighting (only if det>0 and not empty channel) ---
        det_len = int(len(frame_det))
        weight_after = int(engine.get_weight(ch))
        if det_len > 0:
            pf = pred_hist_full[-1] if pred_hist_full else np.zeros((0, 8), dtype=np.float32)
            ph = pred_hist_half[-1] if pred_hist_half else np.zeros((0, 8), dtype=np.float32)
            min_iou_full = _hungarian_min_iou(pf, frame_det)
            min_iou_half = _hungarian_min_iou(ph, frame_det)

            if dynamic_weight and (not src.is_empty):
                if min_iou_full < 0.5:
                    new_w = int(ctx.boost_weight())
                    weight_after = new_w
                    action = "boost" if new_w > weight_before else "boost_sat"
                elif min_iou_half > 0.5:
                    new_w = int(ctx.decay_weight())
                    weight_after = new_w
                    action = "decay"

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
            and det_len > 0
            and (min_iou_full is not None and min_iou_full < 0.5)
            and (action == "boost")
            and (weight_after > weight_before)
            and (ctx.prev_ms is not None)
        ):
            pre_ms = int(ctx.prev_ms)
            now_ms = int(ctx.now_ms)
            mid_ms = int((pre_ms + now_ms) // 2)
            retro_gap_ms = now_ms - mid_ms
            if retro_gap_ms > 200:
                if now_ms - mid_ms > retro_max_gap_ms:
                    mid_ms = now_ms - int(retro_max_gap_ms)
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
                    cost_units = 2
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
            "det_len": int(det_len),
            "half_read_det": bool(half_read_det),
            "min_iou_full": ("" if min_iou_full is None else f"{min_iou_full:.6f}"),
            "min_iou_half": ("" if min_iou_half is None else f"{min_iou_half:.6f}"),
            "weight_before": int(weight_before),
            "weight_after": int(engine.get_weight(ch)),
            "action": action,
            "retro": bool(retro_trigger),
            "retro_mid_ms": ("" if retro_mid_ms is None else int(retro_mid_ms)),
            "retro_mid_frame": ("" if retro_mid_frame is None else int(retro_mid_frame)),
            "cost_units": int(cost_units),
            "total_weight": int(engine.total_weight),
        })

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
            m = compute_motp_idf1(gt_frames[pos], pred_frames[pos], iou_thr=iou_thr)
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
                    sm = compute_motp_idf1(sub_gt, sub_pr, iou_thr=iou_thr) if f_end > 0 else None
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

    gm = compute_motp_idf1(global_gt, global_pr, iou_thr=iou_thr)
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
            sm = compute_motp_idf1(sub_gt_all, sub_pr_all, iou_thr=iou_thr) if sub_gt_all else None
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
            "now_ms", "ch", "name", "frame", "delta", "det_len",
            "half_read_det", "min_iou_full", "min_iou_half",
            "weight_before", "weight_after", "action",
            "retro", "retro_mid_ms", "retro_mid_frame", "cost_units",
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
