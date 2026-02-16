from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class SourceConfig:
    """One simulated channel/video source.

    gt/det are optional.

    **Project rules (updated):**
      - If *either* gt is null/missing *or* det is null/missing, we treat the source as an
        always-empty stream (no objects), BUT it still participates in scheduling/compute.
        We force gt=None and det=None and mark is_empty=True.

    Notes:
      - Such a channel has empty GT/DET lists for every frame, so metrics are NA.
      - Its effective num_frames is determined later by the global simulation duration.
    """
    name: str
    fps: float
    is_empty: bool = False
    gt: Optional[str] = None
    det: Optional[str] = None
    num_frames: Optional[int] = None


@dataclass(frozen=True)
class ProjectConfig:
    tick_ms: int
    iou_thr: float
    det_score_thr: float
    tracker_cfg: Dict[str, Any]
    sources: List[SourceConfig]
    max_frames: int = 0  # optional global fallback length
    empty_channels: int = 0  # auto-insert N empty streams (scheduled, no objects)
    sim_duration_ms: int = 0  # optional requested duration (0 => auto=min across channels)
    # Optional: "half-real-time" retro-fill for evaluation tracker.
    # When enabled, on certain callbacks we will inject an extra mid-point DET update
    # (costing 2 ticks) to reduce large gaps.
    retro_fill: bool = False
    retro_max_gap_ms: int = 200


def load_config(config_path: str) -> Tuple[ProjectConfig, Path]:
    """Load and validate JSON config. Returns (config, base_dir).

    Paths inside the config are treated as *relative to the config file location*.
    """
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(config_path)

    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Config root must be a JSON object")

    tick_ms = int(raw.get("tick_ms", 100))
    iou_thr = float(raw.get("iou_thr", 0.5))
    det_score_thr = float(raw.get("det_score_thr", 0.0))
    tracker_cfg = raw.get("tracker_cfg", {})
    if tracker_cfg is None:
        tracker_cfg = {}
    if not isinstance(tracker_cfg, dict):
        raise ValueError("tracker_cfg must be an object")

    max_frames = int(raw.get("max_frames", 0))
    empty_channels = int(raw.get("empty_channels", 0) or 0)

    # Optional: user-requested simulation duration. If > min(channel durations), we clamp to min with a warning.
    sim_duration_ms = int(raw.get("sim_duration_ms", 0) or raw.get("duration_ms", 0) or 0)
    if sim_duration_ms <= 0:
        dur_s = raw.get("sim_duration_s", None)
        if dur_s is None:
            dur_s = raw.get("duration_s", None)
        if dur_s is not None:
            sim_duration_ms = int(float(dur_s) * 1000.0)

    retro_fill = bool(raw.get("retro_fill", False) or raw.get("retro-fill", False))
    retro_max_gap_ms = int(raw.get("retro_max_gap_ms", raw.get("retro_max_gap", 200)) or 200)
    if retro_max_gap_ms < 1:
        retro_max_gap_ms = 1

    sources_raw = raw.get("sources")
    if not isinstance(sources_raw, list) or not sources_raw:
        raise ValueError("sources must be a non-empty array")

    sources: List[SourceConfig] = []
    for idx, item in enumerate(sources_raw):
        if not isinstance(item, dict):
            raise ValueError(f"sources[{idx}] must be an object")
        name = str(item.get("name", f"ch{idx}"))
        fps = item.get("fps")
        if fps is None:
            raise ValueError(f"sources[{idx}] must contain fps")

        def _norm_path(v):
            if v is None:
                return None
            if isinstance(v, str) and v.strip() == "":
                return None
            return str(v)

        gt = _norm_path(item.get("gt", None))
        det = _norm_path(item.get("det", None))
        num_frames = item.get("num_frames", None)
        # det_mode removed: detections must always come from DET file

        is_empty = bool(item.get("empty", False) or item.get("is_empty", False))  # empty stream flag

        if num_frames is not None:
            num_frames = int(num_frames)
            if num_frames < 0:
                raise ValueError(f"sources[{idx}].num_frames must be >=0")

        # Enforce rules:
        # - If sources[idx].empty==true OR (gt is missing OR det is missing): this is an *empty stream*.
        #   It participates in scheduling but has no objects for the whole duration.
        #   We force gt=None, det=None, and mark is_empty=True.
        if is_empty or gt is None or det is None:
            is_empty = True
            gt = None
            det = None
            # keep num_frames as-is; it may be used as a hint, otherwise global duration will decide.

        sources.append(SourceConfig(
            name=name,
            fps=float(fps),
            is_empty=is_empty,
            gt=gt,
            det=det,
            num_frames=num_frames,
        ))

    cfg = ProjectConfig(
        tick_ms=tick_ms,
        iou_thr=iou_thr,
        det_score_thr=det_score_thr,
        tracker_cfg=tracker_cfg,
        sources=sources,
        max_frames=max_frames,
        empty_channels=empty_channels,
        sim_duration_ms=sim_duration_ms,
        retro_fill=retro_fill,
        retro_max_gap_ms=retro_max_gap_ms,
    )
    return cfg, p.parent
