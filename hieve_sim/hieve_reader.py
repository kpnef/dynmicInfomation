from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

# -----------------------------
# In-memory sequences (MOT-like)
# -----------------------------

# Each GT entry: (gt_id, ltwh) where ltwh=(x,y,w,h) float32
FrameGT = List[Tuple[int, np.ndarray]]

# Each DET entry: (score, ltwh) where ltwh=(x,y,w,h) float32
FrameDet = List[Tuple[float, np.ndarray]]


@dataclass(frozen=True)
class MOTLabelSequence:
    """In-memory MOT-style *GT label* sequence."""
    frames: List[FrameGT]   # index = frame_id (0-based)
    max_frame: int          # last frame index
    path: str

    @property
    def num_frames(self) -> int:
        return self.max_frame + 1

    @property
    def num_boxes(self) -> int:
        return sum(len(f) for f in self.frames)


@dataclass(frozen=True)
class MOTDetSequence:
    """In-memory MOT-style *detection* sequence."""
    frames: List[FrameDet]  # index = frame_id (0-based)
    max_frame: int          # last frame index
    path: str

    @property
    def num_frames(self) -> int:
        return self.max_frame + 1

    @property
    def num_boxes(self) -> int:
        return sum(len(f) for f in self.frames)



def make_empty_labels(num_frames: int, path: str = "") -> MOTLabelSequence:
    frames: List[FrameGT] = [[] for _ in range(max(0, int(num_frames)))]
    max_frame = len(frames) - 1
    return MOTLabelSequence(frames=frames, max_frame=max_frame, path=path)


def make_empty_detections(num_frames: int, path: str = "") -> MOTDetSequence:
    frames: List[FrameDet] = [[] for _ in range(max(0, int(num_frames)))]
    max_frame = len(frames) - 1
    return MOTDetSequence(frames=frames, max_frame=max_frame, path=path)


def _parse_line_to_numbers(parts: List[str]) -> Optional[Tuple[int, float, float, float, float, float]]:
    """Parse a MOTChallenge line:
        frame, id, x, y, w, h, conf, ...
    Returns (frame, x, y, w, h, conf) or None if invalid.
    """
    if len(parts) < 6:
        return None
    try:
        fr = int(float(parts[0]))
        # parts[1] is id (can be -1 for det), ignore here
        x = float(parts[2]); y = float(parts[3]); w = float(parts[4]); h = float(parts[5])
        conf = float(parts[6]) if len(parts) >= 7 else 1.0
        return fr, x, y, w, h, conf
    except Exception:
        return None


def load_mot_labels(path: str, *, allow_empty: bool = False, num_frames_hint: int = 0) -> MOTLabelSequence:
    """Load MOTChallenge-style *GT labels*.

    Expected CSV lines like:
      frame, id, x, y, w, h, conf, -1, -1, -1

    Frame can be 0-based or 1-based. We normalize to 0-based and treat the first frame as time=0.
    """
    p = Path(path)
    if not p.exists():
        if allow_empty:
            # Treat missing file as an empty stream when configured to allow empty inputs.
            return make_empty_labels(int(num_frames_hint), path=str(p))
        raise FileNotFoundError(path)

    rows = []
    min_frame = None
    max_frame = -1
    with p.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = [s.strip() for s in line.split(",")]
            parsed = _parse_line_to_numbers(parts)
            if parsed is None:
                continue
            fr, x, y, w, h, _conf = parsed
            try:
                tid = int(float(parts[1]))
            except Exception:
                continue
            rows.append((fr, tid, x, y, w, h))
            max_frame = max(max_frame, fr)
            min_frame = fr if min_frame is None else min(min_frame, fr)

    if min_frame is None:
        if allow_empty:
            return make_empty_labels(int(num_frames_hint), path=str(p))
        raise ValueError(f"Empty/invalid label file: {path}")

    # Normalize to 0-based if necessary
    offset = 1 if min_frame == 1 else 0
    max_frame -= offset

    frames: List[FrameGT] = [[] for _ in range(max_frame + 1)]
    for fr, tid, x, y, w, h in rows:
        fr0 = fr - offset
        if fr0 < 0:
            continue
        frames[fr0].append((tid, np.asarray([x, y, w, h], dtype=np.float32)))

    return MOTLabelSequence(frames=frames, max_frame=max_frame, path=str(p))


def load_mot_detections(path: str, *, score_thr: float = 0.0, allow_empty: bool = False, num_frames_hint: int = 0) -> MOTDetSequence:
    """Load MOTChallenge-style *detections* (id usually -1).

    Robustness:
    - Skips malformed lines (some files may contain stray header/footer lines).
    - Normalizes frame index to 0-based if the smallest frame id is 1.
    - Applies optional confidence threshold.
    """
    p = Path(path)
    if not p.exists():
        if allow_empty:
            # Treat missing file as an empty stream when configured to allow empty inputs.
            return make_empty_detections(int(num_frames_hint), path=str(p))
        raise FileNotFoundError(path)

    rows = []
    min_frame = None
    max_frame = -1
    with p.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = [s.strip() for s in line.split(",")]
            parsed = _parse_line_to_numbers(parts)
            if parsed is None:
                continue
            fr, x, y, w, h, conf = parsed
            if conf < score_thr:
                continue
            rows.append((fr, x, y, w, h, conf))
            max_frame = max(max_frame, fr)
            min_frame = fr if min_frame is None else min(min_frame, fr)

    if min_frame is None:
        if allow_empty:
            return make_empty_detections(int(num_frames_hint), path=str(p))
        raise ValueError(f"Empty/invalid detection file: {path}")

    offset = 1 if min_frame == 1 else 0
    max_frame -= offset

    frames: List[FrameDet] = [[] for _ in range(max_frame + 1)]
    for fr, x, y, w, h, conf in rows:
        fr0 = fr - offset
        if fr0 < 0:
            continue
        frames[fr0].append((float(conf), np.asarray([x, y, w, h], dtype=np.float32)))

    return MOTDetSequence(frames=frames, max_frame=max_frame, path=str(p))
