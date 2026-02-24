#!/usr/bin/env python3
"""Batch runner for HIEVE simulator configs.

This script loads a JSON config, runs the multi-channel sparse scheduling + ByteTrack fill,
and writes a compact metrics summary to the outputs directory.

Notes:
- Paths inside configs are resolved relative to the config file's directory.
- The simulation duration follows the same rule as hieve_sim/cli.py:
  global duration = min(channel durations), unless a smaller sim_duration_ms is requested.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Make imports work even if you run this script from outside the repo root.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hieve_sim.config import load_config
from hieve_sim.hieve_reader import (
    load_mot_labels,
    load_mot_detections,
    make_empty_labels,
    make_empty_detections,
    MOTLabelSequence,
    MOTDetSequence,
)
from hieve_sim.simulator import run_simulation_sources, SourceSpec


def _slice_labels(seq: MOTLabelSequence, max_frames: int) -> MOTLabelSequence:
    if max_frames <= 0 or seq.num_frames <= 0 or max_frames >= seq.num_frames:
        return seq
    frames = seq.frames[:max_frames]
    return MOTLabelSequence(frames=frames, max_frame=max_frames - 1, path=seq.path)


def _slice_dets(seq: MOTDetSequence, max_frames: int) -> MOTDetSequence:
    if max_frames <= 0 or seq.num_frames <= 0 or max_frames >= seq.num_frames:
        return seq
    frames = seq.frames[:max_frames]
    return MOTDetSequence(frames=frames, max_frame=max_frames - 1, path=seq.path)


def _pad_labels(seq: MOTLabelSequence, num_frames: int) -> MOTLabelSequence:
    if num_frames <= seq.num_frames:
        return seq
    frames = list(seq.frames) + ([[]] * (num_frames - seq.num_frames))
    return MOTLabelSequence(frames=frames, max_frame=num_frames - 1, path=seq.path)


def _pad_dets(seq: MOTDetSequence, num_frames: int) -> MOTDetSequence:
    if num_frames <= seq.num_frames:
        return seq
    frames = list(seq.frames) + ([[]] * (num_frames - seq.num_frames))
    return MOTDetSequence(frames=frames, max_frame=num_frames - 1, path=seq.path)


def _jsonable(x: Any) -> Any:
    # Make numpy scalars / dataclasses JSON friendly.
    try:
        import numpy as np  # type: ignore
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
    except Exception:
        pass
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def run_one_config(config_path: Path, *, seed: int = 0, max_frames: int = 0, iou_min_top_pct: float = 0.95) -> Dict[str, Any]:
    cfg, base_dir = load_config(str(config_path))

    tick_ms = int(cfg.tick_ms)
    iou_thr = float(cfg.iou_thr)
    det_score_thr = float(cfg.det_score_thr)
    tracker_cfg = dict(cfg.tracker_cfg or {})

    # Prepare placeholders (keep stable indices for cfg.sources).
    sources: List[SourceSpec] = []
    for s in cfg.sources:
        sources.append(SourceSpec(name=s.name, fps=float(s.fps), num_frames=int(s.num_frames or 0), gt=None, det=None))

    # Pass 1: load all normal streams (gt+det present) and collect durations.
    loaded_info: List[Tuple[int, MOTLabelSequence, MOTDetSequence, int, int]] = []
    for idx, s in enumerate(cfg.sources):
        if getattr(s, "is_empty", False):
            continue
        if not (s.gt and s.det):
            continue

        gt_path = (base_dir / s.gt).resolve()
        det_path = (base_dir / s.det).resolve()

        hint = int(s.num_frames or (max_frames if max_frames > 0 else 0) or 0)
        gt_seq = load_mot_labels(str(gt_path), allow_empty=True, num_frames_hint=hint)
        det_seq = load_mot_detections(str(det_path), score_thr=det_score_thr, allow_empty=True, num_frames_hint=hint)

        native_frames = int(gt_seq.num_frames)
        if s.num_frames is not None and int(s.num_frames) > 0:
            native_frames = min(native_frames, int(s.num_frames))
        if max_frames > 0:
            native_frames = min(native_frames, int(max_frames))

        duration_ms = int((native_frames / max(1e-6, float(s.fps))) * 1000.0) if native_frames > 0 else 0
        loaded_info.append((idx, gt_seq, det_seq, native_frames, duration_ms))

    if not loaded_info:
        raise RuntimeError(f"No valid gt+det sources found in {config_path}")

    # Global duration = min duration across loaded channels.
    min_ms = min(d for _i, _g, _d, _nf, d in loaded_info if d > 0)
    total_ms = int(min_ms)

    # Empty streams with explicit num_frames can further reduce the global duration.
    for s in cfg.sources:
        if getattr(s, "is_empty", False) and s.num_frames is not None and int(s.num_frames) > 0:
            d_ms = int((int(s.num_frames) / max(1e-6, float(s.fps))) * 1000.0)
            total_ms = min(total_ms, d_ms)

    # Optional requested duration in config (can only shorten).
    req_ms = int(getattr(cfg, "sim_duration_ms", 0) or 0)
    if req_ms > 0:
        total_ms = min(total_ms, req_ms)

    loaded_map = {i: (gt_seq, det_seq, native_frames) for (i, gt_seq, det_seq, native_frames, _dms) in loaded_info}

    # Pass 2: finalize all sources with total_ms.
    for idx, s in enumerate(cfg.sources):
        if getattr(s, "is_empty", False):
            num_frames = int((total_ms * float(s.fps)) / 1000.0)
            num_frames = max(0, num_frames)
            gt_seq = make_empty_labels(num_frames, path=f"<empty:{s.name}>")
            det_seq = make_empty_detections(num_frames, path=f"<empty:{s.name}>")
            sources[idx] = SourceSpec(name=s.name, fps=float(s.fps), num_frames=num_frames, gt=gt_seq, det=det_seq)
            continue

        if idx not in loaded_map:
            continue
        gt_seq, det_seq, native_frames = loaded_map[idx]

        max_by_time = int((total_ms * float(s.fps)) / 1000.0)
        num_frames = min(native_frames, max_by_time) if max_by_time > 0 else 0

        gt_seq = _slice_labels(gt_seq, num_frames)
        gt_seq = _pad_labels(gt_seq, num_frames)
        det_seq = _slice_dets(det_seq, num_frames)
        det_seq = _pad_dets(det_seq, num_frames)

        sources[idx] = SourceSpec(name=s.name, fps=float(s.fps), num_frames=num_frames, gt=gt_seq, det=det_seq)

    # Optional auto-insert empty streams.
    empty_n = int(getattr(cfg, "empty_channels", 0) or 0)
    if empty_n > 0:
        base = next((s for s in sources if s.gt is not None and s.det is not None and s.num_frames > 0 and (not str(s.gt.path).startswith("<empty:"))), None)
        base_fps = float(base.fps) if base is not None else 30.0
        base_frames = int((total_ms * base_fps) / 1000.0)
        for k in range(int(empty_n)):
            name = f"empty{k}"
            gt_seq = make_empty_labels(base_frames, path=f"<empty:{name}>")
            det_seq = make_empty_detections(base_frames, path=f"<empty:{name}>")
            sources.append(SourceSpec(name=name, fps=base_fps, num_frames=base_frames, gt=gt_seq, det=det_seq))

    per_ch, overall = run_simulation_sources(
        sources,
        tick_ms=tick_ms,
        iou_thr=iou_thr,
        iou_min_top_pct=float(iou_min_top_pct),
        tracker_cfg=tracker_cfg,
        seed=int(seed),
        verbose_engine=False,
    )

    return _jsonable({
        "config": str(config_path),
        "tick_ms": tick_ms,
        "iou_thr": iou_thr,
        "det_score_thr": det_score_thr,
        "seed": int(seed),
        "overall": overall.metrics,
        "channels": [
            {
                "name": r.name,
                "fps": r.fps,
                "num_frames": r.num_frames,
                "gt_path": r.gt_path,
                "det_path": r.det_path,
                "metrics": r.metrics,
            }
            for r in per_ch
        ],
    })


def write_results(out_path: Path, results: List[Dict[str, Any]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


def write_results_txt(out_path: Path, results: List[Dict[str, Any]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    for r in results:
        o = r.get("overall", {})
        lines.append(f"=== {Path(r['config']).name} ===")
        lines.append(f"tick_ms={r.get('tick_ms')} seed={r.get('seed')}")
        lines.append(f"OVERALL: motp={o.get('motp')} idf1={o.get('idf1')} idp={o.get('idp')} idr={o.get('idr')} "
                     f"idtp={o.get('idtp')} idfp={o.get('idfp')} idfn={o.get('idfn')} "
                     f"gt={o.get('total_gt')} pred={o.get('total_pred')} matches={o.get('total_matches')}")
        for ch in r.get("channels", []):
            m = ch.get("metrics", {})
            lines.append(f"  [{ch.get('name')}] frames={ch.get('num_frames')} fps={ch.get('fps')} "
                         f"motp={m.get('motp')} idf1={m.get('idf1')} idp={m.get('idp')} idr={m.get('idr')}")
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Batch run 10 configs (scene1-5 + gt_scene1-5)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iou-min-top-pct", type=float, default=0.95,
                    help="Robust min-IoU percentile on DET side for dynamic-weight gating (default 0.95)")
    args = ap.parse_args(argv)

    # 5 scene configs + 5 gt-scene configs = 10 results
    configs = [ROOT / f"scene{i}_config_fps_fixed.json" for i in range(1, 6)] +               [ROOT / f"gt_scene{i}_fps_fixed.json" for i in range(1, 6)]

    results = []
    for cfg_path in configs:
        if not cfg_path.exists():
            raise FileNotFoundError(cfg_path)
        print(f"[RUN] {cfg_path.name}")
        results.append(run_one_config(cfg_path, seed=int(args.seed), iou_min_top_pct=float(args.iou_min_top_pct)))

    out_json = ROOT / "outputs" / "batch_10_results.json"
    out_txt = ROOT / "outputs" / "batch_10_results.txt"
    write_results(out_json, results)
    write_results_txt(out_txt, results)
    print(f"[OK] Wrote: {out_json}")
    print(f"[OK] Wrote: {out_txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
