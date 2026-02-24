#!/usr/bin/env python3
"""Batch runner for HIEVE simulator configs (generic CLI).

This script loads one or more JSON configs, runs the multi-channel sparse scheduling + ByteTrack fill,
and writes compact metrics summaries to the outputs directory.

Key features (CLI overrides > JSON config):
- --tick-ms: override cfg.tick_ms
- --sim-ms: override cfg.sim_duration_ms (can only shorten global duration)
- --sim-cap-ms: hard cap for comparability
- --mode / --modes: static / dyn / dyn-retro (dyn_retro alias accepted)
- --retro-max-gap-ms: dyn-retro only, retro-fill gap control (insert MID2/MID3 if needed)
- --configs: run arbitrary config paths/globs (otherwise run the built-in 10 configs)
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

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

SIM_DURATION_CAP_MS = 47_000  # 47 seconds hard cap


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
    """Make numpy scalars / nested structures JSON friendly."""
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


def _normalize_mode(m: str) -> str:
    return str(m).strip().lower().replace("_", "-")


def _resolve_configs_arg(configs_arg: str) -> List[Path]:
    """Parse --configs 'a.json,b.json,scene*_*.json' into absolute Paths."""
    cfg_paths: List[Path] = []
    for token in [t.strip() for t in configs_arg.split(",") if t.strip()]:
        p = Path(token)
        if not p.is_absolute():
            p = (ROOT / p)
        if any(ch in str(p) for ch in "*?["):
            parent = p.parent
            matches = sorted(parent.glob(p.name))
        else:
            matches = [p]
        cfg_paths.extend(matches)

    # De-dup while keeping order.
    seen = set()
    configs: List[Path] = []
    for p in cfg_paths:
        rp = str(p.resolve())
        if rp not in seen:
            seen.add(rp)
            configs.append(Path(rp))
    return configs


def run_one_config(
    config_path: Path,
    *,
    seed: int = 0,
    max_frames: int = 0,
    sim_cap_ms: int = SIM_DURATION_CAP_MS,
    sim_ms_override: int = 0,
    tick_ms_override: int = 0,
    run_mode: str = "static",
    evolution_steps: int = 0,
    weight_log_dir: Optional[str] = None,
    retro_max_gap_ms: Optional[int] = None,
    iou_min_top_pct: float = 0.95,
    metric_iou_thr_override: float = 0.0,
    gate_iou_thr_override: float = 0.0,
    metrics_backend: str = "simple",

) -> Dict[str, Any]:
    cfg, base_dir = load_config(str(config_path))

    # ---- Overrides (CLI > config) ----
    tick_ms = int(cfg.tick_ms)
    if int(tick_ms_override) > 0:
        tick_ms = int(tick_ms_override)

    # Resolve thresholds:
    # - metric_iou_thr: used for MOTP/IDF1 evaluation
    # - gate_iou_thr: used for dynamic-weight gating (boost/decay/retro)
    # Backward compatible: if config doesn't have metric_iou_thr/gate_iou_thr, fall back to cfg.iou_thr.
    metric_iou_thr = float(getattr(cfg, "metric_iou_thr", cfg.iou_thr))
    gate_iou_thr = float(getattr(cfg, "gate_iou_thr", cfg.iou_thr))
    if float(metric_iou_thr_override) > 0:
        metric_iou_thr = float(metric_iou_thr_override)
    if float(gate_iou_thr_override) > 0:
        gate_iou_thr = float(gate_iou_thr_override)
    
    # Keep legacy name for old simulator signatures that only accept iou_thr.
    iou_thr = float(metric_iou_thr)
    
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

    # Hard cap to keep all experiments comparable.
    total_ms = min(total_ms, int(sim_cap_ms))

    # Optional requested duration:
    # - If CLI provided --sim-ms, it overrides config.sim_duration_ms.
    # - Otherwise, config.sim_duration_ms can only shorten.
    req_ms_cfg = int(getattr(cfg, "sim_duration_ms", 0) or 0)
    req_ms = int(sim_ms_override) if int(sim_ms_override) > 0 else req_ms_cfg
    if req_ms > 0:
        total_ms = min(total_ms, int(req_ms))

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
        base = next(
            (
                s
                for s in sources
                if s.gt is not None
                and s.det is not None
                and s.num_frames > 0
                and (not str(s.gt.path).startswith("<empty:"))
            ),
            None,
        )
        base_fps = float(base.fps) if base is not None else 30.0
        base_frames = int((total_ms * base_fps) / 1000.0)
        for k in range(int(empty_n)):
            name = f"empty{k}"
            gt_seq = make_empty_labels(base_frames, path=f"<empty:{name}>")
            det_seq = make_empty_detections(base_frames, path=f"<empty:{name}>")
            sources.append(SourceSpec(name=name, fps=base_fps, num_frames=base_frames, gt=gt_seq, det=det_seq))

    # Build kwargs for simulator (keep backward compatibility across project versions).
    sim_kwargs: Dict[str, Any] = dict(
        tick_ms=tick_ms,
        iou_thr=iou_thr,
        iou_min_top_pct=float(iou_min_top_pct),
        tracker_cfg=tracker_cfg,
        seed=int(seed),
        verbose_engine=False,
    )

    sig = inspect.signature(run_simulation_sources)
    # Pass-through only if simulator supports these newer parameters.
    if "metric_iou_thr" in sig.parameters:
        sim_kwargs["metric_iou_thr"] = float(metric_iou_thr)
    if "gate_iou_thr" in sig.parameters:
        sim_kwargs["gate_iou_thr"] = float(gate_iou_thr)
    if "metrics_backend" in sig.parameters:
        sim_kwargs["metrics_backend"] = str(metrics_backend)

    # Evolution (stage metrics) can be expensive. Default to OFF unless explicitly requested.
    if "evolution_steps" in sig.parameters:
        sim_kwargs["evolution_steps"] = int(evolution_steps) if int(evolution_steps) >= 2 else 0

    # Run-mode selection:
    #   - static: fixed equal weights
    #   - dyn: dynamic weights, no retro-fill
    #   - dyn-retro: dynamic weights + retro-fill
    mode_norm = _normalize_mode(run_mode)
    if mode_norm not in {"static", "dyn", "dyn-retro"}:
        raise ValueError(f"Unknown run_mode={run_mode!r}. Expected static/dyn/dyn-retro")

    if "run_mode" in sig.parameters:
        sim_kwargs["run_mode"] = mode_norm
    else:
        # Older simulator API (no run_mode): translate to flags if available.
        if mode_norm == "dyn":
            sim_kwargs["dynamic_weight"] = True
        elif mode_norm == "dyn-retro":
            sim_kwargs["dynamic_weight"] = True
            sim_kwargs["retro_fill"] = True

    # Retro-fill clamp (dyn-retro only): CLI > config > default(200ms)
    retro_used: Optional[int] = None
    if mode_norm == "dyn-retro":
        v = retro_max_gap_ms
        if v is None or int(v) <= 0:
            v = int(getattr(cfg, "retro_max_gap_ms", 0) or 0)
        if int(v) <= 0:
            v = 200
        retro_used = int(v)

        # Pass-through only if simulator supports it.
        if "retro_max_gap_ms" in sig.parameters:
            sim_kwargs["retro_max_gap_ms"] = retro_used
        elif "retro_max_gap" in sig.parameters:
            sim_kwargs["retro_max_gap"] = retro_used

    # Optional weight log (one CSV per config+mode).
    if weight_log_dir and ("weight_log_path" in sig.parameters):
        log_dir = Path(weight_log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        mode_tag = mode_norm.replace(":", "_").replace("/", "_")
        log_path = log_dir / f"weights_{config_path.stem}_{mode_tag}.csv"
        sim_kwargs["weight_log_path"] = str(log_path)

    per_ch, overall = run_simulation_sources(sources, **sim_kwargs)

    return _jsonable(
        {
            "config": str(config_path),
            "tick_ms": tick_ms,
            "sim_ms": int(total_ms),
            "iou_thr": iou_thr,
            "metric_iou_thr": float(metric_iou_thr),
            "gate_iou_thr": float(gate_iou_thr),
            "metrics_backend": str(metrics_backend),
            "det_score_thr": det_score_thr,
            "seed": int(seed),
            "run_mode": mode_norm,
            "retro_max_gap_ms": retro_used,
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
        }
    )


def write_results(out_path: Path, results: List[Dict[str, Any]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


def write_results_txt(out_path: Path, results: List[Dict[str, Any]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    for r in results:
        o = r.get("overall", {})
        lines.append(f"=== {Path(r['config']).name} ===")
        extras = []
        if r.get("retro_max_gap_ms") is not None:
            extras.append(f"retro_max_gap_ms={r.get('retro_max_gap_ms')}")
        lines.append(
            f"tick_ms={r.get('tick_ms')} sim_ms={r.get('sim_ms')} seed={r.get('seed')} mode={r.get('run_mode','static')} metric_iou_thr={r.get('metric_iou_thr')} gate_iou_thr={r.get('gate_iou_thr')} backend={r.get('metrics_backend')} "
            + (" ".join(extras) if extras else "")
        )
        lines.append(
            f"OVERALL: motp={o.get('motp')} idf1={o.get('idf1')} idp={o.get('idp')} idr={o.get('idr')} "
            f"idtp={o.get('idtp')} idfp={o.get('idfp')} idfn={o.get('idfn')} "
            f"gt={o.get('total_gt')} pred={o.get('total_pred')} matches={o.get('total_matches')}"
        )
        for ch in r.get("channels", []):
            m = ch.get("metrics", {})
            lines.append(
                f"  [{ch.get('name')}] frames={ch.get('num_frames')} fps={ch.get('fps')} "
                f"motp={m.get('motp')} idf1={m.get('idf1')} idp={m.get('idp')} idr={m.get('idr')}"
            )
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Generic batch runner for HIEVE simulator configs. "
            "By default it runs the 5 scene configs + 5 gt-scene configs shipped in this repo, "
            "but you can pass --configs to run arbitrary JSON config paths/globs."
        )
    )
    ap.add_argument("--seed", type=int, default=0, help="Random seed for scheduling.")

    ap.add_argument("--iou-min-top-pct", type=float, default=0.95,
                    help="Robust min-IoU percentile on DET side for dynamic-weight gating (default 0.95)")
    ap.add_argument("--metric-iou-thr", type=float, default=0.0,
                help="IoU threshold used for evaluation metrics (MOTP/IDF1). 0 = follow config (metric_iou_thr or iou_thr).")
    ap.add_argument("--gate-iou-thr", type=float, default=0.0,
                help="IoU threshold used for dynamic-weight gating (boost/decay/retro). 0 = follow config (gate_iou_thr or iou_thr).")
    ap.add_argument("--metrics-backend", type=str, default="simple", choices=["simple", "motmetrics"],
                help="Evaluator backend: simple (built-in) or motmetrics (3rd-party, requires installing motmetrics).")
    ap.add_argument(
        "--mode",
        type=str,
        default="static",
        choices=["static", "dyn", "dyn-retro", "dyn_retro"],
        help="Run mode: static (equal weights), dyn (dynamic weights), dyn-retro (dynamic+retro).",
    )
    ap.add_argument(
        "--modes",
        type=str,
        default="",
        help="Comma-separated list of modes to run (overrides --mode). Example: static,dyn,dyn-retro",
    )
    ap.add_argument(
        "--sim-cap-ms",
        type=int,
        default=SIM_DURATION_CAP_MS,
        help="Hard cap for simulation duration in ms (default 47000).",
    )
    ap.add_argument(
        "--sim-ms",
        type=int,
        default=0,
        help="Override requested simulation duration in ms (takes precedence over config.sim_duration_ms). 0 = follow config.",
    )
    ap.add_argument(
        "--tick-ms",
        type=int,
        default=0,
        help="Override tick_ms from config. 0 = follow config.",
    )
    ap.add_argument(
        "--retro-max-gap-ms",
        type=int,
        default=0,
        help="Retro-fill mid-sample clamp in ms (dyn-retro only). 0=use config/default(200).",
    )
    ap.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional hard cap on frames per channel (0 = no cap, time cap still applies).",
    )
    ap.add_argument(
        "--evolution-steps",
        type=int,
        default=0,
        help="Stage-metrics steps (>=2 to enable). Default 0 for speed.",
    )
    ap.add_argument(
        "--weight-log-dir",
        type=str,
        default="",
        help="If set, write per-config weight logs (CSV) into this directory.",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default=str(ROOT / "outputs"),
        help="Output directory for batch_*.json and batch_*.txt files.",
    )
    ap.add_argument(
        "--configs",
        type=str,
        default="",
        help=(
            "Comma-separated list of config JSON paths (or globs). "
            "Example: 'scene*_config_fps_fixed.json,gt_scene*_fps_fixed.json'. "
            "If empty, run the built-in 10 configs."
        ),
    )
    args = ap.parse_args()

    # Decide which modes to run.
    if args.modes.strip():
        modes = [_normalize_mode(m) for m in args.modes.split(",") if m.strip()]
    else:
        modes = [_normalize_mode(args.mode)]
    for m in modes:
        if m not in {"static", "dyn", "dyn-retro"}:
            raise ValueError(f"Invalid mode in --modes: {m!r}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Decide which configs to run.
    if args.configs.strip():
        configs = _resolve_configs_arg(args.configs.strip())
        if not configs:
            raise FileNotFoundError(f"No configs matched: {args.configs!r}")
    else:
        configs = [
            *(ROOT / f"scene{i}_config_fps_fixed.json" for i in range(1, 6)),
            *(ROOT / f"gt_scene{i}_fps_fixed.json" for i in range(1, 6)),
        ]

    weight_log_dir = args.weight_log_dir.strip() or None
    retro_arg = int(args.retro_max_gap_ms)
    retro_opt = retro_arg if retro_arg > 0 else None

    for mode in modes:
        results: List[Dict[str, Any]] = []
        for cfg_path in configs:
            if not cfg_path.exists():
                raise FileNotFoundError(cfg_path)
            print(
                f"[RUN] {cfg_path.name}  mode={mode}  seed={args.seed}  "
                f"cap_ms={args.sim_cap_ms}  sim_ms={args.sim_ms or 'cfg'}  tick_ms={args.tick_ms or 'cfg'}  "
                f"retro_max_gap_ms={(retro_arg or 'cfg/default') if mode=='dyn-retro' else 'n/a'}  "
                f"metric_iou_thr={args.metric_iou_thr or 'cfg'}  gate_iou_thr={args.gate_iou_thr or 'cfg'}  backend={args.metrics_backend}"
            )
            results.append(
                run_one_config(
                    cfg_path,
                    seed=int(args.seed),
                    max_frames=int(args.max_frames),
                    sim_cap_ms=int(args.sim_cap_ms),
                    sim_ms_override=int(args.sim_ms),
                    tick_ms_override=int(args.tick_ms),
                    iou_min_top_pct=float(args.iou_min_top_pct),
                    metric_iou_thr_override=float(args.metric_iou_thr),
                    gate_iou_thr_override=float(args.gate_iou_thr),
                    metrics_backend=str(args.metrics_backend),
                    run_mode=mode,
                    evolution_steps=int(args.evolution_steps),
                    weight_log_dir=weight_log_dir,
                    retro_max_gap_ms=retro_opt,
                )
            )

        tag = mode.replace("-", "_")
        n = len(configs)
        out_json = out_dir / f"batch_{n}_results_{tag}.json"
        out_txt = out_dir / f"batch_{n}_results_{tag}.txt"
        write_results(out_json, results)
        write_results_txt(out_txt, results)
        print(f"[OK] Wrote: {out_json}")
        print(f"[OK] Wrote: {out_txt}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())