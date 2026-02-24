from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional

from .config import load_config
from .hieve_reader import (
    load_mot_labels,
    load_mot_detections,
    make_empty_labels,
    make_empty_detections,
    MOTLabelSequence,
    MOTDetSequence,
)
from .simulator import (
    run_simulation_sources,
    save_mot_predictions,
    SourceSpec,
)
from .metrics import pretty_float
from .utils import LOGGER


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


def _fmt_metrics(m: dict) -> str:
    return (
        f"MOTP={pretty_float(m.get('motp'))} "
        f"IDF1={pretty_float(m.get('idf1'))} "
        f"(IDP={pretty_float(m.get('idp'))}, IDR={pretty_float(m.get('idr'))}) "
        f"matches={m.get('total_matches', 0)} "
        f"gt={m.get('total_gt', 0)} pred={m.get('total_pred', 0)} "
        f"idtp={m.get('idtp', 0)} idfp={m.get('idfp', 0)} idfn={m.get('idfn', 0)}"
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description="HIEVE Track1 sparse scheduling + ByteTrack fill + MOTP/IDF1")

    ap.add_argument(
        "--config",
        type=str,
        default=None,
        help="JSON config path (supports multiple sources, per-source fps, optional gt/det)",
    )

    # Legacy mode is disabled by project rule: detections must come from DET files.
    ap.add_argument("--labels", nargs="+", default=None,
                    help="(disabled) Legacy mode is not allowed. Use --config with gt+det per source.")

    ap.add_argument("--tick-ms", type=int, default=None, help="Scheduler tick in ms (override config)")
    ap.add_argument("--iou-thr", type=float, default=None, help="(legacy) IoU threshold override. If --metric-iou-thr/--gate-iou-thr are not set, this value sets BOTH.")
    ap.add_argument("--metric-iou-thr", type=float, default=None, help="IoU threshold used for evaluation metrics (MOTP/IDF1). Overrides config metric_iou_thr.")
    ap.add_argument("--gate-iou-thr", type=float, default=None, help="IoU threshold used for dynamic-weight gating (boost/decay/retro). Overrides config gate_iou_thr.")
    ap.add_argument("--metrics-backend", type=str, default="simple", choices=["simple", "motmetrics"],
                    help="Which evaluator to use for MOTP/IDF1: simple (built-in) or motmetrics (3rd-party, requires installing motmetrics).")
    ap.add_argument("--iou-min-top-pct", type=float, default=None,
                    help="Robust min-IoU percentile on DET side for dynamic-weight gating. Default 0.95 (ignore worst ~5% DETs).")
    ap.add_argument(
        "--iou-det-score-thr",
        type=float,
        default=None,
        help=(
            "Score threshold used ONLY for the dynamic-weight IoU gate. "
            "If omitted, it auto-follows ByteTrack high/new thresholds, so IoU matching "
            "uses the same detection subset that can actually drive tracks."
        ),
    )
    ap.add_argument(
        "--align-tracker-thr",
        action="store_true",
        help=(
            "Force ByteTrack thresholds (track_high/new/low) to align with the same score threshold "
            "used by DET ingestion / IoU gate. This prevents a persistent det-vs-pred count gap when "
            "det_score_thr is much lower than ByteTrack's defaults. (Also supported in config: align_tracker_thr=true)"
        ),
    )
    ap.add_argument("--det-score-thr", type=float, default=None, help="Filter detections below this score")
    ap.add_argument("--max-frames", type=int, default=0, help="If >0, only simulate first N frames per channel; also used as fallback length for empty sources")
    ap.add_argument("--out-dir", type=str, default="outputs", help="Output directory for prediction files")
    ap.add_argument("--save-pred", action="store_true", help="Save MOT-format predictions to out-dir")
    ap.add_argument("--verbose-engine", action="store_true", help="Print per-tick scheduling info")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for weighted scheduler and simulation")

    # Run mode (preferred): controls static/dynamic + retro-fill policy.
    #   - static:        fixed equal weights, single tracker, no retro-fill.
    #   - dyn:           dynamic weights (IoU feedback), no retro-fill (always single tick_ms per progress).
    #   - dyn-retro:     dynamic weights + retro-fill (extra midpoint DET update on effective boost; costs 2 ticks).
    ap.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["static", "dyn", "dyn-retro"],
        help="Scheduling mode: static | dyn | dyn-retro. If set, overrides --dynamic-weight/--retro-fill flags.",
    )

    # Backward compatible switches (kept for older scripts). Prefer --mode.
    ap.add_argument("--dynamic-weight", action="store_true", help="Enable dynamic weight scheduling (IoU feedback)")
    ap.add_argument("--weight-log", type=str, default=None, help="CSV path for weight log (default: out-dir/weights_*.csv)")
    ap.add_argument("--id-log", type=str, default=None, help="JSONL path for logging when new GT IDs first appear (default: out-dir/new_ids_*.jsonl)")
    ap.add_argument("--weight-adjust-log", type=str, default=None, help="JSONL path for logging detailed weight-adjust reasons + boxes (default: out-dir/weight_adjust_*.jsonl)")
    ap.add_argument("--iou-id-log", type=str, default=None, help="JSONL path for per-channel minIoU(full/half) integrated with new-id events (default: out-dir/iou_id_*.jsonl)")
    ap.add_argument("--evolution-steps", type=int, default=10, help="Metric evolution checkpoints (>=2 to enable)")
    ap.add_argument("--empty-channels", type=int, default=None,
                    help="Auto-insert N empty streams (scheduled, no objects). If set, overrides config empty_channels.")
    ap.add_argument("--sim-duration-ms", type=int, default=None,
                    help="Requested simulation duration in ms (clamped to min channel duration)")
    ap.add_argument("--sim-duration-s", type=float, default=None,
                    help="Requested simulation duration in seconds (clamped to min channel duration)")

    # Optional: eval tracker retro-fill (BYTE TRACK C) (legacy; prefer --mode)
    ap.add_argument("--retro-fill", action="store_true", help="Enable eval tracker retro-fill (extra mid-point DET update on effective boost)")
    ap.add_argument("--no-retro-fill", action="store_true", help="Disable retro-fill even if enabled in config")
    ap.add_argument("--retro-max-gap-ms", type=int, default=None, help="Retro-fill gap control: if (now_ms - MID) > this, add MID2; if (now_ms - MID2) > this, add MID3 (default 200ms)")

    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.config:
        cfg, base_dir = load_config(args.config)
        tick_ms = int(args.tick_ms if args.tick_ms is not None else cfg.tick_ms)
        # Resolve IoU thresholds. We separate evaluation threshold vs dynamic-weight gate threshold.
        # Backward-compat: --iou-thr sets BOTH unless separately overridden.
        legacy_iou_thr = args.iou_thr
        metric_iou_thr = float(args.metric_iou_thr) if args.metric_iou_thr is not None else float(getattr(cfg, "metric_iou_thr", cfg.iou_thr))
        gate_iou_thr = float(args.gate_iou_thr) if args.gate_iou_thr is not None else float(getattr(cfg, "gate_iou_thr", cfg.iou_thr))
        if legacy_iou_thr is not None:
            if args.metric_iou_thr is None:
                metric_iou_thr = float(legacy_iou_thr)
            if args.gate_iou_thr is None:
                gate_iou_thr = float(legacy_iou_thr)
        iou_min_top_pct = float(args.iou_min_top_pct if args.iou_min_top_pct is not None else cfg.iou_min_top_pct)
        det_score_thr = float(args.det_score_thr if args.det_score_thr is not None else cfg.det_score_thr)
        iou_det_score_thr = (
            float(args.iou_det_score_thr)
            if args.iou_det_score_thr is not None
            else (float(cfg.iou_det_score_thr) if cfg.iou_det_score_thr is not None else None)
        )

        # Optionally align tracker score thresholds with the DET/IoU-gate threshold so that
        # "what we match" and "what the tracker can actually create/maintain" are in the same system.
        # This is important for your dynamic-weight logic that compares *prediction* vs *HIEVE DET*.
        align_tracker_thr = bool(args.align_tracker_thr) or bool(getattr(cfg, "align_tracker_thr", False))
        tracker_cfg = dict(cfg.tracker_cfg or {})
        if align_tracker_thr:
            thr = float(iou_det_score_thr) if iou_det_score_thr is not None else float(det_score_thr)
            # When aligning, also make IoU gate use the same threshold if not explicitly provided.
            if iou_det_score_thr is None:
                iou_det_score_thr = thr
            tracker_cfg.update({
                "track_high_thresh": thr,
                "new_track_thresh": thr,
                "track_low_thresh": thr,
            })
            LOGGER.info(
                f"[align_tracker_thr] set ByteTrack thresholds to {thr:.3f} (track_high/new/low), "
                f"and iou_det_score_thr={float(iou_det_score_thr):.3f}"
            )

        sources: list[SourceSpec] = []
        for i, s in enumerate(cfg.sources):
            # placeholder; we'll fill after loading
            sources.append(SourceSpec(
                name=s.name,
                fps=s.fps,
                num_frames=max(0, int(s.num_frames or 0)),
                gt=None,
                det=None,
                is_empty=bool(getattr(s, "is_empty", False)),
            ))

        # Pass 1: load all normal streams (gt+det files present) and gather their durations.
        loaded_info = []  # (idx, gt_seq, det_seq, native_frames, duration_ms)
        for idx, s in enumerate(cfg.sources):
            if getattr(s, "is_empty", False):
                continue
            if not (s.gt and s.det):
                # Config loader converts missing gt/det into is_empty==True, but keep safe.
                continue

            gt_path = (base_dir / s.gt).resolve()
            det_path = (base_dir / s.det).resolve()

            hint = int(s.num_frames or (args.max_frames if args.max_frames > 0 else 0) or 0)
            gt_seq = load_mot_labels(str(gt_path), allow_empty=True, num_frames_hint=hint)
            det_seq = load_mot_detections(str(det_path), score_thr=det_score_thr, allow_empty=True, num_frames_hint=hint)

            native_frames = gt_seq.num_frames  # evaluation is against GT length by default
            # Optional: cap by user-provided per-source num_frames
            if s.num_frames is not None and int(s.num_frames) > 0:
                native_frames = min(native_frames, int(s.num_frames))
            # Optional: cap by CLI max_frames
            if args.max_frames > 0:
                native_frames = min(native_frames, int(args.max_frames))

            duration_ms = int((native_frames / max(1e-6, float(s.fps))) * 1000.0) if native_frames > 0 else 0
            loaded_info.append((idx, gt_seq, det_seq, native_frames, duration_ms))

        if not loaded_info:
            ap.error("No valid gt+det sources found in config.")

        # Global simulation duration rule (updated per your latest requirement):
        #   - Default global duration = min duration across ALL channels (real channels + any empty channels with explicit num_frames).
        #   - If user requests a duration, it can only shorten; if it tries to exceed the min duration, we warn and clamp to min.
        min_ms = min(d for _i, _g, _d, _nf, d in loaded_info if d > 0)
        total_ms = min_ms

        # Empty channels with explicit num_frames can further reduce the global min duration.
        for s in cfg.sources:
            if getattr(s, "is_empty", False) and s.num_frames is not None and int(s.num_frames) > 0:
                d_ms = int((int(s.num_frames) / max(1e-6, float(s.fps))) * 1000.0)
                if d_ms < total_ms:
                    LOGGER.warning(f"Empty stream {s.name} duration {d_ms}ms is shorter than current min {total_ms}ms; global duration will follow the min rule and become {d_ms}ms.")
                    total_ms = d_ms

        # Optional user-requested duration (config or CLI)
        req_ms = None
        if args.sim_duration_ms is not None and args.sim_duration_ms > 0:
            req_ms = int(args.sim_duration_ms)
        elif args.sim_duration_s is not None and args.sim_duration_s > 0:
            req_ms = int(float(args.sim_duration_s) * 1000.0)
        elif int(getattr(cfg, "sim_duration_ms", 0) or 0) > 0:
            req_ms = int(getattr(cfg, "sim_duration_ms", 0))

        if req_ms is not None:
            if req_ms > total_ms:
                LOGGER.warning(f"Requested sim duration {req_ms}ms > min channel duration {total_ms}ms; clamping to {total_ms}ms.")
            else:
                total_ms = req_ms

        # Pass 2: finalize ALL sources (normal + empty-from-null + explicit empty) using total_ms.
        loaded_map = {i: (gt_seq, det_seq, native_frames, d_ms) for (i, gt_seq, det_seq, native_frames, d_ms) in loaded_info}

        for idx, s in enumerate(cfg.sources):
            if getattr(s, "is_empty", False):
                # Empty stream participates in scheduling/compute, but has no objects.
                num_frames = int((total_ms * float(s.fps)) / 1000.0)
                num_frames = max(0, num_frames)
                gt_seq = make_empty_labels(num_frames, path=f"<empty:{s.name}>")
                det_seq = make_empty_detections(num_frames, path=f"<empty:{s.name}>")
                sources[idx] = SourceSpec(name=s.name, fps=s.fps, num_frames=num_frames, gt=gt_seq, det=det_seq, is_empty=True)
                continue

            if idx not in loaded_map:
                continue
            gt_seq, det_seq, native_frames, _dms = loaded_map[idx]

            max_by_time = int((total_ms * float(s.fps)) / 1000.0)
            num_frames = min(native_frames, max_by_time) if max_by_time > 0 else 0

            gt_seq = _slice_labels(gt_seq, num_frames)
            gt_seq = _pad_labels(gt_seq, num_frames)
            det_seq = _slice_dets(det_seq, num_frames)
            det_seq = _pad_dets(det_seq, num_frames)

            sources[idx] = SourceSpec(name=s.name, fps=s.fps, num_frames=num_frames, gt=gt_seq, det=det_seq, is_empty=False)

        # Auto-insert empty streams if requested (they participate in scheduling but have no objects).
        empty_n = args.empty_channels if args.empty_channels is not None else int(getattr(cfg, 'empty_channels', 0) or 0)
        if empty_n > 0:
            base = next((s for s in sources if (s.gt is not None and s.det is not None and s.num_frames > 0 and (not str(s.gt.path).startswith('<empty:')))), None)
            # Per design: empty streams use a fixed nominal FPS (30fps) to emulate an always-on camera channel.
            base_fps = 30.0
            # Use ceil so empty streams never become the shortest channel due to rounding.
            base_frames = int(math.ceil((total_ms * base_fps) / 1000.0))
            for k in range(int(empty_n)):
                name = f"empty{k}"
                gt_seq = make_empty_labels(base_frames, path=f"<empty:{name}>")
                det_seq = make_empty_detections(base_frames, path=f"<empty:{name}>")
                sources.append(SourceSpec(name=name, fps=base_fps, num_frames=base_frames, gt=gt_seq, det=det_seq, is_empty=True))

        # Resolve run mode
        if args.mode is not None:
            mode = str(args.mode).strip().lower()
            dynamic_weight = (mode != "static")
            # retro-fill is only meaningful in dynamic mode.
            retro_fill_cli = (mode == "dyn-retro")
        else:
            mode = "dyn" if bool(args.dynamic_weight) else "static"
            dynamic_weight = bool(args.dynamic_weight)
            retro_fill_cli = bool(args.retro_fill)

        weight_log = args.weight_log
        if weight_log is None:
            # default name includes the selected run mode
            tag = str(mode).replace("-", "_")
            stem = Path(args.config).stem if args.config else "run"
            weight_log = str(out_dir / f"weights_{stem}_{tag}.csv")

        id_log = args.id_log
        if id_log is None:
            tag = str(mode).replace("-", "_")
            stem = Path(args.config).stem if args.config else "run"
            id_log = str(out_dir / f"new_ids_{stem}_{tag}.jsonl")
        if isinstance(id_log, str) and (id_log.strip() == ""):
            id_log = None

        weight_adjust_log = args.weight_adjust_log
        if weight_adjust_log is None:
            tag = str(mode).replace("-", "_")
            stem = Path(args.config).stem if args.config else "run"
            weight_adjust_log = str(out_dir / f"weight_adjust_{stem}_{tag}.jsonl")
        if isinstance(weight_adjust_log, str) and (weight_adjust_log.strip() == ""):
            weight_adjust_log = None

        iou_id_log = args.iou_id_log
        if iou_id_log is None:
            tag = str(mode).replace("-", "_")
            stem = Path(args.config).stem if args.config else "run"
            iou_id_log = str(out_dir / f"iou_id_{stem}_{tag}.jsonl")
        if isinstance(iou_id_log, str) and (iou_id_log.strip() == ""):
            iou_id_log = None

        # Retro-fill config (BYTE TRACK C)
        # Base: config value; then legacy CLI; then mode override (if provided).
        retro_fill = bool(getattr(cfg, "retro_fill", False))
        if bool(args.retro_fill):
            retro_fill = True
        if bool(args.no_retro_fill):
            retro_fill = False
        if args.mode is not None:
            # Mode explicitly controls retro behavior.
            retro_fill = bool(retro_fill_cli)
        retro_gap_ms = int(args.retro_max_gap_ms) if args.retro_max_gap_ms is not None else int(getattr(cfg, "retro_max_gap_ms", 200))
        if retro_gap_ms < 1:
            retro_gap_ms = 1

        LOGGER.info(f"[thr] metric_iou_thr={metric_iou_thr:.3f} gate_iou_thr={gate_iou_thr:.3f} (legacy_iou_thr={legacy_iou_thr})")

        results, overall = run_simulation_sources(
            sources,
            tick_ms=tick_ms,
            iou_thr=float(metric_iou_thr),
            metric_iou_thr=float(metric_iou_thr),
            gate_iou_thr=float(gate_iou_thr),
            iou_det_score_thr=iou_det_score_thr,
            iou_min_top_pct=iou_min_top_pct,
            tracker_cfg=tracker_cfg,
            metrics_backend=str(args.metrics_backend),
            verbose_engine=args.verbose_engine,
            seed=int(args.seed),
            dynamic_weight=bool(dynamic_weight),
            weight_log_path=weight_log,
            id_log_path=id_log,
            weight_adjust_log_path=weight_adjust_log,
            iou_id_log_path=iou_id_log,
            evolution_steps=int(args.evolution_steps),
            retro_fill=retro_fill,
            retro_max_gap_ms=retro_gap_ms,
            run_mode=mode,
        )

        for r in results:
            LOGGER.info(f"[{r.name}] frames={r.num_frames} tick_ms={r.tick_ms} fps={r.fps}")
            LOGGER.info(f"[{r.name}] {_fmt_metrics(r.metrics)}")
            if args.save_pred:
                out_path = out_dir / f"pred_{r.name}.txt"
                save_mot_predictions(str(out_path), r.pred_frames)
                LOGGER.info(f"[{r.name}] Saved predictions: {out_path}")

        LOGGER.info("[OVERALL] " + _fmt_metrics(overall.metrics))
        return 0

    # legacy mode disabled
    ap.error("--labels legacy mode is disabled. Please use --config with per-source gt+det.")


if __name__ == "__main__":
    raise SystemExit(main())
