#!/usr/bin/env python3
"""
demo_scaling.py

Run the "insert 0..N empty channels" experiment and print how MOTP/IDF1 degrade.

Definition used here:
- "Empty channel" = has the same number of frames as the main channel, but GT/DET are empty on all frames.
  This means it still participates in scheduling (consumes compute slice), but never outputs detections.

If you instead configure GT/DET as null/missing, the channel is considered inactive and is NOT scheduled,
therefore it will NOT reduce compute for active channels.
"""
from __future__ import annotations

import sys
from pathlib import Path
# Allow running directly from the repo without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from typing import List, Tuple

from hieve_sim.hieve_reader import (
    load_mot_labels, load_mot_detections,
    MOTLabelSequence, MOTDetSequence,
)
from hieve_sim.simulator import SourceSpec, run_simulation_sources


def make_empty_sequences(num_frames: int, tag: str) -> Tuple[MOTLabelSequence, MOTDetSequence]:
    frames = [[] for _ in range(num_frames)]
    gt = MOTLabelSequence(frames=frames, max_frame=num_frames - 1, path=tag)
    det = MOTDetSequence(frames=frames, max_frame=num_frames - 1, path=tag)
    return gt, det


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="GT label file (MOT style)")
    ap.add_argument("--det", required=True, help="DET file (MOT style)")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--tick-ms", type=int, default=300)
    ap.add_argument("--iou-thr", type=float, default=0.5)
    ap.add_argument("--det-score-thr", type=float, default=0.1)
    ap.add_argument("--max-empty", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    gt_seq = load_mot_labels(args.gt, allow_empty=False)
    det_seq = load_mot_detections(args.det, score_thr=args.det_score_thr, allow_empty=False)

    num_frames = gt_seq.num_frames
    print(f"Loaded GT frames={gt_seq.num_frames} boxes={gt_seq.num_boxes}")
    print(f"Loaded DET frames={det_seq.num_frames} boxes(after score_thr)={det_seq.num_boxes}")
    print("")
    print("empty\tchannels\tavg_gap_s\tavg_gap_frames\tMOTP\t\tIDF1")
    print("-" * 72)

    for n_empty in range(0, args.max_empty + 1):
        sources: List[SourceSpec] = [
            SourceSpec(name="main", fps=args.fps, num_frames=num_frames, gt=gt_seq, det=det_seq)
        ]
        for i in range(n_empty):
            egt, edet = make_empty_sequences(num_frames, tag=f"(empty{i})")
            sources.append(SourceSpec(name=f"empty{i}", fps=args.fps, num_frames=num_frames, gt=egt, det=edet))

        _per, overall = run_simulation_sources(
            sources,
            tick_ms=args.tick_ms,
            iou_thr=args.iou_thr,
            tracker_cfg=None,
            seed=args.seed,
            verbose_engine=False,
        )
        C = n_empty + 1
        avg_gap_s = (args.tick_ms / 1000.0) * C
        avg_gap_frames = args.fps * avg_gap_s
        motp = overall.metrics["motp"]
        idf1 = overall.metrics["idf1"]
        print(f"{n_empty}\t{C}\t\t{avg_gap_s:.3f}\t\t{avg_gap_frames:.1f}\t\t{motp:.6f}\t{idf1:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
