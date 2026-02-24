#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Analyze weight (boost/decay/retro) and new GT-ID events.

Usage example:
  python tools_analyze_logs.py \
    --weight-log outputs_dbg_scene5_50ms/weights_scene5_dyn_retro_10s_tick50.csv \
    --id-log outputs_dbg_scene5_50ms/new_ids_scene5_dyn_retro_10s_tick50.jsonl \
    --topk 20
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weight-log', required=True)
    ap.add_argument('--id-log', required=True)
    ap.add_argument('--topk', type=int, default=10)
    args = ap.parse_args()

    w = pd.read_csv(args.weight_log)
    boosts = w[w['action'].isin(['boost', 'boost_sat'])]
    decays = w[w['action'].isin(['decay', 'decay_sat'])]

    print('=== Weight events summary (per channel) ===')
    names = sorted(w['name'].unique(), key=lambda x: int(x[2:]) if x.startswith('ch') else 999)
    for name in names:
        sub = w[w['name'] == name]
        b = boosts[boosts['name'] == name]
        d = decays[decays['name'] == name]
        retro = sub[sub.get('retro', False) == True] if 'retro' in sub.columns else sub.iloc[0:0]

        bf = b[b['action'] == 'boost']['frame'].tolist()
        bs = b[b['action'] == 'boost_sat']['frame'].tolist()
        df_ = d[d['action'] == 'decay']['frame'].tolist()
        ds = d[d['action'] == 'decay_sat']['frame'].tolist()

        print(f"{name}: boost={len(bf)} boost_sat={len(bs)} decay={len(df_)} decay_sat={len(ds)} retro={len(retro)}")
        if bf:
            print('  boost frames:', bf[:args.topk])
        if df_:
            print('  decay frames:', df_[:args.topk])
        if len(retro):
            tmp = retro[['frame', 'retro_mid_frame', 'retro_mid_ms', 'delta', 'now_ms']].head(args.topk)
            print('  retro examples:')
            print(tmp.to_string(index=False))

    # Load id log
    print('\n=== New GT-ID events (first N per channel) ===')
    events_by_name = defaultdict(list)
    with open(args.id_log, 'r', encoding='utf-8') as f:
        for line in f:
            e = json.loads(line)
            events_by_name[e['name']].append(e)

    for name in names:
        es = events_by_name.get(name, [])
        print(f"{name}: {len(es)} events")
        for e in es[: min(args.topk, len(es))]:
            print(f"  frame={e['frame']:<4d} t_ms={e['t_ms']:<5d} new_ids={e['new_ids']} gt_box_cnt={e['gt_box_cnt']}")


if __name__ == '__main__':
    main()
