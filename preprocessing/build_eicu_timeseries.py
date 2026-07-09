#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_eicu_timeseries.py -- additional vital-sign channels appended to the backbone input.

Extracts extra time-series channels on the SAME grid as eicu_X (first 24h, 5-min bins, 288 steps):
  NIBP systolic / diastolic / mean  (vitalAperiodic)
  GCS Total                         (nurseCharting)
Row order matches eicu_meta.csv / eicu_X. Values are raw (clipped only); missing = 0 with a separate
observed mask. Loaded and concatenated at train time via --extra_channels.
Output: eicu_X_extra.npy (N, 288, M), eicu_mask_extra.npy (N, 288, M), eicu_extra_names.json
Usage : python build_eicu_timeseries.py --data_dir <data> --vitalaperiodic <path> --nursecharting <path>
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

ID = "patientunitstayid"
VA_OFFSET = ["observationoffset"]
NC_OFFSET = ["nursingchartoffset", "observationoffset"]
NC_NAME = "nursingchartcelltypevalname"
NC_VAL = "nursingchartvalue"

# (source, key, out_name, clip_lo, clip_hi)   source: "va"=vitalAperiodic column, "nc"=nurseCharting valname
CORE = [
    ("va", "noninvasivesystolic",  "nibp_sys",  40, 250),
    ("va", "noninvasivediastolic", "nibp_dia",  10, 200),
    ("va", "noninvasivemean",      "nibp_mean", 20, 220),
    ("nc", "GCS Total",            "gcs_total",  3,  15),
]
OPTIONAL = [
    ("nc", "Pain Score",      "pain",            0,   10),
    ("nc", "Bedside Glucose", "glucose_bedside", 20, 1000),
]


def find_col(cols, cands):
    for c in cands:
        if c in cols:
            return c
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="mtsfl_eicu", help="location of eicu_meta.csv / eicu_X.npy")
    ap.add_argument("--vitalaperiodic", required=True)
    ap.add_argument("--nursecharting", required=True)
    ap.add_argument("--window_min", type=int, default=1440)
    ap.add_argument("--bin_min", type=int, default=5)
    ap.add_argument("--chunksize", type=int, default=2_000_000)
    ap.add_argument("--all_extra", action="store_true", help="also include Pain/Bedside Glucose (ablation)")
    ap.add_argument("--out_suffix", default="", help="output filename suffix (e.g. _all)")
    a = ap.parse_args()

    n_bins = a.window_min // a.bin_min
    signals = CORE + (OPTIONAL if a.all_extra else [])
    names = [s[2] for s in signals]
    print(f"[cfg] window={a.window_min}m bin={a.bin_min}m bins={n_bins}  channels={names}")

    # ---- stay -> row index (eicu_X row order = eicu_meta row order) ----
    meta = pd.read_csv(os.path.join(a.data_dir, "eicu_meta.csv"), usecols=[ID])
    stay_ids = meta[ID].astype(np.int64).to_numpy()
    N = len(stay_ids)
    row_of = {int(s): i for i, s in enumerate(stay_ids)}
    stayset = set(row_of)
    # alignment check: matches the number of eicu_X rows
    xp = os.path.join(a.data_dir, "eicu_X.npy")
    if os.path.exists(xp):
        xrows = np.load(xp, mmap_mode="r").shape[0]
        assert xrows == N, f"eicu_X rows({xrows}) != eicu_meta rows({N}) -- alignment mismatch"
        print(f"[align] eicu_X rows={xrows} == meta rows={N}  OK")
    print(f"[meta] stays={N:,}")

    # per-channel sum/cnt accumulation (float32)
    SUM = {nm: np.zeros((N, n_bins), np.float32) for nm in names}
    CNT = {nm: np.zeros((N, n_bins), np.float32) for nm in names}

    def accumulate(rows_idx, bins, vals, lo, hi, nm):
        ok = np.isfinite(vals) & (vals >= lo) & (vals <= hi)   # values outside the physiological range are dropped (missing)
        if not ok.any():
            return
        ri = rows_idx[ok]; bi = bins[ok]; vv = vals[ok].astype(np.float32)
        np.add.at(SUM[nm], (ri, bi), vv)
        np.add.at(CNT[nm], (ri, bi), 1.0)

    # ---- vitalAperiodic: NIBP ----
    va_sig = [s for s in signals if s[0] == "va"]
    if va_sig:
        print("[vitalAperiodic] scanning...")
        off_col = None
        for ch in pd.read_csv(a.vitalaperiodic, chunksize=a.chunksize):
            if off_col is None:
                off_col = find_col(ch.columns, VA_OFFSET)
                if off_col is None:
                    raise SystemExit(f"offset column not found: {VA_OFFSET}")
            ch = ch[ch[ID].isin(stayset)]
            ch = ch[(ch[off_col] >= 0) & (ch[off_col] < a.window_min)]
            if ch.empty:
                continue
            ri = np.array([row_of[int(s)] for s in ch[ID].to_numpy()], dtype=np.int64)
            bi = (ch[off_col].to_numpy() // a.bin_min).astype(np.int64)
            bi = np.clip(bi, 0, n_bins - 1)
            for _, key, nm, lo, hi in va_sig:
                if key in ch.columns:
                    accumulate(ri, bi, pd.to_numeric(ch[key], errors="coerce").to_numpy(), lo, hi, nm)

    # ---- nurseCharting: GCS Total (+ optional) ----
    nc_sig = [s for s in signals if s[0] == "nc"]
    if nc_sig:
        want = {s[1]: s for s in nc_sig}   # valname -> spec
        wantset = set(want)
        print("[nurseCharting] scanning...")
        off_col = None
        for ch in pd.read_csv(a.nursecharting, chunksize=a.chunksize):
            if off_col is None:
                off_col = find_col(ch.columns, NC_OFFSET)
                if off_col is None:
                    raise SystemExit(f"offset column not found: {NC_OFFSET}")
            sub = ch[ch[ID].isin(stayset)]
            sub = sub[(sub[off_col] >= 0) & (sub[off_col] < a.window_min)]
            if sub.empty:
                continue
            sub = sub[sub[NC_NAME].astype(str).isin(wantset)]
            if sub.empty:
                continue
            vnum = pd.to_numeric(sub[NC_VAL], errors="coerce").to_numpy()
            ri = np.array([row_of[int(s)] for s in sub[ID].to_numpy()], dtype=np.int64)
            bi = np.clip((sub[off_col].to_numpy() // a.bin_min).astype(np.int64), 0, n_bins - 1)
            vn = sub[NC_NAME].astype(str).to_numpy()
            for valname, (_, _, nm, lo, hi) in want.items():
                m = (vn == valname)
                if m.any():
                    accumulate(ri[m], bi[m], vnum[m], lo, hi, nm)

    # ---- assemble channels (raw mean, missing=0, mask) ----
    M = len(names)
    X = np.zeros((N, n_bins, M), np.float32)
    Mk = np.zeros((N, n_bins, M), np.float32)
    report = {}
    for ci, nm in enumerate(names):
        cnt = CNT[nm]
        obs = cnt > 0
        val = np.zeros_like(cnt)
        val[obs] = SUM[nm][obs] / cnt[obs]
        X[:, :, ci] = val
        Mk[:, :, ci] = obs.astype(np.float32)
        cov = float((obs.any(axis=1)).mean())   # fraction of stays with >=1 observation
        spec = next(s for s in signals if s[2] == nm)
        report[nm] = {"source": spec[0], "key": spec[1], "clip": [spec[3], spec[4]],
                      "coverage_frac": round(cov, 4),
                      "obs_per_covered_stay": round(float(cnt[obs.any(axis=1)].sum(axis=1).mean()), 2)
                      if obs.any() else 0.0}
        print(f"  ch[{ci}] {nm:<16s} coverage={cov*100:5.1f}%  obs/stay={report[nm]['obs_per_covered_stay']:.1f}")

    suf = a.out_suffix
    np.save(os.path.join(a.data_dir, f"eicu_X_extra{suf}.npy"), X)
    np.save(os.path.join(a.data_dir, f"eicu_mask_extra{suf}.npy"), Mk)
    json.dump({"channels": names, "n_bins": n_bins, "bin_min": a.bin_min,
               "window_min": a.window_min, "all_extra": a.all_extra, "detail": report},
              open(os.path.join(a.data_dir, f"eicu_extra_names{suf}.json"), "w"),
              indent=2, ensure_ascii=False)
    print(f"\n[done] eicu_X_extra{suf}.npy {X.shape}  +  eicu_mask_extra{suf}.npy  →  {a.data_dir}/")
    print("       Loaded/concatenated in the model via --extra_channels. (If coverage matches the probe, alignment is correct.)")


if __name__ == "__main__":
    main()
