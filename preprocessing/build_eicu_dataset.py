#!/usr/bin/env python3
"""
build_eicu_dataset.py -- core eICU dataset builder (MT-SFL JSAC revision).

Produces the base cohort and the 4-channel vital time-series tensor that all other
preprocessing steps and the training code build on.

Task   : first-24h ICU vital time series -> primary-diagnosis category (multi-class base label)
Input  : heartrate, respiration, spo2 (= sao2 in eICU), temperature  (+ missingness mask)
         + static: age, gender, admissionweight
Window : 24h, 5-min grid (288 bins); only stays with >= 12h (144 bins) of observations
Label  : primary diagnosis (organ|category); classes with global support < 100 dropped
Client : hospital x unittype; clients with < 100 usable stays dropped
Split  : global 80/20 split on patienthealthsystemstayid (patient-level, no stay leakage)

Nine steps:
 1) read patient / diagnosis / hospital
 2) drop hospitals with a missing census region
 3) build the primary-diagnosis label (earliest diagnosisoffset)
 4) drop rare classes (global support < 100)
 5) build the 24h 4-channel vitalPeriodic tensor
 6) filter usable stays (coverage >= 12h)
 7) drop clients with < 100 stays
 8) global patient-level train/test split
 9) print final statistics and save

Notes:
 - eICU stores SpO2 in a column named 'sao2' (we expose it as the channel 'spo2').
 - systemic BP from vitalPeriodic is arterial-line based and thus severity-correlated in its
   missingness, so it is intentionally excluded to avoid learning "sicker => more measured".
 - vitalPeriodic (~1.7 GB) is read in chunks; tensor accumulation uses ~1-1.5 GB RAM.

NOTE (MT-SFL phenotype experiments): this builder writes a 72-class primary-diagnosis label as the
base cohort label. The paper's 25-label PHENOTYPE target is produced on top of this cohort by
build_eicu_phenotype.py, and the temporal split key is added by add_eicu_admit_time.py.

Usage:
  python build_eicu_dataset.py --eicu_dir <raw eICU csv dir>   # defaults reproduce the paper spec
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

# output channel name -> raw vitalPeriodic column name
CHANNELS = ["heartrate", "respiration", "spo2", "temperature"]
SRC_COLS = {"heartrate": "heartrate", "respiration": "respiration",
            "spo2": "sao2", "temperature": "temperature"}
LABEL_DEPTH = 2

# physiologically plausible ranges (values outside are sensor artifacts -> treated as missing)
VRANGE = {"heartrate": (20, 250), "respiration": (3, 80),
          "spo2": (50, 100), "temperature": (25, 45)}


def clean_vitals(ch, vals):
    """Per-reading cleaning: convert temperature F->C, then set out-of-range values to NaN (missing)."""
    if ch == "temperature":
        vals = np.where(vals > 45, (vals - 32.0) * 5.0 / 9.0, vals)  # looks like F -> convert to C
    lo, hi = VRANGE[ch]
    return np.where((vals >= lo) & (vals <= hi), vals, np.nan)


def find_file(eicu_dir, basename):
    # case-insensitive match (eICU filenames use mixed case, e.g. vitalPeriodic)
    targets = {(basename + ".csv.gz").lower(): 0, (basename + ".csv").lower(): 1}
    best = None
    for fn in os.listdir(eicu_dir):
        rank = targets.get(fn.lower())
        if rank is not None and (best is None or rank < best[0]):
            best = (rank, os.path.join(eicu_dir, fn))
    if best:
        return best[1]
    raise FileNotFoundError(os.path.join(eicu_dir, basename + ".csv[.gz]"))


def parse_label(dx_string, depth=LABEL_DEPTH):
    parts = [p.strip() for p in str(dx_string).split("|") if p.strip()]
    if not parts:
        return None
    return "|".join(parts[:depth])


def clean_age(series):
    """eICU age: integer strings plus '> 89' (89 or older). Map '>89' -> 90, otherwise numeric."""
    s = series.astype(str).str.replace(">", "", regex=False).str.strip()
    s = s.replace({"89": "89"})  # no-op, kept for readability
    s = pd.to_numeric(s, errors="coerce")
    s = s.where(series.astype(str).str.contains("89", na=False) | s.notna(), other=s)
    # entries that were '> 89' became 89 above; bump them to 90
    over = series.astype(str).str.contains(">", na=False)
    s = s.mask(over, 90)
    return s


def ffill_2d(a):
    """Forward-fill a (N, T) array along the time axis (axis=1). Leading NaNs are left as-is."""
    n, t = a.shape
    mask = ~np.isnan(a)
    idx = np.where(mask, np.arange(t)[None, :], 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    return a[np.arange(n)[:, None], idx]


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eicu_dir", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--out_dir", default="mtsfl_eicu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--window_min", type=int, default=1440)   # 24h
    ap.add_argument("--bin_min", type=int, default=5)          # 5-min grid
    ap.add_argument("--min_coverage_min", type=int, default=720)  # 12h
    ap.add_argument("--rare_class_min", type=int, default=100)
    ap.add_argument("--client_min_stays", type=int, default=100)
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--vp_chunksize", type=int, default=1_000_000)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    n_bins = args.window_min // args.bin_min          # 288
    min_cov_bins = args.min_coverage_min // args.bin_min  # 144
    rng = np.random.default_rng(args.seed)

    # ---- 1) + 2) read hospital table, drop hospitals with a missing region ----
    section("[1-2] hospital table / drop hospitals without a census region")
    hosp = pd.read_csv(find_file(args.eicu_dir, "hospital"))
    hosp.columns = [c.strip().lower() for c in hosp.columns]
    good_hosp = set(hosp.loc[hosp["region"].notna(), "hospitalid"])
    print(f"total hospitals {hosp['hospitalid'].nunique()} -> with region {len(good_hosp)}")

    pcols = ["patientunitstayid", "patienthealthsystemstayid", "hospitalid",
             "unittype", "age", "gender", "admissionweight"]
    pat = pd.read_csv(find_file(args.eicu_dir, "patient"),
                      usecols=lambda c: c.strip().lower() in pcols)
    pat.columns = [c.strip().lower() for c in pat.columns]
    pat = pat[pat["hospitalid"].isin(good_hosp)].copy()
    pat["client"] = pat["hospitalid"].astype(str) + "::" + pat["unittype"].fillna("UNK")
    print(f"stays {pat['patientunitstayid'].nunique():,} / clients {pat['client'].nunique()}")

    # ---- 3) build primary-diagnosis label ----
    section("[3] primary diagnosis (earliest diagnosisoffset)")
    dcols = ["patientunitstayid", "diagnosisstring", "diagnosisoffset", "diagnosispriority"]
    dx = pd.read_csv(find_file(args.eicu_dir, "diagnosis"),
                     usecols=lambda c: c.strip().lower() in dcols)
    dx.columns = [c.strip().lower() for c in dx.columns]
    dx = dx[dx["diagnosispriority"].astype(str).str.strip().str.lower() == "primary"]
    dx = dx.dropna(subset=["diagnosisstring"])
    dx = dx.sort_values("diagnosisoffset").drop_duplicates("patientunitstayid", keep="first")
    dx["label"] = dx["diagnosisstring"].map(parse_label)
    dx = dx.dropna(subset=["label"])
    stay_label = dict(zip(dx["patientunitstayid"], dx["label"]))

    pat_ids = set(pat["patientunitstayid"])
    have_primary = sum(1 for s in pat_ids if s in stay_label)
    print(f"patient stays with a primary dx: {have_primary:,} / {len(pat_ids):,} "
          f"({have_primary/max(len(pat_ids),1):.1%})")

    # ---- 4) drop rare classes (global support < 100) ----
    section(f"[4] drop rare classes (support < {args.rare_class_min})")
    pat = pat[pat["patientunitstayid"].isin(stay_label)].copy()
    pat["label"] = pat["patientunitstayid"].map(stay_label)
    support = pat["label"].value_counts()
    kept_classes = set(support[support >= args.rare_class_min].index)
    pat = pat[pat["label"].isin(kept_classes)].copy()
    print(f"classes {len(support)} -> kept {len(kept_classes)} / stays {len(pat):,}")

    # candidate stay set S (targets for the vitals tensor below)
    S = pat["patientunitstayid"].tolist()
    stay_to_idx = {s: i for i, s in enumerate(S)}
    n_s = len(S)

    # ---- 5) 24h 4-channel vitalPeriodic tensor ----
    section(f"[5] build vitalPeriodic tensor (N={n_s:,}, bins={n_bins}, ch=4)")
    sum_flat = np.zeros((n_s * n_bins, 4), dtype=np.float64)
    cnt_flat = np.zeros((n_s * n_bins, 4), dtype=np.int32)

    vcols = ["patientunitstayid", "observationoffset"] + [SRC_COLS[c] for c in CHANNELS]
    reader = pd.read_csv(find_file(args.eicu_dir, "vitalperiodic"),
                         usecols=lambda c: c.strip().lower() in vcols,
                         chunksize=args.vp_chunksize)
    rows_used = 0
    for ci, chunk in enumerate(reader):
        chunk.columns = [c.strip().lower() for c in chunk.columns]
        chunk = chunk[(chunk["observationoffset"] >= 0) &
                      (chunk["observationoffset"] < args.window_min)]
        idx = chunk["patientunitstayid"].map(stay_to_idx)
        keep = idx.notna()
        if not keep.any():
            continue
        chunk = chunk[keep]
        idx = idx[keep].to_numpy().astype(np.int64)
        b = (chunk["observationoffset"].to_numpy() // args.bin_min).astype(np.int64)
        np.clip(b, 0, n_bins - 1, out=b)
        flat = idx * n_bins + b
        for c, ch in enumerate(CHANNELS):
            vals = clean_vitals(ch, chunk[SRC_COLS[ch]].to_numpy(dtype=np.float64))
            ok = ~np.isnan(vals)
            if not ok.any():
                continue
            sum_flat[:, c] += np.bincount(flat[ok], weights=vals[ok],
                                          minlength=n_s * n_bins)
            cnt_flat[:, c] += np.bincount(flat[ok],
                                          minlength=n_s * n_bins).astype(np.int32)
        rows_used += len(chunk)
        if (ci + 1) % 20 == 0:
            print(f"  ...chunk {ci+1}, cumulative rows used {rows_used:,}")
    print(f"vital observation rows used within the window: {rows_used:,}")

    cnt = cnt_flat.reshape(n_s, n_bins, 4)

    # ---- 6) filter usable stays (coverage >= 12h) ----
    section(f"[6] usable filter (coverage >= {args.min_coverage_min} min = {min_cov_bins} bins)")
    coverage = (cnt > 0).any(axis=2).sum(axis=1)        # number of observed 5-min bins
    pat = pat.reset_index(drop=True)
    pat["coverage_bins"] = coverage[[stay_to_idx[s] for s in pat["patientunitstayid"]]]
    usable = pat["coverage_bins"] >= min_cov_bins
    print(f"usable stays: {int(usable.sum()):,} / {len(pat):,}")
    pat = pat[usable].copy()

    # ---- 7) drop clients with < 100 stays ----
    section(f"[7] drop clients with < {args.client_min_stays} stays")
    csize = pat["client"].value_counts()
    kept_clients = set(csize[csize >= args.client_min_stays].index)
    pat = pat[pat["client"].isin(kept_clients)].copy()
    print(f"clients {len(csize)} -> kept {len(kept_clients)} / stays {len(pat):,}")

    # ---- 8) global patient-level 80/20 split ----
    section(f"[8] global patient-level (healthsystemstayid) {1-args.test_frac:.0%}/{args.test_frac:.0%} split")
    patients = pat["patienthealthsystemstayid"].unique()
    perm = rng.permutation(len(patients))
    n_test = int(round(len(patients) * args.test_frac))
    test_pat = set(patients[perm[:n_test]])
    pat["split"] = np.where(pat["patienthealthsystemstayid"].isin(test_pat),
                            "test", "train")
    print(f"patients {len(patients):,} -> test {len(test_pat):,} / "
          f"train stays {int((pat['split']=='train').sum()):,}, "
          f"test stays {int((pat['split']=='test').sum()):,}")

    # ---- assemble the final tensor subset + mean/mask/ffill ----
    final_idx = np.array([stay_to_idx[s] for s in pat["patientunitstayid"]])
    cnt_f = cnt[final_idx]                               # (Nf, T, 4)
    sum_f = sum_flat.reshape(n_s, n_bins, 4)[final_idx]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_f = np.where(cnt_f > 0, sum_f / cnt_f, np.nan)
    mask_f = (cnt_f > 0).astype(np.uint8)
    for c in range(4):                                  # forward-fill along the time axis
        mean_f[:, :, c] = ffill_2d(mean_f[:, :, c])
    X = np.nan_to_num(mean_f, nan=0.0).astype(np.float32)  # leading NaN -> 0 (distinguished by mask)

    # static feature cleanup
    pat["age_num"] = clean_age(pat["age"])
    pat["gender"] = pat["gender"].fillna("Unknown")

    # ---- 9) save + final statistics ----
    section("[9] save + final statistics")
    vocab = {lab: i for i, lab in enumerate(sorted(pat["label"].unique()))}
    pat["y"] = pat["label"].map(vocab)

    meta = pat[["patientunitstayid", "patienthealthsystemstayid", "client",
                "hospitalid", "unittype", "label", "y", "split",
                "age_num", "gender", "admissionweight", "coverage_bins"]].reset_index(drop=True)

    np.save(os.path.join(args.out_dir, "eicu_X.npy"), X)
    np.save(os.path.join(args.out_dir, "eicu_mask.npy"), mask_f)
    meta.to_csv(os.path.join(args.out_dir, "eicu_meta.csv"), index=False)
    with open(os.path.join(args.out_dir, "eicu_label_vocab.json"), "w") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    tr = meta[meta["split"] == "train"]
    maj = tr["y"].value_counts(normalize=True).max() if len(tr) else float("nan")
    summary = {
        "n_stays": int(len(meta)),
        "n_clients": int(meta["client"].nunique()),
        "n_hospitals": int(meta["hospitalid"].nunique()),
        "n_classes": len(vocab),
        "n_train": int((meta["split"] == "train").sum()),
        "n_test": int((meta["split"] == "test").sum()),
        "X_shape": list(X.shape),
        "majority_class_baseline_acc": float(maj),
        "channels": CHANNELS,
        "bins": n_bins,
        "min_coverage_bins": min_cov_bins,
    }
    with open(os.path.join(args.out_dir, "eicu_summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"  stays={summary['n_stays']:,}  clients={summary['n_clients']}  "
          f"classes={summary['n_classes']}")
    print(f"  train={summary['n_train']:,}  test={summary['n_test']:,}")
    print(f"  X shape={summary['X_shape']}  (forward-filled; mask stored separately)")
    print(f"  majority-class baseline acc (train) = {maj:.3f}")
    print(f"  -> proceed past step 8 once centralized macro-F1 clearly exceeds this baseline")
    print(f"\n[saved] {args.out_dir}/ : eicu_X.npy, eicu_mask.npy, eicu_meta.csv, "
          f"eicu_label_vocab.json, eicu_summary.json")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"File not found: {e}", file=sys.stderr)
        print("Check that hospital/patient/diagnosis/vitalPeriodic (.csv or .csv.gz) exist under --eicu_dir.",
              file=sys.stderr)
        sys.exit(1)
