"""
add_eicu_admit_time.py -- inject a temporal ordering key into eicu_meta.csv.

eICU exposes only a coarse discharge year (patient.hospitaldischargeyear). We define
time_order = integer rank of (hospitaldischargeyear, patientunitstayid) in ascending order,
using the (roughly monotone) stay id to break ties within a year. Writing this column enables
the temporal split used in the paper:
  train_eicu_mtsfl.py --test_mode temporal --test_time_key time_order
Usage: python add_eicu_admit_time.py --raw_dir <raw eICU csv dir> --data_dir <data>
"""
import argparse, os
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True,
                    help="eICU-CRD raw CSV directory (location of patient.csv)")
    ap.add_argument("--data_dir", default="mtsfl_eicu", help="location of eicu_meta.csv")
    ap.add_argument("--patient_csv", default="patient.csv")
    ap.add_argument("--out_col", default="time_order")
    ap.add_argument("--dry_run", action="store_true", help="print statistics only, do not write")
    a = ap.parse_args()

    meta_path = os.path.join(a.data_dir, "eicu_meta.csv")
    meta = pd.read_csv(meta_path)
    pt_path = os.path.join(a.raw_dir, a.patient_csv)
    usecols = ["patientunitstayid", "hospitaldischargeyear"]
    pt = pd.read_csv(pt_path, usecols=usecols).drop_duplicates("patientunitstayid")
    print(f"[load] meta rows={len(meta):,}  patient rows={len(pt):,}")
    print(f"[year] hospitaldischargeyear distribution:\n{pt['hospitaldischargeyear'].value_counts().sort_index()}")

    m = meta.merge(pt, on="patientunitstayid", how="left")
    miss = int(m["hospitaldischargeyear"].isna().sum())
    if miss:
        print(f"[warn] {miss} missing years -> filled with the max year (treated as most recent)")
        m["hospitaldischargeyear"] = m["hospitaldischargeyear"].fillna(
            m["hospitaldischargeyear"].max())

    # ascending rank of (year, unitstayid) = time_order (larger = more recent)
    order = np.lexsort((m["patientunitstayid"].to_numpy(),
                        m["hospitaldischargeyear"].to_numpy()))
    time_order = np.empty(len(m), dtype=np.int64)
    time_order[order] = np.arange(len(m))
    m[a.out_col] = time_order

    # diagnostics: check recency by year + which years the most-recent 20% per client fall in
    print(f"[{a.out_col}] range=[{time_order.min()},{time_order.max()}]")
    yr = m["hospitaldischargeyear"]
    print(f"  year distribution of the most-recent-20% (by time_order) patients:")
    thr = np.percentile(time_order, 80)
    print(m.loc[time_order >= thr, "hospitaldischargeyear"].value_counts().sort_index())

    if a.dry_run:
        print("[dry_run] skipping save")
        return
    # add only the new column to meta and overwrite (row order preserved)
    meta[a.out_col] = m[a.out_col].values
    meta.to_csv(meta_path, index=False)
    print(f"[write] {meta_path}: added column '{a.out_col}'.")
    print(f"  -> now use: --test_mode temporal --test_time_key {a.out_col}")


if __name__ == "__main__":
    main()
