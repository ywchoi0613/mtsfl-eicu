#!/usr/bin/env python3
"""
add_eicu_labs.py -- first-24h laboratory summary features.

For the final stay set (eicu_meta.csv), extracts per-lab [min, max, last] plus a measured mask
from lab.csv (offset in [0, window)); a wide candidate panel is auto-filtered to labs whose
measurement coverage is >= --min_coverage (K = 45 labs are retained by default). Each lab is
winsorized at p0.5/p99.5. Row order matches eicu_meta.csv.
Output: eicu_labs.npy (N, K, 3), eicu_lab_mask.npy (N, K), eicu_lab_names.json
Usage : python add_eicu_labs.py --data_dir <data> --eicu_dir <raw eICU csv dir> --min_coverage 0.1
"""
import argparse, json, os
import numpy as np, pandas as pd

# wide candidate panel (eICU labname). auto-filtered by coverage.
CANDIDATES = [
    # chemistry / metabolic
    "glucose", "bicarbonate", "BUN", "creatinine", "sodium", "potassium",
    "chloride", "calcium", "anion gap", "magnesium", "phosphate",
    "total protein", "albumin", "lactate",
    # hematology (CBC)
    "Hgb", "Hct", "WBC x 1000", "platelets x 1000", "RBC", "MCV", "MCH",
    "MCHC", "RDW", "MPV",
    # liver
    "total bilirubin", "direct bilirubin", "ALT (SGPT)", "AST (SGOT)", "alkaline phos.",
    # blood gas
    "pH", "paO2", "paCO2", "Base Excess", "O2 Sat (%)", "HCO3", "FiO2",
    # cardiac
    "troponin - I", "troponin - T", "CPK", "CPK-MB", "BNP",
    # coagulation
    "PT", "PTT", "PT - INR",
    # WBC differential
    "-polys", "-lymphs", "-monos", "-eos", "-basos",
]


def find_file(eicu_dir, basename):
    tgt = {(basename + ".csv.gz").lower(): 0, (basename + ".csv").lower(): 1}
    best = None
    for fn in os.listdir(eicu_dir):
        r = tgt.get(fn.lower())
        if r is not None and (best is None or r < best[0]):
            best = (r, os.path.join(eicu_dir, fn))
    if best:
        return best[1]
    raise FileNotFoundError(os.path.join(eicu_dir, basename + ".csv[.gz]"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="mtsfl_eicu", help="location of eicu_meta.csv")
    ap.add_argument("--eicu_dir", default=".", help="location of lab.csv")
    ap.add_argument("--out_dir", default=None, help="default: data_dir")
    ap.add_argument("--window_min", type=int, default=1440)
    ap.add_argument("--min_coverage", type=float, default=0.10,
                    help="labs measured in fewer than this fraction of stays are excluded")
    ap.add_argument("--chunksize", type=int, default=2_000_000)
    args = ap.parse_args()
    out_dir = args.out_dir or args.data_dir

    meta = pd.read_csv(os.path.join(args.data_dir, "eicu_meta.csv"))
    stay2row = {s: i for i, s in enumerate(meta["patientunitstayid"])}
    N = len(meta)
    cand_lc = {c.strip().lower(): c for c in CANDIDATES}
    print(f"stays={N:,}  candidate labs={len(CANDIDATES)}")

    # ---- read lab.csv in chunks and collect (row, lab, offset, value) ----
    cols = ["patientunitstayid", "labname", "labresult", "labresultoffset"]
    rows, labs, offs, vals = [], [], [], []
    lab_index = {}  # labname (original) -> integer idx
    reader = pd.read_csv(find_file(args.eicu_dir, "lab"),
                         usecols=lambda c: c.strip().lower() in cols,
                         chunksize=args.chunksize)
    nrow = 0
    for ci, ch in enumerate(reader):
        ch.columns = [c.strip().lower() for c in ch.columns]
        ch = ch[(ch["labresultoffset"] >= 0) & (ch["labresultoffset"] < args.window_min)]
        ch = ch.dropna(subset=["labresult"])
        lname = ch["labname"].astype(str).str.strip().str.lower()
        keep = lname.isin(cand_lc)
        ch = ch[keep]; lname = lname[keep]
        if len(ch) == 0:
            continue
        r = ch["patientunitstayid"].map(stay2row)
        m = r.notna()
        ch, lname, r = ch[m], lname[m], r[m].astype(int)
        lcanon = lname.map(cand_lc)   # lowercase match -> restore canonical spelling
        for ln in lcanon.unique():
            if ln not in lab_index:
                lab_index[ln] = len(lab_index)
        rows.append(r.to_numpy())
        labs.append(lcanon.map(lab_index).to_numpy())
        offs.append(ch["labresultoffset"].to_numpy())
        vals.append(pd.to_numeric(ch["labresult"], errors="coerce").to_numpy())
        nrow += len(ch)
        if (ci + 1) % 5 == 0:
            print(f"  ...chunk {ci+1}, cumulative lab rows {nrow:,}")
    print(f"lab observations within window/panel: {nrow:,}")

    df = pd.DataFrame({
        "row": np.concatenate(rows), "lab": np.concatenate(labs),
        "off": np.concatenate(offs), "val": np.concatenate(vals)}).dropna(subset=["val"])
    idx2lab = {v: k for k, v in lab_index.items()}
    K0 = len(lab_index)

    # ---- per-lab winsorize ----
    for li in range(K0):
        mask = df["lab"] == li
        v = df.loc[mask, "val"]
        if len(v) > 20:
            lo, hi = np.percentile(v, [0.5, 99.5])
            df.loc[mask, "val"] = v.clip(lo, hi)

    # ---- min/max/last/cnt per (row, lab) ----
    gmin = df.groupby(["row", "lab"])["val"].min()
    gmax = df.groupby(["row", "lab"])["val"].max()
    glast = df.sort_values("off").groupby(["row", "lab"])["val"].last()
    gcnt = df.groupby(["row", "lab"]).size()

    Lmin = np.zeros((N, K0)); Lmax = np.zeros((N, K0))
    Llast = np.zeros((N, K0)); LM = np.zeros((N, K0), dtype=np.uint8)
    for (r, l), v in gmin.items(): Lmin[r, l] = v
    for (r, l), v in gmax.items(): Lmax[r, l] = v
    for (r, l), v in glast.items(): Llast[r, l] = v
    for (r, l), v in gcnt.items(): LM[r, l] = 1

    # ---- filter by coverage ----
    coverage = LM.mean(0)
    keep = [li for li in range(K0) if coverage[li] >= args.min_coverage]
    keep.sort(key=lambda li: -coverage[li])
    print(f"\nmeasurement coverage (top; kept if >={args.min_coverage:.0%}):")
    for li in sorted(range(K0), key=lambda x: -coverage[x]):
        mark = "✓" if li in keep else "·"
        print(f"  {mark} {idx2lab[li]:<18s} {coverage[li]:.1%}")

    K = len(keep)
    L = np.stack([Lmin[:, keep], Lmax[:, keep], Llast[:, keep]], axis=2).astype(np.float32)  # (N,K,3)
    Lm = LM[:, keep]
    names = [idx2lab[li] for li in keep]

    np.save(os.path.join(out_dir, "eicu_labs.npy"), L)
    np.save(os.path.join(out_dir, "eicu_lab_mask.npy"), Lm)
    json.dump({"lab_names": names, "stats": ["min", "max", "last"],
               "coverage": {idx2lab[li]: float(coverage[li]) for li in keep}},
              open(os.path.join(out_dir, "eicu_lab_names.json"), "w"),
              ensure_ascii=False, indent=2)
    print(f"\n{K} labs retained. saved: eicu_labs.npy {L.shape}, eicu_lab_mask.npy {Lm.shape}")
    print("Next: the model script fuses this lab vector into each tier's classifier head.")


if __name__ == "__main__":
    main()
