#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_eicu_phenotype.py -- 25-label ICU phenotype target from diagnosis.csv.

Maps ICD-9 codes to the 25 HCUP CCS single-level phenotype categories used in the
clinical-time-series benchmarks of Harutyunyan et al. (2019) and Sheikhalishahi et al.
(2020). Only in-unit diagnoses (diagnosisoffset > 0) are used; the first comma token of
icd9code (the ICD-9 code, dots removed) is matched exactly against phen_code.json. A stay
is multi-hot: label j is 1 if any matched diagnosis maps to phenotype j.

Requires:  phen_code.json (Sheikhalishahi eICU_Benchmark) in the same folder or via --phen_code.
Output  :  eicu_phenotype.npy (N, 25) uint8, aligned to the row order of eicu_meta.csv.
Usage   :  python build_eicu_phenotype.py --data_dir <data> --diagnosis <path>/diagnosis.csv
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

ID = "patientunitstayid"

# (phen_code.json key, human-readable name, type) -- output order of the 25 columns (acute 15 / chronic 10; per the YAML)
PHENO = [
    ("septicemia", "Septicemia (except in labor)", "acute"),
    ("t2dmwocomp", "Diabetes mellitus without complication", "chronic"),
    ("t2dmcomp", "Diabetes mellitus with complications", "chronic"),
    ("lipidmetab", "Disorders of lipid metabolism", "chronic"),
    ("fluiddiso", "Fluid and electrolyte disorders", "acute"),
    ("essehyper", "Essential hypertension", "chronic"),
    ("hypercomp", "Hypertension with complications and secondary hypertension", "chronic"),
    ("myocar_infarction", "Acute myocardial infarction", "acute"),
    ("Coronary_ath", "Coronary atherosclerosis and other heart disease", "chronic"),
    ("Conduction_dis", "Conduction disorders", "chronic"),
    ("Cardiac_dysr", "Cardiac dysrhythmias", "chronic"),
    ("Congestive_hf", "Congestive heart failure; nonhypertensive", "acute"),
    ("Acute_cerebrovascular", "Acute cerebrovascular disease", "acute"),
    ("Pneumonia", "Pneumonia", "acute"),
    ("COPD", "COPD and bronchiectasis", "chronic"),
    ("Pleurisy", "Pleurisy; pneumothorax; pulmonary collapse", "acute"),
    ("Resp_failure", "Respiratory failure; insufficiency; arrest (adult)", "acute"),
    ("lower_respiratory", "Other lower respiratory disease", "acute"),
    ("upper_respiratory", "Other upper respiratory disease", "acute"),
    ("Other_liver_dis", "Other liver diseases", "acute"),
    ("Gastroint_hemorrhage", "Gastrointestinal hemorrhage", "acute"),
    ("renal_failure", "Acute and unspecified renal failure", "acute"),
    ("ckd", "Chronic kidney disease", "chronic"),
    ("Compl_surgical", "Complications of surgical procedures or medical care", "acute"),
    ("Shock", "Shock", "acute"),
]

# Sheikhalishahi 2020 Table 7 prevalence (for sanity check only; may not match exactly for a different cohort)
REF_PREV = {
    "Respiratory failure; insufficiency; arrest (adult)": 0.241,
    "Fluid and electrolyte disorders": 0.156,
    "Septicemia (except in labor)": 0.145,
    "Acute and unspecified renal failure": 0.142,
    "Pneumonia": 0.120,
    "Acute cerebrovascular disease": 0.108,
    "Acute myocardial infarction": 0.090,
    "Gastrointestinal hemorrhage": 0.079,
    "Shock": 0.068,
    "Pleurisy; pneumothorax; pulmonary collapse": 0.039,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="mtsfl_eicu", help="location of eicu_meta.csv / eicu_X.npy")
    ap.add_argument("--diagnosis", required=True, help="path to diagnosis.csv")
    ap.add_argument("--phen_code", default=None, help="default: {data_dir}/phen_code.json or ./phen_code.json")
    ap.add_argument("--chunksize", type=int, default=1_000_000)
    a = ap.parse_args()

    # phen_code.json (Sheikhalishahi)
    pc_path = a.phen_code or next((p for p in [os.path.join(a.data_dir, "phen_code.json"), "phen_code.json"]
                                   if os.path.exists(p)), None)
    if pc_path is None:
        raise SystemExit("phen_code.json not found (specify with --phen_code)")
    codes = json.load(open(pc_path))
    code2idx = {}
    for idx, (key, _, _) in enumerate(PHENO):
        for c in codes[key]:
            code2idx[str(c)] = idx        # CCS categories are mutually exclusive -> one phenotype per code
    names = [nm for _, nm, _ in PHENO]
    print(f"[phen] {pc_path}  25 phenotypes, {sum(len(codes[k]) for k, _, _ in PHENO)} ICD-9 codes")

    # eicu_meta -> stay -> row (row order of eicu_X)
    meta = pd.read_csv(os.path.join(a.data_dir, "eicu_meta.csv"), usecols=[ID])
    stay_ids = meta[ID].astype(np.int64).to_numpy()
    N = len(stay_ids)
    row_of = {int(s): i for i, s in enumerate(stay_ids)}
    stayset = set(row_of)
    xp = os.path.join(a.data_dir, "eicu_X.npy")
    if os.path.exists(xp):
        xr = np.load(xp, mmap_mode="r").shape[0]
        assert xr == N, f"eicu_X rows({xr}) != meta rows({N}) -- alignment mismatch"
        print(f"[align] eicu_X rows={xr} == meta rows={N}  OK")
    print(f"[meta] stays={N:,}")

    Y = np.zeros((N, 25), np.uint8)
    n_rows = n_match = n_icd10_present = 0
    for ch in pd.read_csv(a.diagnosis, usecols=[ID, "diagnosisoffset", "icd9code"],
                          dtype={"icd9code": str}, chunksize=a.chunksize):
        ch = ch[ch["diagnosisoffset"] > 0]
        ch = ch[ch["icd9code"].notnull() & ch[ID].isin(stayset)]
        if ch.empty:
            continue
        n_rows += len(ch)
        raw = ch["icd9code"].astype(str)
        n_icd10_present += int(raw.str.contains(",").sum())              # rows that also carry a second token (ICD-10)
        icd = raw.str.split(",", n=1).str[0].str.replace(".", "", regex=False).str.strip()
        idxs = icd.map(code2idx)                                         # unmapped (e.g. ICD-10) = NaN
        m = idxs.notna()
        n_match += int(m.sum())
        if m.any():
            rows = ch.loc[m, ID].map(row_of).to_numpy().astype(np.int64)
            cols = idxs[m].to_numpy().astype(np.int64)
            Y[rows, cols] = 1

    # ---- save ----
    np.save(os.path.join(a.data_dir, "eicu_phenotype.npy"), Y)
    prev = Y.mean(axis=0)
    detail = {}
    for i, (_, nm, typ) in enumerate(PHENO):
        detail[nm] = {"type": typ, "n_icd9_codes": len(codes[PHENO[i][0]]),
                      "prevalence": round(float(prev[i]), 4),
                      "ref_sheikhalishahi": REF_PREV.get(nm)}
    json.dump({"phenotypes": names, "n_phenotypes": 25,
               "diagnosis_rows_used": int(n_rows), "rows_matched_icd9": int(n_match),
               "rows_with_icd10_token": int(n_icd10_present),
               "stays_with_any_phenotype": round(float((Y.sum(1) > 0).mean()), 4),
               "mean_phenotypes_per_stay": round(float(Y.sum(1).mean()), 3),
               "detail": detail},
              open(os.path.join(a.data_dir, "eicu_phenotype_names.json"), "w"),
              indent=2, ensure_ascii=False)

    # ---- console summary + comparison with Sheikhalishahi Table 7 ----
    print(f"\n[labels] diagnosis rows used={n_rows:,}  ICD-9 matched={n_match:,}  "
          f"rows with an ICD-10 token={n_icd10_present:,} (unused)")
    print(f"[labels] fraction of stays with >=1 phenotype={float((Y.sum(1) > 0).mean()):.1%}  "
          f"mean phenotypes/stay={float(Y.sum(1).mean()):.2f}")
    print(f"\n{'phenotype':<52s}{'prev':>7}{'ref(S20)':>9}{'type':>9}")
    order = sorted(range(25), key=lambda i: -prev[i])
    for i in order:
        _, nm, typ = PHENO[i]
        ref = REF_PREV.get(nm)
        rs = f"{ref:.3f}" if ref is not None else "-"
        print(f"  {nm[:50]:<50s}{prev[i]:>7.3f}{rs:>9}{typ:>9}")
    print(f"\n[done] eicu_phenotype.npy ({N},25) + eicu_phenotype_names.json → {a.data_dir}/")
    print("       Use --task phenotype in the model (BCEWithLogits/AUROC/AUPRC). "
          "If prevalence is close to the reference, the mapping is correct.")


if __name__ == "__main__":
    main()