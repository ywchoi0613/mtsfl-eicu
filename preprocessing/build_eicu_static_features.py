#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_eicu_static_features.py -- per-stay static clinical side-features.

Builds eicu_static.npy (N, D) aligned to the stay order of eicu_meta.csv, concatenated at
each exit head alongside the lab and vital-summary features. Feature blocks are drawn from
five raw eICU tables:
  A. apacheApsVar    -- APACHE physiology (GCS, intubated/vent/dialysis, 24h-worst vitals/labs + missing masks)
  B. patient         -- ethnicity/stay-type/admit-source one-hot, height/weight/BMI/age/gender/visit
  C. respiratoryCare -- 24h ventilation flag, airway type, respiratory event counts
  D. pastHistory     -- major comorbidity multi-hot + count
  E. infusionDrug    -- 24h drug-group indicators (vasopressor/inotrope/sedative/opioid/anticoagulant/insulin/...) + count

Conventions: first-24h window (offset in [0, 1440)); standardization uses train-split statistics
only; partition keys (hospitalid/region/unittype), admission diagnosis, and the target diagnosis
are deliberately excluded to prevent leakage.
Usage: python build_eicu_static_features.py --data_dir <data> --raw_dir <raw eICU csv dir>
"""
import argparse, gc, json, os
import numpy as np, pandas as pd

W = 1440  # first 24h (minutes)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="folder containing eicu_meta.csv etc.")
    ap.add_argument("--raw_dir", required=True, help="raw CSV folder")
    a = ap.parse_args()
    R = lambda f: os.path.join(a.raw_dir, f)

    meta = pd.read_csv(os.path.join(a.data_dir, "eicu_meta.csv"))
    sids = meta["patientunitstayid"].to_numpy()
    N = len(sids)
    stays = set(sids.tolist())
    is_train = (meta["split"] == "train").to_numpy()
    sid_index = pd.Index(sids)
    print(f"[meta] stay={N:,}  train={int(is_train.sum()):,}")

    feats, numeric = [], set()
    def add(name, arr, num=False):
        feats.append((name, np.asarray(arr, dtype=np.float64)))
        if num: numeric.add(name)
    def memb(id_set):  # stay->{0,1} aligned
        return np.fromiter((1.0 if s in id_set else 0.0 for s in sids), dtype=np.float64, count=N)
    def reindex_series(s):  # per-stay Series -> (N,) in meta order
        return s.reindex(sid_index).to_numpy()

    # ---------- A. apacheApsVar ----------
    av = pd.read_csv(R("apacheApsVar.csv"))
    av = av.drop_duplicates("patientunitstayid").set_index("patientunitstayid").reindex(sid_index)
    gcs = av[["eyes", "motor", "verbal"]].replace(-1, np.nan).sum(axis=1, min_count=3)
    add("apache_gcs", gcs.to_numpy(), num=True)
    for c in ["intubated", "vent", "dialysis"]:
        add(f"apache_{c}", av[c].fillna(0).clip(0, 1).to_numpy())
    num_cols = ["temperature", "respiratoryrate", "sodium", "heartrate", "meanbp", "wbc",
                "hematocrit", "creatinine", "bun", "glucose", "urine", "albumin",
                "bilirubin", "ph", "pao2", "pco2", "fio2"]
    sparse = {"ph", "pao2", "pco2", "fio2", "bilirubin", "albumin", "urine"}
    for c in num_cols:
        v = av[c].replace(-1, np.nan)
        add(f"apache_{c}", v.to_numpy(), num=True)
        if c in sparse:
            add(f"apache_{c}_miss", v.isna().astype(float).to_numpy())
    print(f"  A. apacheApsVar coverage {av['vent'].notna().mean():.0%}")
    del av; gc.collect()

    # ---------- B. patient ----------
    pt = pd.read_csv(R("patient.csv"), low_memory=False,
                     usecols=["patientunitstayid", "ethnicity", "admissionheight",
                              "unitstaytype", "unitadmitsource", "unitvisitnumber"])
    pt = pt.drop_duplicates("patientunitstayid").set_index("patientunitstayid").reindex(sid_index)
    eths = ["Caucasian", "African American", "Hispanic", "Asian", "Native American"]
    for e in eths:
        add(f"eth_{e.replace(' ', '_')}", (pt["ethnicity"] == e).astype(float).to_numpy())
    add("eth_other", (~pt["ethnicity"].isin(eths)).astype(float).to_numpy())
    for u in ["admit", "transfer", "readmit", "stepdown/other"]:
        add(f"staytype_{u.replace('/', '_')}", (pt["unitstaytype"] == u).astype(float).to_numpy())
    srcs = ["Emergency Department", "Operating Room", "Floor", "Direct Admit",
            "Recovery Room", "Acute Care/Floor", "ICU to SDU"]
    for s in srcs:
        add(f"src_{s.split()[0].lower()}", (pt["unitadmitsource"] == s).astype(float).to_numpy())
    add("src_other", (~pt["unitadmitsource"].isin(srcs)).astype(float).to_numpy())
    h = pd.to_numeric(pt["admissionheight"], errors="coerce").to_numpy()
    h = np.where((h < 120) | (h > 230), np.nan, h)
    w = pd.to_numeric(meta["admissionweight"], errors="coerce").to_numpy()
    w = np.where((w < 25) | (w > 350), np.nan, w)
    bmi = w / ((h / 100.0) ** 2)
    add("height", h, num=True); add("weight", w, num=True); add("bmi", bmi, num=True)
    add("age", pd.to_numeric(meta["age_num"], errors="coerce").to_numpy(), num=True)
    add("gender_male", (meta["gender"] == "Male").astype(float).to_numpy())
    add("unitvisit", pd.to_numeric(pt["unitvisitnumber"], errors="coerce").clip(upper=10).to_numpy(), num=True)
    print("  B. join patient")
    del pt; gc.collect()

    # ---------- C. respiratoryCare ----------
    rc = pd.read_csv(R("respiratoryCare.csv"), low_memory=False,
                     usecols=["patientunitstayid", "respcarestatusoffset", "ventstartoffset", "airwaytype"])
    rc = rc[rc["patientunitstayid"].isin(stays)]
    vent_ids = set(rc[(rc["ventstartoffset"].notna()) & (rc["ventstartoffset"] < W)
                      & (rc["ventstartoffset"] >= -720)]["patientunitstayid"].unique())
    add("vent_24h", memb(vent_ids))
    rc24 = rc[(rc["respcarestatusoffset"] >= -720) & (rc["respcarestatusoffset"] < W)]
    for at, key in [("Oral ETT", "ett"), ("Tracheostomy", "trach"), ("No Artificial Airway", "noair")]:
        add(f"airway_{key}", memb(set(rc24[rc24["airwaytype"] == at]["patientunitstayid"].unique())))
    cnt = rc24.groupby("patientunitstayid").size()
    add("resp_count_log", np.log1p(np.nan_to_num(reindex_series(cnt))), num=True)
    print(f"  C. respiratoryCare (vent24h={len(vent_ids):,})")
    del rc, rc24; gc.collect()

    # ---------- D. pastHistory ----------
    ph = pd.read_csv(R("pastHistory.csv"), engine="python", on_bad_lines="skip",
                     usecols=["patientunitstayid", "pasthistoryvalue"])
    ph = ph[ph["patientunitstayid"].isin(stays)].copy()
    v = ph["pasthistoryvalue"].fillna("").str.lower()
    hx = {
        "hypertension": "hypertension", "diabetes": "diabetes",
        "chf": "chf|congestive heart failure", "afib": "atrial fibrillation",
        "copd": "copd", "asthma": "asthma", "pvd": "peripheral vascular",
        "mi": r"\bmi\b|myocardial", "stroke": "stroke|cva|tia",
        "ckd_esrd": "hemodialysis|renal failure|renal insufficiency|ckd|esrd",
        "dementia": "dementia|alzheimer", "seizure": "seizure",
        "cad": "cabg|coronary intervention|coronary artery|cad|ASHD".lower(),
        "hypothyroid": "hypothyroid", "cancer": "cancer|malignan|leukemia|lymphoma|metasta",
        "home_o2": "home oxygen",
    }
    hx_arrs = []
    for name, pat in hx.items():
        ids = set(ph.loc[v.str.contains(pat, regex=True, na=False), "patientunitstayid"].unique())
        arr = memb(ids); add(f"hx_{name}", arr); hx_arrs.append(arr)
    add("hx_count", np.clip(np.sum(hx_arrs, axis=0), 0, 12), num=True)
    print(f"  D. pastHistory ({len(hx)} condition flags)")
    del ph, v; gc.collect()

    # ---------- E. infusionDrug ----------
    inf = pd.read_csv(R("infusionDrug.csv"), low_memory=False,
                      usecols=["patientunitstayid", "infusionoffset", "drugname"])
    inf = inf[inf["patientunitstayid"].isin(stays) & (inf["infusionoffset"] >= -60) & (inf["infusionoffset"] < W)].copy()
    d = inf["drugname"].fillna("").str.lower()
    classes = {
        "vasopressor": "norepinephrine|phenylephrine|vasopressin|epinephrine|dopamine|levophed|neo-synephrine",
        "inotrope": "dobutamine|milrinone|dobutrex",
        "sedative": "propofol|midazolam|dexmedetomidine|lorazepam|versed|precedex|diprivan",
        "opioid": "fentanyl|hydromorphone|morphine|dilaudid",
        "anticoag": "heparin|argatroban|bivalirudin",
        "antiarrhythmic": "amiodarone|cardizem|diltiazem",
        "insulin": "insulin",
        "vasodilator": "nicardipine|nitroglycerin|nitroprusside|cardene|ntg|nipride",
    }
    for cls, pat in classes.items():
        ids = set(inf.loc[d.str.contains(pat, regex=True, na=False), "patientunitstayid"].unique())
        add(f"drug_{cls}", memb(ids))
    cnt = inf.groupby("patientunitstayid")["drugname"].nunique()
    add("drug_nunique_log", np.log1p(np.nan_to_num(reindex_series(cnt))), num=True)
    print(f"  E. infusionDrug 24h (stay={inf['patientunitstayid'].nunique():,})")
    del inf, d; gc.collect()

    # ---------- standardization (train statistics) + missing handling ----------
    names = [n for n, _ in feats]
    M = np.stack([arr for _, arr in feats], axis=1)  # (N,D) float64
    for i, n in enumerate(names):
        col = M[:, i]
        if n in numeric:
            tr = col[is_train]
            mu = np.nanmean(tr) if np.isfinite(np.nanmean(tr)) else 0.0
            sd = np.nanstd(tr); sd = sd if sd > 1e-6 else 1.0
            col = np.where(np.isnan(col), mu, col)
            M[:, i] = np.clip((col - mu) / sd, -10, 10)
        else:
            M[:, i] = np.nan_to_num(col, nan=0.0)
    M = M.astype(np.float32)

    out = os.path.join(a.data_dir, "eicu_static.npy")
    np.save(out, M)
    json.dump(names, open(os.path.join(a.data_dir, "eicu_static_names.json"), "w"), indent=1)
    print(f"\n[done] eicu_static.npy shape={M.shape}  (numeric {len(numeric)} / binary {len(names)-len(numeric)})")
    print(f"  -> {out}")
    print(f"  features: {names}")

if __name__ == "__main__":
    main()
