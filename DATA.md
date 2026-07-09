# Data preparation

All preprocessing operates on **eICU-CRD v2.0** obtained from PhysioNet under a signed
Credentialed Health Data Use Agreement. No raw or processed data files are distributed
with this repository.

## Raw eICU tables used

Place the following tables (`.csv` or `.csv.gz`) in your raw eICU directory. Each row
lists what the table provides and which script consumes it.

| eICU table | Provides | Used by |
|---|---|---|
| `hospital` | `hospitalid` → census `region` (Tier-3 mapping) | `build_eicu_dataset.py`; training (`--hospital_csv`) |
| `patient` | stay/patient IDs, `hospitalid`, `unittype`, age, gender, admission weight, `hospitaldischargeyear` | `build_eicu_dataset.py`, `add_eicu_admit_time.py`, `build_eicu_static_features.py` |
| `diagnosis` | ICD-9 diagnosis strings/codes and offsets | `build_eicu_dataset.py` (base label), `build_eicu_phenotype.py` (25-label target) |
| `vitalPeriodic` | continuous vitals: heart rate, respiration, SpO2 (`sao2`), temperature | `build_eicu_dataset.py` |
| `vitalAperiodic` | NIBP systolic / diastolic / mean | `build_eicu_timeseries.py` |
| `nurseCharting` | GCS Total | `build_eicu_timeseries.py` |
| `lab` | laboratory results within the first 24h | `add_eicu_labs.py` |
| `apacheApsVar` | APACHE physiology (GCS, ventilation, dialysis, 24h-worst vitals/labs) | `build_eicu_static_features.py` |
| `respiratoryCare` | ventilation / airway support | `build_eicu_static_features.py` |
| `pastHistory` | comorbidities | `build_eicu_static_features.py` |
| `infusionDrug` | infused drug groups (vasopressor, sedative, ...) | `build_eicu_static_features.py` |

`preprocessing/phen_code.json` is the ICD-9 → HCUP-CCS phenotype mapping from the eICU
benchmark of Sheikhalishahi et al. (2020) (`eICU_Benchmark`); it contains no patient
data and is included here for convenience with attribution to the original authors.

## Pipeline

Run the full pipeline with the helper script:

```bash
bash scripts/run_preprocess.sh <raw_eicu_dir> data
```

or run the steps individually (all write into the data directory, `data/` by default):

| # | Command | Output artifacts |
|---|---|---|
| 1 | `python preprocessing/build_eicu_dataset.py --eicu_dir <raw> --out_dir data` | `eicu_X.npy` (N,288,4), `eicu_mask.npy`, `eicu_meta.csv`, `eicu_label_vocab.json`, `eicu_summary.json` |
| 2 | `python preprocessing/add_eicu_admit_time.py --raw_dir <raw> --data_dir data` | adds `time_order` column to `eicu_meta.csv` |
| 3 | `python preprocessing/build_eicu_phenotype.py --data_dir data --diagnosis <raw>/diagnosis.csv` | `eicu_phenotype.npy` (N,25), `eicu_phenotype_names.json` |
| 4 | `python preprocessing/build_eicu_timeseries.py --data_dir data --vitalaperiodic <raw>/vitalAperiodic.csv --nursecharting <raw>/nurseCharting.csv` | `eicu_X_extra.npy`, `eicu_mask_extra.npy`, `eicu_extra_names.json` |
| 5 | `python preprocessing/add_eicu_labs.py --data_dir data --eicu_dir <raw> --min_coverage 0.1` | `eicu_labs.npy`, `eicu_lab_mask.npy`, `eicu_lab_names.json` |
| 6 | `python preprocessing/build_eicu_static_features.py --data_dir data --raw_dir <raw>` | `eicu_static.npy`, `eicu_static_names.json` |

Step 1 defines the cohort (the 4-channel vital tensor, the client/hospital/region
hierarchy, and a patient-level split). Step 3 replaces the base diagnosis label with the
25-label phenotype target used in the paper. All arrays are kept **row-aligned to
`eicu_meta.csv`**.

## Feature construction

The model input is the **vital time series with a missingness mask** (`eicu_X`): first 24h
on a 5-min grid (288 bins), 4 channels (heart rate, respiration, SpO2, temperature).
Per-reading physiological range cleaning, temperature °F→°C conversion, bin-averaging, then
forward-fill along time; a separate observed mask distinguishes imputed values.
`time_pool=3` reduces 288→96 at train time. Systemic BP is intentionally excluded
(severity-correlated missingness).

## Reproduced cohort statistics

After running the pipeline you should obtain the following (aggregate counts only):

| Census region (Tier 3) | Hospitals | Clients | Stays |
|---|---|---|---|
| Midwest | 47 | 72 | 33,736 |
| South | 40 | 68 | 37,215 |
| West | 37 | 47 | 29,864 |
| Northeast | 11 | 15 | 10,211 |
| **Total** | **135** | **202** | **111,026** |

For the phenotype experiments, the **temporal** split holds out each client's most-recent
20% of patients, giving **88,865 train / 22,161 test** stays. (`eicu_summary.json` reports
the base random split used for the diagnosis cohort; the temporal split is applied at
training time via `--test_mode temporal --test_time_key time_order`.)
