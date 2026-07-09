# MT-SFL on eICU-CRD

Reproducibility code for the eICU experiments in **"Multi-Tier Split Federated
Learning for Multi-Level Personalization" (MT-SFL)**, corresponding to the paper
section *Evaluation on a Real Hierarchical Dataset*.

MT-SFL is a four-tier **split federated learning** framework (client → edge → region
→ cloud) with tier-wise model aggregation, per-tier personalization weights, and a
Shannon-entropy **early-exit routing** mechanism for adaptive, low-latency inference.
This repository reproduces the eICU phenotype-classification experiment used to
validate MT-SFL on a real, naturally hierarchical dataset.

> [!IMPORTANT]
> **This repository does not contain any data.** The eICU Collaborative Research
> Database is credentialed, restricted health data. Under the PhysioNet Credentialed
> Health Data Use Agreement, we may release **code** but may **not** redistribute the
> data or any data-derived files. You must obtain eICU-CRD yourself (see below) and run
> the preprocessing locally.

---

## The eICU four-tier hierarchy

eICU induces a natural four-tier hierarchy that maps directly onto MT-SFL's tiers:

| MT-SFL tier | eICU level | Count |
|---|---|---|
| Tier 1 — Client | ICU care unit (hospital × unit type) | 202 |
| Tier 2 — Edge | Hospital | 135 |
| Tier 3 — Region | U.S. census region | 4 |
| Tier 4 — Cloud | Global population | 1 |

The processed cohort has **88,865 training** and **22,161 held-out test** ICU stays.
A *client* is a specific ICU unit type within a hospital, which is why there are more
clients (202) than hospitals (135).

**Task.** Predict which of **25 clinical phenotypes** are present during an ICU stay
(multi-label classification), following the clinical-time-series benchmarks of
Harutyunyan et al. (2019) and Sheikhalishahi et al. (2020). The backbone is a **GRU**
suited to irregular clinical time series.

**Distribution shift.** To emulate a realistic deployment-time shift, we use a
**temporal split**: within each client, the most recent 20% of patients are held out
for testing (a natural, non-manual shift).

**Metric.** Macro-AUROC over the 25 phenotypes (each phenotype's AUROC is computed
separately and the 25 values are averaged). MT-SFL is scored on its **routed,
label-wise** predictions — each (sample, phenotype) pair is read from the tier where
the entropy-based early-exit criterion is first satisfied — while each baseline is
scored on its single full model.

---

## 1. Data access (PhysioNet DUA)

1. Obtain credentialed access to **eICU-CRD v2.0** on PhysioNet:
   https://physionet.org/content/eicu-crd/2.0/ (requires CITI training + a signed DUA).
2. Download the raw tables and place them (as `.csv` or `.csv.gz`) in `data/`.
   The exact tables used are listed in [`DATA.md`](DATA.md).

We do not, and you must not, commit any of these files or their derivatives to a public
repository. `.gitignore` is configured to block them.

## 2. Environment

Python 3.10. The code depends on only four external packages:

```bash
pip install -r requirements.txt
```

Experiments were run on 4× NVIDIA RTX 3090 (24 GB) GPUs with a CUDA 12.x build of
PyTorch. Install the `torch` wheel matching your CUDA version.

## 3. Data preparation

See [`DATA.md`](DATA.md) for the full pipeline, the raw eICU tables consumed at each
step, and the artifacts produced. In short:

```bash
bash scripts/run_preprocess.sh <raw_eicu_dir> data
```

This runs, in order: `build_eicu_dataset.py` → `add_eicu_admit_time.py` →
`build_eicu_phenotype.py` → `build_eicu_timeseries.py` → `add_eicu_labs.py` →
`build_eicu_static_features.py`, writing all processed arrays into `data/`.

## 4. Reproducing the paper

The reported experiments use the **GRU backbone** and the **temporal** split.

**MT-SFL (main):**

```bash
bash scripts/run_mtsfl.sh
```

which trains with the final configuration:

| Setting | Value |
|---|---|
| Optimizer / LR / schedule | Adam / `1e-3` / cosine annealing (schedule length 300) |
| Weight decay / batch / local epochs | `1e-4` / 64 / 1 |
| Aggregation periods **τ** | `[3, 6, 12]` |
| Loss weights **γ** (client, edge, region, cloud) | `[0.4, 0.2, 0.2, 0.2]` |
| Personalization weights **λ** | `[0.8, 0.6, 0.4]` |
| Entropy thresholds **E_th** (client, edge, region) | `[0.6044, 0.6440, 0.5672]` |
| Training length | 276 epochs; cosine curve of length 300 (`--num_epochs 276 --sched_epochs 300`) |

Training stops at epoch 276 while the cosine LR curve keeps its full length
(`--sched_epochs 300`), so the reported numbers are those a 300-epoch run reaches at epoch
276, before the annealing tail.

**Baselines (six methods):**

```bash
bash scripts/run_baselines.sh
```

All baselines share the optimizer and cosine schedule above, **except HSFL**, which uses
`lr = 5e-4` and `weight_decay = 5e-4`.

---

## Repository layout

```
mtsfl-eicu/
├── README.md, DATA.md                        # documentation
├── requirements.txt, LICENSE
├── data/                                     # you place raw eICU tables here (git-ignored)
├── preprocessing/
│   ├── build_eicu_dataset.py         # base cohort + 24h 4-channel vital tensor + patient-level split
│   ├── add_eicu_admit_time.py        # temporal ordering key (time_order) for the temporal split
│   ├── build_eicu_phenotype.py       # 25-label phenotype target (ICD-9 -> HCUP CCS)
│   ├── build_eicu_timeseries.py      # extra vital channels (NIBP, GCS)
│   ├── add_eicu_labs.py              # first-24h lab summaries (K=45 labs)
│   ├── build_eicu_static_features.py # static clinical side-features (APACHE, demographics, ...)
│   └── phen_code.json                # ICD-9 -> phenotype map (Sheikhalishahi eICU_Benchmark)
├── src/
│   ├── train_eicu_mtsfl.py           # main entry point: MT-SFL and, via --baseline, all baselines
│   ├── baselines_eicu.py             # the six baseline training loops
│   ├── baselines_common.py           # shared baseline utilities
│   └── multiexit_models.py           # multi-exit GRU/Transformer backbones (phi_{k,l}, beta_{k,l})
└── scripts/
    ├── run_preprocess.sh             # runs the full preprocessing pipeline in order
    ├── run_mtsfl.sh                  # reproduces MT-SFL (GRU, temporal)
    └── run_baselines.sh              # reproduces the six baselines (GRU, temporal)
```

## License

Code in this repository is released under the MIT License (see `LICENSE`). This license
applies to the code only; use of eICU-CRD is governed by the PhysioNet Credentialed
Health Data Use Agreement.
