#!/usr/bin/env bash
# Run the full eICU preprocessing pipeline in order.
# Usage: bash scripts/run_preprocess.sh <raw_eicu_dir> [data_dir]
set -euo pipefail
RAW="${1:?usage: run_preprocess.sh <raw_eicu_dir> [data_dir]}"
DATA="${2:-data}"
cd "$(dirname "$0")/.."

echo "[1/6] base cohort + 24h 4-channel vital tensor"
python preprocessing/build_eicu_dataset.py --eicu_dir "$RAW" --out_dir "$DATA"
echo "[2/6] temporal ordering key (time_order)"
python preprocessing/add_eicu_admit_time.py --raw_dir "$RAW" --data_dir "$DATA"
echo "[3/6] 25-label phenotype target"
python preprocessing/build_eicu_phenotype.py --data_dir "$DATA" --diagnosis "$RAW"/diagnosis.csv
echo "[4/6] extra vital channels (NIBP, GCS)"
python preprocessing/build_eicu_timeseries.py --data_dir "$DATA" \
  --vitalaperiodic "$RAW"/vitalAperiodic.csv --nursecharting "$RAW"/nurseCharting.csv
echo "[5/6] first-24h lab summaries"
python preprocessing/add_eicu_labs.py --data_dir "$DATA" --eicu_dir "$RAW" --min_coverage 0.1
echo "[6/6] static clinical features"
python preprocessing/build_eicu_static_features.py --data_dir "$DATA" --raw_dir "$RAW"

# training needs the eICU hospital table as <data>/hospital.csv
cp -f "$RAW"/hospital.csv "$DATA"/hospital.csv 2>/dev/null || \
  echo "NOTE: place the eICU hospital table at $DATA/hospital.csv before training."
echo "preprocessing complete -> $DATA/"
