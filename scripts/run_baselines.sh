#!/usr/bin/env bash
# Reproduce the six baselines (GRU backbone, temporal split). Each prints the
# full-model macro-AUROC per epoch (no per-exit breakdown).
# Usage: bash scripts/run_baselines.sh [data_dir]
set -euo pipefail
cd "$(dirname "$0")/.."
DATA="${1:-data}"

SHARED=(--data_dir "$DATA" --task phenotype
  --hospital_csv "$DATA"/hospital.csv --phenotype_npy eicu_phenotype.npy
  --rnn gru --hidden 256 256 256 256
  --optimizer adam --lr 0.001 --weight_decay 1e-4 --lr_sched cosine
  --num_epochs 300 --local_epochs 1 --batch_size 64
  --tau2 3 --tau3 6 --tau4 12 --extra_channels --time_pool 3
  --test_mode temporal --test_time_key time_order --seed 0)

python src/train_eicu_mtsfl.py "${SHARED[@]}" --baseline hfl      --run_tag base_hfl_gru_temporal
python src/train_eicu_mtsfl.py "${SHARED[@]}" --baseline ebfl     --run_tag base_ebfl_gru_temporal
python src/train_eicu_mtsfl.py "${SHARED[@]}" --baseline pfl      --run_tag base_pfl_gru_temporal
python src/train_eicu_mtsfl.py "${SHARED[@]}" --baseline lgfedavg --run_tag base_lgfedavg_gru_temporal
# HSFL uses a smaller learning rate and larger weight decay
python src/train_eicu_mtsfl.py "${SHARED[@]}" --baseline hsfl --lr 5e-4 --weight_decay 5e-4 --run_tag base_hsfl_gru_temporal
# PH-SFL fine-tunes exit heads with temperature 10
python src/train_eicu_mtsfl.py "${SHARED[@]}" --baseline phsfl --phsfl_temp_tune 10 --phsfl_final_tune 10 --run_tag base_phsfl_gru_temporal
