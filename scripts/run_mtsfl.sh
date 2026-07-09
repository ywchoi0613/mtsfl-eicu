#!/usr/bin/env bash
# Reproduce MT-SFL (GRU backbone, temporal split) with the final configuration.
# The reported operating point uses the fixed entropy thresholds below.
# Usage: bash scripts/run_mtsfl.sh [data_dir]
set -euo pipefail
cd "$(dirname "$0")/.."
DATA="${1:-data}"

python src/train_eicu_mtsfl.py \
  --data_dir "$DATA" --task phenotype \
  --hospital_csv "$DATA"/hospital.csv --phenotype_npy eicu_phenotype.npy \
  --rnn gru --hidden 256 256 256 256 \
  --optimizer adam --lr 0.001 --weight_decay 1e-4 --lr_sched cosine \
  --num_epochs 300 --local_epochs 1 --batch_size 64 \
  --tau2 3 --tau3 6 --tau4 12 --extra_channels --time_pool 3 --top_m 15 30 45 \
  --gamma 0.4 0.2 0.2 --lambda1 0.8 --lambda2 0.6 --lambda3 0.4 \
  --test_mode temporal --test_time_key time_order \
  --tune_eth --eth_fixed 0.6044,0.6440,0.5672 \
  --seed 0 --run_tag mtsfl_gru_temporal
