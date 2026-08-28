#!/usr/bin/env bash
set -euo pipefail

# Usage: bash TimeMixer_HCAN_ETTh1_PCHIP15.sh {mse|ps|facl}
loss=${1:-mse}
case "$loss" in
  mse|ps|facl) ;;
  *)
    echo "Usage: $0 {mse|ps|facl}" >&2
    exit 2
    ;;
esac

python_bin=${PYTHON_BIN:-python}
gpu=${GPU:-0}
batch_size=${BATCH_SIZE:-16}
train_epochs=${TRAIN_EPOCHS:-10}
num_workers=${NUM_WORKERS:-4}
ps_lambda=${PS_LAMBDA:-3.0}
ps_delta=${PS_DELTA:-24}
facl_alpha=${FACL_ALPHA:-0.1}

exec "$python_bin" -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id "ETTh1_PCHIP15_MS_OT_384_384_TimeMixer_HCAN_${loss}" \
  --model TimeMixer_HCAN \
  --data ETTh1_PCHIP15 \
  --features MS \
  --target OT \
  --freq t \
  --seq_len 384 \
  --label_len 0 \
  --pred_len 384 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --e_layers 2 \
  --d_model 16 \
  --d_ff 32 \
  --channel_independence 1 \
  --down_sampling_layers 3 \
  --down_sampling_method avg \
  --down_sampling_window 2 \
  --hcan_kc 2 \
  --hcan_kf 4 \
  --hcan_hidden_dim 512 \
  --hcan_alpha 1 \
  --hcan_beta 1 \
  --hcan_gamma 1 \
  --hcan_annealing_steps 10 \
  --loss "$loss" \
  --ps_lambda "$ps_lambda" \
  --ps_delta "$ps_delta" \
  --facl_alpha "$facl_alpha" \
  --batch_size "$batch_size" \
  --train_epochs "$train_epochs" \
  --num_workers "$num_workers" \
  --patience 10 \
  --learning_rate 0.01 \
  --lradj type1 \
  --eval_mask_mode observed \
  --des "TimeMixer_HCAN_${loss}" \
  --itr 1 \
  --gpu "$gpu"
