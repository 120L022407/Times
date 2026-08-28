#!/usr/bin/env bash
set -euo pipefail

python_bin=${PYTHON_BIN:-python}
gpu=${GPU:-0}
batch_size=${BATCH_SIZE:-128}
train_epochs=${TRAIN_EPOCHS:-10}
num_workers=${NUM_WORKERS:-4}

"$python_bin" -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_RAW_MS_OT_96_96_TimeMixer_HCAN \
  --model TimeMixer_HCAN \
  --data ETTh1 \
  --features MS \
  --target OT \
  --freq h \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
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
  --loss mse \
  --batch_size "$batch_size" \
  --train_epochs "$train_epochs" \
  --num_workers "$num_workers" \
  --patience 10 \
  --learning_rate 0.01 \
  --lradj type1 \
  --eval_mask_mode all \
  --des TimeMixer_HCAN_RAW \
  --itr 1 \
  --gpu "$gpu"
