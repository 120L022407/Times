#!/bin/bash
set -euo pipefail

python_bin=${PYTHON_BIN:-python}
gpu=${GPU:-0}
batch_size=${BATCH_SIZE:-128}
train_epochs=${TRAIN_EPOCHS:-100}
num_workers=${NUM_WORKERS:-10}

"$python_bin" -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_MS_OT_96_96_TFPS \
  --model TFPS \
  --data ETTh1 \
  --features MS \
  --target OT \
  --freq h \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 1 \
  --e_layers 3 \
  --n_heads 4 \
  --d_model 16 \
  --d_ff 128 \
  --dropout 0.3 \
  --patch_len 16 \
  --stride 8 \
  --tfps_t_num_experts 16 \
  --tfps_t_top_k 1 \
  --tfps_f_num_experts 16 \
  --tfps_f_top_k 1 \
  --tfps_eta 5 \
  --tfps_beta 0.1 \
  --batch_size "$batch_size" \
  --train_epochs "$train_epochs" \
  --num_workers "$num_workers" \
  --patience 20 \
  --learning_rate 0.0005 \
  --lradj type3 \
  --eval_mask_mode all \
  --des TFPS \
  --itr 1 \
  --gpu "$gpu"
