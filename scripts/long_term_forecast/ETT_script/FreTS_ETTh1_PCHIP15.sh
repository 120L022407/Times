model_name=FreTS
train_epochs=${TRAIN_EPOCHS:-10}
batch_size=${BATCH_SIZE:-16}
patience=${PATIENCE:-3}
num_workers=${NUM_WORKERS:-4}

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id ETTh1_PCHIP15_384_384 \
  --model $model_name \
  --data ETTh1_PCHIP15 \
  --root_path ./dataset/ETT-small \
  --data_path ETTh1.csv \
  --features MS \
  --target OT \
  --freq 15min \
  --seq_len 384 \
  --label_len 192 \
  --pred_len 384 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 1 \
  --eval_mask_mode observed \
  --batch_size $batch_size \
  --learning_rate 0.0001 \
  --train_epochs $train_epochs \
  --patience $patience \
  --num_workers $num_workers \
  --gpu 0 \
  --itr 1 \
  --des 'Exp'
