model_name=Informer
train_epochs=${TRAIN_EPOCHS:-10}
batch_size=${BATCH_SIZE:-8}
patience=${PATIENCE:-3}
num_workers=${NUM_WORKERS:-4}
gpu=${GPU:-0}
log_dir=${LOG_DIR:-./logs}
log_file=${LOG_FILE:-$log_dir/etth1_pchip15_informer_ms_384_384.log}

mkdir -p "$log_dir"

nohup python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id ETTh1_PCHIP15_384_384 \
  --model $model_name \
  --data ETTh1_PCHIP15 \
  --root_path ./dataset/ETT-small \
  --data_path ETTh1.csv \
  --features MS \
  --target OT \
  --freq t \
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
  --gpu $gpu \
  --itr 1 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --n_heads 4 \
  --d_model 128 \
  --d_ff 256 \
  --des 'Exp' > "$log_file" 2>&1 &

echo "Started $model_name training in background."
echo "PID: $!"
echo "Log: $log_file"
