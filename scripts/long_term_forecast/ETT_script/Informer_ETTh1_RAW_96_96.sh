model_name=Informer
train_epochs=${TRAIN_EPOCHS:-10}
batch_size=${BATCH_SIZE:-16}
patience=${PATIENCE:-3}
num_workers=${NUM_WORKERS:-4}
gpu=${GPU:-0}
log_dir=${LOG_DIR:-./logs}
log_file=${LOG_FILE:-$log_dir/etth1_raw_informer_ms_ot_96_96.log}

mkdir -p "$log_dir"

nohup python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id ETTh1_RAW_MS_OT_96_96 \
  --model $model_name \
  --data ETTh1 \
  --root_path ./dataset/ETT-small \
  --data_path ETTh1.csv \
  --features MS \
  --target OT \
  --freq h \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --eval_mask_mode all \
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
  --des 'Exp' > "$log_file" 2>&1 &

echo "Started $model_name training on raw ETTh1 in background."
echo "PID: $!"
echo "Log: $log_file"
