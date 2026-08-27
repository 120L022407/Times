#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 MODEL LOSS"
  echo "  MODEL: PatchTST | Informer | DLinear | iTransformer | TimeMixer | TimesNet | FEDformer | TFPS"
  echo "  LOSS:  mse | facl | ps"
}

if [[ $# -ne 2 ]]; then
  usage >&2
  exit 2
fi

model_key=$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')
case "$model_key" in
  patchtst) model=PatchTST ;;
  informer) model=Informer ;;
  dlinear) model=DLinear ;;
  itransformer) model=iTransformer ;;
  timemixer) model=TimeMixer ;;
  timesnet) model=TimesNet ;;
  fedformer) model=FEDformer ;;
  tfps) model=TFPS ;;
  *)
    echo "Unsupported model: $1" >&2
    usage >&2
    exit 2
    ;;
esac

loss_key=$(printf '%s' "$2" | tr '[:upper:]' '[:lower:]')
case "$loss_key" in
  mse|original) loss=mse ;;
  facl) loss=facl ;;
  ps|psloss) loss=ps ;;
  *)
    echo "Unsupported loss: $2" >&2
    usage >&2
    exit 2
    ;;
esac

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_root=$(cd "$script_dir/../../.." && pwd)
cd "$project_root"

python_bin=${PYTHON_BIN:-python}
root_path=${ROOT_PATH:-./dataset/ETT-small}
data_path=${DATA_PATH:-ETTh1.csv}
gpu=${GPU:-0}
num_workers=${NUM_WORKERS:-4}
train_epochs_default=10
batch_size_default=16
patience_default=3
learning_rate=0.0001
label_len=192
c_out=7
model_args=()

case "$model" in
  PatchTST)
    c_out=1
    model_args=(--patch_len 64 --stride 32 --e_layers 3 --n_heads 4 --d_model 128 --d_ff 256)
    ;;
  Informer)
    batch_size_default=8
    c_out=1
    model_args=(--e_layers 2 --d_layers 1 --factor 3 --n_heads 4 --d_model 128 --d_ff 256)
    ;;
  DLinear)
    model_args=(--moving_avg 25)
    ;;
  iTransformer)
    model_args=(--e_layers 2 --d_layers 1 --factor 3 --d_model 128 --d_ff 128)
    ;;
  TimeMixer)
    patience_default=10
    learning_rate=0.01
    label_len=0
    model_args=(
      --e_layers 2 --d_model 16 --d_ff 32
      --down_sampling_layers 3 --down_sampling_method avg --down_sampling_window 2
    )
    ;;
  TimesNet)
    model_args=(--e_layers 2 --d_layers 1 --factor 3 --d_model 16 --d_ff 32 --top_k 5)
    ;;
  FEDformer)
    batch_size_default=8
    c_out=1
    model_args=(--e_layers 2 --d_layers 1 --factor 3 --n_heads 4 --d_model 128 --d_ff 256)
    ;;
  TFPS)
    train_epochs_default=100
    batch_size_default=16
    patience_default=20
    learning_rate=0.0005
    c_out=1
    model_args=(
      --e_layers 3 --n_heads 4 --d_model 16 --d_ff 128 --dropout 0.3
      --patch_len 64 --stride 32
      --tfps_t_num_experts 16 --tfps_t_top_k 1
      --tfps_f_num_experts 16 --tfps_f_top_k 1
      --tfps_eta 5 --tfps_beta 0.1 --lradj type3
    )
    ;;
esac

train_epochs=${TRAIN_EPOCHS:-$train_epochs_default}
batch_size=${BATCH_SIZE:-$batch_size_default}
patience=${PATIENCE:-$patience_default}
loss_args=(--loss "$loss")
if [[ "$loss" == facl ]]; then
  loss_args+=(--facl_alpha "${FACL_ALPHA:-0.1}" --facl_eps "${FACL_EPS:-1e-8}")
elif [[ "$loss" == ps ]]; then
  loss_args+=(--ps_lambda "${PS_LAMBDA:-3.0}" --ps_delta "${PS_DELTA:-24}")
fi

command=(
  "$python_bin" -u run.py
  --task_name long_term_forecast
  --is_training 1
  --model_id "ETTh1_PCHIP15_384_384_${model}_${loss}"
  --model "$model"
  --data ETTh1_PCHIP15
  --root_path "$root_path"
  --data_path "$data_path"
  --features MS
  --target OT
  --freq t
  --seq_len 384
  --label_len "$label_len"
  --pred_len 384
  --enc_in 7
  --dec_in 7
  --c_out "$c_out"
  --eval_mask_mode observed
  --batch_size "$batch_size"
  --learning_rate "$learning_rate"
  --train_epochs "$train_epochs"
  --patience "$patience"
  --num_workers "$num_workers"
  --gpu "$gpu"
  --itr 1
  --des "PCHIP15_${loss}"
  "${loss_args[@]}"
  "${model_args[@]}"
)

if [[ ${DRY_RUN:-0} == 1 ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi

echo "Starting model=$model loss=$loss gpu=$gpu epochs=$train_epochs batch_size=$batch_size"
exec "${command[@]}"
