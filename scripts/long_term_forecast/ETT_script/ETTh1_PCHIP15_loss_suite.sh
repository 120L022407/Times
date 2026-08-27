#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 mse|facl|ps" >&2
  exit 2
fi

loss=$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')
case "$loss" in
  mse|facl|ps) ;;
  *)
    echo "Unsupported loss: $1 (expected mse, facl, or ps)" >&2
    exit 2
    ;;
esac

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
runner="$script_dir/ETTh1_PCHIP15_experiment.sh"
models=${MODELS:-"PatchTST Informer DLinear iTransformer TimeMixer TimesNet FEDformer TFPS"}

for model in $models; do
  echo "===== $(date '+%F %T') model=$model loss=$loss gpu=${GPU:-0} ====="
  "$runner" "$model" "$loss"
done
