#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."

# Each output resolution has its own directory; input size comes from the checkpoint.
read -r -a sizes <<< "${SIZES:-128 384 896}"
for size in "${sizes[@]}"; do
    "${PYTHON:-python}" main.py --mode eval --dataset polyp \
        --data_path "${DATA_PATH:-data/sessile-Kvasir}" \
        --resume "${GAUSAM_CHECKPOINT:-outputs/polyp/checkpoint_best_dice.pth}" \
        --work_dir "${WORK_DIR:-outputs}" "$@" \
        --label_size "${size}" --task_name "${TASK_NAME:-polyp_multiscale}_${size}"
done
