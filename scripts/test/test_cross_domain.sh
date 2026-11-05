#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
args=()
if [[ -n "${DATA_PATH:-}" ]]; then args+=(--data_path "$DATA_PATH"); fi
"${PYTHON:-python}" main.py --mode eval --dataset cvc \
    --resume "${GAUSAM_CHECKPOINT:-outputs/polyp/checkpoint_best_dice.pth}" \
    --work_dir "${WORK_DIR:-outputs}" --task_name "${TASK_NAME:-polyp_to_cvc}" \
    "${args[@]}" "$@"
