#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."

"${PYTHON:-python}" main.py --mode eval --dataset polyp \
    --data_path "${DATA_PATH:-data/sessile-Kvasir}" \
    --resume "${GAUSAM_CHECKPOINT:-outputs/polyp/checkpoint_best_dice.pth}" \
    --work_dir "${WORK_DIR:-outputs}" --task_name "${TASK_NAME:-polyp_test}" "$@"
