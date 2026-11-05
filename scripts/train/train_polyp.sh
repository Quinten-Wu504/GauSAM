#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."

# NPROC_PER_NODE=4 bash scripts/train/train_polyp.sh ... enables multi-GPU training.
launcher=("${PYTHON:-python}")
if (( ${NPROC_PER_NODE:-1} > 1 )); then
    launcher+=(-m torch.distributed.run --standalone --nproc_per_node "${NPROC_PER_NODE}")
fi
"${launcher[@]}" main.py --config configs/polyp.json \
    --data_path "${DATA_PATH:-data/sessile-Kvasir}" \
    --work_dir "${WORK_DIR:-outputs}" --task_name "${TASK_NAME:-polyp}" "$@"
