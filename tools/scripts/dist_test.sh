#!/usr/bin/env bash
# Modified by RV-SDTM contributors for this public release; see NOTICE.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: bash scripts/dist_test.sh <gpus> [test.py arguments...]" >&2
    exit 2
fi

NUM_GPUS=$1
shift
MASTER_PORT=${MASTER_PORT:-$((10000 + RANDOM % 50000))}

exec torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --nproc_per_node="${NUM_GPUS}" \
    --master_addr=127.0.0.1 \
    --master_port="${MASTER_PORT}" \
    test.py --launcher pytorch "$@"
