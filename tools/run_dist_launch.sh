#!/usr/bin/env bash

set -x

RUN_COMMAND=${@:2}
if [ $GPUS -lt 2]; then
    GPUS_PER_NODE=${GPUS_PER_NODE:-2}
else
    GPUS_PER_NODE=${GPUS_PER_NODE:-2}
fi
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-"29500"}
NODE_RANK=${NODE_RANK:-0}

let "NNODES=GPUS/GPUS_PER_NODE"

python ./tools/launch.py \
    --nnodes ${NNODES} \
    --node_rank ${NODE_RANK} \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT} \
    --nproc_per_node ${GPUS_PER_NODE} \
    ${RUN_COMMAND}