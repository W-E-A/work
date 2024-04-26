#!/usr/bin/env bash
WORLD_SIZE=$1
GPUS=$2
CONFIG=$3
RANK=${RANK:='0'}
MASTER_ADDR=${MASTER_ADDR:='127.0.0.1'}
MASTER_PORT=${MASTER_PORT:='123456'}

echo "WORLD_SIZE: ${WORLD_SIZE}"
echo "RANK: ${RANK}"
echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "MASTER_PORT: ${MASTER_PORT}"
echo "GPUS: ${GPUS}"

PYTHONPATH="$(dirname $0)/../..":$PYTHONPATH \
python -m torch.distributed.launch --nnodes=${WORLD_SIZE} \
    --node_rank=$RANK --nproc_per_node=$GPUS \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    $(dirname "$0")/train_v2x.py $CONFIG --launcher pytorch ${@:4} #--deterministic
