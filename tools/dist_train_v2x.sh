cd "/ai/volume/work"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mm

CONFIG=projects/MyProject/configs/exp1_dist.py
GPUS=1
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    tools/train_v2x.py \
    $CONFIG \
    --launcher pytorch ${@:3}
