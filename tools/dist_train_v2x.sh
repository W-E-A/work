# cd "/ai/volume/work"

# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate mm

# CONFIG=projects/MyProject/configs/exp1.py
# GPUS=8
# NNODES=${NNODES:-1}
# NODE_RANK=${NODE_RANK:-0}
# PORT=${PORT:-29500}
# MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python -m torch.distributed.launch \
#     --nnodes=$NNODES \
#     --node_rank=$NODE_RANK \
#     --master_addr=$MASTER_ADDR \
#     --nproc_per_node=$GPUS \
#     --master_port=$PORT \
#     tools/train_v2x.py \
#     $CONFIG \
#     # --amp \
#     # --auto-scale-lr \
#     # --checkpoint /ai/volume/work/work_dirs/exp1/single_epoch_20.pth \
#     --launcher pytorch ${@:3}

python -m torch.distributed.launch --nproc_per_node=8 tools/train_v2x.py projects/MyProject/configs/exp1.py --launcher pytorch --checkpoint /ai/volume/work/work_dirs/exp1/single_epoch_20.pth --auto-scale-lr