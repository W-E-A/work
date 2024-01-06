source ~/miniconda3/etc/profile.d/conda.sh
conda activate mm

cd "/ai/volume/work"
python tools/train_v2x.py projects/Where2comm/configs/where2comm_dair-v2x-c_my2.py