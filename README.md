## command

```Shell
# without checkpoint
nohup python -m torch.distributed.launch --nproc_per_node=3 tools/train_v2x.py projects/MyProject/configs/exp1.py --launcher pytorch &

# with checkpoint
nohup python -m torch.distributed.launch --nproc_per_node=3 tools/train_v2x.py projects/MyProject/configs/exp1.py --launcher pytorch --checkpoint /ai/volume/work/work_dirs/exp1/single_epoch_20.pth &

# test
python tools/test_v2x.py projects/MyProject/configs/exp1.py work_dirs/exp1/epoch_20.pth --work-dir work_dirs/uni_temppp

#create data
# dair-v2x-c
python tools/create_data.py dair-v2x --root-path /ai/volume/dataset/cooperative-vehicle-infrastructure --version c --out-dir ./data/dair
# deepaccident
python tools/create_data.py deepaccident --root-path /ai/datasets/DeepAccident_data --sample-interval 5 --out-dir ./data/deepaccident
```
**工程文件夹：`projects/MyProject`**

**配置文件：`projects/MyProject/configs/exp1.py`**


## 这里是源代码的根目录，放置打包工具和一些其他的文件

#### `configs`是develop的原生配置，**将会在打包的时候放置在`mmdet3d/.mim`下以供`mim`来使用**

#### `data`是存放所有的数据集、数据标签、临时数据、权重数据、中间数据的地方

#### `demo`是develop的示例代码，有一些数据样本和数据处理、推理的代码，**将会在打包的时候放置在`mmdet3d/.mim`下以供`mim`程序来使用**

#### `mmdet3d`作为项目的主体文件，**注意并不是作为第三方库使用，而是作为开发库使用，详情可以见`setup.py`文件**

#### `projects`是每个模型自带的特殊文件，存放每个模型的配置、自定义操作、自定义信息等代码，内部结构可以自定义

#### `tools`是develop的工具代码，涵盖数据处理、训练测试的总代码，**将会在打包的时候放置在`mmdet3d/.mim`下以供`mim`来使用**

## 注意，如果系统环境中同时存在多个版本的`mmdet3d`，注意在使用前于项目根目录运行 `pip install -e .`

## TODO

- [ ] 分离dair-v2x-c数据
- [ ] 解析v2x-seq数据
- [ ] 给出where2comm原理下的精度
- [ ] 给出新方法的精度
- [ ] 给出合适的可视化程序
- [ ] 给出合适的推理程序
- [ ] 看检测部分的后处理是否有问题
- [ ] 解决其他FIXME的部分

## description

# 基于预测轨迹、自车规划的相关性筛选

## 场景：V2I

## 视角：路侧视角下

## 已知信息

### 1、路侧检测范围内所有目标的3D检测信息（位置，尺寸等）、轨迹预测信息
### 2、EGO的规划轨迹

## 预测时长：3s，6帧 [2、3、4]s [4、6、8]帧

## 危险距离：10m [5、10、15]m

## 方法：对于每一条路侧所检测到的目标，在每一帧时刻下，如果目标与EGO的距离低于危险距离，表明了具有碰撞的可能，标记该目标为具有相关性的目标