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