## `tools`目录

存放了基本的训练测试脚本、数据处理代码、其他工具文件

`analysis_tools`：用于日志、模型推理速速benchmark、模型info的分析

`dataset_converters`：用于不同的数据集的处理操作，以及转化为mm的标准数据格式，包括对于每个不同数据集的原生处理

`deployment`：用于沟通mm和torch的一些代码，例如torchserver等一些工具

`model_converters`：用于开发的时候转化不同来源的模型权重，以及模型发布等