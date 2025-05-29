# 使用CNN识别CIFAR10

## 开始
- 在model中定义了可配置的CNN，具体配置可前往config中自定义，参照其中格式可实现自己的CNN
- 在train.py中引用config中的模型定义，调用main中的训练函数，可以进行对比训练以及单次训练
- loss中定义了两种损失，可以在CNN中使用：Focal、LabelSmoothing
- visualize中定义了可视化训练过程（训练时结束后自动调用）以及获取已有模型可视化卷积核以及loss landscape
- visualization中，train中的是单个模型的图像，comparison中的是config各个类定义的模型对比图像