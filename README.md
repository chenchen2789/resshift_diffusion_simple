# 说明
本仓库是resshift diffusion的简化版，主要提取其中的关键操作，辅助对原论文的理解，重写代码有助于对整个模型的训练流程有更好的把握。
目前推理部分代码效果完全对齐论文仓库代码效果，利用简单下采样的退化方式重训模型，效果符合预期，放心食用。

## 论文unet权重推理结果
![](./results/eval.png)

## 简单下采样退化重训模型推理结果
![](./results/retrain_simple.png)

# 配置环境
同resshift 官方的配置文件，官网地址：https://github.com/zsyOAOA/ResShift

# 运行配置文件
config/realsr_swinunet_realesrgan256_journal_simple.yaml

需要将其中的数据集替换为本地的数据集路径，数据集格式可参考dataloader的输入和输出情况，示例中使用rgb的jpg图像
此外，配置文件中的autoencoder和unet的网络权重路径根据需要情况进行修改，注释掉路径不使用预训练权重，预训练权重参考论文的github库

# 模型推理测试
在main.py中调用
Trainer.evaluate()

# 模型训练
在main.py中调用
Trainer.train()
