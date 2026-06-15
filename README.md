# CIFAR-10 ResNet Data Augmentation Experiment

本项目用于完成 CIFAR-10 图像分类实验，重点比较普通 CNN、数据增强和 ResNet18 结构改造对分类效果的影响。

## 实验设置

| 实验 | 模型 | 数据增强 | 目标 |
| --- | --- | --- | --- |
| Exp1 | SimpleCNN | 否 | 建立 baseline |
| Exp2 | SimpleCNN | 是 | 分析数据增强对泛化能力的影响 |
| Exp3 | ResNet18_CIFAR | 是 | 分析残差网络相对普通 CNN 的性能提升 |

## 目录结构

```text
源代码/
├── configs/
├── models/
├── train.py
├── test.py
├── visualize.py
├── plot_curves.py
└── utils.py

实验结果/
├── logs/
├── checkpoints/
├── curves/
├── confusion_matrix/
├── predictions/
├── metrics.csv
└── class_accuracy.csv
```

`data/`、`源代码/data/`、`实验结果/checkpoints/` 和 `*.pth` 不建议提交到 Git。权重文件可以留在服务器；如需复现实验，可重新运行训练脚本生成。

## 环境安装

```bash
python -m pip install -r requirements.txt
```

服务器训练前建议先确认 CUDA：

```bash
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA not available')"
```

## Smoke Test

正式训练前，每组实验先跑 1 个 epoch：

```bash
python 源代码/train.py --config 源代码/configs/exp1_cnn_baseline.yaml --epochs 1
python 源代码/train.py --config 源代码/configs/exp2_cnn_aug.yaml --epochs 1
python 源代码/train.py --config 源代码/configs/exp3_resnet18.yaml --epochs 1
```

如果显存不足，可以临时加 `--batch-size 128`。

## 正式训练

```bash
python 源代码/train.py --config 源代码/configs/exp1_cnn_baseline.yaml
python 源代码/train.py --config 源代码/configs/exp2_cnn_aug.yaml
python 源代码/train.py --config 源代码/configs/exp3_resnet18.yaml
```

训练日志会保存到：

```text
实验结果/logs/exp1_train_log.csv
实验结果/logs/exp2_train_log.csv
实验结果/logs/exp3_train_log.csv
```

最佳权重默认保存到 `实验结果/checkpoints/`，该目录不作为强制提交内容。

## 测试与指标统计

```bash
python 源代码/test.py --config 源代码/configs/exp1_cnn_baseline.yaml
python 源代码/test.py --config 源代码/configs/exp2_cnn_aug.yaml
python 源代码/test.py --config 源代码/configs/exp3_resnet18.yaml
```

输出：

```text
实验结果/metrics.csv
实验结果/class_accuracy.csv
```

## 曲线与可视化

生成三组训练曲线：

```bash
python 源代码/plot_curves.py
```

生成 Exp3 混淆矩阵、正确分类样例和错误分类样例：

```bash
python 源代码/visualize.py --config 源代码/configs/exp3_resnet18.yaml
```

输出：

```text
实验结果/dataset_examples.png
实验结果/curves/exp1_loss_acc.png
实验结果/curves/exp2_loss_acc.png
实验结果/curves/exp3_loss_acc.png
实验结果/confusion_matrix/exp3_confusion_matrix.png
实验结果/predictions/correct_examples.png
实验结果/predictions/wrong_examples.png
```

## 结果回收

推荐使用 VS Code Remote SSH 从服务器右键下载轻量结果：

```text
实验结果/metrics.csv
实验结果/class_accuracy.csv
实验结果/dataset_examples.png
实验结果/curves/
实验结果/confusion_matrix/
实验结果/predictions/
```

不需要下载或提交：

```text
源代码/data/
实验结果/checkpoints/
*.pth
```
