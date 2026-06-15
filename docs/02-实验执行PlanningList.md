## 实验执行 Planning List

本文档只规划“做实验”部分，包括代码实现、训练、测试、可视化、服务器运行和实验结果整理。不包含实验报告正文编写，实验报告后续单独制定 planning list。

本实验采用“本机开发 + Git 同步 + RTX 4090 服务器训练 + 结果回收”的方式执行：

```text
本机修改代码和配置
→ 本机做轻量检查
→ git commit / push
→ 服务器 git pull
→ 服务器 smoke test
→ 服务器正式训练 Exp1、Exp2、Exp3
→ 服务器生成测试指标和可视化结果
→ 回收轻量结果到本机最终目录
```

---

## 一、执行目标

完成 CIFAR-10 图像分类实验的三组核心实验：

| 实验编号 | 实验内容 | 目标 |
| -------- | -------- | ---- |
| Exp1 | SimpleCNN，无数据增强 | 建立 baseline |
| Exp2 | SimpleCNN，有数据增强 | 分析数据增强对泛化能力的影响 |
| Exp3 | ResNet18_CIFAR，有数据增强 | 分析残差网络相对普通 CNN 的性能提升 |

最终需要产出：

- 可运行的训练、测试和可视化代码；
- 三组实验的训练日志；
- 三组实验的训练曲线；
- 测试集总体准确率；
- 每类准确率；
- 混淆矩阵；
- 正确分类样例图；
- 错误分类样例图；
- 可选模型权重文件。

---

## 二、阶段 0：本机与服务器协作约定

### 1. 本机负责内容

本机主要负责代码开发和轻量检查：

- 修改 `train.py`、`test.py`、`visualize.py`、`plot_curves.py`；
- 修改 `models/` 下的模型代码；
- 修改 `configs/` 下的实验配置；
- 维护 `README.md` 和 `requirements.txt`；
- 做语法检查、导入检查和小规模逻辑检查；
- 通过 Git 将代码同步到服务器。

本机不作为正式训练环境，除非服务器临时不可用。

### 2. 服务器负责内容

RTX 4090 服务器主要负责正式实验运行：

- 存放或下载 CIFAR-10 数据集；
- 执行 Exp1、Exp2、Exp3 的 smoke test；
- 正式训练 Exp1、Exp2、Exp3；
- 保存训练日志、权重、曲线、混淆矩阵和预测样例；
- 执行测试集评价；
- 保留大体积权重文件。

### 3. Git 同步原则

Git 主要同步代码、配置、文档和轻量结果。

本实验使用的远程仓库为：

```text
https://github.com/TomorrowlandD/cifar10-resnet-data-augmentation
```

当前约定：

- 本机仓库 `origin` 指向该 GitHub 仓库；
- 服务器已通过该远程仓库读取到本机提交的代码；
- 后续本机修改代码后，通过 `git push` 同步；
- 服务器通过 `git pull` 获取最新代码。

建议纳入 Git：

```text
源代码/
docs/
README.md
requirements.txt
小体积 csv
小体积 png
```

不建议纳入 Git：

```text
data/
*.pth
checkpoints/
__pycache__/
临时日志
大体积训练输出
```

说明：

- 权重文件可以保留在服务器；
- 最终提交压缩包不强制包含权重；
- 如果需要提交权重，优先只放 `exp3_resnet18_best.pth`；
- README 中需要说明：如需复现实验，可运行 `train.py` 重新生成 Exp1、Exp2、Exp3 权重。

### 4. 阶段验收标准

- [ ] 本机和服务器使用同一个 Git 仓库；
- [ ] 服务器能够 `git pull` 获取本机最新代码；
- [ ] 大文件不会误提交到 Git；
- [ ] README 中写明训练和复现实验方式；
- [ ] 数据集路径、结果路径不依赖本机绝对路径。

---

## 三、需要用户手动操作的终端命令

本节列出需要手动执行的命令。后续代码文件写好后，命令中的路径和配置名应保持一致。

### 1. 本机：检查远程仓库

在本机 PowerShell 中执行：

```powershell
git remote -v
git status --short --branch
```

预期：

- `origin` 指向 `https://github.com/TomorrowlandD/cifar10-resnet-data-augmentation`；
- 当前分支为 `main`；
- 本机改动清楚可见。

### 2. 本机：提交代码和文档

在本机 PowerShell 中执行：

```powershell
git status --short
git add .gitignore docs README.md requirements.txt
git add 源代码
git status --short
git commit -m "Add CIFAR-10 experiment code and planning docs"
git push origin main
```

注意：

- 如果 `README.md`、`requirements.txt` 或 `源代码/` 尚未创建，对应 `git add` 命令会报路径不存在，等文件创建后再执行；
- 不要执行 `git add .`，避免误提交数据集、权重或临时输出；
- 不要提交 `data/`、`checkpoints/`、`*.pth`。

### 3. 服务器：拉取最新代码

在服务器终端中执行：

```bash
cd ~/cifar10-resnet-data-augmentation
git status --short --branch
git pull origin main
git status --short --branch
```

如果服务器上的仓库路径不同，先进入实际仓库目录再执行 `git pull`。

### 4. 服务器：检查 GPU 和 Python 环境

在服务器终端中执行：

```bash
nvidia-smi
python --version
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA not available')"
```

预期：

- 能看到 RTX 4090；
- `torch.cuda.is_available()` 输出 `True`；
- PyTorch 能识别 GPU。

### 5. 服务器：安装依赖

如果服务器还没有安装依赖，在服务器终端中执行：

```bash
python -m pip install -r requirements.txt
```

如果服务器使用虚拟环境，需要先激活虚拟环境，再安装依赖。具体激活命令以服务器环境为准。

### 6. 服务器：执行 smoke test

正式训练前，先每组跑 1 个 epoch。后续 `train.py` 写好后，建议支持 `--epochs 1` 覆盖配置中的 epoch。

```bash
python 源代码/train.py --config 源代码/configs/exp1_cnn_baseline.yaml --epochs 1
python 源代码/train.py --config 源代码/configs/exp2_cnn_aug.yaml --epochs 1
python 源代码/train.py --config 源代码/configs/exp3_resnet18.yaml --epochs 1
```

smoke test 后检查：

```bash
ls -R 实验结果
```

需要确认：

- 日志目录已生成；
- checkpoint 保存逻辑正常；
- 没有 CUDA、路径或数据集错误。

### 7. 服务器：正式训练三组实验

smoke test 通过后，在服务器终端中执行：

```bash
python 源代码/train.py --config 源代码/configs/exp1_cnn_baseline.yaml
python 源代码/train.py --config 源代码/configs/exp2_cnn_aug.yaml
python 源代码/train.py --config 源代码/configs/exp3_resnet18.yaml
```

如果显存或其他任务占用导致训练失败，将配置中的 `batch_size` 从 `256` 调整为 `128` 后重新运行对应实验。

### 8. 服务器：测试并生成指标

三组训练完成后，在服务器终端中执行：

```bash
python 源代码/test.py --config 源代码/configs/exp1_cnn_baseline.yaml
python 源代码/test.py --config 源代码/configs/exp2_cnn_aug.yaml
python 源代码/test.py --config 源代码/configs/exp3_resnet18.yaml
```

然后检查指标文件：

```bash
ls 实验结果
```

需要看到：

```text
metrics.csv
class_accuracy.csv
```

### 9. 服务器：生成曲线和可视化结果

在服务器终端中执行：

```bash
python 源代码/plot_curves.py
python 源代码/visualize.py --config 源代码/configs/exp3_resnet18.yaml
```

需要生成：

```text
实验结果/curves/exp1_loss_acc.png
实验结果/curves/exp2_loss_acc.png
实验结果/curves/exp3_loss_acc.png
实验结果/confusion_matrix/exp3_confusion_matrix.png
实验结果/predictions/correct_examples.png
实验结果/predictions/wrong_examples.png
```

### 10. 服务器：确认大文件没有进入 Git

在服务器终端中执行：

```bash
git status --short
git check-ignore -v 实验结果/checkpoints/exp3_resnet18_best.pth
```

预期：

- `*.pth` 权重文件被 `.gitignore` 忽略；
- `git status` 中不应出现 `checkpoints/` 或 `.pth` 文件。

### 11. 服务器：只提交轻量结果，可选

如果决定通过 Git 同步轻量结果，在服务器终端中只选择性添加 CSV 和 PNG：

```bash
git add 实验结果/metrics.csv 实验结果/class_accuracy.csv
git add 实验结果/curves 实验结果/confusion_matrix 实验结果/predictions
git status --short
git commit -m "Add experiment metrics and visualizations"
git push origin main
```

注意：

- 不提交 `实验结果/checkpoints/`；
- 不提交 `.pth`；
- 如果轻量结果改为手动下载回本机，这一步可以不执行。

### 12. 本机：用 VS Code Remote SSH 下载轻量结果，推荐

如果使用 VS Code Remote SSH 连接服务器，推荐直接在远程资源管理器中右键下载结果文件。

在 VS Code 远程窗口中找到服务器仓库：

```text
cifar10-resnet-data-augmentation/
└── 实验结果/
```

需要右键下载到本机最终目录的内容：

```text
实验结果/metrics.csv
实验结果/class_accuracy.csv
实验结果/curves/
实验结果/confusion_matrix/
实验结果/predictions/
```

不需要下载：

```text
实验结果/checkpoints/
data/
*.pth
```

### 13. 本机：拉取服务器提交的轻量结果，可选

如果服务器已经通过 Git 提交轻量结果，在本机 PowerShell 中执行：

```powershell
git pull origin main
git status --short --branch
```

如果不通过 Git 回收结果，也可以使用 VS Code Remote SSH 右键下载，或使用 `scp` 手动下载轻量结果到本机最终目录。

### 14. 手动操作验收标准

- [ ] 本机 `git remote -v` 显示正确 GitHub 仓库；
- [ ] 本机可以 `git push origin main`；
- [ ] 服务器可以 `git pull origin main`；
- [ ] 服务器 `nvidia-smi` 能看到 RTX 4090；
- [ ] 服务器 PyTorch 能识别 CUDA；
- [ ] 服务器 smoke test 命令可以执行；
- [ ] 正式训练、测试、可视化命令已经整理清楚；
- [ ] 权重文件和 `checkpoints/` 没有被加入 Git；
- [ ] 轻量结果的回收方式已经确定，优先使用 VS Code Remote SSH 右键下载。

---

## 四、阶段 1：准备代码目录

### 1. 建立实验目录结构

需要整理出以下结构：

```text
源代码/
├── data/
│   └── cifar10/
├── models/
│   ├── simple_cnn.py
│   └── resnet_cifar.py
├── configs/
│   ├── exp1_cnn_baseline.yaml
│   ├── exp2_cnn_aug.yaml
│   └── exp3_resnet18.yaml
├── train.py
├── test.py
├── visualize.py
├── plot_curves.py
├── utils.py
└── requirements.txt
```

### 2. 建立结果目录结构

需要整理出以下结构：

```text
实验结果/
├── checkpoints/
├── curves/
├── confusion_matrix/
├── predictions/
├── metrics.csv
└── class_accuracy.csv
```

说明：

- `checkpoints/` 可选提交；
- 如果压缩包过大，可以只保留 `exp3_resnet18_best.pth`；
- 如果不提交权重，需要保证 `train.py` 可以重新生成 Exp1、Exp2、Exp3 的权重。

### 3. 阶段验收标准

- [ ] 目录结构清晰；
- [ ] 代码和结果分开放置；
- [ ] 不把临时缓存、无关文件放入最终提交目录；
- [ ] 不批量删除文件或目录，如需清理大文件，先单个确认后处理。

---

## 五、阶段 2：实现数据加载与预处理

### 1. 本机实现 CIFAR-10 数据加载代码

需要完成：

- 加载 CIFAR-10 训练集和测试集；
- 从原始训练集中划分训练集和验证集；
- 固定随机种子，保证划分可复现；
- 支持 `batch_size = 256`，如显存占用异常可降为 `128`；
- 支持 RTX 4090 服务器 CUDA 训练。

需要注意：

- 数据集实际下载和缓存建议放在服务器；
- 本机代码中不要写死服务器私有绝对路径；
- 数据集路径应通过配置文件或命令行参数指定。

### 2. 实现两套 transform

Exp1 使用无增强版本：

```python
transforms.ToTensor()
transforms.Normalize(mean, std)
```

Exp2、Exp3 使用增强版本：

```python
transforms.RandomCrop(32, padding=4)
transforms.RandomHorizontalFlip()
transforms.ColorJitter(brightness=0.2, contrast=0.2)
transforms.ToTensor()
transforms.Normalize(mean, std)
```

### 3. 在服务器生成数据集样例图

需要输出：

```text
实验结果/dataset_examples.png
```

### 4. 阶段验收标准

- [ ] 能正常读取 CIFAR-10；
- [ ] 能输出训练集、验证集、测试集数量；
- [ ] 能生成数据集样例图；
- [ ] DataLoader 能正常迭代一个 batch；
- [ ] CPU / CUDA 设备选择逻辑正常。

---

## 六、阶段 3：实现模型

### 1. 实现 SimpleCNN

文件：

```text
源代码/models/simple_cnn.py
```

模型结构建议：

```text
Input: 3×32×32
Conv-BN-ReLU
Conv-BN-ReLU
MaxPool
Conv-BN-ReLU
Conv-BN-ReLU
MaxPool
Conv-BN-ReLU
Global Average Pooling
Fully Connected
Output: 10 classes
```

### 2. 实现 ResNet18_CIFAR

文件：

```text
源代码/models/resnet_cifar.py
```

关键修改：

```python
model.conv1 = nn.Conv2d(
    3, 64, kernel_size=3, stride=1, padding=1, bias=False
)
model.maxpool = nn.Identity()
model.fc = nn.Linear(model.fc.in_features, 10)
```

### 3. 阶段验收标准

- [ ] `SimpleCNN` 输入 `[B, 3, 32, 32]`，输出 `[B, 10]`；
- [ ] `ResNet18_CIFAR` 输入 `[B, 3, 32, 32]`，输出 `[B, 10]`；
- [ ] 两个模型都能完成一次前向传播；
- [ ] 参数量和模型名称能正常打印或记录。

---

## 七、阶段 4：实现训练脚本与本机轻量检查

### 1. 编写配置文件

需要完成：

```text
源代码/configs/exp1_cnn_baseline.yaml
源代码/configs/exp2_cnn_aug.yaml
源代码/configs/exp3_resnet18.yaml
```

每个配置文件至少包含：

- 实验名称；
- 模型类型；
- 是否使用数据增强；
- batch size；
- epoch；
- optimizer；
- learning rate；
- weight decay；
- scheduler；
- 输出路径。

### 2. 编写 train.py

训练脚本需要支持：

- 读取配置文件；
- 设置随机种子；
- 自动选择 CUDA；
- 构建数据集和模型；
- 训练与验证；
- 保存每个 epoch 的 loss 和 accuracy；
- 保存最佳模型权重；
- 输出训练日志 CSV。

建议日志命名：

```text
实验结果/logs/exp1_train_log.csv
实验结果/logs/exp2_train_log.csv
实验结果/logs/exp3_train_log.csv
```

### 3. 本机轻量检查

正式推送到服务器前，本机先做轻量检查：

```text
检查 Python 文件是否能正常导入
检查配置文件是否能正常读取
检查模型能否完成一次前向传播
检查输出路径是否可配置
```

如果本机没有合适的 PyTorch / CUDA 环境，至少完成静态检查和配置文件检查。

### 4. Git 推送到服务器

本机确认代码可用后：

```text
git status
git add 源代码 docs README.md requirements.txt
git commit
git push
```

服务器执行：

```text
git pull
```

注意：

- 不提交 `data/`；
- 不提交 `*.pth`；
- 不提交 `checkpoints/`；
- 不提交无关缓存文件。

### 5. 阶段验收标准

- [ ] 三个配置文件都能被 `train.py` 正常读取；
- [ ] 本机完成轻量检查；
- [ ] 最新代码已经通过 Git 同步到服务器；
- [ ] 服务器可以正常看到最新配置和代码；
- [ ] 大文件没有误提交。

---

## 八、阶段 5：服务器 smoke test

正式训练前，服务器先分别跑短训练：

```text
Exp1：1 epoch
Exp2：1 epoch
Exp3：1 epoch
```

### 1. 服务器 smoke test 重点

需要确认：

- CUDA 可用；
- CIFAR-10 数据路径正确；
- `batch_size = 256` 能正常运行；
- 训练日志能保存；
- 权重保存路径正常；
- 曲线和结果目录能够创建；
- `test.py` 能加载 smoke test 权重完成一次测试。

### 2. smoke test 失败处理

如果失败，优先检查：

- 依赖版本；
- CUDA / PyTorch 是否匹配；
- 数据集路径；
- 配置文件路径；
- 输出目录权限；
- batch size 是否过大。

如果显存不足，先将 batch size 调整为 `128`，不改变实验主线。

### 3. 阶段验收标准

- [ ] Exp1、Exp2、Exp3 都能在服务器完成 1 epoch；
- [ ] 日志、权重、输出目录都正常；
- [ ] 服务器环境确认可用于正式训练。

---

## 九、阶段 6：服务器正式训练三组实验

### 1. 服务器正式运行 Exp1

配置：

```text
model: SimpleCNN
augmentation: False
input_size: 32×32
batch_size: 256
epochs: 15
```

目标产物：

- `exp1_train_log.csv`
- `exp1_loss_acc.png`
- 可选：`exp1_best.pth`

### 2. 服务器正式运行 Exp2

配置：

```text
model: SimpleCNN
augmentation: True
input_size: 32×32
batch_size: 256
epochs: 20
```

目标产物：

- `exp2_train_log.csv`
- `exp2_loss_acc.png`
- 可选：`exp2_best.pth`

### 3. 服务器正式运行 Exp3

配置：

```text
model: ResNet18_CIFAR
augmentation: True
input_size: 32×32
batch_size: 256
epochs: 25
```

目标产物：

- `exp3_train_log.csv`
- `exp3_loss_acc.png`
- 建议保留：`exp3_resnet18_best.pth`

### 4. 阶段验收标准

- [ ] 三组实验都完整训练结束；
- [ ] 每组都有训练日志；
- [ ] 每组都有 loss / accuracy 曲线；
- [ ] 每组都有验证集最佳记录；
- [ ] 如果训练中断，可以从对应配置重新运行。

---

## 十、阶段 7：服务器测试与指标统计

### 1. 编写 test.py

测试脚本需要支持：

- 加载指定模型配置；
- 加载对应最佳权重；
- 在 CIFAR-10 测试集上预测；
- 计算总体测试准确率；
- 计算每类准确率；
- 保存预测结果和真实标签。

### 2. 生成总体指标表

输出：

```text
实验结果/metrics.csv
```

建议字段：

```text
experiment, model, augmentation, epochs, best_val_acc, test_acc
```

### 3. 生成每类准确率表

输出：

```text
实验结果/class_accuracy.csv
```

建议字段：

```text
class_name, exp1_acc, exp2_acc, exp3_acc
```

### 4. 阶段验收标准

- [ ] `metrics.csv` 包含 Exp1、Exp2、Exp3 三行结果；
- [ ] `class_accuracy.csv` 包含 CIFAR-10 十个类别；
- [ ] Exp3 测试准确率应明显高于或接近 Exp2；
- [ ] 如果个别结果不符合预期，需要保留真实结果，并在后续报告分析中解释。

---

## 十一、阶段 8：服务器可视化结果生成

### 1. 编写 plot_curves.py

需要生成：

```text
实验结果/curves/exp1_loss_acc.png
实验结果/curves/exp2_loss_acc.png
实验结果/curves/exp3_loss_acc.png
```

每张图建议包含：

- train loss；
- val loss；
- train accuracy；
- val accuracy。

### 2. 生成混淆矩阵

建议使用 Exp3 作为主要展示模型：

```text
实验结果/confusion_matrix/exp3_confusion_matrix.png
```

### 3. 编写 visualize.py

需要生成：

```text
实验结果/predictions/correct_examples.png
实验结果/predictions/wrong_examples.png
```

正确样例图应包含：

- 原图；
- 真实类别；
- 预测类别。

错误样例图应包含：

- 原图；
- 真实类别；
- 预测类别；
- 便于后续分析的典型混淆类别。

### 4. 阶段验收标准

- [ ] 三组训练曲线都能正常打开；
- [ ] 混淆矩阵类别标签完整；
- [ ] 正确分类样例不少于 8 张；
- [ ] 错误分类样例不少于 8 张；
- [ ] 图片清晰，能直接放入实验报告。

---

## 十二、阶段 9：服务器结果回收到本机

### 1. 需要回收的轻量结果

服务器训练和可视化完成后，需要将以下文件回收到本机最终目录：

```text
实验结果/metrics.csv
实验结果/class_accuracy.csv
实验结果/curves/exp1_loss_acc.png
实验结果/curves/exp2_loss_acc.png
实验结果/curves/exp3_loss_acc.png
实验结果/confusion_matrix/exp3_confusion_matrix.png
实验结果/predictions/correct_examples.png
实验结果/predictions/wrong_examples.png
```

这些文件体积较小，适合放入最终提交目录，也可以按需纳入 Git。

如果需要通过 Git 同步轻量结果，只选择性提交 `metrics.csv`、`class_accuracy.csv` 和必要 PNG，不提交 `checkpoints/` 或 `.pth` 权重文件。

### 2. 推荐回收方式：VS Code Remote SSH 右键下载

本实验推荐使用 VS Code Remote SSH 的远程文件资源管理器下载结果：

1. 在 VS Code Remote SSH 窗口中打开服务器仓库；
2. 找到 `实验结果/` 目录；
3. 右键下载 `metrics.csv` 和 `class_accuracy.csv`；
4. 右键下载 `curves/`、`confusion_matrix/`、`predictions/` 三个目录；
5. 下载到本机仓库的 `实验结果/` 目录；
6. 本机打开 PNG 和 CSV，确认文件完整。

不建议右键下载 `checkpoints/`，除非确实需要保留代表性权重。

### 3. 建议留在服务器的文件

以下文件可以留在服务器，不强制回收到本机：

```text
data/
实验结果/checkpoints/
*.pth
大体积中间输出
```

如果需要在最终压缩包中放一个代表性权重，优先只回收：

```text
实验结果/checkpoints/exp3_resnet18_best.pth
```

### 4. 回收后的本机检查

本机需要确认：

- 所有轻量结果文件都能正常打开；
- CSV 文件编码和字段正常；
- PNG 图片清晰可读；
- 文件路径与最终提交目录一致；
- README 中对权重不强制提交的说明准确。

### 5. 阶段验收标准

- [ ] 本机已经拿到报告所需的全部轻量实验结果；
- [ ] 使用 VS Code Remote SSH 右键下载的 CSV 和 PNG 文件均可打开；
- [ ] 服务器保留完整训练输出和权重；
- [ ] Git 中没有误提交大文件；
- [ ] 最终提交目录可以不依赖服务器直接查看实验结果。

---

## 十三、阶段 10：实验结果自检

### 1. 结果文件检查

必须存在：

```text
实验结果/metrics.csv
实验结果/class_accuracy.csv
实验结果/curves/exp1_loss_acc.png
实验结果/curves/exp2_loss_acc.png
实验结果/curves/exp3_loss_acc.png
实验结果/confusion_matrix/exp3_confusion_matrix.png
实验结果/predictions/correct_examples.png
实验结果/predictions/wrong_examples.png
```

可选存在：

```text
实验结果/checkpoints/exp3_resnet18_best.pth
```

### 2. 指标合理性检查

重点检查：

- Exp1 是否能作为 baseline；
- Exp2 相比 Exp1 是否体现数据增强效果；
- Exp3 相比 Exp2 是否体现 ResNet18 的结构优势；
- 每类准确率是否存在明显短板类别；
- 错误样例是否能支撑后续失败案例分析。

### 3. 可复现性检查

需要确认：

- `requirements.txt` 能覆盖主要依赖；
- `README.md` 中写明训练、测试、可视化运行顺序；
- README 中注明：如需复现实验，可运行 `train.py` 重新生成 Exp1、Exp2、Exp3 权重；
- 不依赖本机绝对路径；
- 数据集路径可以通过配置文件或命令行参数修改。

### 4. 阶段验收标准

- [ ] 必须存在的 CSV 和 PNG 结果文件均已检查；
- [ ] 可选权重文件策略已经确认；
- [ ] Exp1、Exp2、Exp3 的指标关系已经初步检查；
- [ ] 错误样例能够支撑后续失败案例分析；
- [ ] README、requirements 和路径配置满足复现实验要求。

---

## 十四、最终实验完成标准

当以下条件全部满足时，实验部分可以认为完成：

- [ ] Exp1、Exp2、Exp3 三组实验已经跑通；
- [ ] 三组实验有训练日志和曲线；
- [ ] 测试集准确率已经统计；
- [ ] 每类准确率已经统计；
- [ ] 混淆矩阵已经生成；
- [ ] 正确和错误分类样例已经生成；
- [ ] 代码可以通过 README 说明重新运行；
- [ ] 权重文件不作为强制提交内容；
- [ ] 实验结果足够支撑后续实验报告撰写。
