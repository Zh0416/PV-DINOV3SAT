# 实验 A：冻结 DINOv3-SAT 原型匹配

本实验不训练 DINOv3，也不训练分割头。它使用一张带像素掩膜的支持影像建立光伏/背景特征原型，再用余弦相似度差对查询影像分割。

## 当前数据审计

首次审计发现 `datasets/images` 和 `datasets/mask` 中有 663 对 BMP，影像为 RGB `1024×1024`，掩膜为单通道 `1024×1024`，掩膜值 `0` 表示背景、非零值 `12` 表示光伏。掩膜名以 `_label.bmp` 结尾。文件名含两个空间坐标数值；默认会排除支持图坐标 1000 个坐标单位以内的查询图，减少相邻切片泄漏。正式记录实验前请确认这两个坐标以米为单位；不确定时可将 `--spatial-buffer` 设为 `0`，并在实验记录中说明。

## SAT 权重

官方 GitHub 仓库只发布加载代码。ViT-L/16 的 SAT-493M 权重需先通过 [DINOv3 官方下载页面](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/)申请并接受其条款。获得下载链接后，将权重保存为官方文件名 `dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth`，并通过 `--weights` 指定路径。不要用随机初始化权重或其他预训练权重把结果记作实验 A。

## 环境

官方仓库当前要求 Python 3.11 或更新版本。建议 Kaggle GPU Notebook 使用 Python 3.11、CUDA 版 PyTorch。工作区里的 DINOv3 官方源码位于 `codex运行文件/dinov3`；GitHub 项目中会只提交本实验代码，Kaggle 上需另行 clone 官方仓库。

```bash
pip install pillow numpy
```

Kaggle 通常已安装 PyTorch。若本地运行，按本机 CUDA 版本安装 PyTorch，再安装 DINOv3 官方仓库依赖。

## 先检查数据

```bash
python codex运行文件/experiment_a/run_experiment_a.py --validate-only
```

自动支持图按文件名中的空间坐标选取地理位置居中的影像，不读取查询掩膜。正式实验建议明确指定支持影像，便于复现：

```bash
python "codex运行文件/experiment_a/run_experiment_a.py" --support PV08_331824_1195530.bmp --weights /path/to/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth --device cuda
```

默认使用 512×512 分块、ViT-L/16 最后四层特征、前景/背景加权原型，分数为 `cosine(fg) - cosine(bg)`，阈值为 0。每次运行会在 `codex运行文件/experiment_a/runs/<时间戳>/` 下保存 `summary.json`、逐图指标 CSV 和预测掩膜。

默认快速测试固定使用 1 张支持图和 20 张查询图。若要对完整数据集做一次性诊断，可显式指定 `--data-root datasets --all-data`；再加 `--limit-query 1 --no-save-predictions` 可缩短诊断时间。正式报告至少同时记录支持影像、空间缓冲距离、mIoU、Dice、精确率和召回率；阈值若需要调整，只能用验证区域选择，不能根据测试标签选择。

## 固定 1-shot / 20-query 测试集

后续模型版本统一使用 [`benchmark_v1.json`](benchmark_v1.json)：支持图固定为 `PV08_331824_1195530.bmp`，查询图固定为清单中的 20 张。清单按文件名坐标做确定性最远点抽样，并排除距支持图不超过 1000 个坐标单位的图像；抽样不查看掩膜或模型结果。坐标的物理单位尚未独立核实，因此这里把缓冲记作“文件名坐标单位”。更换支持图或查询图应创建新的 benchmark 版本，不覆盖 v1。

为后续快速测试，清单中的 1 张支持图、20 张查询图及其掩膜已单独整理到 `benchmark_v1_data/images/` 和 `benchmark_v1_data/mask/`，共 21 对。主实验脚本默认读取这份子集和 `benchmark_v1.json`；固定清单之外的完整数据仍保留在项目根目录 `datasets/`。层权重实验只用支持图及其掩膜反向传播更新四个标量权重；查询图的标签只在权重训练完成后用于评估。

用本地 GPU 按固定清单运行：

```bash
python "codex运行文件/experiment_a/run_experiment_a.py" --weights /path/to/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth --device cuda
```

Kaggle Notebook 使用同一组影像和掩膜，结果保存在 `codex运行文件/kaggle_run_output/benchmark_v1/`。每个版本都应保留逐图指标，便于确认变化是普遍提升还是由少数查询图带动。

### Benchmark v1 初始基线

2026-11-05 在 Kaggle Notebook 第 6 版、Tesla T4 上完成 20 张查询测试。冻结 DINOv3-SAT 原型匹配得到 micro 前景 IoU `0.4743`、micro mIoU `0.6761`、micro Dice `0.6434`、Precision `0.4809`、Recall `0.9719`；macro mIoU `0.6980`、macro Dice `0.6314`。逐图指标、20 张预测掩膜和完整运行日志均保存在上述目录，且逐图混淆矩阵计数已用本地真值掩膜复核。

该固定集可作为后续版本的开发对比基准。由于反复据此调整模型会逐渐对这 20 张图过拟合，最终报告仍应另留未参与调参的区域级测试集。

### 中间层与权重实验 v1

2026-11-05 在 Kaggle Notebook 第 7 版、Tesla T4 上测试第 6、12、18、24 层、等权融合和可学习权重。DINOv3-SAT 全程冻结；反向传播只更新四个 softmax 标量。训练与内部空间验证都由唯一支持图 `PV08_331824_1195530.bmp` 的四个不重叠象限构造。20 张查询图及其掩膜只在训练结束后用于评估。

| 方法 | 层权重（6、12、18、24） | micro mIoU | macro mIoU | micro Dice |
|---|---|---:|---:|---:|
| 第 6 层 | 1、0、0、0 | 0.5714 | 0.5865 | 0.4929 |
| 第 12 层 | 0、1、0、0 | 0.7008 | 0.7076 | 0.6705 |
| 第 18 层 | 0、0、1、0 | 0.7060 | **0.7465** | 0.6786 |
| 第 24 层 | 0、0、0、1 | 0.7338 | 0.7371 | 0.7152 |
| 等权融合 | 0.25、0.25、0.25、0.25 | 0.6751 | 0.7039 | 0.6400 |
| 可学习融合 | 0.0590、0.0119、0.0052、0.9238 | **0.7344** | 0.7397 | **0.7160** |

当前清单上，第 24 层是最强的单层方案；可学习融合将约 92.4% 权重分配给第 24 层，相比单用第 24 层只小幅提高 micro mIoU `0.0006` 和 micro Dice `0.0007`。若关注逐图 macro mIoU，第 18 层最好。等权融合明显较弱。后续快速测试建议保留第 24 层、第 18 层和可学习融合进行对照；这是一张支持图和固定 20 张查询图上的开发结果，不能视为独立区域泛化结论。

权重训练的内部验证损失从等权初始值 `0.4884` 降到 `0.4742`。完整权重历史、逐图指标及 120 张预测掩膜保存在 `codex运行文件/kaggle_run_output/layer_selection_v1/`，该目录由 `.gitignore` 排除。

## 实验边界

A 是冻结特征的直接匹配诊断；层权重实验只使用一张支持图训练四个融合系数。快速测试默认只读固定 benchmark v1 的 21 对样本，完整 663 对数据仍保留在项目根目录 `datasets/`。当前没有明确的独立区域级测试划分，因此这些结果属于开发对比，不能单独作为跨区域泛化结论。应根据真实区域/日期信息建立独立划分后再报告跨区域结果。

## Kaggle 冒烟测试

2026-11-04 在私有 Kaggle GPU Notebook `zh0416/pv-dinov3sat-experiment-a-run` 运行第 5 版。Notebook 的数据挂载根目录是 `/kaggle/input/datasets`，其中影像、掩膜和代码按子目录展开；不能假设挂载目录名等于数据集 slug，也不能假设代码一定以 ZIP 形式存在。启动脚本会在挂载目录中定位恰有 663 对影像/掩膜的目录，只暂存一张支持图和两张查询图，然后调用本实验脚本。

本次使用 Tesla T4、官方 SAT-493M 权重（SHA-256 前缀 `eadcf0ff`），支持图为 `PV08_331824_1195530.bmp`，查询图为 `PV08_332270_1196670.bmp` 和 `PV08_333154_1195787.bmp`。两张查询图各自的 mIoU / Dice 为 `0.6533 / 0.6195` 和 `0.9484 / 0.9516`；合并像素计算的前景 IoU / mIoU / Dice 为 `0.5927 / 0.7583 / 0.7443`。预测掩膜下载后与本地标签重新计算的混淆矩阵计数一致。

这只是检查挂载、解包/筛选、权重加载、GPU 推理和指标保存的两图冒烟测试。样本太少，且没有区域级独立测试划分，不能据此判断模型的跨区域或跨时间泛化能力。完整输出保存在本地 `codex运行文件/kaggle_run_output/version5/`，并由 `.gitignore` 排除。
