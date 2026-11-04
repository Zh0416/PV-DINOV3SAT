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
python "codex运行文件/experiment_a/run_experiment_a.py" --support PV08_312621_1197085.bmp --weights /path/to/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth --device cuda
```

默认使用 512×512 分块、ViT-L/16 最后四层特征、前景/背景加权原型，分数为 `cosine(fg) - cosine(bg)`，阈值为 0。每次运行会在 `codex运行文件/experiment_a/runs/<时间戳>/` 下保存 `summary.json`、逐图指标 CSV 和预测掩膜。

快速检查单张查询图时可加 `--limit-query 1 --no-save-predictions`。正式报告至少同时记录支持影像、空间缓冲距离、mIoU、Dice、精确率和召回率；阈值若需要调整，只能用验证区域选择，不能根据测试标签选择。

## 实验边界

A 是冻结特征的直接匹配诊断，不进行光伏数据训练。这个目录当前没有明确的 train/validation/test 区域清单；因此，当前默认的全数据查询结果是初步诊断，不能单独作为无空间泄漏的正式泛化结果。应根据真实区域/日期信息建立独立划分后再报告跨区域结果。
