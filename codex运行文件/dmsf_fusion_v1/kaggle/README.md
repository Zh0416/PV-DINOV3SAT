# Kaggle 本地推理 Notebook 配置

这个目录用于在 Kaggle GPU Notebook 中运行固定的 `dmsf-fusion-v1`。它只做推理，不训练。输入影像可以是任意宽高，支持图和支持掩膜只需要尺寸相同；查询图可以与支持图尺寸不同。

## 1. 准备输入 Dataset

在 Kaggle 创建一个私有 Dataset，例如 `my-selected-images`，放入：

```text
support.png          # 支持影像，任意宽高，RGB/RGBA 均可
support_mask.png     # 支持掩膜，尺寸必须与 support.png 相同，非零像素视为前景
query.png            # 查询影像，任意宽高，RGB/RGBA 均可
```

掩膜可以使用 0/12、0/255 或任意“0=背景、非零=前景”的灰度编码。Notebook 还需要添加 DINOv3 SAT 模型数据集 `yadavdamodar/dinov3-vitl16-pretrain-sat493m`，并打开 Internet 和 GPU。

## 2. Notebook 单元格

如果代码分支已经推送到 GitHub，在第一个单元格运行：

```python
!git clone -b codex/dmsf-global-fusion-v1 https://github.com/Zh0416/PV-DINOV3SAT.git /kaggle/working/PV-DINOV3SAT
!pip install -q open_clip_torch==2.24.0
```

如果分支尚未推送，把本分支的代码目录作为 Kaggle Dataset 挂载，并把 `CODE_ROOT` 改为该目录。当前固定权重在 `codex运行文件/dmsf_fusion_v1/checkpoints/best.pt`，也可以单独作为 Dataset 挂载。

下面的单元格只需要修改三个输入文件名：

```python
import os

os.environ['DMSF_CODE_ROOT'] = '/kaggle/working/PV-DINOV3SAT'
os.environ['DMSF_SUPPORT_IMAGE'] = '/kaggle/input/my-selected-images/support.png'
os.environ['DMSF_SUPPORT_MASK'] = '/kaggle/input/my-selected-images/support_mask.png'
os.environ['DMSF_QUERY_IMAGE'] = '/kaggle/input/my-selected-images/query.png'
os.environ['DMSF_OUTPUT'] = '/kaggle/working/dmsf_prediction.npz'
os.environ['DMSF_DEVICE'] = 'cuda'
```

运行入口：

```python
%run /kaggle/working/PV-DINOV3SAT/codex运行文件/dmsf_fusion_v1/kaggle/run-kaggle-inference.py
```

如果脚本目录来自挂载的代码 Dataset，把上面的路径替换成挂载路径即可。脚本会自动 clone DINOv3 并切换到提交 `6876159a11b4df116f30f667f8c9888617df0751`，自动搜索模型 Dataset 中的 `dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth`，并在 `/kaggle/working` 下载和校验 GeoRSCLIP `RS5M_ViT-B-32.pt`。

## 3. 输出

输出文件为：

```text
/kaggle/working/dmsf_prediction.npz
/kaggle/working/dmsf_prediction.json
```

`.npz` 中的 `spatial_cam`、`semantic_cam`、`fused_cam`、`logits_full`、`mask_full` 都是查询图的原始尺寸；`logits_patch` 仍是模型的 `64×64` patch 输出，`logits_square` 是去除前的 `1024×1024` letterbox 画布输出。这样可以直接用原图尺寸显示或保存掩膜。

当前适配方式是等比例缩放到 1024×1024 画布，短边两侧用 DINO SAT 均值填充；推理后去掉填充并恢复到查询图原始宽高。它避免了拉伸变形，但超宽或超高影像的有效内容会占画布较小区域，目标很小时仍可能损失细节。需要更高分辨率时，应增加滑窗/多尺度推理，这属于下一版推理策略。

## 4. 直接用 Kaggle CLI 提交

把 `kernel-metadata.json` 中的 `dataset_sources` 改成你自己的输入 Dataset ID，然后在本地运行：

```powershell
kaggle kernels push -p "codex运行文件/dmsf_fusion_v1/kaggle"
```

Notebook 中的环境变量可改成脚本顶部的默认路径，或把 `run-kaggle-inference.py` 复制到 Kaggle Notebook 单元格执行。推荐先在 Notebook 中运行并确认输入路径，再使用 CLI 提交批量任务。

## 5. PV-512 指定样本

如果 `PV-512` 已作为 Kaggle Dataset 挂载，并且数据集内包含以下两张正样本影像：

```text
0807_Q_934ae716b5714f_r00024_c00069.tif  # support
0807_Q_934ae716b5714f_r00024_c00075.tif  # query
```

可以直接运行针对这对影像的入口。脚本会在 `PV-512` 下自动定位两张 TIFF 和 `c00069` 对应的标签文件（标签可以位于 `labels`/`mask`/`annotations` 目录，或使用 `_label`、`_mask` 等后缀）。

```python
import os
os.environ['DMSF_CODE_ROOT'] = '/kaggle/working/PV-DINOV3SAT'
os.environ['DMSF_PV512_ROOT'] = '/kaggle/input/pv-512'  # 按 Kaggle 右侧 Input 路径修改
os.environ['DMSF_OUTPUT_DIR'] = '/kaggle/working/pv512_cam_outputs'
os.environ['DMSF_DEVICE'] = 'cuda'
# 如果标签文件命名不符合自动搜索规则，可以显式指定：
# os.environ['DMSF_SUPPORT_MASK'] = '/kaggle/input/pv-512/.../c00069_label.tif'
%run /kaggle/working/PV-DINOV3SAT/codex运行文件/dmsf_fusion_v1/kaggle/run-pv512-cams.py
```

输出目录中包含：

```text
spatial_cam.png
semantic_cam.png
fused_cam.png
cam_panel.png
prediction.npz
prediction.json
```

脚本会自动将 TIFF 读取为 RGB，按比例放入 1024×1024 编码画布，推理后再恢复到查询图的原始尺寸；因此 512×512 输入会得到 512×512 CAM。`cam_panel.png` 使用同一个 0..1 色标显示三张 CAM，便于直接比较。
