# 固定版：空间、语义与等权融合 CAM

推理固定使用第三阶段 `seed_42/fusion_minmax_beta07_real` 的第 18 轮检查点。CAM 融合已改为固定等权；检查点中旧的融合标量只为兼容权重文件而保留，在等权推理中冻结且不参与计算。Experiment A 的 CAM-only runner 会加载原检查点中的分割头权重以保持结构兼容，但跳过分割头计算和评估。原训练使用的融合权重约为 0.66/0.34，因此这里的等权结果属于推理期对照。冻结的 DINOv3-SAT 与 GeoRSCLIP 编码器权重需另行挂载。

计算流程：DINOv3-SAT 第 24 层冻结特征经共享投影 `phi` 后，支持图前景特征平均形成原型。查询特征与原型的逐 patch 余弦相似经逐图 min-max 和空间修正器得到 `spatial_cam`。固定提示 `a satellite image of ground-mounted solar panel arrays` 的 GeoRSCLIP 文本向量经可训练 `g` 和独立 `psi` 投影，再与查询特征逐 patch 余弦相似，经逐图 min-max 和语义修正器得到 `semantic_cam`。现在使用 `fused_cam = 0.5 * spatial_cam + 0.5 * semantic_cam`。Experiment A 等权 CAM 评估直接对上采样后的 `fused_cam` 按阈值 0.5 生成预测，不调用分割头；分割头权重仍保留在 checkpoint 中以维持模型结构兼容。此处的语义 CAM 是 patch 级文本方向相似度，区别于早期 GeoRSCLIP 滑窗图；也不是完整 SSPNet 复现。

`version.json` 锁定检查点 SHA-256、源码哈希、输入契约和模型身份。`checkpoints/best.pt` 是 13.2 MB 的可训练模块权重，不包含冻结编码器。源码分别位于相邻的 `dmsf_v1/dmsf.py`、`dmsf_v1/feature_cache.py`、`dmsf_stage3/stage3_model.py`，以及文本编码器加载器 `georsclip_semantic_cam_e1/run_e1.py`。训练入口 `dmsf_stage3/run_stage3.py` 和固定协议、数据清单也在本分支。

已有冻结特征的推理：

```powershell
python "codex运行文件/dmsf_fusion_v1/infer.py" --features-npz "PATH/inputs.npz" --output "PATH/prediction.npz"
```

输入 `.npz` 需要 `support`、`query`、`mask`、`text` 四个 float32 数组，形状依次为 `[1,1024,64,64]`、`[1,1024,64,64]`、`[1,1,64,64]`、`[1,512]`。`mask` 是非空 0/1 支持前景，`text` 必须是上述固定提示的归一化 GeoRSCLIP 向量。此路径仅检查张量格式，调用者负责保证冻结特征和文本的来源。仓库本地已有的验证探针可用 `codex运行文件/dmsf_stage2/kaggle_output/version1/dmsf_stage2/verification_probe/inputs.npz` 测试，但它是被 Git 忽略的实验输出。

原图推理：

```powershell
python "codex运行文件/dmsf_fusion_v1/infer.py" --support-image "PATH/support.bmp" --support-mask "PATH/support_label.bmp" --query-image "PATH/query.bmp" --dino-repo "PATH/dinov3" --dino-weights "PATH/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth" --geo-weights "PATH/RS5M_ViT-B-32.pt" --output "PATH/prediction.npz"
```

原图可以是任意宽高的 RGB/RGBA 影像，支持标签尺寸必须与支持图相同，掩膜按“0=背景、非零=前景”读取。推理会将每张图等比例缩放到 1024x1024 画布，短边两侧用 SAT 均值填充，再使用四个独立 512px crop 拼成 64x64 patch 网格；输出 CAM、logits 和二值掩膜会裁掉填充并恢复到查询图原始宽高。超宽或超高影像的有效区域较小，目标很小时可能损失细节，需要滑窗/多尺度策略时应单独扩展推理协议。DINO 源码需位于固定提交 `6876159a11b4df116f30f667f8c9888617df0751`。GeoRSCLIP 权重加载时校验 SHA-256，若给定路径不存在，原加载器会尝试下载。需要 PyTorch、NumPy、Pillow、open_clip_torch 及 DINOv3 上游依赖；CUDA 推荐用于原图编码。

输出 `.npz` 在特征输入模式下包含 64x64 的 CAM 和 1024x1024 的 logits；在原图模式下 `spatial_cam`、`semantic_cam`、`fused_cam`、`logits_full`、`mask_full` 都是查询图原始尺寸，另外保留 `logits_patch` 和去填充前的 `logits_square`。旁边的 `.json` 记录版本、letterbox 几何和阈值。CAM 是模型内部响应图，不等同于独立阈值分割结果。运行 `python -m unittest discover -s "codex运行文件/dmsf_fusion_v1" -v` 检查固定权重和导出关系；本地验证探针缺失时相应测试跳过。第三阶段的完整评估与局限见 `dmsf_stage3/STAGE3_REPORT.md`。

Kaggle Notebook 的输入、GPU、依赖和运行单元格见 [`kaggle/README.md`](kaggle/README.md)，入口脚本为 [`kaggle/run-kaggle-inference.py`](kaggle/run-kaggle-inference.py)。
