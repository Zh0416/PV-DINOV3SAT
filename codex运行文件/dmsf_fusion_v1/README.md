# 固定版：空间、语义与融合 CAM

这个分支固定第三阶段 `seed_42/fusion_minmax_beta07_real` 的第 18 轮检查点。它有已训练的空间 CAM、真实 GeoRSCLIP 文本生成的语义 CAM，以及一张送入分割头的融合 CAM。选择此版是为了固定原定 `beta_init=0.7` 的真实文本实现，不以是否优于空间单路作为条件。检查点包含可训练投影、CAM 修正器、融合标量和分割头；冻结 DINOv3-SAT 与 GeoRSCLIP 编码器权重需另行获取。

计算流程：DINOv3-SAT 第 24 层冻结特征经共享投影 `phi` 后，支持图前景特征平均形成原型。查询特征与原型的逐 patch 余弦相似经逐图 min-max 和空间修正器得到 `spatial_cam`。固定提示 `a satellite image of ground-mounted solar panel arrays` 的 GeoRSCLIP 文本向量经可训练 `g` 和独立 `psi` 投影，再与查询特征逐 patch 余弦相似，经逐图 min-max 和语义修正器得到 `semantic_cam`。`fused_cam = sigmoid(beta) * spatial_cam + (1-sigmoid(beta)) * semantic_cam`，其中 beta 是已训练标量，不再是初值 0.7。分割头输入仅为 `[phi(Fq), fused_cam]`，阈值为双线性上采样后的 logits `>= 0`。此处的语义 CAM 是 patch 级文本方向相似度，区别于早期 GeoRSCLIP 滑窗图；也不是完整 SSPNet 复现。

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

原图必须为 1024x1024 RGB，支持标签为 1024x1024 灰度 0/12。DINO 源码需位于固定提交 `6876159a11b4df116f30f667f8c9888617df0751`；编码使用四个独立 512px crop 拼成 64x64 patch 网格。GeoRSCLIP 权重加载时校验 SHA-256，若给定路径不存在，原加载器会尝试下载。需要 PyTorch、NumPy、Pillow、open_clip_torch 及 DINOv3 上游依赖；CUDA 推荐用于原图编码。

输出 `.npz` 包含 64x64 的 `spatial_cam`、`semantic_cam`、`fused_cam`、`logits_patch`，1024x1024 的 `logits_full` 和 0/1 `mask_full`，以及标量 `spatial_weight`；旁边的 `.json` 记录版本与阈值。CAM 是模型内部响应图，不等同于独立阈值分割结果。运行 `python -m unittest discover -s "codex运行文件/dmsf_fusion_v1" -v` 检查固定权重和导出关系；本地验证探针缺失时相应测试跳过。第三阶段的完整评估与局限见 `dmsf_stage3/STAGE3_REPORT.md`。
