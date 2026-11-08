# 全局标量融合受控实验

本目录实现第三阶段开发集实验。冻结 DINOv3-SAT ViT-L/16 第24层和 GeoRSCLIP RS5M ViT-B/32 文本编码器，仅使用固定提示 `a satellite image of ground-mounted solar panel arrays` 的文本向量。DINO 支持/查询特征共享视觉投影 `phi`；支持前景投影特征汇聚原型并与查询特征做余弦相似。文本经过三层 MLP `g` 和独立 `psi` 投影到 256 维，再与查询特征逐位置余弦相似。GeoRSCLIP 图像塔在严格加载后删除，不参与此实验。

受控分割头始终为 `Conv3x3(257,64) -> ReLU -> Conv1x1(64,1)`。空间对照输入 `[phi(Fq), A_spatial]`，融合输入 `[phi(Fq), A_fused]`，其中 `A_fused = sigmoid(beta) * A_spatial + (1-sigmoid(beta)) * A_semantic`。头部只见一张 CAM，无法绕开融合权重分别读取空间和语义 CAM。损失、AdamW、训练 episode 和每种子初始化与第二阶段保持一致；各组独立训练20轮、按验证查询组宏Dice选择最佳 checkpoint。

消融覆盖两种 CAM 尺度、两种初始空间权重以及真实/固定随机文本。`minmax` 使用 SSPNet DMSF 融合参考中的逐图 min-max；这不是完整 SSPNet 复现。`calibrated` 使用源训练 episode 的查询标注初始化每支 CAM 的源阈值和温度，训练中继续优化。目标验证时只用固定支持图及其掩膜做 0.25 收缩阈值更新。空间支持分数由 4×4 分块留出原型计算，各 patch 所在块不参与自己的原型；如果留出后无前景，该 patch 不参与支持阈值估计。语义支持分数来自同一冻结文本方向，无空间自原型问题。查询真值只在训练损失或验证评分时读取，不用于支持校准、阈值选择或模型前向。逐图 min-max 对无光伏查询可能把最高相对响应拉到近1，结果中保留空查询 CAM 峰值/高响应面积诊断。

`protocol.json` 固定三个种子 42/43/44。每种子两个空间对照和八个融合组，共30次训练。数据只使用原候选 train 260 张和 validation 111 张（16固定支持、95查询）；旧90张 test 已被第二阶段查看，本阶段访问控制保持其锁闭。文件名数值坐标分组不能证明地理/时间独立。另在训练结束前锁定独立来源 PV-896 的支持/查询划分、标签定义、区域、时间与近似GSD；外部测试仅用目标支持图与掩膜建原型和校准，不用查询标签调权重或阈值。实测结果与限制见 [报告](STAGE3_REPORT.md)。

运行：

```powershell
python -m unittest discover -s "codex运行文件/dmsf_stage3" -v
python "codex运行文件/dmsf_stage3/build_kaggle_bundle.py"
kaggle kernels push -p "codex运行文件/dmsf_stage3/kaggle_bundle" --accelerator NvidiaTeslaT4
```

Kaggle 私有作业：[PV DMSF Global Scalar Fusion](https://www.kaggle.com/code/zh0416/pv-dmsf-global-scalar-fusion) 和 [PV DMSF External PV896](https://www.kaggle.com/code/zh0416/pv-dmsf-external-pv896)，均已在 Tesla T4 完成。`run_stage3.py` 做真实编码、训练与验证，输出每轮验证、梯度、参数变化、检查点和源码哈希；`run_external.py` 原样评估全部30个已选检查点。生成包中的 `source_manifest.json` 固定源码与协议；历史第二阶段文件和检查点均不覆盖。

另有 `external_protocol.json`：在训练结束前依据 PV-896 源清单预锁 8 张支持、48 张查询。支持场景的范围全部在北纬24度以南，查询场景在北纬26度以北；支持用原始场景中有光伏标注的样本，查询样本只按 ID 哈希选取。清单记录原 GeoTIFF 文件名中的拍摄日期、EPSG:4326 范围、近似 GSD 和文件哈希。`run_external.py` 将原生896像素影像分为四块448像素输入冻结 DINO，输出56×56 patch 网格，只允许目标支持掩膜参与建原型和校准，查询标签仅用于锁定后评分。PV-896 来源声明未使用 PV08；但其部分 SHP 文件名带“不确定”，锁定样本中有7/8张支持和24/48张查询带该标记，标签质量限制必须随结果报告。
