# S0：PV-512 空间原型基线

本分支的 **185 张合并集随训练逐轮验证**流程和 Kaggle 命令见 [INTEGRATED_VALIDATION.md](INTEGRATED_VALIDATION.md)。下文保留原始 S0 v1 训练协议，供比较与审计；新的集成验证运行使用 v2 协议和独立输出目录。

用于用户随后在 Kaggle 单张 T4 上运行。本次只完成代码、CPU 测试和真实数据的 prepare 校验，未运行 DINO 编码、CUDA FP16 或正式训练。无 GeoRSCLIP、RGB 解码支路、局部交叉注意力、Mamba 或频域模块。

## 已锁定数据

真实 PV-512 manifest 的列为 `sample_id,sample_type,source_scene,image,label,row_off,col_off,height,width,label_pixels,image_valid_fraction,crs`。源码读取真实 `source_scene`，不从 tile ID 猜测场景。

- support 固定为 `0807_Q_934ae716b5714f_r00024_c00076`，占 1200 张正样本预算中的一张，不作为 query。
- 其余选 1199 张正 query、50 张困难负 query，每轮每张 query 使用一次，共 1249 个 episode。
- 从 manifest 列出的正/困难负样本中，分别按 `SHA256("s0-pv512-v1|42|<positive|hard_negative>|<id>")`、其次按 ID 排序。正样本排除 support 后取前 1199；困难负样本取前 50。用户没有预先给定这些 query ID；本规则已在真实数据上执行并锁定。
- `protocol/s0_selection.json` 保存完整 ID、真实场景、相对路径、每份影像/标签 SHA-256、源 manifest/summary 摘要、选择规则和清单哈希。`protocol/s0_selection_ids.csv` 是便于查看的表格版本。
- 清单哈希：`f591c05f62cf4c60e31bd4d19a97284d48323df36059bd5afb915430530c2062`。
- `protocol/data_audit.json` 记录实际通过校验的 1250 对样本、16 个源场景，support 有 55762 个前景像素。源数据统计为 1277 正样本和 2625 困难负样本。入选影像全部为 512×512 uint8 RGB，标签值全部为 0/1，50 张困难负 query 全空。

每次 prepare、训练和恢复都重新计算清单与源文件摘要，并与已提交的参考清单及输出目录中的锁文件比较；不匹配即停止。普通负样本不参与 S0。影像和编码器权重不提交 Git。

## 模型与训练策略

冻结官方 DINOv3-SAT ViT-L/16 第 24 块，用 SAT mean/std 对整张 512×512 输入归一化，取 final-LN 后、额外 L2 归一化前的 `1024×32×32` token。可缓存为 FP16，训练读取后在 CUDA autocast 中使用。DINO eval、不计算梯度、不进入优化器。只接受官方 `dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth`，校验完整权重摘要及官方哈希前缀。DINO 源码固定为 `6876159a11b4df116f30f667f8c9888617df0751`。

共享可训练投影为 `Conv1x1(1024,256)+GroupNorm(32)+ReLU`。support 的 0/1 掩膜经 16×16 平均池化成为 token 前景占比，前景占比及其补集分别加权汇聚前景、背景原型。query 与两原型的余弦相似度差为原始 `1×32×32` CAM，范围约为 [-2,2]，每次前向都随可训练投影重新计算，不做逐图 min-max，不读旧灰度 CAM。

共用 S0–S3 解码器：拼接 256 通道 query 投影和 1 通道原始 CAM → `Conv3x3(257,64)+GroupNorm+ReLU` → 两个 64 通道残差块 → 四次 `bilinear×2(align_corners=False)+Conv3x3+GroupNorm+ReLU`，输出通道依次 64、32、16、8 → `Conv1x1(8,1)`，得到 512×512 logits。

总上限 20 轮，包含以下两个阶段：

1. 第 1–3 轮 **warmup**：只训练视觉投影及由其实时计算的空间 CAM；解码器冻结并跳过。正、负 query 均监督原始 CAM BCE，正像素权重 2，固定温度 1.0；不添加 Dice。日志中的预测指标来自 CAM 零阈值，只作热身诊断。
2. 第 4–20 轮 **joint**：联合训练视觉投影与解码器。正 query 用最终分割 `weighted BCE(pos_weight=2)+Dice+0.1×CAM BCE`；困难负 query 用最终分割 BCE 加 `0.1×CAM BCE`，不计算 Dice。两类 query 的 CAM 辅助监督均使用上采样后的原始相似度差、固定温度 1.0；解码器输入始终为未经温度变换的原始 CAM。

固定 seed=42；在编码/缓存全部准备完成后、构造训练模型前再次设种子，保证冷/热缓存不改变训练初始化。每轮按 `42+epoch` 固定打乱 query。batch=1、梯度累积=4、CUDA FP16 GradScaler、AdamW lr=`1e-4`、weight decay=`1e-4`、梯度裁剪=1。1249 个 episode 对应每轮 313 组更新，最后一组只有 1 个 episode，按实际组大小归一化。AMP 溢出跳过与累计成功更新数分别记录。

运行前以 **joint 阶段**做一次完整 forward/backward/AdamW 显存预检，包含优化器状态分配，之后恢复模型、优化器、scaler、随机状态和可训练标记。每个训练阶段首次完成 **100 次成功 optimizer update** 后记录计时与峰值显存；正常对应约 **400 个 episode**，日志同时保存实际 episode 数。计时不包括编码器缓存生成，轮时估计也不包括验证。

## 外部验证与选模

验证/测试由用户另外提供；不从训练清单抽取数据冒充独立验证，不把 PV-896 标称为独立外测。无外部验证时可 smoke 或训练，保存 `last.pt` 和每轮 `epoch_NN.pt`，不产生 `best.pt`。

外部验证 CSV 必须有 `id,category,source_scene,image,mask` 列。category 为 `positive` 或 `empty`，两类都必须存在；路径相对 `--validation-root`，默认相对 CSV 所在目录；影像/标签为 512×512，标签为 0/1。验证样本 ID、影像内容摘要或 source_scene 与选定训练/support 重叠会被拒绝。空标签文件内容一致是正常现象，不用于重叠判定。场景名比较统一斜杠、取路径末段并忽略大小写；应沿用训练 manifest 的真实场景命名，不能重新起名绕过检查。这不代替来源、时间和真实地理范围的核验。

分割阈值锁定 logits ≥ 0。预测与 GT 同为空时 Dice/IoU=1；GT 空但有误报时 Dice/IoU=0。输出 Dice、IoU、precision、recall、TP/FP/FN/TN，以及全空图像误报率、误报像素数/比例、逐场景指标。

**选模 score = 正样本场景宏 Dice − 全空图场景宏误报像素率**。先在每个源场景内分别平均正 query Dice、空 query 误报像素率，再对相应场景等权平均。完全相同分数保留更早轮次。只在第 4 轮起验证选模，并在第 4 轮初始化/重置最佳分数与 patience；联合阶段连续 5 轮无提升才早停。全空图像“至少一个误报像素”的比率仍报告，但不用于选模。

## Kaggle 操作

挂载 PV-512 和包含官方 DINOv3-SAT 权重的输入。入口会扫描 `/kaggle/input`，通过 `manifest.csv`、类别目录和完整权重文件名确定路径；如出现多个候选，用 `--data-root`、`--dino-weights` 显式选择。无需硬编码截图中被截断的 slug。

Notebook 中先取得新分支并安装轻量依赖（保留 Kaggle 自带 CUDA PyTorch，要求 ≥2.4）：

```python
import pathlib, subprocess, sys
repo = pathlib.Path('/kaggle/working/PV-DINOV3SAT')
subprocess.run(['git', 'clone', '--single-branch', '--branch', 'codex/s0-pv512-baseline',
                'https://github.com/Zh0416/PV-DINOV3SAT.git', str(repo)], check=True)
code = repo / 'codex运行文件' / 's0_baseline'
subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', str(code / 'requirements.txt')], check=True)
```

先校验精确清单，再运行 8-episode smoke；两者与正式训练的输出分开：

```python
subprocess.run([sys.executable, str(code / 'train_s0.py'), '--mode', 'prepare',
                '--output', '/kaggle/working/s0_prepare'], check=True)
subprocess.run([sys.executable, str(code / 'train_s0.py'), '--mode', 'smoke', '--smoke-steps', '8',
                '--output', '/kaggle/working/s0_smoke'], check=True)
```

Smoke 用少量正/负 query 检查联合解码器训练，输出 `smoke_results.json`，不导出正式/最佳模型。它不等于完整的一轮训练。首次编码时可联网 clone 固定版本 DINO；离线时用 `--dino-repo` 指向已处于上述提交的干净源码目录，代码不会改写用户提供的其他版本。

用户准备好后自行启动正式训练：

```python
subprocess.run([sys.executable, str(code / 'train_s0.py'), '--mode', 'train', '--epochs', '20',
                '--output', '/kaggle/working/s0_train'], check=True)
```

有外部验证时在**首次正式训练**就加 `--validation-manifest <CSV绝对路径> --validation-root <验证根目录>`；同一次运行不能中途更换/增加验证清单。如果原始 manifest 的真实场景列名不同，可通过 `--source-scene-column` 明确指定，不能留空或从 tile ID 推断。

## 跨 Kaggle 会话恢复

每轮保存 `last.pt` 和 `epoch_NN.pt`，包含模型、优化器、scaler、完成轮次、阶段、成功/跳过更新数、RNG、最佳轮次、patience、清单/验证/协议/源码/编码器摘要。初始 `last.pt` 为 epoch 0；轮中中断时从最近完整轮重新开始。先保存并保留上一 Notebook 的输出，再在新会话把旧输出挂载为只读输入。必须复制到可写目录后恢复：

```python
import shutil
candidates = [p.parent for p in pathlib.Path('/kaggle/input').rglob('last.pt')
              if (p.parent / 's0_selection.json').is_file()]
assert len(candidates) == 1, candidates  # 多候选时手动选定正确旧训练目录
old_run = candidates[0]
new_run = pathlib.Path('/kaggle/working/s0_resumed')
assert not new_run.exists()
shutil.copytree(old_run, new_run, ignore=shutil.ignore_patterns('dinov3'))
subprocess.run([sys.executable, str(code / 'train_s0.py'), '--mode', 'train', '--epochs', '20',
                '--output', str(new_run)], check=True)
```

同时重新挂载原 PV-512 和编码器权重；原运行有验证时必须再提供完全相同的验证清单和数据（根路径可变化）。使用相同代码版本和 PyTorch 版本，勿用新代码覆盖旧运行。复制的 token 缓存会校验摘要并复用；缺少缓存条目时才编码补齐。若只备份了 `epoch_NN.pt`，需将选定快照放为新输出目录的 `last.pt`，并保留原清单/协议/验证锁文件。

## 打包与本地测试

```powershell
python -m unittest discover -s "codex运行文件/s0_baseline" -p "test_s0.py" -v
python "codex运行文件/s0_baseline/build_kaggle_bundle.py"
```

`kaggle_bundle/run-s0-baseline.py` 含源码和已锁定的精确清单，支持相同 CLI 参数；构建器不上传或启动 Kaggle。元数据的 dataset/model sources 留空，由用户附加真实输入。源文件及生成脚本的摘要见 `source_manifest.json`。测试覆盖真实 S0 形状/梯度、正负损失、热身、场景计分/隔离、断点恢复与下一步一致性、冷/热缓存初始化、预检回滚、联合阶段早停和打包入口。实际 T4 显存、FP16 数值稳定性和 20 轮训练性能尚未实测。
