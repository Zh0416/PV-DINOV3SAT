# S0 训练时验证：PV-512-MERGED-185

## 固定规则

模型和训练样本沿用 S0：固定 support `0807_Q_934ae716b5714f_r00024_c00076`，1199 张正 query、50 张困难负 query；冻结 DINOv3-SAT，训练共享视觉投影、空间 CAM 和统一解码器。前 3 轮仅训练投影/CAM；第 4 轮起联合训练，最多 20 轮。batch 1、梯度累积 4、CUDA FP16、AdamW `1e-4`、梯度裁剪 1。训练集才计算损失和反向传播。

已上传的 `PV-512-MERGED-185` 中，`images/` 的 **185 张全是验证图**，`labels/` 为同名 0/255 掩膜；代码在内存中转为 0/1。首次运行核对 `manifest.csv`、185 对文件 SHA-256、原始场景、前景像素数以及仓库中的参考摘要。文件变动或续训时换验证集会被拒绝。

从第 4 轮起，每轮对全部 185 张执行固定 support 的前向推理，阈值固定为 logits ≥ 0。保存逐图 TP/FP/FN/TN、Dice、IoU、precision、recall，以及按来源、类别和场景聚合的指标：

- `PV-512TEST`：108 张、3 个新原始场景（48 正样本、60 困难正样本）。**仅这 108 张的场景宏 Dice 用于选择 `best.pt`**；每个场景先平均图像 Dice，再对 3 个场景等权平均。
- `PV-512-unused`：77 张未参与训练的正样本，与训练图共享原始场景。它们也全部验证并报告，但不影响选模分数。

同分保留更早轮次；第 4 轮起连续 5 轮无提升则早停。两组均含光伏，不能据此评估整张无光伏图的误报。新场景 108 张用于选模后不再是独立最终测试集。本分支运行协议为 `s0-pv512-training-v2`，必须用新的输出目录开始；旧 v1 训练断点不能直接续入本协议。

## Kaggle 拉取和运行

Notebook 已挂载原 `PV-512`、`PV-512-MERGED-185` 和官方 DINOv3-SAT 权重。开启 GPU；首次克隆固定版本 DINO 源码需开启 Internet，也可用 `--dino-repo` 提供已准备的源码。旧频域实验单元不要用于 S0 训练。

```python
from pathlib import Path
import json
import subprocess
import sys

repo = Path('/kaggle/working/PV-DINOV3SAT-S0-v2')
subprocess.run([
    'git', 'clone', '--single-branch', '--branch', 'codex/s0-integrated-validation',
    'https://github.com/Zh0416/PV-DINOV3SAT.git', str(repo),
], check=True)
code = repo / 'codex运行文件' / 's0_baseline'
subprocess.run([sys.executable, '-m', 'pip', 'install', '-r',
                str(code / 'requirements.txt')], check=True)

candidates = [p.parent for p in Path('/kaggle/input').rglob('dataset_summary.json')
              if json.loads(p.read_text(encoding='utf-8')).get('sample_count') == 185
              and (p.parent / 'images').is_dir() and (p.parent / 'labels').is_dir()]
assert len(candidates) == 1, candidates
merged_root = candidates[0]
print('Validation:', merged_root)
```

先运行只读数据检查；`prepare` 不编码 DINO，也不训练：

```python
subprocess.run([sys.executable, str(code / 'train_s0.py'), '--mode', 'prepare',
                '--merged-validation-root', str(merged_root),
                '--output', '/kaggle/working/s0_prepare_v2'], check=True)
```

再运行 8 个 episode 的训练冒烟测试，包含正 query 和困难负 query；结果只作诊断，不产生正式 checkpoint：

```python
subprocess.run([sys.executable, str(code / 'train_s0.py'), '--mode', 'smoke',
                '--smoke-steps', '8', '--merged-validation-root', str(merged_root),
                '--output', '/kaggle/working/s0_smoke_v2'], check=True)
```

正式训练从头开始，验证与每轮 checkpoint 同步保存：

```python
subprocess.run([sys.executable, str(code / 'train_s0.py'), '--mode', 'train',
                '--epochs', '20', '--merged-validation-root', str(merged_root),
                '--output', '/kaggle/working/s0_train_v2'], check=True)
```

每轮 `validation_epoch_NN.json` 包含 185 张逐图结果和分组统计；`best.pt` 是新场景宏 Dice 最佳模型，`last.pt` 和 `epoch_NN.pt` 可恢复训练。输出需通过 Kaggle 保存版本保留。跨会话恢复时，将上一运行的完整输出目录从只读 `/kaggle/input` 复制到新的 `/kaggle/working` 目录，重新挂载同版本三份输入，并以相同 `--merged-validation-root` 和新可写输出路径运行上面的 `train` 命令；源码和 PyTorch 版本保持一致。

本地验证：

```powershell
python -m unittest discover -s "codex运行文件/s0_baseline" -p "test_s0*.py" -v
python "codex运行文件/s0_baseline/build_kaggle_bundle.py"
```

本地只有 CPU 版 PyTorch；T4 显存、FP16 数值和实际训练耗时需要由 Kaggle 的 smoke 与前 100 次成功优化器更新记录确认。
