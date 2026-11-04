# PV-DINOV3SAT

Code for few-shot photovoltaic segmentation experiments using DINOv3-SAT. Images and masks stay in the local `datasets/` directory and are excluded from Git.

The first experiment is in [`codex运行文件/experiment_a/README.md`](codex运行文件/experiment_a/README.md). It freezes the official DINOv3 ViT-L/16 SAT-493M encoder and matches foreground/background prototypes from one labeled support image.

The official DINOv3 source checkout is kept under `codex运行文件/dinov3/` locally and is excluded from this repository. The SAT checkpoint is licensed and must be obtained separately from Meta.
