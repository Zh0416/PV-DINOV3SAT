# DINOv3-SAT Patch-Token Local Frequency CAM

This bundle evaluates a fixed, support-guided local frequency prior over frozen DINOv3-SAT patch tokens. It adds no trainable frequency module. PV is the only foreground class; every mask-zero pixel is one shared background class.

## Encoder Contract

The runner imports the branch's existing `dmsf_v1/feature_cache.py` implementation. It uses official DINOv3-SAT ViT-L/16 block 24 (zero-based block index 23), final LayerNorm output before extra L2 normalization, the existing SAT mean/std, and four independent 512x512 crops from each 1024x1024 RGB tile. The four 32x32 outputs are stitched into a 64x64 grid. `get_intermediate_layers(..., reshape=True)` supplies only spatial patch tokens; CLS and register tokens are not included in the grid. Crop attention/RoPE coordinates reset independently, matching the existing branch contract.

## Frequency CAM

For each token center and each 4, 6, or 8 token square neighborhood, the implementation removes each channel's local spatial mean, applies a separable Hann window, then computes `fft2` over the neighborhood's two spatial axes. It never flattens the token map into a sequence or transforms along the channel axis. Per-channel power is normalized across local frequency bins and converted to log power. The descriptor records the channel mean and population standard deviation at every local frequency bin, then L2-normalizes the result.

The support mask is averaged onto the 64x64 token grid and onto each local window. Windows with at least 0.60 PV coverage form the positive prototype; windows with at most 0.05 coverage form a unified background prototype; mixed windows are ignored. A query token's continuous frequency score is the cosine similarity to the positive prototype minus its similarity to the background prototype. The default does not mine hard negatives; `--hard-negatives K` enables a deterministic support-only top-K option.

The DINO spatial baseline uses the same frozen patch tokens and the existing Experiment A foreground/background prototype rule. Its CAM is recomputed during this run. It does not depend on a pre-existing CAM artifact in the notebook. The frequency CAM averages the three fixed scales. Fusion is an equal-weight average of the DINO spatial and frequency probabilities. Every map uses `sigmoid(5 * score)` and a fixed threshold of 0.5; query masks do not tune the threshold.

## Data And Evaluation

The Kaggle dataset `zh0416/pv-dinov3sat-experiment-a` contains 21 paired 1024x1024 BMP tiles and masks: one fixed support image and 20 fixed query images in `spatial_cam_code/benchmark_v1.json`. Image/mask pairs are matched by basename and `_label.bmp`. The pre-existing manifest uses a whole-image split and excludes queries within 1000 filename coordinate units of the support; those coordinate units have not been independently established as physical distances. The benchmark is a development set already used in prior experiments, so all query-mask metrics are reported as development metrics, not independent-test results.

`summary.json` includes micro and per-image IoU, Dice, precision, recall, confusion counts, frozen encoder provenance, split manifest with file hashes, and class-window counts. `per_image_metrics.csv` supports image-by-image review. `support_spectral_profiles.json` records the support PV/background log-power summaries. `fusion_cam/` contains one full-resolution float32 probability map per query as `.npy` in `[0, 1]`, plus a color heatmap PNG and an RGB overlay PNG for direct inspection. `diagnostics/` contains per-query RGB, mask, spatial CAM, FFT CAM, fused CAM, and error panels (green=TP, blue=FN, red=FP). Bare soil and cropland have no separate labels in this dataset, so their individual error rates cannot be measured; all non-PV pixels remain the background class.

## Local Checks

```bash
python -m unittest discover -s "codex运行文件/dinov3_patch_token_frequency_cam" -p "test_frequency_cam.py" -v
```

## Kaggle Run

The notebook needs the Experiment A Dataset and the existing private encoder-checkpoint Dataset attached, plus GPU and Internet enabled for the pinned DINOv3 source checkout. Clone this repository branch into `/kaggle/working`, then run only the new entry point:

```python
%run /kaggle/working/PV-DINOV3SAT/codex运行文件/dinov3_patch_token_frequency_cam/run_experiment.py --dataset-root /kaggle/input/datasets/zh0416/pv-dinov3sat-experiment-a --output-dir /kaggle/working/dinov3_patch_token_frequency_cam --device auto
```

The runner searches the attached inputs for the exact official SAT-493M checkpoint filename and validates it using the existing loader. Results stay under `/kaggle/working/dinov3_patch_token_frequency_cam`.
