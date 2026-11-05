# DINOv3-SAT PV Spatial CAM v1

This is a frozen, single-layer, support-prototype CAM implementation inspired by the spatial branch of SSPNet. It is not a full SSPNet reproduction. The learned projection and residual refinement modules from SSPNet are deliberately omitted; this run directly uses the official pretrained DINOv3-SAT features without training.

## Fixed protocol

- Read `experiment_a/benchmark_v1.json` and only its 21 image/mask pairs.
- Support: `PV08_331824_1195530.bmp`; the other 20 manifest images are queries.
- Use the frozen/eval DINOv3 ViT-L/16 SAT-493M checkpoint. Request only block index 23 (the 24th block), normalized by the model's final LayerNorm and then L2-normalize each patch token.
- Normalize RGB inputs with SAT mean `(0.430, 0.411, 0.296)` and standard deviation `(0.213, 0.156, 0.143)`.
- Encode each 1024x1024 image as four non-overlapping 512x512 crops. Each crop gives a 32x32 patch grid; stitch top-left/top-right and bottom-left/bottom-right by their image positions into 64x64.
- The support mask uses pixel values 0 (background) and 12 (PV). Average-pool its binary PV pixels over 16x16 cells, use the resulting PV fractions to weight support tokens, and L2-normalize the weighted sum to form `p_fg`.
- The primary response is `S_fg(u,v) = cosine(F_query(u,v), p_fg)`. Min-max normalize each 64x64 response independently, then bilinearly upsample it to 1024x1024.
- Query masks are not opened by this script and do not affect the prototype, CAM, or any parameter choice.

The per-image min-max CAM shows relative response within that image. It is not a PV probability, and a bright region alone does not establish that the image contains PV. Use the saved raw cosine scores for cross-image interpretation. The raw cosine color map uses the same fixed `[-1, 1]` range for every query; the normalized CAM uses `[0, 1]`.

## Run on Kaggle

The private Kaggle runner mounts the fixed dataset and official model, checks the checkpoint SHA-256 prefix, clones the official DINOv3 repository, and calls this script. To run locally, install the DINOv3 dependencies and pass their paths:

```powershell
python "codex运行文件/spatial_cam_v1/generate_spatial_cam.py" `
  --data-root "codex运行文件/experiment_a/benchmark_v1_data" `
  --manifest "codex运行文件/experiment_a/benchmark_v1.json" `
  --dinov3-repo "codex运行文件/dinov3" `
  --weights "C:/path/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth" `
  --device cuda
```

## Outputs

`benchmark_v1_results/summary.json` and `metadata.json` record the fixed inputs, checkpoint, layer, preprocessing, tiling, prototype, formulas, device, and raw-score ranges. For each query, `images/<stem>/scores.npz` stores float32 patch-grid and full-resolution `S_fg` and `A_fg` arrays. The same directory contains grayscale and color CAMs, a color overlay, and a three-panel comparison of the original image, raw cosine score, and normalized CAM overlay. `overview.jpg` and `index.html` provide a 20-query index.

After inference, run `python "codex运行文件/spatial_cam_v1/inspect_tile_seams.py"` to write `tile_seam_diagnostics.json`. It compares average adjacent-patch response changes across the vertical and horizontal center joins with within-tile changes. A larger ratio can indicate a visible join, but can also reflect actual scene content at that location, so review the corresponding `panel.jpg` as well.

## Benchmark v1 run

Completed on the private Kaggle GPU Notebook `zh0416/pv-dinov3sat-spatial-cam-v1` with a Tesla T4 and the official checkpoint (SHA-256 prefix `eadcf0ff`). All 20 manifest queries were processed. Across the 64x64 raw patch score maps, per-image cosine ranges were between `0.0130` and `0.9298`. This is a raw score range only; the separately saved min-max maps remain relative responses, not calibrated probabilities.

All 20 outputs passed checks for manifest coverage, finite float32 arrays, 64x64 patch grids, 1024x1024 upsampled maps and images, and normalized CAM values within `[0,1]`. The query masks were not opened by inference. The support mask values were verified as `0` background and `12` PV.

Visual review of the overview and full-resolution panels found no consistent cross-image seam pattern, though some center joins are visible. In the raw cosine grid, median mean absolute response change at the vertical and horizontal center joins was respectively `1.27x` and `1.22x` the within-tile adjacent-patch mean. `PV08_326513_1201232.bmp` had the largest vertical ratio (`2.73x`); its panel should be kept in view when interpreting results. These ratios also include real scene changes at the center lines. No seam correction or smoothing was applied. The per-image details are in `benchmark_v1_results/tile_seam_diagnostics.json`.

The full-resolution values, previews, overview, metadata, summary and seam diagnostics are stored locally in `benchmark_v1_results/`. The resulting CAMs are continuous spatial responses only; no binary mask or segmentation output was produced.

No segmentation mask, threshold, decoder, semantic branch, or training step is produced by this experiment.
