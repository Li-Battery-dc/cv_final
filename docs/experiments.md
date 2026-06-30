# Experiment Plan and Run Audit

This document is the source of truth for final-report experiments. Use explicit
run directories only. Do not rely on `latest` symlinks; the current scripts print
the created run directory and record it in `run_config.json`.

## Current Usable Results

Baseline scene: `data/scene/images`, 64 frames uniformly sampled from
`data/raw/3_scene.mp4`. Validation uses every 8th frame as held-out view.

| ID | Reconstruction | Renderer / Trainer | Run | PSNR | SSIM | LPIPS | Status |
|---|---|---|---|---:|---:|---:|---|
| A | VGGT raw | official 3DGS | `data/scene/gs_official_raw/runs/20260627_135803_gaussian_official` | 20.729 | 0.754 | 0.247 | usable |
| B | custom BA | official 3DGS | `data/scene/gs_official_ba/runs/20260627_133737_gaussian_official` | 22.471 | 0.820 | 0.197 | main baseline |
| C | VGGT raw | official 3DGS random init | `data/scene/gs_official_random_raw/runs/20260627_141252_gaussian_official` | 20.317 | 0.742 | 0.264 | usable |
| D | custom BA | official 3DGS random init | `data/scene/gs_official_random_ba/runs/20260627_140637_gaussian_official` | 21.983 | 0.806 | 0.215 | usable |
| E | custom BA | custom 3DGS | `data/scene/gs_custom_ba` | 16.399 | 0.570 | - | engineering baseline |

BA effect on the same 64-frame scene:

| Stage | Run | Images | Points | Observations | RMSE px | Median px | P90 px | Runtime |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| VGGT raw | `data/scene/vggt_raw/runs/20260627_074913_vggt_export` | 64 | 9,760 | 116,725 | 3.816 | 2.801 | 6.423 | VGGT 19.7s + track 34.1s |
| custom BA | `data/scene/ba_custom/runs/20260627_075839_ba_custom` | 64 | 9,679 | 112,366 | 1.577 | 0.792 | 2.788 | 122.1s |

Human report results:

| Scene | Reconstruction / Renderer | Run | PSNR | SSIM | LPIPS | Notes |
|---|---|---|---:|---:|---:|---|
| `1-human` | HQ custom BA + official 3DGS + mask-white | `data/1-human/gs_official_ba_hq_masked/runs/20260629_084243_gaussian_official` | 26.435 | 0.959 | 0.031 | 8 test views |
| `2-human` | HQ custom BA + official 3DGS + mask-white | `data/2-human/gs_official_ba_hq_masked/runs/20260629_085004_gaussian_official` | 28.624 | 0.968 | 0.024 | 8 test views |

Useful negative / support result:

| Scene | Method | Run | PSNR | SSIM | LPIPS | Interpretation |
|---|---|---|---:|---:|---:|---|
| `scene_32` | raw HQ | `data/scene_32/gs_official_raw_hq_768/runs/20260629_092440_gaussian_official` | 18.233 | 0.678 | 0.317 | fewer frames hurt coverage |
| `scene_32` | BA HQ | `data/scene_32/gs_official_ba_hq_768/runs/20260629_092117_gaussian_official` | 19.740 | 0.750 | 0.265 | BA still helps |

## Current Gaps and Invalid Runs

- Dense Geometry Ablation for the final report must be run on the original
  uniform `data/scene` setting, not on `data/scene_selected`. The selected dense
  runs are useful implementation checks, but they should not be used as the main
  I2 ablation conclusion because the frame-selection change confounds the dense
  geometry change.
- The original `data/scene` VGGT cache
  `data/scene/vggt_raw/runs/20260627_074913_vggt_export/vggt_predictions.npz`
  does not contain `point_map`. Therefore it can export only `depth_only` and
  `reprojection_only`. The variants `pointmap_only`, `disagreement_only`, and
  `filtered_full` require a new original-scene VGGT cache with
  `ENABLE_POINT_HEAD=1`.
- A point-head cache attempt on GPU 7,
  `data/scene/vggt_dense_ablation_scene_pointhead_cache/runs/20260630_025621_vggt_export`,
  failed with CUDA OOM under current shared hardware load. At the time GPU 7 had
  only about 5.4 GiB free and two resident processes using about 26 GiB total.
- `data/scene_selected/vggt_improved/runs/20260629_125715_vggt_export` is incomplete for sparse-track final claims: it has `vggt_predictions.npz` and `points3d_dense.ply`, but no `reconstruction.npz` or `summary.json`. It is still valid as the cached VGGT prediction source for dense ablation export because it contains depth, cameras, and point-map predictions.
- That same selected run used `init_points_source=point_head`, which conflicts with the final sparse method definition. Final selected-frame sparse tracks must use `INIT_POINTS_SOURCE=depth`; point head is only a consistency checker for dense filtering.
- `data/scene_selected/vggt_improved_depth_q256/runs/20260629_150641_vggt_export` is a debug artifact only. It lowered `MAX_QUERY_PTS` to 256 and should not be used for final comparison against the uniform baseline.
- The 512-query selected sparse retry `data/scene_selected/vggt_improved_depth_retry512/runs/20260629_151605_vggt_export` matched the uniform baseline tracking parameters (`img_load_resolution=448`, `max_query_pts=512`, `query_frame_num=12`, `fine_tracking=true`, `vis_thresh=0.1`, `max_reproj_error=8.0`, `min_visible_frames=2`) but failed during fine tracking with CUDA OOM. At failure time GPU 6 had a separate resident process using about 6.91 GiB, leaving the run with insufficient headroom. This is a hardware availability blocker, not evidence that the selected-frame method requires lower query count.
- Full sparse selected-frame claims are not report-ready until the 512-query track export, selected BA, and selected sparse official 3DGS runs complete on a GPU with enough free memory.

## Improvement Experiment Matrix

The final improvement branch uses:

```text
data/raw/3_scene.mp4
  -> src.improvement.video_select
  -> data/scene_selected/images
  -> src.vggt_export with depth sparse tracks + point_head cache
  -> dense ablation Reconstructions
  -> custom BA on selected sparse tracks
  -> official 3DGS metrics and render sheets
```

### I1: Video Frame Selection

Purpose: test whether training-free frame selection improves VGGT/BA geometry
and downstream rendering against uniform 64-frame sampling.

| Method | Input | Candidate frames | Final frames | VGGT points | Observations | Raw RMSE | BA RMSE | 3DGS PSNR | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Uniform baseline | `data/scene/images` | - | 64 | 9,760 | 116,725 | 3.816 | 1.577 | 20.729 raw / 22.471 BA | done |
| Selected final | `data/raw/3_scene.mp4` | 192 preferred, 128 fallback | 64 | blocked | blocked | blocked | blocked | see dense results | sparse 512 run OOM on current GPU 6 availability |

Frame selection diagnostics to record:

| Metric | Source |
|---|---|
| decoded frames, FPS, candidate count, final count | `frame_selection_summary.json` |
| selected reliability mean vs candidate mean | `reliability_stats` |
| centrality mean/min/max | `reliability_stats` |
| pose jump mean/P90 and gap edges | `reliability_stats` |
| selected frame indices/timestamps | `final_frames` |

### I2: Dense Geometry Ablation

Purpose: evaluate whether depth/camera dense points and consistency filtering
provide better 3DGS initialization than sparse tracks or direct point maps.

Final-report setting: original uniform `data/scene`, fixed 64 frames, fixed
official 3DGS training setup. All dense variants are exported as
`Reconstruction` `.npz` files with empty observation graphs and can be passed
directly to official 3DGS.

| Variant | Point source | Filtering | Expected output | 3DGS PSNR | SSIM | LPIPS | Status |
|---|---|---|---|---:|---:|---:|---|
| sparse track baseline | VGGT tracker points | visibility + reprojection | `data/scene/vggt_raw/.../reconstruction.npz` | 20.729 | 0.754 | 0.247 | done |
| sparse BA baseline | VGGT tracker points | custom BA | `data/scene/ba_custom/.../reconstruction.npz` | 22.471 | 0.820 | 0.197 | done |
| `depth_only` | depth + camera unprojection | none | `reconstruction_dense_depth_only.npz` | TBD | TBD | TBD | exported, 3DGS pending |
| `pointmap_only` | VGGT direct point map | none | `reconstruction_dense_pointmap_only.npz` | TBD | TBD | TBD | needs original-scene point-head cache |
| `disagreement_only` | depth + camera | point-map disagreement | `reconstruction_dense_disagreement_only.npz` | TBD | TBD | TBD | needs original-scene point-head cache |
| `reprojection_only` | depth + camera | neighbor depth reprojection | `reconstruction_dense_reprojection_only.npz` | TBD | TBD | TBD | exported, 3DGS pending |
| `filtered_full` | depth + camera | disagreement + reprojection voting | `reconstruction_dense_filtered_full.npz` | TBD | TBD | TBD | needs original-scene point-head cache |

Partial original-scene dense export that does not require point maps:

```text
data/scene/vggt_dense_ablation_scene_200k/runs/20260630_025655_vggt_export
```

Partial original-scene dense geometry statistics:

| Variant | Finite points | After disagreement | After reprojection | Output points | Mean kept votes |
|---|---:|---:|---:|---:|---:|
| `depth_only` | 17,172,736 | 17,172,736 | 17,172,736 | 200,000 | 0.000 |
| `reprojection_only` | 17,172,736 | 17,172,736 | 14,092,514 | 200,000 | 2.918 |

Selected-frame dense runs already completed but should be treated as support
material, not as the main dense ablation:

| Variant | Scene | PSNR | SSIM | LPIPS | Run |
|---|---|---:|---:|---:|---|
| `depth_only` | `scene_selected` | 24.834 | 0.859 | 0.141 | `data/scene_selected/gs_official_dense_depth_only/runs/20260629_152539_gaussian_official` |
| `pointmap_only` | `scene_selected` | 24.914 | 0.859 | 0.141 | `data/scene_selected/gs_official_dense_pointmap_only/runs/20260629_152909_gaussian_official` |
| `disagreement_only` | `scene_selected` | 24.953 | 0.860 | 0.139 | `data/scene_selected/gs_official_dense_disagreement_only/runs/20260629_153239_gaussian_official` |
| `reprojection_only` | `scene_selected` | 24.920 | 0.860 | 0.141 | `data/scene_selected/gs_official_dense_reprojection_only/runs/20260629_153608_gaussian_official` |
| `filtered_full` | `scene_selected` | 24.903 | 0.860 | 0.141 | `data/scene_selected/gs_official_dense_filtered_full/runs/20260629_152158_gaussian_official` |

Geometry diagnostics to record from `geometry_filter_*_stats.json`:

| Metric | Meaning |
|---|---|
| `total_pixels` | all dense pixels across selected frames |
| `finite_points` | valid finite dense points before filtering |
| `after_disagreement` | retained by point-map consistency |
| `after_reprojection` | retained by final keep rule |
| `output_points` | sampled output points, capped by `MAX_DENSE_POINTS` |
| `disagreement_threshold`, `reproj_threshold` | adaptive thresholds |
| `points_with_valid_neighbors`, `mean_reproj_votes_kept` | multi-view support |

Selected-frame dense geometry statistics from
`report/metrics/geometry_filter_metrics.csv`:

| Variant | Finite points | After disagreement | After reprojection | Output points | Mean kept votes |
|---|---:|---:|---:|---:|---:|
| `depth_only` | 17,172,736 | 17,172,736 | 17,172,736 | 200,000 | 0.000 |
| `pointmap_only` | 17,172,736 | 17,172,736 | 17,172,736 | 200,000 | 0.000 |
| `disagreement_only` | 17,172,736 | 12,020,915 | 12,020,915 | 200,000 | 0.000 |
| `reprojection_only` | 17,172,736 | 17,172,736 | 14,513,982 | 200,000 | 3.087 |
| `filtered_full` | 17,172,736 | 12,020,915 | 11,068,970 | 200,000 | 3.235 |

Reason output points are identical: every variant keeps far more than the
`MAX_DENSE_POINTS=200000` cap, so the final exported reconstruction is a
confidence-weighted sample of 200,000 points. These experiments compare point
quality under a fixed point budget, not final point count.

### I3: Full Method

Purpose: compare the final combined method against the uniform raw/BA baselines.

| Method | Frames | Cameras | Init points | BA | PSNR | SSIM | LPIPS | Status |
|---|---:|---|---|---|---:|---:|---:|---|
| Uniform + raw sparse | 64 | VGGT raw | sparse tracks | no | 20.729 | 0.754 | 0.247 | done |
| Uniform + BA sparse | 64 | custom BA | sparse tracks | yes | 22.471 | 0.820 | 0.197 | done |
| Selected + raw sparse | 64 | selected VGGT | sparse tracks | no | TBD | TBD | TBD | blocked by 512-query tracking OOM |
| Selected + BA sparse | 64 | selected custom BA | sparse tracks | yes | TBD | TBD | TBD | blocked by selected sparse export |
| Selected + filtered dense | 64 | selected VGGT | filtered dense depth | no | 24.903 | 0.860 | 0.141 | support only; confounded by frame selection |
| Selected + filtered dense + BA cameras | 64 | selected custom BA cameras | filtered dense depth | yes | TBD | TBD | TBD | optional if code path is available |

## Commands

Frame selection:

```bash
CUDA_VISIBLE_DEVICES=0 OVERWRITE=1 CANDIDATE_COUNT=192 FINAL_COUNT=64 \
bash scripts/video_select.sh
```

If the scout VGGT pass cannot fit in memory, rerun with `CANDIDATE_COUNT=128`
and record that as a hardware fallback.

Final selected VGGT export with depth sparse tracks and all dense ablations.
Keep these tracking parameters aligned with the uniform baseline; do not lower
`MAX_QUERY_PTS` for final claims:

```bash
CUDA_VISIBLE_DEVICES=6 \
SCENE_DIR=/home/dhr/cv_final/data/scene_selected \
OUTPUT_ROOT=/home/dhr/cv_final/data/scene_selected/vggt_improved_depth \
INIT_POINTS_SOURCE=depth \
ENABLE_POINT_HEAD=1 \
DENSE_RECONSTRUCTION_VARIANTS=depth_only,pointmap_only,disagreement_only,reprojection_only,filtered_full \
MAX_QUERY_PTS=512 \
IMAGE_RESOLUTION=448 \
MAX_DENSE_POINTS=200000 \
bash scripts/vggt_export.sh
```

If `vggt_predictions.npz` is already valid and contains `point_map`, reuse it
for tracks/export only:

```bash
CUDA_VISIBLE_DEVICES=6 \
SCENE_DIR=/home/dhr/cv_final/data/scene_selected \
OUTPUT_ROOT=/home/dhr/cv_final/data/scene_selected/vggt_improved_depth \
STAGE=tracks \
VGGT_CACHE=/home/dhr/cv_final/data/scene_selected/vggt_improved_depth/runs/<run>/vggt_predictions.npz \
INIT_POINTS_SOURCE=depth \
ENABLE_POINT_HEAD=1 \
DENSE_RECONSTRUCTION_VARIANTS=depth_only,pointmap_only,disagreement_only,reprojection_only,filtered_full \
MAX_QUERY_PTS=512 \
IMAGE_RESOLUTION=448 \
MAX_DENSE_POINTS=200000 \
bash scripts/vggt_export.sh
```

Selected-scene dense export from an existing VGGT cache. This is completed
support material only; it is not the final I2 dense ablation setting:

```bash
CUDA_VISIBLE_DEVICES=6 \
SCENE_DIR=/home/dhr/cv_final/data/scene_selected \
OUTPUT_ROOT=/home/dhr/cv_final/data/scene_selected/vggt_dense_ablation_200k \
STAGE=dense \
VGGT_CACHE=/home/dhr/cv_final/data/scene_selected/vggt_improved/runs/20260629_125715_vggt_export/vggt_predictions.npz \
INIT_POINTS_SOURCE=depth \
ENABLE_POINT_HEAD=1 \
DENSE_RECONSTRUCTION_VARIANTS=depth_only,pointmap_only,disagreement_only,reprojection_only,filtered_full \
MAX_DENSE_POINTS=200000 \
IMAGE_RESOLUTION=448 \
bash scripts/vggt_export.sh
```

Selected-frame BA:

```bash
CUDA_VISIBLE_DEVICES=0 \
INPUT_RECON=/home/dhr/cv_final/data/scene_selected/vggt_improved_depth/runs/<run>/reconstruction.npz \
OUTPUT_ROOT=/home/dhr/cv_final/data/scene_selected/ba_custom_depth \
MAX_NFEV=100 \
bash scripts/ba_run.sh
```

Official 3DGS selected sparse raw / BA:

```bash
CUDA_VISIBLE_DEVICES=0 \
RECONSTRUCTION=/home/dhr/cv_final/data/scene_selected/vggt_improved_depth/runs/<run>/reconstruction.npz \
IMAGE_DIR=/home/dhr/cv_final/data/scene_selected/images \
OUTPUT_ROOT=/home/dhr/cv_final/data/scene_selected/gs_official_selected_raw \
ITERATIONS=10000 RESOLUTION=768 SH_DEGREE=2 TEST_EVERY=8 \
bash scripts/gs_train_official.sh

CUDA_VISIBLE_DEVICES=0 \
RECONSTRUCTION=/home/dhr/cv_final/data/scene_selected/ba_custom_depth/runs/<run>/reconstruction.npz \
IMAGE_DIR=/home/dhr/cv_final/data/scene_selected/images \
OUTPUT_ROOT=/home/dhr/cv_final/data/scene_selected/gs_official_selected_ba \
ITERATIONS=10000 RESOLUTION=768 SH_DEGREE=2 TEST_EVERY=8 \
bash scripts/gs_train_official.sh
```

Official 3DGS dense ablations on the original `data/scene` setting:

```bash
for variant in depth_only pointmap_only disagreement_only reprojection_only filtered_full; do
  CUDA_VISIBLE_DEVICES=6 \
  RECONSTRUCTION=/home/dhr/cv_final/data/scene/vggt_dense_ablation_scene_200k/runs/<run>/reconstruction_dense_${variant}.npz \
  IMAGE_DIR=/home/dhr/cv_final/data/scene/images \
  OUTPUT_ROOT=/home/dhr/cv_final/data/scene/gs_official_dense_${variant} \
  ITERATIONS=10000 RESOLUTION=768 SH_DEGREE=2 TEST_EVERY=8 \
  bash scripts/gs_train_official.sh
done
```

Complete remaining original-scene dense ablation when GPU memory is available:

```bash
# 1. Generate an original-scene VGGT cache with point_map.
CUDA_VISIBLE_DEVICES=<free_gpu> \
SCENE_DIR=/home/dhr/cv_final/data/scene \
OUTPUT_ROOT=/home/dhr/cv_final/data/scene/vggt_dense_ablation_scene_pointhead_cache \
STAGE=vggt \
ENABLE_POINT_HEAD=1 \
IMAGE_RESOLUTION=448 \
bash scripts/vggt_export.sh

# 2. Export all dense variants from that cache.
CUDA_VISIBLE_DEVICES=<free_gpu> \
SCENE_DIR=/home/dhr/cv_final/data/scene \
OUTPUT_ROOT=/home/dhr/cv_final/data/scene/vggt_dense_ablation_scene_200k \
STAGE=dense \
VGGT_CACHE=/home/dhr/cv_final/data/scene/vggt_dense_ablation_scene_pointhead_cache/runs/<run>/vggt_predictions.npz \
INIT_POINTS_SOURCE=depth \
ENABLE_POINT_HEAD=1 \
DENSE_RECONSTRUCTION_VARIANTS=depth_only,pointmap_only,disagreement_only,reprojection_only,filtered_full \
MAX_DENSE_POINTS=200000 \
IMAGE_RESOLUTION=448 \
bash scripts/vggt_export.sh

# 3. Train official 3DGS for each exported variant with the original scene split.
for variant in depth_only pointmap_only disagreement_only reprojection_only filtered_full; do
  CUDA_VISIBLE_DEVICES=<free_gpu> \
  RECONSTRUCTION=/home/dhr/cv_final/data/scene/vggt_dense_ablation_scene_200k/runs/<run>/reconstruction_dense_${variant}.npz \
  IMAGE_DIR=/home/dhr/cv_final/data/scene/images \
  OUTPUT_ROOT=/home/dhr/cv_final/data/scene/gs_official_dense_${variant} \
  ITERATIONS=10000 RESOLUTION=768 SH_DEGREE=2 TEST_EVERY=8 \
  bash scripts/gs_train_official.sh
done
```

Generate report metrics and image sheets from all completed runs:

```bash
bash scripts/report_assets.sh
```

Outputs:

- `report/metrics/official_3dgs_metrics.csv`
- `report/metrics/ba_metrics.csv`
- `report/metrics/vggt_metrics.csv`
- `report/metrics/custom_3dgs_metrics.csv`
- `report/metrics/all_metrics.json`
- `report/metrics/asset_audit.md`
- `report/assets/scene_metric_bars.png`
- `report/assets/selected_dense_metric_bars.png`
- `report/assets/scene_main_render_comparison.png`
- `report/assets/human_render_comparison.png`
- `report/assets/selected_dense_render_comparison.png`

## Acceptance Criteria

- Existing tests pass: `PYTHONPATH=src:vggt python -m pytest src/tests/ -v`.
- Every report claim points to an explicit run directory.
- Every completed 3DGS experiment has `summary.json`, `results.json`,
  `per_view.json`, and `test/ours_10000/{gt,renders}`.
- Every dense ablation has `.npz`, `.ply`, and `geometry_filter_*_stats.json`.
- `report/metrics/asset_audit.md` has no unexpected missing baseline outputs;
  remaining warnings are only for intentionally pending improvement runs.
