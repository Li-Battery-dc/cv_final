# Experiment Plan and Run Audit

This document is the source of truth for final-report experiments. Use explicit
run directories only. Do not rely on `latest` symlinks; the current scripts print
the created run directory and record it in `run_config.json`.

## 2026-06-30 Reset

The previous sparse VGGT reconstructions used `max_reproj_error=8.0` during
`Reconstruction.from_tracks`. That removed high-error tracker observations
before custom BA, so the reported "BA before" reprojection error was already
measured on a cleaned VGGT observation graph.

For the final report, this pre-filter is removed. New final sparse runs must use:

```text
MAX_REPROJ_ERROR=0.0
```

or omit `MAX_REPROJ_ERROR`, because the default is now `0.0`. In this setting,
VGGT export keeps all observations that pass visibility and minimum-visible-frame
filters. Custom BA then demonstrates its effect through robust optimization and
its own post-round-1 outlier removal.

All old numeric results are obsolete for final claims and must be treated as:

```text
[PLACEHOLDER: rerun required after disabling VGGT reprojection pre-filter]
```

Do not mix old `max_reproj_error=8.0` results with new `max_reproj_error=0.0`
results in the report.

## Experiment Matrix

### Main Scene

Baseline scene: `data/scene/images`, 64 frames uniformly sampled from
`data/raw/3_scene.mp4`. Validation uses every 8th frame as held-out view.

| ID | Reconstruction | Renderer / Trainer | Run | PSNR | SSIM | LPIPS | Status |
|---|---|---|---|---:|---:|---:|---|
| A | VGGT raw, no reproj pre-filter | official 3DGS | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | rerun |
| B | custom BA from A | official 3DGS | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | rerun |
| C | VGGT raw cameras, random init | official 3DGS | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | rerun |
| D | custom BA cameras, random init | official 3DGS | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | rerun |
| E | custom BA | custom 3DGS | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | optional rerun |

BA effect on the same 64-frame scene:

| Stage | Run | Images | Points | Observations | RMSE px | Median px | P90 px | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| VGGT raw | `[PLACEHOLDER]` | 64 | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | rerun |
| custom BA | `[PLACEHOLDER]` | 64 | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | rerun |

### Human Scenes

| Scene | Reconstruction / Renderer | Run | PSNR | SSIM | LPIPS | Status |
|---|---|---|---:|---:|---:|---|
| `1-human` | custom BA + official 3DGS + mask-white | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | rerun |
| `2-human` | custom BA + official 3DGS + mask-white | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | rerun |

Human BA statistics:

| Scene | VGGT points | VGGT observations | BA points | BA observations | RMSE before | RMSE after | P90 before | P90 after | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `1-human` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | rerun |
| `2-human` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | rerun |

Human track-density ablation:

Purpose: check whether human quality changes mainly come from denser tracks
rather than mask compositing alone.

| Scene | Density | MAX_QUERY_PTS | Query frames | Mask background | Mask points | BA RMSE after | PSNR | SSIM | LPIPS | Status |
|---|---|---:|---:|---|---|---:|---:|---:|---:|---|
| `1-human` | low | 512 | 8 | white | optional | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | pending |
| `1-human` | high | 1024 | 16 | white | optional | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | pending |
| `2-human` | low | 512 | 8 | white | optional | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | pending |
| `2-human` | high | 1024 | 16 | white | optional | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | pending |

### Scene 32 Negative Test

This remains useful only if time permits. It should also be rerun with
`MAX_REPROJ_ERROR=0.0`, otherwise it is not comparable to the main scene.

| Scene | Method | Tracking setting | Run | Raw RMSE | PSNR | SSIM | LPIPS | Status |
|---|---|---|---|---:|---:|---:|---:|---|
| `scene_32` | raw HQ | 768 pts x 16 query frames | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | optional rerun |
| `scene_32` | BA HQ | 768 pts x 16 query frames | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | optional rerun |

### Improvement Experiments

The improvement branch uses:

```text
data/raw/3_scene.mp4
  -> src.improvement.video_select
  -> data/scene_selected/images
  -> src.vggt_export with MAX_REPROJ_ERROR=0.0
  -> custom BA on selected sparse tracks
  -> dense ablation Reconstructions
  -> official 3DGS metrics and render sheets
```

I1 video frame selection:

| Method | Input | Candidate frames | Final frames | VGGT points | Observations | Raw RMSE | BA RMSE | 3DGS PSNR | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Uniform baseline | `data/scene/images` | - | 64 | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | rerun |
| Selected final | `data/raw/3_scene.mp4` | `[PLACEHOLDER]` | 64 | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | rerun |

I2 dense geometry ablation:

| Variant | Point source | Filtering | Expected output | PSNR | SSIM | LPIPS | Status |
|---|---|---|---|---:|---:|---:|---|
| sparse track baseline | VGGT tracker points | visibility only + min frames | `reconstruction.npz` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | rerun |
| sparse BA baseline | VGGT tracker points | custom BA | `reconstruction.npz` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | rerun |
| `depth_only` | depth + camera unprojection | none | `reconstruction_dense_depth_only.npz` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | rerun |
| `pointmap_only` | VGGT direct point map | none | `reconstruction_dense_pointmap_only.npz` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | rerun |
| `disagreement_only` | depth + camera | point-map disagreement | `reconstruction_dense_disagreement_only.npz` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | rerun |
| `reprojection_only` | depth + camera | neighbor depth reprojection | `reconstruction_dense_reprojection_only.npz` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | rerun |
| `filtered_full` | depth + camera | disagreement + reprojection voting | `reconstruction_dense_filtered_full.npz` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | rerun |

Full method:

| Method | Frames | Cameras | Init points | BA | PSNR | SSIM | LPIPS | Status |
|---|---:|---|---|---|---:|---:|---:|---|
| Uniform + raw sparse | 64 | VGGT raw | sparse tracks | no | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | rerun |
| Uniform + BA sparse | 64 | custom BA | sparse tracks | yes | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | rerun |
| Selected + raw sparse | 64 | selected VGGT | sparse tracks | no | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | rerun |
| Selected + BA sparse | 64 | selected custom BA | sparse tracks | yes | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | rerun |
| Selected + filtered dense | 64 | selected VGGT | dense filtered | no | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | rerun |

## Rerun Pipeline

Use unique output roots so stale 8px-filtered results cannot be confused with
new runs. Suggested suffix: `_nofilter`.

### 1. Main Scene VGGT Raw

```bash
CUDA_VISIBLE_DEVICES=<gpu> \
SCENE_DIR=/home/dhr/cv_final/data/scene \
OUTPUT_ROOT=/home/dhr/cv_final/data/scene/vggt_raw_nofilter \
STAGE=all \
MAX_REPROJ_ERROR=0.0 \
MAX_QUERY_PTS=512 \
QUERY_FRAME_NUM=12 \
VIS_THRESH=0.1 \
MIN_VISIBLE_FRAMES=2 \
IMAGE_RESOLUTION=448 \
bash scripts/vggt_export.sh
```

Record the printed run directory as:

```text
SCENE_VGGT_RUN=<run_dir>
```

### 2. Main Scene Custom BA

```bash
CUDA_VISIBLE_DEVICES=<gpu> \
INPUT_RECON=<SCENE_VGGT_RUN>/reconstruction.npz \
OUTPUT_ROOT=/home/dhr/cv_final/data/scene/ba_custom_nofilter \
HUBER_DELTA=1.0 \
OUTLIER_THRESHOLD=5.0 \
MAX_NFEV=100 \
bash scripts/ba_run.sh
```

Record:

```text
SCENE_BA_RUN=<run_dir>
```

### 3. Main Scene Official 3DGS

```bash
CUDA_VISIBLE_DEVICES=<gpu> \
RECONSTRUCTION=<SCENE_VGGT_RUN>/reconstruction.npz \
IMAGE_DIR=/home/dhr/cv_final/data/scene/images \
OUTPUT_ROOT=/home/dhr/cv_final/data/scene/gs_official_raw_nofilter \
ITERATIONS=10000 RESOLUTION=768 SH_DEGREE=2 TEST_EVERY=8 \
bash scripts/gs_train_official.sh

CUDA_VISIBLE_DEVICES=<gpu> \
RECONSTRUCTION=<SCENE_BA_RUN>/reconstruction.npz \
IMAGE_DIR=/home/dhr/cv_final/data/scene/images \
OUTPUT_ROOT=/home/dhr/cv_final/data/scene/gs_official_ba_nofilter \
ITERATIONS=10000 RESOLUTION=768 SH_DEGREE=2 TEST_EVERY=8 \
bash scripts/gs_train_official.sh
```

Optional random-init camera ablation:

```bash
CUDA_VISIBLE_DEVICES=<gpu> \
RECONSTRUCTION=<SCENE_VGGT_RUN>/reconstruction.npz \
IMAGE_DIR=/home/dhr/cv_final/data/scene/images \
OUTPUT_ROOT=/home/dhr/cv_final/data/scene/gs_official_random_raw_nofilter \
INIT_MODE=random ITERATIONS=10000 RESOLUTION=768 SH_DEGREE=2 TEST_EVERY=8 \
bash scripts/gs_train_official.sh

CUDA_VISIBLE_DEVICES=<gpu> \
RECONSTRUCTION=<SCENE_BA_RUN>/reconstruction.npz \
IMAGE_DIR=/home/dhr/cv_final/data/scene/images \
OUTPUT_ROOT=/home/dhr/cv_final/data/scene/gs_official_random_ba_nofilter \
INIT_MODE=random ITERATIONS=10000 RESOLUTION=768 SH_DEGREE=2 TEST_EVERY=8 \
bash scripts/gs_train_official.sh
```

### 4. Human Scenes

Run VGGT and BA for each human scene with no reprojection pre-filter:

```bash
for scene in 1-human 2-human; do
  CUDA_VISIBLE_DEVICES=<gpu> \
  SCENE_DIR=/home/dhr/cv_final/data/${scene} \
  OUTPUT_ROOT=/home/dhr/cv_final/data/${scene}/vggt_raw_hq_unmasked_nofilter \
  STAGE=tracks \
  VGGT_CACHE=/home/dhr/cv_final/data/${scene}/vggt_raw/runs/<existing_vggt_cache_run>/vggt_predictions.npz \
  MAX_REPROJ_ERROR=0.0 \
  MAX_QUERY_PTS=1024 \
  QUERY_FRAME_NUM=16 \
  VIS_THRESH=0.1 \
  MIN_VISIBLE_FRAMES=2 \
  IMAGE_RESOLUTION=448 \
  bash scripts/vggt_export.sh
done
```

Then run BA and official 3DGS for each printed VGGT run:

```bash
CUDA_VISIBLE_DEVICES=<gpu> \
INPUT_RECON=<HUMAN_VGGT_RUN>/reconstruction.npz \
OUTPUT_ROOT=/home/dhr/cv_final/data/<scene>/ba_custom_hq_nofilter \
MAX_NFEV=100 \
bash scripts/ba_run.sh

CUDA_VISIBLE_DEVICES=<gpu> \
RECONSTRUCTION=<HUMAN_BA_RUN>/reconstruction.npz \
IMAGE_DIR=/home/dhr/cv_final/data/<scene>/images_masked_white \
OUTPUT_ROOT=/home/dhr/cv_final/data/<scene>/gs_official_ba_hq_masked_nofilter \
ITERATIONS=10000 RESOLUTION=768 SH_DEGREE=2 TEST_EVERY=8 \
bash scripts/gs_train_official.sh
```

If the masked-white image directory name differs, use the actual directory used
by the human preprocessing step and record it in the run notes.

Human track-density rerun template:

```bash
# Low-density example. Change scene to 1-human / 2-human.
CUDA_VISIBLE_DEVICES=<gpu> \
SCENE_DIR=/home/dhr/cv_final/data/<scene> \
OUTPUT_ROOT=/home/dhr/cv_final/data/<scene>/vggt_raw_low_nofilter \
STAGE=tracks \
VGGT_CACHE=/home/dhr/cv_final/data/<scene>/vggt_raw/runs/<existing_vggt_cache_run>/vggt_predictions.npz \
MAX_REPROJ_ERROR=0.0 \
MAX_QUERY_PTS=512 \
QUERY_FRAME_NUM=8 \
VIS_THRESH=0.1 \
MIN_VISIBLE_FRAMES=2 \
IMAGE_RESOLUTION=448 \
bash scripts/vggt_export.sh

# High-density example.
CUDA_VISIBLE_DEVICES=<gpu> \
SCENE_DIR=/home/dhr/cv_final/data/<scene> \
OUTPUT_ROOT=/home/dhr/cv_final/data/<scene>/vggt_raw_high_nofilter \
STAGE=tracks \
VGGT_CACHE=/home/dhr/cv_final/data/<scene>/vggt_raw/runs/<existing_vggt_cache_run>/vggt_predictions.npz \
MAX_REPROJ_ERROR=0.0 \
MAX_QUERY_PTS=1024 \
QUERY_FRAME_NUM=16 \
VIS_THRESH=0.1 \
MIN_VISIBLE_FRAMES=2 \
IMAGE_RESOLUTION=448 \
bash scripts/vggt_export.sh
```

For each low/high VGGT run, continue with BA and official 3DGS:

```bash
# Change <density> to low / high, and use the printed run directory.
CUDA_VISIBLE_DEVICES=<gpu> \
INPUT_RECON=/home/dhr/cv_final/data/<scene>/vggt_raw_<density>_nofilter/runs/<vggt_run>/reconstruction.npz \
OUTPUT_ROOT=/home/dhr/cv_final/data/<scene>/ba_custom_<density>_nofilter \
MAX_NFEV=100 \
bash scripts/ba_run.sh

CUDA_VISIBLE_DEVICES=<gpu> \
RECONSTRUCTION=/home/dhr/cv_final/data/<scene>/ba_custom_<density>_nofilter/runs/<ba_run>/reconstruction.npz \
IMAGE_DIR=/home/dhr/cv_final/data/<scene>/images_masked_white \
OUTPUT_ROOT=/home/dhr/cv_final/data/<scene>/gs_official_ba_<density>_masked_nofilter \
ITERATIONS=10000 RESOLUTION=768 SH_DEGREE=2 TEST_EVERY=8 \
bash scripts/gs_train_official.sh
```

### 5. Optional Scene 32

```bash
CUDA_VISIBLE_DEVICES=<gpu> \
SCENE_DIR=/home/dhr/cv_final/data/scene_32 \
OUTPUT_ROOT=/home/dhr/cv_final/data/scene_32/vggt_raw_hq_768_nofilter \
STAGE=tracks \
VGGT_CACHE=/home/dhr/cv_final/data/scene_32/vggt_raw_hq/runs/<cache_run>/vggt_predictions.npz \
MAX_REPROJ_ERROR=0.0 \
MAX_QUERY_PTS=768 \
QUERY_FRAME_NUM=16 \
VIS_THRESH=0.1 \
MIN_VISIBLE_FRAMES=2 \
IMAGE_RESOLUTION=448 \
bash scripts/vggt_export.sh
```

Then run BA and official 3DGS using the same pattern as the main scene.

### 6. Selected-Frame Sparse Branch

Frame selection:

```bash
CUDA_VISIBLE_DEVICES=<gpu> \
OVERWRITE=1 CANDIDATE_COUNT=192 FINAL_COUNT=64 \
bash scripts/video_select.sh
```

VGGT sparse export with no reprojection pre-filter:

```bash
CUDA_VISIBLE_DEVICES=<gpu> \
SCENE_DIR=/home/dhr/cv_final/data/scene_selected \
OUTPUT_ROOT=/home/dhr/cv_final/data/scene_selected/vggt_improved_depth_nofilter \
INIT_POINTS_SOURCE=depth \
ENABLE_POINT_HEAD=1 \
MAX_REPROJ_ERROR=0.0 \
MAX_QUERY_PTS=512 \
QUERY_FRAME_NUM=12 \
IMAGE_RESOLUTION=448 \
MAX_DENSE_POINTS=200000 \
bash scripts/vggt_export.sh
```

Then run selected BA and selected sparse official 3DGS using the main-scene
commands with selected paths.

### 7. Dense Ablation

Dense variants do not use sparse track observations, but rerun them after the
main sparse baseline so the report has one consistent set of run directories.

```bash
CUDA_VISIBLE_DEVICES=<gpu> \
SCENE_DIR=/home/dhr/cv_final/data/scene \
OUTPUT_ROOT=/home/dhr/cv_final/data/scene/vggt_dense_ablation_scene_pointhead_cache_nofilter \
STAGE=vggt \
ENABLE_POINT_HEAD=1 \
IMAGE_RESOLUTION=448 \
bash scripts/vggt_export.sh

CUDA_VISIBLE_DEVICES=<gpu> \
SCENE_DIR=/home/dhr/cv_final/data/scene \
OUTPUT_ROOT=/home/dhr/cv_final/data/scene/vggt_dense_ablation_scene_200k_nofilter \
STAGE=dense \
VGGT_CACHE=/home/dhr/cv_final/data/scene/vggt_dense_ablation_scene_pointhead_cache_nofilter/runs/<run>/vggt_predictions.npz \
INIT_POINTS_SOURCE=depth \
ENABLE_POINT_HEAD=1 \
DENSE_RECONSTRUCTION_VARIANTS=depth_only,pointmap_only,disagreement_only,reprojection_only,filtered_full \
MAX_DENSE_POINTS=200000 \
IMAGE_RESOLUTION=448 \
bash scripts/vggt_export.sh
```

Train each dense variant:

```bash
for variant in depth_only pointmap_only disagreement_only reprojection_only filtered_full; do
  CUDA_VISIBLE_DEVICES=<gpu> \
  RECONSTRUCTION=/home/dhr/cv_final/data/scene/vggt_dense_ablation_scene_200k_nofilter/runs/<run>/reconstruction_dense_${variant}.npz \
  IMAGE_DIR=/home/dhr/cv_final/data/scene/images \
  OUTPUT_ROOT=/home/dhr/cv_final/data/scene/gs_official_dense_${variant}_nofilter \
  ITERATIONS=10000 RESOLUTION=768 SH_DEGREE=2 TEST_EVERY=8 \
  bash scripts/gs_train_official.sh
done
```

### 8. Regenerate Report Assets

After all run directories are known, update `src/tools/report_assets.py` run specs
to the new `_nofilter` runs, then regenerate:

```bash
bash scripts/report_assets.sh
```

Expected outputs:

- `report/metrics/official_3dgs_metrics.csv`
- `report/metrics/ba_metrics.csv`
- `report/metrics/vggt_metrics.csv`
- `report/metrics/custom_3dgs_metrics.csv`
- `report/metrics/all_metrics.json`
- `report/metrics/asset_audit.md`
- updated figures under `report/assets/`

## Acceptance Criteria

- All final sparse VGGT runs have `max_reproj_error=0.0` in `run_config.json`.
- Every final report claim points to an explicit new run directory.
- BA tables use the new no-pre-filter `rmse_before / median_before / p90_before`.
- 3DGS comparisons use matching raw and BA reconstructions from the same VGGT run.
- Existing tests pass:

```bash
PYTHONPATH=src:vggt python -m pytest src/tests/ -v
```
