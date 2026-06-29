# Experiment Plan

## Baseline Results

Baseline scene: `data/scene/images`, 64 frames uniformly sampled from
`data/raw/3_scene.mp4`. Unless otherwise stated, validation uses every 8th
frame as held-out view.

| ID | Reconstruction | Renderer / Trainer | PSNR | SSIM | LPIPS | Notes |
|---|---|---|---:|---:|---:|---|
| A | VGGT raw | official 3DGS | 20.729 | 0.754 | 0.247 | no BA |
| B | custom BA | official 3DGS | 22.471 | 0.820 | 0.197 | main BA baseline |
| C | VGGT raw | official 3DGS random init | 20.317 | 0.742 | 0.264 | no geometry init |
| D | custom BA | official 3DGS random init | 21.983 | 0.806 | 0.215 | camera benefit without point init |
| E | custom BA | custom 3DGS | 16.399 | 0.570 | - | self renderer baseline |

BA effect on the same 64-frame scene:

| Stage | Images | Points | Observations | RMSE px | Median px | P90 px | Runtime |
|---|---:|---:|---:|---:|---:|---:|---:|
| VGGT raw | 64 | 9,760 | 116,725 | 3.816 | 2.801 | 6.423 | VGGT 19.7s + track 34.1s |
| custom BA | 64 | 9,679 | 112,366 | 1.577 | 0.792 | 2.788 | 122.1s |

## Improvement Experiments

The improvement branch uses `data/raw/3_scene.mp4` as input and writes selected
frames to `data/scene_selected/images`.

```text
data/raw/3_scene.mp4
  -> python -m src.improvement.video_select
  -> data/scene_selected/images
  -> python -m src.vggt_export --enable_point_head --save_dense_filtered_reconstruction
  -> sparse track reconstruction for BA
  -> dense filtered reconstruction for 3DGS initialization
```

### I1: Video Frame Selection

Purpose: test whether training-free frame selection improves VGGT camera and
track quality compared with uniform 64-frame sampling.

| Method | Input frames | Candidate frames | Final frames | VGGT points | Observations | RMSE px | P90 px | 3DGS PSNR | 3DGS SSIM | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Uniform baseline | existing `data/scene/images` | - | 64 | 9,760 | 116,725 | 3.816 | 6.423 | 20.729 | 0.754 | done |
| Quality + VGGT selected | `data/raw/3_scene.mp4` | 192 | 64 | TBD | TBD | TBD | TBD | TBD | TBD | pending |

Primary artifacts after running:

- `data/scene_selected/frame_selection_candidates.json`
- `data/scene_selected/frame_selection_summary.json`
- `data/scene_selected/images/*.jpg`
- `data/scene_selected/vggt_improved/.../reconstruction.npz`

Record these frame-selection diagnostics:

| Metric | Value |
|---|---:|
| decoded video frames | TBD |
| candidate frames | 192 |
| final frames | 64 |
| mean selected centrality | TBD |
| mean candidate centrality | TBD |
| pose jump P90 before selection | TBD |
| pose jump P90 after selection | TBD |

### I2: Dense Geometry Initialization and Filtering

Purpose: compare VGGT sparse tracks, depth-camera dense points, point-map dense
points, and filtered depth-camera dense points for 3DGS initialization.

| Method | Main point source | Filtering | Points before sampling | Init points | 3DGS PSNR | 3DGS SSIM | LPIPS | Notes | Status |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| Sparse track baseline | VGGT tracks | reprojection filtering | 9,760 | 9,760 | 20.729 | 0.754 | 0.247 | original sparse baseline | done |
| Depth only | depth + camera | none | TBD | 200k max | TBD | TBD | TBD | tests depth-camera representation | pending |
| Point map only | point map | none | TBD | 200k max | TBD | TBD | TBD | tests direct point-map quality | pending |
| Depth filtered | depth + camera | depth-point disagreement + reprojection voting | TBD | 200k max | TBD | TBD | TBD | tests consistency filtering benefit | pending |

Primary geometry-filter artifacts:

- `reconstruction_dense_filtered.npz`
- `points3d_dense_filtered.ply`
- `geometry_filter_stats.json`

Record these filtering diagnostics:

| Metric | Value |
|---|---:|
| total dense pixels | TBD |
| finite depth points | TBD |
| after depth-point disagreement | TBD |
| after reprojection voting | TBD |
| sampled output points | 200,000 max |
| disagreement percentile | 70 |
| reprojection percentile | 70 |
| min neighbor votes | 1 |

Current implementation note:

- `filtered dense` is already exported as a ready-to-train dense reconstruction.
- `pointmap-only init` does not yet have a standalone dense reconstruction export command.
- `depth-only dense init` does not yet have a standalone dense reconstruction export command.

### I3: Full Method

Purpose: evaluate the final method that combines both improvements:

1. video frame selection from 192 candidates to 64 selected frames;
2. final VGGT rerun on selected frames;
3. depth-camera point construction;
4. point-map disagreement filtering;
5. neighbor-view reprojection voting;
6. 3DGS initialization from filtered dense points.

| Method | Frames | Cameras | Init point source | BA | 3DGS PSNR | SSIM | LPIPS | Status |
|---|---:|---|---|---|---:|---:|---:|---|
| Uniform + raw track | 64 | VGGT raw | sparse tracks | no | 20.729 | 0.754 | 0.247 | done |
| Uniform + BA | 64 | custom BA | sparse tracks | yes | 22.471 | 0.820 | 0.197 | done |
| Selected + raw track | 64 | VGGT selected | sparse tracks | no | TBD | TBD | TBD | pending |
| Selected + BA | 64 | custom BA | sparse tracks | yes | TBD | TBD | TBD | pending |
| Selected + filtered dense | 64 | VGGT selected or BA | filtered depth dense | optional | TBD | TBD | TBD | pending |

## Commands

Frame selection:

```bash
.venv/bin/python -m src.improvement.video_select \
  --video data/raw/3_scene.mp4 \
  --output_scene_dir data/scene_selected \
  --candidate_count 192 \
  --final_count 64 \
  --overwrite
```

Final VGGT export with point-map cache and dense geometry filtering:

```bash
INIT_POINTS_SOURCE=depth ENABLE_POINT_HEAD=1 SAVE_DENSE_FILTERED_RECONSTRUCTION=1 \
SCENE_DIR=/home/dhr/cv_final/data/scene_selected \
OUTPUT_ROOT=/home/dhr/cv_final/data/scene_selected/vggt_improved \
bash scripts/vggt_export.sh
```

This keeps sparse reconstruction and BA on the `depth`-initialized VGGSfM track
path, while enabling `point_head` only for consistency checking and dense 3DGS
initialization.

BA on selected-frame sparse reconstruction:

```bash
.venv/bin/python -m src.ba.run \
  --input data/scene_selected/vggt_improved/runs/<run>/reconstruction.npz \
  --output data/scene_selected/ba_custom
```

3DGS from filtered dense reconstruction:

```bash
.venv/bin/python -m src.gaussian.train \
  --reconstruction data/scene_selected/vggt_improved/runs/<run>/reconstruction_dense_filtered.npz \
  --image_dir data/scene_selected/images \
  --output data/scene_selected/gs_filtered_dense \
  --max_init_gaussians 200000
```

## Metrics

- Frame selection: selected frame indices, centrality distribution, redundancy,
  pose jump P90, final camera path visualization.
- VGGT/BA: point count, observation count, RMSE, median, P90, removed outliers,
  runtime.
- Geometry filtering: dense pixels, finite points, retained ratio after each
  filter, final sampled points, disagreement threshold, reprojection threshold,
  neighbor-vote distribution.
- 3DGS: PSNR, SSIM, LPIPS, training time, final Gaussian count, validation sheet,
  visible floating artifacts.

## Priority

1. Fill I1 selected-frame VGGT and BA statistics.
2. Fill I2 dense filtering statistics and point-cloud screenshots.
3. Run I2 depth-only / pointmap-only / filtered-depth 3DGS comparison.
4. Run I3 full method against the existing uniform + BA baseline.
