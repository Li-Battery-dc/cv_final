# Workspace Guide

## Project Goal

This repository is for the course final project: multi-view 3D reconstruction and Gaussian rendering.

The assignment requirements from `大作业.pdf` are:

- Use scene multi-view images without provided camera calibration.
- Use VGGT to estimate camera parameters and an initial point cloud.
- Implement Bundle Adjustment to optimize camera extrinsics and 3D points.
- Implement 3D Gaussian Splatting optimization and show real-time interactive rendering.
- Research and evaluate an improvement method for VGGT to improve reconstruction accuracy or speed.
- Prepare PPT and defense materials, including implementation explanation, demo, BA impact analysis, improvement experiment analysis, and future research directions.

The expected final pipeline is:

```text
data/scene/images
  -> scripts/vggt_export.py
  -> outputs/vggt_raw/reconstruction.npz
  -> python -m src.ba.run
  -> outputs/ba_custom/reconstruction.npz
  -> python -m src.gaussian.train
  -> outputs/gs_custom_ba
  -> python -m src.gaussian.viewer
```

## Current Implementation State

The repository already has an initial pipeline framework:

- `vggt/`: upstream VGGT code vendored into the repo. Treat it as read-only unless there is a clear reason to patch upstream behavior.
- `scripts/vggt_export.py`: runs VGGT inference, track prediction, filtering, `.npz` export, COLMAP sparse export, and dense PLY export.
- `src/data/reconstruction.py`: central `Reconstruction` dataclass used as the handoff format between VGGT, BA, COLMAP conversion, and 3DGS.
- `src/data/colmap_io.py`: COLMAP import/export compatibility layer.
- `src/ba/`: custom BA implementation.
- `src/gaussian/`: custom Gaussian model, renderer wrapper, trainer, and viewer.
- `docs/design.md`: detailed design, pipeline, and experiment plan.
- `record.md`: short project progress note.

The important architectural choice is that all stages should exchange data through `Reconstruction` and compressed `.npz` files. Avoid adding stage-specific ad hoc formats unless they are external compatibility exports such as COLMAP sparse models or PLY files.

## Reconstruction Format Contract

`src/data/reconstruction.py` is the project-level data contract:

- `image_names`: image filename array.
- `image_size_hw`: original image size as `[height, width]`.
- `intrinsics`: per-image `3x3` camera intrinsics.
- `extrinsics`: per-image OpenCV camera-from-world `[R|t]`.
- `points3d`, `points_rgb`, `points_conf`: sparse point cloud and confidence.
- `obs_camera_id`, `obs_point_id`, `obs_xy`, `obs_conf`: flat observation graph for BA.
- `metadata`: extra stage settings and provenance.

When changing a pipeline stage, preserve this contract unless the change updates all downstream users and tests.

## Development Rules

- Prefer minimal, scoped changes that keep the VGGT -> BA -> 3DGS stages independently runnable.
- Keep upstream `vggt/` code untouched by default. Use wrapper scripts or project modules for integration logic.
- Keep image and geometry coordinate systems explicit. Current extrinsics are OpenCV camera-from-world.
- Keep output directories under `outputs/`; they are generated artifacts.
- Do not commit generated caches, checkpoints, large outputs, or `__pycache__`.
- Use structured geometry code instead of string parsing or loosely shaped arrays.
- Add or update focused tests when changing projection, SO(3), COLMAP conversion, BA packing/unpacking, or renderer-facing data.

## Environment

The project uses `uv` and Python 3.11.

```bash
uv python install 3.11
uv sync --index-url https://download.pytorch.org/whl/cu128
source .venv/bin/activate
```

The local `pyproject.toml` expects `packages/LightGlue` as an editable path dependency. If it is missing, clone or install it before a full sync.

## Common Commands

Run VGGT initialization:

```bash
bash scripts/vggt_export.sh
```

Run the VGGT export script directly:

```bash
python scripts/vggt_export.py \
  --scene_dir data/scene \
  --output_dir outputs/vggt_raw \
  --max_query_pts 2048 \
  --query_frame_num 5 \
  --vis_thresh 0.2 \
  --max_reproj_error 0.0 \
  --min_visible_frames 3
```

Run custom BA:

```bash
python -m src.ba.run \
  --input outputs/vggt_raw/reconstruction.npz \
  --output outputs/ba_custom \
  --huber_delta 1.0 \
  --outlier_threshold 5.0
```

Train custom 3DGS:

```bash
python -m src.gaussian.train \
  --reconstruction outputs/ba_custom/reconstruction.npz \
  --image_dir data/scene/images \
  --output outputs/gs_custom_ba \
  --n_iterations 10000 \
  --resolution 768 432 \
  --sh_degree 2
```

Open the interactive viewer:

```bash
python -m src.gaussian.viewer \
  --checkpoint outputs/gs_custom_ba/checkpoints/latest.pt \
  --port 8080
```

Run tests:

```bash
PYTHONPATH=src:vggt python -m pytest src/tests/ -v
```

Or run individual script-style tests:

```bash
python src/tests/test_so3.py
python src/tests/test_projection.py
python src/tests/test_colmap_roundtrip.py
python src/tests/test_ba_synthetic.py
```

## Experiment Priorities

The defense should have evidence for these comparisons:

- A: VGGT raw reconstruction -> custom 3DGS.
- B: custom BA reconstruction -> custom 3DGS.
- C: PyCOLMAP BA reconstruction -> custom 3DGS, if time permits.
- D: custom BA reconstruction -> official or reference gsplat pipeline, if time permits.
- E: one VGGT improvement experiment, focused on accuracy or speed.

The most important comparison is A vs B, because the assignment explicitly asks to analyze whether BA improves Gaussian Splatting results.

## Reporting Notes

For the PPT and defense, keep track of:

- VGGT output statistics: image count, sparse points, observations, dense points.
- BA statistics: initial/final reprojection error, number of removed outliers, runtime.
- 3DGS metrics: PSNR/SSIM/LPIPS if available, training time, final Gaussian count, visual comparisons.
- Viewer demo path and checkpoint used.
- VGGT improvement method, implementation scope, and measured effect.
