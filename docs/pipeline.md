# Pipeline Guide

## Goal

Build a full office-scene pipeline:

```text
scene images
  -> VGGT initialization / tracking
  -> sparse reconstruction (.npz + COLMAP sparse)
  -> bundle adjustment
  -> Gaussian Splatting training
  -> interactive / headless rendering
```

The project keeps one internal handoff format:

- [src/data/reconstruction.py](/home/dhr/cv_final/src/data/reconstruction.py)
- `Reconstruction` is the contract between VGGT, BA, COLMAP conversion, and 3DGS.

## Output Convention

Each stage writes to its stage root with timestamped runs created in Python:

```text
<stage_root>/
  latest -> runs/<timestamp>_<stage_name>
  runs/
    <timestamp>_<stage_name>/
```

Typical stage roots:

- `data/scene/vggt_raw`
- `data/scene/ba_custom`
- `data/scene/ba_pycolmap`
- `data/scene/gs_custom_ba`

## Stage Layout

### 1. VGGT export

Main code:

- [src/vggt_export.py](/home/dhr/cv_final/src/vggt_export.py)
- [scripts/vggt_export.sh](/home/dhr/cv_final/scripts/vggt_export.sh)

Outputs:

- `reconstruction.npz`
- `sparse/`
- `vggt_predictions.npz`
- `points3d_dense.ply`
- `run_config.json`
- `summary.json`

Notes:

- Reconstruction observations are stored in the VGGT image-load coordinate system.
- COLMAP export now goes through [src/data/colmap_io.py](/home/dhr/cv_final/src/data/colmap_io.py), which converts to original image coordinates using `metadata["original_coords"]` and `metadata["img_load_resolution"]`.

### 2. Custom BA

Main code:

- [src/ba/run.py](/home/dhr/cv_final/src/ba/run.py)
- [scripts/ba_run.sh](/home/dhr/cv_final/scripts/ba_run.sh)

Outputs:

- `reconstruction.npz`
- `ba_stats.json`
- `sparse/`
- `run_config.json`
- `summary.json`

Current solver:

- SciPy `least_squares`
- Huber loss
- two rounds with optional outlier filtering
- first two cameras fixed as gauge anchor

### 3. PyCOLMAP BA baseline

Main code:

- [src/ba/pycolmap_run.py](/home/dhr/cv_final/src/ba/pycolmap_run.py)
- [scripts/ba_pycolmap.sh](/home/dhr/cv_final/scripts/ba_pycolmap.sh)

Notes:

- Uses PyCOLMAP + Ceres as a stronger BA baseline.
- Keeps first two camera poses fixed.
- Keeps intrinsics fixed by default.
- Supports an optional second round after outlier filtering.

### 4. Custom Gaussian Splatting

Main code:

- [src/gaussian/train.py](/home/dhr/cv_final/src/gaussian/train.py)
- [src/gaussian/trainer.py](/home/dhr/cv_final/src/gaussian/trainer.py)
- [src/gaussian/model.py](/home/dhr/cv_final/src/gaussian/model.py)
- [scripts/gs_train.sh](/home/dhr/cv_final/scripts/gs_train.sh)

Outputs:

- `checkpoints/latest.pt`
- `final.ply`
- `validation/*.png`
- `metrics.json`
- `run_config.json`
- `summary.json`

Recent implementation updates:

- Densification thresholds are now relative to the scene scale reference instead of hard-coded world units.
- Resume now continues from the stored iteration instead of restarting from 1.
- Added `init_mode=random` for a random-initialization baseline.

### 5. Official 3DGS baseline wrapper

Main code:

- [src/gaussian/official_train.py](/home/dhr/cv_final/src/gaussian/official_train.py)
- [scripts/gs_train_official.sh](/home/dhr/cv_final/scripts/gs_train_official.sh)
- [scripts/install_external_baselines.sh](/home/dhr/cv_final/scripts/install_external_baselines.sh)

External sources:

- `gaussian-splatting`
- `packages/pycolmap`

The wrapper keeps the project handoff format as `reconstruction.npz`. It uses
the project environment to convert `Reconstruction` into a temporary
COLMAP-style scene, then calls the official 3DGS repository with
`OFFICIAL_PYTHON` from a separate environment, for example
`$PROJECT_ROOT/.venv-3dgs/bin/python`.

## Metadata Files

Each run directory keeps two small JSON files:

- `run_config.json`: key parameters and input paths for that run
- `summary.json`: output paths and main metrics
