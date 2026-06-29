#!/usr/bin/env bash
set -e

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/cv_final}"
VENV_PATH="${VENV_PATH:-$PROJECT_ROOT/.venv}"
OFFICIAL_VENV_PATH="${OFFICIAL_VENV_PATH:-$PROJECT_ROOT/.venv_3dgs}"
OFFICIAL_PYTHON="${OFFICIAL_PYTHON:-$OFFICIAL_VENV_PATH/bin/python}"
SCENE_DIR="${SCENE_DIR:-$PROJECT_ROOT/data/scene}"

RECONSTRUCTION="${RECONSTRUCTION:-/home/dhr/cv_final/data/scene/vggt_raw/runs/20260627_074913_vggt_export/reconstruction.npz}"
SPARSE_DIR="${SPARSE_DIR:-}"
IMAGE_DIR="${IMAGE_DIR:-$SCENE_DIR/images}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCENE_DIR/gs_official_raw}"
OUTPUT_RUN_DIR="${OUTPUT_RUN_DIR:-}"
REPO="${REPO:-$PROJECT_ROOT/gaussian-splatting}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
ITERATIONS="${ITERATIONS:-10000}"
RESOLUTION="${RESOLUTION:-768}"
SH_DEGREE="${SH_DEGREE:-2}"
CAMERA_TYPE="${CAMERA_TYPE:-PINHOLE}"
INIT_MODE="${INIT_MODE:-reconstruction}"
RANDOM_INIT_POINTS="${RANDOM_INIT_POINTS:-15000}"
RANDOM_SEED="${RANDOM_SEED:-42}"
SKIP_RENDER_METRICS="${SKIP_RENDER_METRICS:-0}"

source "$VENV_PATH/bin/activate"
cd "$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES

echo "============================================"
echo "  Official Gaussian Splatting"
echo "============================================"
echo "  Repo:              $REPO"
echo "  Project venv:      $VENV_PATH"
echo "  Official Python:   $OFFICIAL_PYTHON"
echo "  Reconstruction:    $RECONSTRUCTION"
echo "  Sparse dir:        $SPARSE_DIR"
echo "  Image dir:         $IMAGE_DIR"
echo "  Output root:       $OUTPUT_ROOT"
echo "  CUDA devices:      $CUDA_VISIBLE_DEVICES"
echo "  Iterations:        $ITERATIONS"
echo "  Resolution:        $RESOLUTION"
echo "  SH degree:         $SH_DEGREE"
echo "  Init mode:         $INIT_MODE"
echo "  Random points:     $RANDOM_INIT_POINTS"
echo "============================================"

CMD=(python -m src.gaussian.official_train
    --repo "$REPO"
    --official_python "$OFFICIAL_PYTHON"
    --image_dir "$IMAGE_DIR"
    --output "$OUTPUT_ROOT"
    --camera_type "$CAMERA_TYPE"
    --init_mode "$INIT_MODE"
    --random_init_points "$RANDOM_INIT_POINTS"
    --random_seed "$RANDOM_SEED"
    --iterations "$ITERATIONS"
    --resolution "$RESOLUTION"
    --sh_degree "$SH_DEGREE")

if [[ -n "$OUTPUT_RUN_DIR" ]]; then
    CMD+=(--output_run_dir "$OUTPUT_RUN_DIR")
fi

if [[ -n "$RECONSTRUCTION" ]]; then
    CMD+=(--reconstruction "$RECONSTRUCTION")
fi
if [[ -n "$SPARSE_DIR" ]]; then
    CMD+=(--sparse_dir "$SPARSE_DIR")
fi
if [[ "$SKIP_RENDER_METRICS" == "1" || "$SKIP_RENDER_METRICS" == "true" || "$SKIP_RENDER_METRICS" == "TRUE" ]]; then
    CMD+=(--skip_render_metrics)
fi

"${CMD[@]}"

echo ""
echo "Done! Output root: $OUTPUT_ROOT"
echo "  - prepared_scene/"
echo "  - summary.json"
echo "  - results.json / per_view.json (if metrics ran)"
echo "Latest run: $OUTPUT_ROOT/latest"
