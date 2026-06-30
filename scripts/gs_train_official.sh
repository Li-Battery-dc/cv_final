#!/usr/bin/env bash
set -e

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/cv_final}"
VENV_PATH="${VENV_PATH:-$PROJECT_ROOT/.venv}"
OFFICIAL_VENV_PATH="${OFFICIAL_VENV_PATH:-$PROJECT_ROOT/.venv_3dgs}"
OFFICIAL_PYTHON="${OFFICIAL_PYTHON:-$OFFICIAL_VENV_PATH/bin/python}"
SCENE_DIR="${SCENE_DIR:-$PROJECT_ROOT/data/scene}"

RECONSTRUCTION="${RECONSTRUCTION:-}"
SPARSE_DIR="${SPARSE_DIR:-}"
IMAGE_DIR="${IMAGE_DIR:-$SCENE_DIR/images}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCENE_DIR/gs_official_raw}"
OUTPUT_RUN_DIR="${OUTPUT_RUN_DIR:-}"
REPO="${REPO:-$PROJECT_ROOT/gaussian-splatting}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
ITERATIONS="${ITERATIONS:-10000}"
RESOLUTION="${RESOLUTION:-768}"
SH_DEGREE="${SH_DEGREE:-2}"
CAMERA_TYPE="${CAMERA_TYPE:-PINHOLE}"
INIT_MODE="${INIT_MODE:-reconstruction}"
RANDOM_INIT_POINTS="${RANDOM_INIT_POINTS:-15000}"
RANDOM_SEED="${RANDOM_SEED:-42}"
SKIP_RENDER_METRICS="${SKIP_RENDER_METRICS:-0}"
WHITE_BACKGROUND="${WHITE_BACKGROUND:-0}"
MASK_DIR="${MASK_DIR:-}"
MASK_BACKGROUND="${MASK_BACKGROUND:-none}"
MASK_POINTS="${MASK_POINTS:-0}"
MASK_FOREGROUND_THRESHOLD="${MASK_FOREGROUND_THRESHOLD:-0.5}"
MASK_MIN_OBSERVATIONS="${MASK_MIN_OBSERVATIONS:-2}"
MASK_MIN_RATIO="${MASK_MIN_RATIO:-0.5}"
TEST_EVERY="${TEST_EVERY:-}"
TEST_LIST="${TEST_LIST:-}"
TEST_NAMES="${TEST_NAMES:-}"

source "$VENV_PATH/bin/activate"
cd "$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES

if [[ -z "$RECONSTRUCTION" && -z "$SPARSE_DIR" ]]; then
    echo "ERROR: set RECONSTRUCTION to an explicit reconstruction.npz path or SPARSE_DIR to an explicit COLMAP sparse directory." >&2
    exit 2
fi

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
echo "  Mask dir:          $MASK_DIR"
echo "  Mask background:   $MASK_BACKGROUND"
echo "  Mask points:       $MASK_POINTS"
echo "  Test every:        $TEST_EVERY"
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
if [[ "$WHITE_BACKGROUND" == "1" || "$WHITE_BACKGROUND" == "true" || "$WHITE_BACKGROUND" == "TRUE" ]]; then
    CMD+=(--white_background)
fi
if [[ -n "$MASK_DIR" ]]; then
    CMD+=(--mask_dir "$MASK_DIR"
        --mask_background "$MASK_BACKGROUND"
        --mask_foreground_threshold "$MASK_FOREGROUND_THRESHOLD"
        --mask_min_observations "$MASK_MIN_OBSERVATIONS"
        --mask_min_ratio "$MASK_MIN_RATIO")
fi
if [[ "$MASK_POINTS" == "1" || "$MASK_POINTS" == "true" || "$MASK_POINTS" == "TRUE" ]]; then
    CMD+=(--mask_points)
fi
if [[ -n "$TEST_EVERY" ]]; then
    CMD+=(--test_every "$TEST_EVERY")
fi
if [[ -n "$TEST_LIST" ]]; then
    CMD+=(--test_list "$TEST_LIST")
fi
if [[ -n "$TEST_NAMES" ]]; then
    read -r -a TEST_NAME_ARRAY <<< "$TEST_NAMES"
    CMD+=(--test_names "${TEST_NAME_ARRAY[@]}")
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
echo "Run directory is printed above and recorded in run_config.json. latest symlinks are not updated."
