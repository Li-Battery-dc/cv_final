#!/usr/bin/env bash
set -e

# VGGT Export Shell Script
# Runs VGGT inference + track prediction + COLMAP export on the office scene.

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/cv_final}"
VENV_PATH="${VENV_PATH:-$PROJECT_ROOT/.venv}"
SCENE_DIR="${SCENE_DIR:-$PROJECT_ROOT/data/scene}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCENE_DIR/vggt_raw}"
OUTPUT_RUN_DIR="${OUTPUT_RUN_DIR:-}"

# 分阶段跑省显存
STAGE="${STAGE:-all}" # tracks, vggt, or all
VGGT_CACHE="${VGGT_CACHE:-}"

MAX_QUERY_PTS="${MAX_QUERY_PTS:-512}"
QUERY_FRAME_NUM="${QUERY_FRAME_NUM:-12}"
VIS_THRESH="${VIS_THRESH:-0.1}"
MAX_REPROJ_ERROR="${MAX_REPROJ_ERROR:-8.0}"
MIN_VISIBLE_FRAMES="${MIN_VISIBLE_FRAMES:-2}"
CAMERA_TYPE="${CAMERA_TYPE:-PINHOLE}"
FINE_TRACKING="${FINE_TRACKING:-1}"

IMAGE_RESOLUTION="${IMAGE_RESOLUTION:-448}"

source "$VENV_PATH/bin/activate"
cd "$PROJECT_ROOT"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export CUDA_VISIBLE_DEVICES

PREV_VGGT_CACHE="$OUTPUT_ROOT/latest/vggt_predictions.npz"
if [[ ! -e "$PREV_VGGT_CACHE" ]]; then
    PREV_VGGT_CACHE="$OUTPUT_ROOT/vggt_predictions.npz"
fi
if [[ -z "$VGGT_CACHE" ]]; then
    if [[ "$STAGE" == "tracks" && -e "$PREV_VGGT_CACHE" ]]; then
        VGGT_CACHE="$PREV_VGGT_CACHE"
    else
        VGGT_CACHE=""
    fi
fi

echo "============================================"
echo "  VGGT Export Pipeline"
echo "============================================"
echo "  Scene dir:      $SCENE_DIR"
echo "  Output root:    $OUTPUT_ROOT"
echo "  Stage:          $STAGE"
echo "  VGGT cache:     $VGGT_CACHE"
echo "  Image res:      $IMAGE_RESOLUTION"
echo "  Query frames:   $QUERY_FRAME_NUM"
echo "  Max query pts:  $MAX_QUERY_PTS"
echo "  Fine tracking:  $FINE_TRACKING"
echo "  Vis thresh:     $VIS_THRESH"
echo "  Reproj error:   $MAX_REPROJ_ERROR"
echo "  Min vis frames: $MIN_VISIBLE_FRAMES"
echo "============================================"

CMD=(python -m src.vggt_export
    --scene_dir "$SCENE_DIR"
    --output_dir "$OUTPUT_ROOT"
    --stage "$STAGE"
    --img_load_resolution "$IMAGE_RESOLUTION"
    --max_query_pts "$MAX_QUERY_PTS"
    --query_frame_num "$QUERY_FRAME_NUM"
    --vis_thresh "$VIS_THRESH"
    --max_reproj_error "$MAX_REPROJ_ERROR"
    --min_visible_frames "$MIN_VISIBLE_FRAMES"
    --conf_thres_value 2.0 # dense point export confidence threshold, 2.0 for this specific scene. 
    --camera_type "$CAMERA_TYPE")

if [[ -n "$OUTPUT_RUN_DIR" ]]; then
    CMD+=(--output_run_dir "$OUTPUT_RUN_DIR")
fi
if [[ -n "$VGGT_CACHE" ]]; then
    CMD+=(--vggt_cache "$VGGT_CACHE")
fi
if [[ "$FINE_TRACKING" = "1" ]]; then
    CMD+=(--fine_tracking)
else
    CMD+=(--no-fine_tracking)
fi

"${CMD[@]}"

echo ""
echo "Done! Output root: $OUTPUT_ROOT"
echo "Latest run: $OUTPUT_ROOT/latest"
