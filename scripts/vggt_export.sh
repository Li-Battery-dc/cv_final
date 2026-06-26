#!/usr/bin/env bash
set -e

# VGGT Export Shell Script
# Runs VGGT inference + track prediction + COLMAP export on the office scene.

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/cv_final}"
VENV_PATH="${VENV_PATH:-$PROJECT_ROOT/.venv}"
SCENE_DIR="${SCENE_DIR:-$PROJECT_ROOT/data/scene}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCENE_DIR/vggt_raw}"

# 分阶段跑省显存
STAGE="${STAGE:-all}"
VGGT_CACHE="${VGGT_CACHE:-$OUTPUT_DIR/vggt_predictions.npz}"

MAX_QUERY_PTS="${MAX_QUERY_PTS:-2048}"
QUERY_FRAME_NUM="${QUERY_FRAME_NUM:-5}"
VIS_THRESH="${VIS_THRESH:-0.2}"
MAX_REPROJ_ERROR="${MAX_REPROJ_ERROR:-8.0}"
MIN_VISIBLE_FRAMES="${MIN_VISIBLE_FRAMES:-2}"
CAMERA_TYPE="${CAMERA_TYPE:-SIMPLE_PINHOLE}"
FINE_TRACKING="${FINE_TRACKING:-1}"

IMAGE_RESOLUTION="${IMAGE_RESOLUTION:-512}"

source "$VENV_PATH/bin/activate"
cd "$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}

echo "============================================"
echo "  VGGT Export Pipeline"
echo "============================================"
echo "  Scene dir:      $SCENE_DIR"
echo "  Output dir:     $OUTPUT_DIR"
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

python -m src.vggt_export \
    --scene_dir "$SCENE_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --stage "$STAGE" \
    --vggt_cache "$VGGT_CACHE" \
    --img_load_resolution "$IMAGE_RESOLUTION" \
    --max_query_pts "$MAX_QUERY_PTS" \
    --query_frame_num "$QUERY_FRAME_NUM" \
    --vis_thresh "$VIS_THRESH" \
    --max_reproj_error "$MAX_REPROJ_ERROR" \
    --min_visible_frames "$MIN_VISIBLE_FRAMES" \
    --camera_type "$CAMERA_TYPE" \
    $(if [ "$FINE_TRACKING" = "1" ]; then echo "--fine_tracking"; else echo "--no-fine_tracking"; fi)

echo ""
echo "Done! Output saved to: $OUTPUT_DIR"
