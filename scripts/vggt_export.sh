#!/usr/bin/env bash
set -e

# VGGT Export Shell Script
# Runs VGGT inference + track prediction + COLMAP export on the office scene.

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/cv_final}"
VENV_PATH="${VENV_PATH:-$PROJECT_ROOT/.venv}"
SCENE_DIR="${SCENE_DIR:-$PROJECT_ROOT/data/scene}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCENE_DIR/vggt_raw}"

MAX_QUERY_PTS="${MAX_QUERY_PTS:-4096}"
QUERY_FRAME_NUM="${QUERY_FRAME_NUM:-8}"
VIS_THRESH="${VIS_THRESH:-0.2}"
MAX_REPROJ_ERROR="${MAX_REPROJ_ERROR:-8.0}"
MIN_VISIBLE_FRAMES="${MIN_VISIBLE_FRAMES:-2}"
CAMERA_TYPE="${CAMERA_TYPE:-SIMPLE_PINHOLE}"

source "$VENV_PATH/bin/activate"
cd "$PROJECT_ROOT"

echo "============================================"
echo "  VGGT Export Pipeline"
echo "============================================"
echo "  Scene dir:      $SCENE_DIR"
echo "  Output dir:     $OUTPUT_DIR"
echo "  Query frames:   $QUERY_FRAME_NUM"
echo "  Max query pts:  $MAX_QUERY_PTS"
echo "  Vis thresh:     $VIS_THRESH"
echo "  Reproj error:   $MAX_REPROJ_ERROR"
echo "  Min vis frames: $MIN_VISIBLE_FRAMES"
echo "============================================"

python scripts/vggt_export.py \
    --scene_dir "$SCENE_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --max_query_pts "$MAX_QUERY_PTS" \
    --query_frame_num "$QUERY_FRAME_NUM" \
    --vis_thresh "$VIS_THRESH" \
    --max_reproj_error "$MAX_REPROJ_ERROR" \
    --min_visible_frames "$MIN_VISIBLE_FRAMES" \
    --camera_type "$CAMERA_TYPE" \
    --fine_tracking

echo ""
echo "Done! Output saved to: $OUTPUT_DIR"
echo "  - reconstruction.npz  (unified format)"
echo "  - sparse/             (COLMAP cameras/images/points3D.bin)"
echo "  - points3d_dense.ply  (dense point cloud)"
